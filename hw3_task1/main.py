"""
主程序模块 - 机器人控制系统的入口点
集成了AI模型、仿真环境、机器人API和用户交互界面
"""
import os
import signal
import sys
import time
import threading
import re
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
except ImportError:
    print("警告: 未安装google-generativeai包，AI功能将不可用")
    genai = None

from simulation import Simulation
from robot_apis import RobotAPI

class CodeExecutor:
    """安全的代码执行器"""
    
    def __init__(self, available_functions: Dict[str, Any]):
        self.available_functions = available_functions
        self.execution_timeout = 30.0
        
        # 禁用的导入和函数
        self.restricted_imports = [
            "os", "sys", "subprocess", "shutil", "glob",
            "socket", "urllib", "requests", "ftplib",
            "smtplib", "telnetlib", "threading", "multiprocessing"
        ]
    
    def _validate_code_safety(self, code: str) -> bool:
        """验证代码安全性"""
        # 检查禁用的导入
        for restricted in self.restricted_imports:
            if re.search(rf'\b{re.escape(restricted)}\b', code):
                print(f"检测到禁用的导入或函数: {restricted}")
                return False
        
        # 检查危险的函数调用
        dangerous_patterns = [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\s*\(',
            r'\bopen\s*\(',
            r'\bfile\s*\(',
            r'\binput\s*\(',
            r'\braw_input\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                print(f"检测到危险的函数调用: {pattern}")
                return False
        
        return True
    
    def execute_code(self, code: str) -> tuple[bool, Optional[str]]:
        """安全执行代码"""
        try:
            # 验证代码安全性
            if not self._validate_code_safety(code):
                return False, "代码包含不安全的操作"
            
            # 创建受限的执行环境
            restricted_builtins = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                }
            }
            
            # 合并可用函数
            execution_globals = {**restricted_builtins, **self.available_functions}
            
            # 在超时限制内执行代码
            self._execute_with_timeout(code, execution_globals)
            
            print(f"代码执行成功: {code[:50]}...")
            return True, None
            
        except Exception as e:
            error_msg = f"代码执行失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def _execute_with_timeout(self, code: str, globals_dict: Dict[str, Any]):
        """在超时限制内执行代码"""
        result = [None]
        exception = [None]
        
        def execute_target():
            try:
                exec(code, globals_dict)
                result[0] = "success"
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=execute_target, daemon=True)
        thread.start()
        thread.join(timeout=self.execution_timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"代码执行超时: {self.execution_timeout}秒")
        
        if exception[0]:
            raise exception[0]

class AIModelManager:
    """AI模型管理器"""
    
    def __init__(self):
        self.model = None
        self.generation_count = 0
        self.model_name = "gemini-2.5-flash"
        self.temperature = 0.3
        self.max_output_tokens = 2048
        self.max_retry_attempts = 3
        
    def initialize_model(self) -> bool:
        """初始化AI模型"""
        if genai is None:
            print("Google Generative AI库未安装，无法使用AI功能")
            return False
            
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("请设置GEMINI_API_KEY环境变量")
                return False
                
            genai.configure(api_key=api_key)
            
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }
            
            self.model = genai.GenerativeModel(
                self.model_name,
                generation_config=generation_config
            )
            
            print(f"AI模型 {self.model_name} 初始化成功")
            return True
            
        except Exception as e:
            print(f"AI模型初始化失败: {e}")
            return False
    
    def generate_code(self, instruction: str) -> str:
        """生成策略代码"""
        if not self.model:
            raise Exception("AI模型未初始化")
        
        prompt = self._build_enhanced_prompt(instruction)
        
        for attempt in range(self.max_retry_attempts):
            try:
                print(f"向AI模型发送请求 (尝试 {attempt + 1}/{self.max_retry_attempts})...")
                
                response = self.model.generate_content(prompt)
                
                if not response.text:
                    raise Exception("模型返回空响应")
                
                self.generation_count += 1
                
                # 提取代码
                code = self._extract_code_from_response(response.text)
                
                if code:
                    print(f"成功生成代码 ({len(code)} 字符)")
                    return code
                else:
                    print("响应中未找到有效代码")
                    
            except Exception as e:
                print(f"AI生成代码失败 (尝试 {attempt + 1}): {e}")
                if attempt == self.max_retry_attempts - 1:
                    raise Exception(f"所有重试都失败了: {e}")
                time.sleep(1)
        
        raise Exception("生成代码失败")
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """从AI响应中提取代码"""
        # 查找代码块
        code_patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'`([^`\n]+)`'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                # 返回最长的匹配
                return max(matches, key=len).strip()
        
        # 如果没有找到代码块，尝试其他方法
        lines = response_text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(func) for func in 
                   ['detect_objects', 'pick_place', 'move_to_position', 'is_empty', 
                    'get_object_position', 'stack_objects', 'clear_container']):
                in_code = True
                code_lines.append(line)
            elif in_code and (stripped == '' or stripped.startswith(' ')):
                code_lines.append(line)
            elif in_code and not stripped.startswith('#'):
                break
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return ""
    
    def _build_enhanced_prompt(self, user_instruction: str) -> str:
        """构建增强的提示词"""
        prompt = f"""
你是一个机器人控制专家。你需要根据用户的指令，生成Python代码来控制机器人完成任务。

可用的机器人API函数：

感知函数：
- detect_objects(object_type) -> List[str]: 检测特定类型的物体，返回名称列表
- is_empty(container_name) -> bool: 检查容器是否为空
- get_object_position(object_name) -> Tuple[x, y, z]: 获取物体位置
- count_objects(object_type=None) -> int: 计算物体数量

控制函数：
- pick_place(object_to_move, target_location) -> bool: 将物体放置到目标位置上
- move_to_position(object_name, (x, y, z)) -> bool: 移动物体到绝对坐标
- stack_objects(objects_list, base_location) -> bool: 堆叠多个物体
- clear_container(container_name) -> bool: 清空容器

场景中的物体类型：
- "block" 类型: red_block, blue_block, green_block
- "bowl" 类型: tray

用户指令: {user_instruction}

请生成Python代码来完成这个任务。只返回代码，不要解释。代码应该：
1. 使用上述API函数
2. 包含适当的错误检查
3. 逻辑清晰，步骤明确

代码：
```python
"""
        
        return prompt
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        return {
            "generation_count": self.generation_count,
            "model_name": self.model_name,
            "is_initialized": self.model is not None
        }

class RobotControlSystem:
    """机器人控制系统主类"""
    
    def __init__(self):
        self.simulation = None
        self.robot_api = None
        self.ai_manager = AIModelManager()
        self.code_executor = None
        self.running = False
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在关闭系统...")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """初始化系统"""
        print("初始化机器人控制系统...")
        
        try:
            # 初始化仿真环境
            print("初始化仿真环境...")
            self.simulation = Simulation(use_gui=True)
            
            if not self.simulation.connect():
                print("连接仿真器失败")
                return False
            
            if not self.simulation.setup_scene():
                print("设置仿真场景失败")
                return False
            
            # 初始化机器人API
            print("初始化机器人API...")
            self.robot_api = RobotAPI(self.simulation)
            
            # 设置可用函数
            available_functions = self._setup_available_functions()
            self.code_executor = CodeExecutor(available_functions)
            
            # 初始化AI模型
            print("初始化AI模型...")
            ai_success = self.ai_manager.initialize_model()
            if not ai_success:
                print("AI模型初始化失败，但系统仍可手动使用")
            
            # 启动后台仿真
            self.simulation.start_background_simulation()
            
            print("系统初始化完成！")
            return True
            
        except Exception as e:
            print(f"系统初始化失败: {e}")
            return False
    
    def _setup_available_functions(self) -> Dict[str, Any]:
        """设置代码执行时可用的函数"""
        return {
            # 机器人API函数
            'detect_objects': self.robot_api.detect_objects,
            'is_empty': self.robot_api.is_empty,
            'get_object_position': self.robot_api.get_object_position,
            'get_objects_in_area': self.robot_api.get_objects_in_area,
            'count_objects': self.robot_api.count_objects,
            'pick_place': self.robot_api.pick_place,
            'move_to_position': self.robot_api.move_to_position,
            'stack_objects': self.robot_api.stack_objects,
            'clear_container': self.robot_api.clear_container,
            'list_available_objects': self.robot_api.list_available_objects,
            'reset_environment': self.robot_api.reset_environment,
        }
    
    def run_interactive_mode(self):
        """运行交互模式"""
        print("\n" + "="*60)
        print("欢迎使用机器人控制系统!")
        print("="*60)
        print("可用命令:")
        print("  - 直接输入自然语言指令，AI将生成并执行代码")
        print("  - 'help' - 显示帮助信息")
        print("  - 'status' - 显示系统状态")
        print("  - 'reset' - 重置环境")
        print("  - 'stats' - 显示统计信息")
        print("  - 'quit' 或 'exit' - 退出系统")
        print("="*60 + "\n")
        
        self.running = True
        
        while self.running:
            try:
                instruction = input("请输入指令: ").strip()
                
                if not instruction:
                    continue
                
                # 处理特殊命令
                if self._handle_special_commands(instruction):
                    continue
                
                # 处理用户指令
                self._process_user_instruction(instruction)
                
            except KeyboardInterrupt:
                print("\n用户中断，退出...")
                break
            except EOFError:
                print("\n输入结束，退出...")
                break
            except Exception as e:
                print(f"处理指令时发生错误: {e}")
    
    def _handle_special_commands(self, instruction: str) -> bool:
        """处理特殊命令"""
        instruction_lower = instruction.lower()
        
        if instruction_lower in ['quit', 'exit', 'q']:
            print("退出系统...")
            self.running = False
            return True
        
        elif instruction_lower == 'help':
            self._show_help()
            return True
        
        elif instruction_lower == 'status':
            self._show_status()
            return True
        
        elif instruction_lower == 'reset':
            self._reset_environment()
            return True
        
        elif instruction_lower == 'stats':
            self._show_statistics()
            return True
        
        return False
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n" + "="*50)
        print("帮助信息")
        print("="*50)
        print("系统指令:")
        print("  help   - 显示此帮助信息")
        print("  status - 显示系统状态")
        print("  reset  - 重置环境到初始状态")
        print("  stats  - 显示系统统计信息")
        print("  quit   - 退出系统")
        print()
        print("机器人指令示例:")
        print("  '把红色方块放到托盘里'")
        print("  '检测所有方块'")
        print("  '将绿色方块移动到(0.5, 0.3, 0.2)'")
        print("  '清空托盘'")
        print("  '堆叠所有方块到托盘上'")
        print()
        print("可用物体:")
        if self.robot_api:
            objects = self.robot_api.list_available_objects()
            for name, obj_type in objects.items():
                print(f"  {name} ({obj_type})")
        print("="*50 + "\n")
    
    def _show_status(self):
        """显示系统状态"""
        print("\n" + "="*40)
        print("系统状态")
        print("="*40)
        
        if self.simulation:
            sim_stats = self.simulation.get_simulation_stats()
            print(f"仿真状态: {'运行中' if sim_stats['is_running'] else '已停止'}")
            print(f"仿真步数: {sim_stats['step_count']}")
            print(f"物体数量: {sim_stats['object_count']}")
        else:
            print("仿真: 未初始化")
        
        if self.ai_manager:
            ai_stats = self.ai_manager.get_model_stats()
            print(f"AI模型: {'已初始化' if ai_stats['is_initialized'] else '未初始化'}")
            print(f"代码生成次数: {ai_stats['generation_count']}")
        else:
            print("AI管理器: 未初始化")
        
        print("="*40 + "\n")
    
    def _reset_environment(self):
        """重置环境"""
        print("重置环境...")
        if self.robot_api:
            success = self.robot_api.reset_environment()
            if success:
                print("环境重置成功")
            else:
                print("环境重置失败")
        else:
            print("机器人API未初始化")
    
    def _show_statistics(self):
        """显示统计信息"""
        print("\n" + "="*50)
        print("系统统计信息")
        print("="*50)
        
        if self.robot_api:
            stats = self.robot_api.get_api_stats()
            print(f"API调用次数: {stats['operation_count']}")
            print(f"最后操作时间: {stats['last_operation_time']}")
            print(f"可用物体: {stats['available_objects']}")
        
        if self.simulation:
            sim_stats = self.simulation.get_simulation_stats()
            print(f"仿真步数: {sim_stats['step_count']}")
            print(f"仿真连接状态: {sim_stats['is_connected']}")
        
        if self.ai_manager:
            ai_stats = self.ai_manager.get_model_stats()
            print(f"AI生成次数: {ai_stats['generation_count']}")
            print(f"AI模型: {ai_stats['model_name']}")
        
        print("="*50 + "\n")
    
    def _process_user_instruction(self, instruction: str):
        """处理用户指令"""
        print(f"\n处理指令: {instruction}")
        
        try:
            if not self.ai_manager.model:
                print("AI模型未初始化，无法自动生成代码")
                return
            
            # 生成代码
            print("正在生成代码...")
            code = self.ai_manager.generate_code(instruction)
            
            if not code:
                print("未能生成有效代码")
                return
            
            print(f"生成的代码:\n{code}\n")
            
            # 执行代码
            print("执行代码...")
            success, error = self.code_executor.execute_code(code)
            
            if success:
                print("指令执行成功！")
            else:
                print(f"指令执行失败: {error}")
                
        except Exception as e:
            print(f"处理指令失败: {e}")
    
    def shutdown(self):
        """关闭系统"""
        print("正在关闭系统...")
        
        self.running = False
        
        if self.simulation:
            self.simulation.disconnect()
        
        print("系统已关闭")
    
    def start(self) -> bool:
        """启动系统"""
        try:
            if not self.initialize():
                print("系统初始化失败")
                return False
            
            # 运行交互模式
            self.run_interactive_mode()
            
            return True
            
        except Exception as e:
            print(f"系统启动失败: {e}")
            return False
        finally:
            self.shutdown()

def main():
    """主函数"""
    print("机器人控制系统启动中...")
    
    try:
        system = RobotControlSystem()
        success = system.start()
        
        if success:
            print("系统正常退出")
        else:
            print("系统异常退出")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断，退出...")
    except Exception as e:
        print(f"系统发生严重错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()