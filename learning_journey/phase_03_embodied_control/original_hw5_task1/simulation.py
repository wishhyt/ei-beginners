"""
仿真环境模块 - 提供PyBullet仿真环境封装
"""
import pybullet as p
import pybullet_data
import time
import threading
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ObjectInfo:
    """物体信息数据类"""
    name: str
    object_id: int
    object_type: str
    initial_position: Tuple[float, float, float]
    current_position: Optional[Tuple[float, float, float]] = None
    mass: float = 1.0
    scale: float = 1.0

class Simulation:
    """完善的PyBullet仿真环境封装类"""
    
    def __init__(self, use_gui: Optional[bool] = None):
        """
        初始化仿真环境
        
        Args:
            use_gui: 是否使用图形界面，None表示使用默认设置
        """
        self.use_gui = use_gui if use_gui is not None else True
        self.client_id = None
        self.is_connected = False
        self.is_running = False
        self.simulation_thread = None
        
        # 仿真配置参数
        self.time_step = 1.0/240.0
        self.gravity = (0, 0, -9.8)
        self.default_object_scale = 0.3
        self.tray_scale = 0.6
        self.workspace_bounds = (
            (0.0, 1.0),  # x范围
            (-0.5, 0.5), # y范围  
            (0.0, 1.0)   # z范围
        )
        
        # 物体管理
        self.object_ids: Dict[str, int] = {}
        self.object_info: Dict[str, ObjectInfo] = {}
        self.name_to_type: Dict[str, str] = {}
        
        # 性能监控
        self.step_count = 0
        self.last_step_time = 0
        
        print("仿真环境已初始化")
    
    def connect(self) -> bool:
        """
        连接到PyBullet仿真器
        
        Returns:
            连接是否成功
        """
        try:
            if self.use_gui:
                self.client_id = p.connect(p.GUI)
                print("已连接到PyBullet GUI模式")
            else:
                self.client_id = p.connect(p.DIRECT)
                print("已连接到PyBullet DIRECT模式")
            
            if self.client_id < 0:
                print("无法连接到PyBullet仿真器")
                return False
            
            self.is_connected = True
            self._configure_simulation()
            return True
            
        except Exception as e:
            print(f"连接仿真器失败: {e}")
            self.is_connected = False
            return False
    
    def _configure_simulation(self):
        """配置仿真参数"""
        try:
            # 设置搜索路径
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # 设置重力
            p.setGravity(*self.gravity)
            
            # 设置时间步长
            p.setTimeStep(self.time_step)
            
            # 设置实时仿真
            p.setRealTimeSimulation(0)  # 禁用实时仿真，手动控制
            
            print("仿真参数配置完成")
            
        except Exception as e:
            print(f"配置仿真参数失败: {e}")
    
    def setup_scene(self):
        """设置仿真场景"""
        if not self.is_connected and not self.connect():
            print("未连接到仿真器")
            return False
        
        try:
            print("开始设置仿真场景...")
            
            # 加载地面
            plane_id = p.loadURDF("plane.urdf")
            print(f"已加载地面 (ID: {plane_id})")
            
            # 加载物体
            self._load_objects()
            
            # 等待稳定
            self._stabilize_scene()
            
            print(f"场景设置完成，共加载 {len(self.object_ids)} 个物体")
            print(f"可用物体: {list(self.object_ids.keys())}")
            return True
            
        except Exception as e:
            print(f"设置场景失败: {e}")
            return False
    
    def _load_objects(self):
        """加载场景中的物体"""
        objects_to_load = [
            {
                "name": "red_block",
                "urdf": "cube.urdf",
                "position": [0.6, 0.2, 0.1],
                "scale": self.default_object_scale,
                "color": [1, 0, 0, 1],  # 红色
                "type": "block",
                "mass": 0.5
            },
            {
                "name": "blue_block", 
                "urdf": "cube.urdf",
                "position": [0.6, -0.2, 0.1],
                "scale": self.default_object_scale,
                "color": [0, 0, 1, 1],  # 蓝色
                "type": "block",
                "mass": 0.5
            },
            {
                "name": "green_block",
                "urdf": "cube.urdf", 
                "position": [0.4, 0.0, 0.1],
                "scale": self.default_object_scale,
                "color": [0, 1, 0, 1],  # 绿色
                "type": "block",
                "mass": 0.5
            },
            {
                "name": "tray",
                "urdf": "tray/tray.urdf",
                "position": [0.8, 0, 0],
                "scale": self.tray_scale,
                "color": [0.8, 0.8, 0.8, 1],  # 灰色
                "type": "bowl",
                "mass": 2.0
            }
        ]
        
        for obj_config in objects_to_load:
            try:
                # 加载URDF
                obj_id = p.loadURDF(
                    obj_config["urdf"],
                    obj_config["position"],
                    globalScaling=obj_config["scale"]
                )
                
                # 设置颜色
                p.changeVisualShape(obj_id, -1, rgbaColor=obj_config["color"])
                
                # 设置物理属性
                if obj_config.get("mass"):
                    p.changeDynamics(obj_id, -1, mass=obj_config["mass"])
                
                # 增加物理稳定性设置
                p.changeDynamics(obj_id, -1, 
                                lateralFriction=0.8,        # 侧向摩擦
                                spinningFriction=0.3,       # 旋转摩擦
                                rollingFriction=0.1,        # 滚动摩擦
                                restitution=0.2,            # 弹性系数
                                linearDamping=0.9,          # 线性阻尼
                                angularDamping=0.9)         # 角度阻尼
                
                # 保存物体信息
                self.object_ids[obj_config["name"]] = obj_id
                self.name_to_type[obj_config["name"]] = obj_config["type"]
                
                self.object_info[obj_config["name"]] = ObjectInfo(
                    name=obj_config["name"],
                    object_id=obj_id,
                    object_type=obj_config["type"],
                    initial_position=tuple(obj_config["position"]),
                    mass=obj_config.get("mass", 1.0),
                    scale=obj_config["scale"]
                )
                
                print(f"已加载物体: {obj_config['name']} (ID: {obj_id})")
                
            except Exception as e:
                print(f"加载物体 {obj_config['name']} 失败: {e}")
    
    def _stabilize_scene(self, steps: int = 100):
        """稳定场景"""
        print(f"稳定场景中，执行 {steps} 步...")
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(self.time_step)
        print("场景稳定完成")
    
    def get_object_position(self, object_name: str) -> Optional[Tuple[float, float, float]]:
        """
        获取物体位置
        
        Args:
            object_name: 物体名称
            
        Returns:
            物体位置 (x, y, z)
        """
        if object_name not in self.object_ids:
            return None
        
        try:
            obj_id = self.object_ids[object_name]
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            
            # 更新物体信息中的当前位置
            if object_name in self.object_info:
                self.object_info[object_name].current_position = pos
            
            return pos
        except Exception as e:
            print(f"获取物体 {object_name} 位置失败: {e}")
            return None
    
    def get_object_info(self, object_name: str) -> Optional[ObjectInfo]:
        """获取物体信息"""
        return self.object_info.get(object_name)
    
    def get_all_objects(self) -> List[str]:
        """获取所有物体名称"""
        return list(self.object_ids.keys())
    
    def get_objects_by_type(self, object_type: str) -> List[str]:
        """根据类型获取物体列表"""
        return [name for name, obj_type in self.name_to_type.items() 
                if obj_type == object_type]
    
    def is_position_in_bounds(self, position: Tuple[float, float, float]) -> bool:
        """检查位置是否在工作空间边界内"""
        x, y, z = position
        x_bounds, y_bounds, z_bounds = self.workspace_bounds
        
        return (x_bounds[0] <= x <= x_bounds[1] and
                y_bounds[0] <= y <= y_bounds[1] and
                z_bounds[0] <= z <= z_bounds[1])
    
    def move_object(self, object_name: str, target_pos: List[float]) -> bool:
        """
        移动物体到指定位置
        
        Args:
            object_name: 物体名称
            target_pos: 目标位置 [x, y, z]
            
        Returns:
            移动是否成功
        """
        if object_name not in self.object_ids:
            print(f"物体 '{object_name}' 不存在")
            return False
        
        try:
            # 检查边界
            if not self.is_position_in_bounds(tuple(target_pos)):
                print(f"目标位置 {target_pos} 超出工作空间边界")
                return False
            
            obj_id = self.object_ids[object_name]
            
            # 重置物体位置
            p.resetBasePositionAndOrientation(
                obj_id, 
                target_pos, 
                [0, 0, 0, 1]  # 默认四元数方向
            )
            
            # 重置物体速度，防止飞走
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])
            
            # 增加稳定物体的步骤
            for _ in range(120):  # 增加到120步
                p.stepSimulation()
                time.sleep(self.time_step / 20)  # 更短的时间间隔
            
            # 验证移动是否成功
            new_pos = self.get_object_position(object_name)
            if new_pos:
                distance = np.linalg.norm(np.array(new_pos) - np.array(target_pos))
                success = distance < 0.1  # 10cm容差
                
                if success:
                    print(f"成功移动物体 '{object_name}' 到位置 {target_pos}")
                else:
                    print(f"移动物体 '{object_name}' 位置偏差过大: {distance}")
                
                return success
            else:
                return False
                
        except Exception as e:
            print(f"移动物体 '{object_name}' 失败: {e}")
            return False
    
    def run(self):
        """运行仿真循环"""
        print("开始仿真循环...")
        self.is_running = True
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # 执行一步仿真
                p.stepSimulation()
                self.step_count += 1
                
                # 更新物体位置信息
                self._update_object_positions()
                
                # 控制仿真频率
                elapsed = time.time() - start_time
                sleep_time = max(0, self.time_step - elapsed)
                time.sleep(sleep_time)
                
                self.last_step_time = time.time()
                
            except Exception as e:
                print(f"仿真循环错误: {e}")
                break
        
        print("仿真循环结束")
    
    def _update_object_positions(self):
        """更新所有物体的位置信息"""
        for object_name in self.object_ids:
            self.get_object_position(object_name)
    
    def start_background_simulation(self):
        """在后台线程中启动仿真"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            print("仿真已在运行中")
            return
        
        self.simulation_thread = threading.Thread(target=self.run, daemon=True)
        self.simulation_thread.start()
        print("后台仿真已启动")
    
    def stop_simulation(self):
        """停止仿真"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1)
        print("仿真已停止")
    
    def reset_scene(self):
        """重置场景到初始状态"""
        print("重置场景到初始状态...")
        
        # 重置所有物体到初始位置
        for object_name, obj_info in self.object_info.items():
            if object_name in self.object_ids:
                self.move_object(object_name, list(obj_info.initial_position))
        
        self.step_count = 0
        print("场景重置完成")
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """获取仿真统计信息"""
        return {
            "step_count": self.step_count,
            "last_step_time": self.last_step_time,
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "object_count": len(self.object_ids),
            "time_step": self.time_step
        }
    
    def disconnect(self):
        """断开与仿真器的连接"""
        try:
            self.stop_simulation()
            if self.is_connected and self.client_id is not None:
                p.disconnect(self.client_id)
                self.is_connected = False
                print("已断开与仿真器的连接")
        except Exception as e:
            print(f"断开连接时发生错误: {e}")
    
    def __del__(self):
        """析构函数"""
        self.disconnect()