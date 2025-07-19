"""
机器人API模块 - 提供完善的机器人控制API接口
"""
import time
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

class RobotAPI:
    """完善的机器人API接口类"""
    
    def __init__(self, simulation_instance):
        """
        初始化机器人API
        
        Args:
            simulation_instance: 仿真环境实例
        """
        self.sim = simulation_instance
        self.last_operation_time = 0
        self.operation_count = 0
        
        # 支持的物体类型
        self.valid_object_types = ["block", "bowl", "tray"]
        
        print("机器人API已初始化")
    
    def _validate_object_type(self, object_type: str) -> str:
        """
        验证并规范化物体类型
        
        Args:
            object_type: 输入的物体类型
            
        Returns:
            规范化的物体类型
        """
        # 去除复数形式 (blocks -> block)
        normalized_type = object_type.lower().rstrip('s')
        
        if normalized_type not in self.valid_object_types:
            raise ValueError(f"无效的物体类型 '{object_type}', 支持的类型: {self.valid_object_types}")
        
        return normalized_type
    
    def _check_object_exists(self, object_name: str) -> bool:
        """检查物体是否存在"""
        return object_name in self.sim.object_ids
    
    # --- 感知API (Perception APIs) ---
    
    def detect_objects(self, object_type: str) -> List[str]:
        """
        检测场景中所有属于特定类型的物体，并返回它们的名称列表
        
        Args:
            object_type: 物体类型 ("block", "bowl", "tray")
            
        Returns:
            物体名称列表
        """
        try:
            # 验证物体类型
            normalized_type = self._validate_object_type(object_type)
            
            # 检测物体
            detected = []
            for name, type_val in self.sim.name_to_type.items():
                if type_val == normalized_type:
                    detected.append(name)
            
            # 按名称排序以保证一致性
            detected.sort()
            print(f"检测到 {len(detected)} 个 {object_type} 类型的物体: {detected}")
            return detected
            
        except Exception as e:
            print(f"检测物体失败: {e}")
            return []

    def is_empty(self, container_name: str) -> bool:
        """
        检查一个容器（如碗或托盘）是否为空
        
        Args:
            container_name: 容器名称
            
        Returns:
            如果容器为空返回True，否则返回False
        """
        try:
            # 检查容器是否存在
            if not self._check_object_exists(container_name):
                print(f"容器 '{container_name}' 不存在")
                return False
            
            container_pos = self.sim.get_object_position(container_name)
            if not container_pos:
                return False
            
            # 检查是否有物体在容器内
            objects_in_container = []
            container_detection_radius = 0.25
            container_height_threshold = 0.05
            
            for name, obj_id in self.sim.object_ids.items():
                if name == container_name:
                    continue
                
                obj_pos = self.sim.get_object_position(name)
                if not obj_pos:
                    continue
                
                # 计算xy平面的距离
                dist_xy = np.sqrt((obj_pos[0] - container_pos[0])**2 + 
                                 (obj_pos[1] - container_pos[1])**2)
                
                # 检查是否在容器上方
                is_above = obj_pos[2] > container_pos[2] + container_height_threshold
                
                if (dist_xy < container_detection_radius and is_above):
                    objects_in_container.append(name)
            
            is_empty_result = len(objects_in_container) == 0
            print(f"容器 '{container_name}' {'为空' if is_empty_result else '不为空'}, 包含物体: {objects_in_container}")
            return is_empty_result
            
        except Exception as e:
            print(f"检查容器状态失败: {e}")
            return False
    
    def get_object_position(self, object_name: str) -> Optional[Tuple[float, float, float]]:
        """
        获取物体的当前位置
        
        Args:
            object_name: 物体名称
            
        Returns:
            物体位置坐标 (x, y, z)，如果物体不存在返回None
        """
        try:
            if not self._check_object_exists(object_name):
                print(f"物体 '{object_name}' 不存在")
                return None
            
            position = self.sim.get_object_position(object_name)
            print(f"物体 '{object_name}' 位置: {position}")
            return position
            
        except Exception as e:
            print(f"获取物体位置失败: {e}")
            return None
    
    def get_objects_in_area(self, center: Tuple[float, float, float], 
                           radius: float) -> List[str]:
        """
        获取指定区域内的所有物体
        
        Args:
            center: 区域中心坐标 (x, y, z)
            radius: 搜索半径
            
        Returns:
            区域内物体名称列表
        """
        try:
            objects_in_area = []
            center_array = np.array(center)
            
            for object_name in self.sim.object_ids:
                obj_pos = self.sim.get_object_position(object_name)
                if obj_pos:
                    distance = np.linalg.norm(np.array(obj_pos) - center_array)
                    if distance <= radius:
                        objects_in_area.append(object_name)
            
            print(f"在区域 {center} 半径 {radius} 内找到 {len(objects_in_area)} 个物体")
            return objects_in_area
            
        except Exception as e:
            print(f"搜索区域内物体失败: {e}")
            return []
    
    def count_objects(self, object_type: Optional[str] = None) -> int:
        """
        计算特定类型的物体数量，如果不指定类型则计算所有物体
        
        Args:
            object_type: 物体类型，None表示所有类型
            
        Returns:
            物体数量
        """
        try:
            if object_type is None:
                count = len(self.sim.object_ids)
            else:
                objects = self.detect_objects(object_type)
                count = len(objects)
            
            print(f"物体数量统计: {count}")
            return count
            
        except Exception as e:
            print(f"统计物体数量失败: {e}")
            return 0
    
    # --- 控制API (Control APIs) ---
    
    def pick_place(self, object_to_move: str, target_location: str) -> bool:
        """
        抓取一个物体并将其放置到另一个物体的位置之上
        
        Args:
            object_to_move: 要移动的物体名称
            target_location: 目标位置物体名称
            
        Returns:
            操作是否成功
        """
        try:
            # 检查物体是否存在
            if not self._check_object_exists(object_to_move):
                print(f"要移动的物体 '{object_to_move}' 不存在")
                return False
            
            if not self._check_object_exists(target_location):
                print(f"目标位置物体 '{target_location}' 不存在")
                return False
            
            # 获取目标位置
            target_pos = self.sim.get_object_position(target_location)
            if not target_pos:
                return False

            # 计算放置位置（目标物体上方）
            placement_height_offset = 0.03  # 方块变小，进一步减小高度偏移
            final_pos = list(target_pos)
            final_pos[2] += placement_height_offset
            
            # 移动物体
            success = self.sim.move_object(object_to_move, final_pos)
            
            if success:
                print(f"成功将 '{object_to_move}' 放置到 '{target_location}' 上方")
            else:
                print(f"移动物体 '{object_to_move}' 失败")
            
            return success
            
        except Exception as e:
            print(f"pick_place操作失败: {e}")
            return False
    
    def move_to_position(self, object_name: str, position: Tuple[float, float, float]) -> bool:
        """
        将物体移动到指定的绝对位置
        
        Args:
            object_name: 物体名称
            position: 目标位置 (x, y, z)
            
        Returns:
            操作是否成功
        """
        try:
            if not self._check_object_exists(object_name):
                print(f"物体 '{object_name}' 不存在")
                return False
            
            success = self.sim.move_object(object_name, list(position))
            
            if success:
                print(f"成功将 '{object_name}' 移动到位置 {position}")
            else:
                print(f"移动物体 '{object_name}' 到位置 {position} 失败")
            
            return success
            
        except Exception as e:
            print(f"move_to_position操作失败: {e}")
            return False
    
    def stack_objects(self, objects: List[str], base_location: str) -> bool:
        """
        将多个物体堆叠在指定位置
        
        Args:
            objects: 要堆叠的物体名称列表（从下到上的顺序）
            base_location: 堆叠的基础位置
            
        Returns:
            操作是否成功
        """
        try:
            if not objects:
                return True
            
            # 验证所有物体都存在
            for obj in objects:
                if not self._check_object_exists(obj):
                    print(f"堆叠物体 '{obj}' 不存在")
                    return False
            
            if not self._check_object_exists(base_location):
                print(f"基础位置物体 '{base_location}' 不存在")
                return False
            
            # 获取基础位置
            base_pos = self.sim.get_object_position(base_location)
            if not base_pos:
                return False
            
            # 依次堆叠物体
            placement_height_offset = 0.05  # 堆叠高度也相应减小
            current_height = base_pos[2]
            for i, obj in enumerate(objects):
                stack_pos = [base_pos[0], base_pos[1], 
                           current_height + placement_height_offset * (i + 1)]
                
                success = self.sim.move_object(obj, stack_pos)
                if not success:
                    print(f"堆叠物体 '{obj}' 失败")
                    return False
            
            print(f"成功将 {len(objects)} 个物体堆叠在 '{base_location}' 上")
            return True
            
        except Exception as e:
            print(f"stack_objects操作失败: {e}")
            return False
    
    def clear_container(self, container_name: str, 
                       target_area: Optional[Tuple[float, float, float]] = None) -> bool:
        """
        清空容器中的所有物体
        
        Args:
            container_name: 容器名称
            target_area: 物体移动到的目标区域中心，None表示使用默认位置
            
        Returns:
            操作是否成功
        """
        try:
            if not self._check_object_exists(container_name):
                print(f"容器 '{container_name}' 不存在")
                return False
            
            # 找到容器中的所有物体
            if self.is_empty(container_name):
                print(f"容器 '{container_name}' 已经是空的")
                return True
            
            container_pos = self.sim.get_object_position(container_name)
            objects_to_move = []
            
            container_detection_radius = 0.25
            container_height_threshold = 0.05
            
            for name, obj_id in self.sim.object_ids.items():
                if name == container_name:
                    continue
                
                obj_pos = self.sim.get_object_position(name)
                if not obj_pos:
                    continue
                
                # 检查是否在容器内
                dist_xy = np.sqrt((obj_pos[0] - container_pos[0])**2 + 
                                 (obj_pos[1] - container_pos[1])**2)
                is_above = obj_pos[2] > container_pos[2] + container_height_threshold
                
                if (dist_xy < container_detection_radius and is_above):
                    objects_to_move.append(name)
            
            # 移动物体
            if target_area is None:
                # 使用默认区域（容器旁边）
                target_area = (container_pos[0] + 0.3, container_pos[1], container_pos[2] + 0.1)
            
            success_count = 0
            for i, obj in enumerate(objects_to_move):
                # 分散放置物体
                offset_x = (i % 2) * 0.1 - 0.05
                offset_y = (i // 2) * 0.1
                target_pos = (target_area[0] + offset_x, 
                             target_area[1] + offset_y, 
                             target_area[2])
                
                if self.move_to_position(obj, target_pos):
                    success_count += 1
            
            all_success = success_count == len(objects_to_move)
            
            print(f"清空容器完成: 移动了 {success_count}/{len(objects_to_move)} 个物体")
            return all_success
            
        except Exception as e:
            print(f"clear_container操作失败: {e}")
            return False
    
    # --- 状态查询API ---
    
    def get_api_stats(self) -> Dict[str, Any]:
        """获取API使用统计信息"""
        return {
            "operation_count": self.operation_count,
            "last_operation_time": self.last_operation_time,
            "available_objects": list(self.sim.object_ids.keys()),
            "valid_object_types": self.valid_object_types,
            "simulation_stats": self.sim.get_simulation_stats()
        }
    
    def list_available_objects(self) -> Dict[str, str]:
        """列出所有可用物体及其类型"""
        return dict(self.sim.name_to_type)
    
    def reset_environment(self) -> bool:
        """重置环境到初始状态"""
        try:
            self.sim.reset_scene()
            self.operation_count = 0
            self.last_operation_time = 0
            
            print("环境已重置到初始状态")
            return True
            
        except Exception as e:
            print(f"重置环境失败: {e}")
            return False