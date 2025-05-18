import pybullet as p
import time
import pybullet_data
import math
import random

# Helper function to calculate IK and ensure it's a valid list of floats
def calculate_ik(body_id, end_effector_id, target_pos, target_orn=None, joint_indices=None, max_iterations=100, threshold=1e-4):
    """ Calculates Inverse Kinematics. Returns joint poses or None if no solution found. """
    if joint_indices is None:
        joint_indices = range(p.getNumJoints(body_id))

    if target_orn is None:
        # Calculate IK for position only
        joint_poses = p.calculateInverseKinematics(body_id,
                                               end_effector_id,
                                               target_pos,
                                               maxNumIterations=max_iterations,
                                               residualThreshold=threshold)
    else:
         # Calculate IK for position and orientation
        joint_poses = p.calculateInverseKinematics(body_id,
                                               end_effector_id,
                                               target_pos,
                                               targetOrientation=target_orn,
                                               maxNumIterations=max_iterations,
                                               residualThreshold=threshold)

    # Ensure the result is a valid tuple/list of floats
    if joint_poses is None or not isinstance(joint_poses, (list, tuple)):
        print("IK solution not found or invalid.")
        return None
    # Verify elements are floats (sometimes it might return unexpected types if fails badly)
    if not all(isinstance(j, float) for j in joint_poses):
         print(f"IK solution contains non-float elements: {joint_poses}")
         return None

    return list(joint_poses) # Convert tuple to list

# Helper function to move arm and wait until it reaches the target
def move_arm_to_pose(body_id, joint_indices, target_joint_poses, speed=0.01, timeout=10):  # 增加超时时间
    """ Moves the arm to a target configuration and waits."""
    if target_joint_poses is None:
        print("Cannot move arm, target poses are None.")
        return False

    start_time = time.time()
    prev_joint_states = None
    
    # 先打印当前关节状态
    current_states = p.getJointStates(body_id, joint_indices)
    current_pos = [state[0] for state in current_states]
    print(f"Current joint positions: {current_pos}")
    print(f"Target joint positions: {target_joint_poses}")

    while time.time() - start_time < timeout:
        p.setJointMotorControlArray(bodyUniqueId=body_id,
                                    jointIndices=joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=target_joint_poses,
                                    forces=[500] * len(joint_indices))  # 进一步增加关节力

        p.stepSimulation()
        time.sleep(1./240.)

        # Check if joints have reached the target
        current_joint_states = p.getJointStates(body_id, joint_indices)
        actual_positions = [state[0] for state in current_joint_states]

        diff = sum([(actual_positions[i] - target_joint_poses[i])**2 for i in range(len(joint_indices))])
        if diff < 0.001:  # 保持较低的到达目标阈值
            print("Arm reached target pose.")
            return True

        # 每1秒打印一次当前位置
        if int(time.time() * 10) % 10 == 0 and prev_joint_states != actual_positions:
            print(f"Current positions: {actual_positions}")
            print(f"Distance to target: {diff}")
            prev_joint_states = actual_positions.copy()

    print("Timeout reached while moving arm.")
    return False

# 1. 连接到物理引擎
physicsClient = p.connect(p.GUI)

# 2. 设置环境
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1./240.)

# 3. 加载模型
planeId = p.loadURDF("plane.urdf")

# 将机械臂抬高以确保与方块在同一高度范围内
kukaStartPos = [0, 0, 0]  # 原始位置
kukaStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
kukaId = p.loadURDF("kuka_iiwa/model.urdf", kukaStartPos, kukaStartOrientation, useFixedBase=True)

# 设置相机视角，使机械臂和方块都可见
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.5, 0, 0.2])

# 检查机械臂模型是否加载正确
num_joints = p.getNumJoints(kukaId)
print(f"Number of joints in KUKA model: {num_joints}")
for i in range(num_joints):
    joint_info = p.getJointInfo(kukaId, i)
    print(f"Joint {i}: {joint_info[1].decode('utf-8')}, type: {joint_info[2]}")

# Define cube colors and positions - 把方块放在机械臂可以更容易到达的位置
cube_colors = {
    "red": [1, 0, 0, 1],
    "green": [0, 1, 0, 1],
    "blue": [0, 0, 1, 1],
    "yellow": [1, 1, 0, 1]
}
cube_positions = [
    [0.5, 0.0, 0.025],  # 移动红色方块到机械臂前方
    [0.7, -0.1, 0.025],
    [0.8, 0.1, 0.025], 
    [0.5, -0.2, 0.025]
]
cubes = {} # Dictionary to store cube_id: color_name

print("Loading cubes...")
cube_keys = list(cube_colors.keys())
for i in range(len(cube_positions)):
    color_name = cube_keys[i % len(cube_keys)] # Cycle through colors
    pos = cube_positions[i]
    orn = p.getQuaternionFromEuler([0, 0, random.uniform(0, math.pi)]) # Random orientation
    cube_id = p.loadURDF("cube_small.urdf", pos, orn)
    color_rgba = cube_colors[color_name]
    p.changeVisualShape(cube_id, -1, rgbaColor=color_rgba)
    cubes[cube_id] = color_name
    print(f"Loaded {color_name} cube with ID {cube_id} at {pos}")
    # Allow objects to settle
    for _ in range(50):
        p.stepSimulation()

# 确保红色方块放在位置[0.5, 0.0, 0.025]
red_cube_id = None
for cube_id, color_name in cubes.items():
    if color_name == "red":
        red_cube_id = cube_id
        # 重新设置红色方块位置为确保可达的位置
        p.resetBasePositionAndOrientation(red_cube_id, [0.5, 0.0, 0.025], 
                                         p.getQuaternionFromEuler([0, 0, 0]))
        pos, orn = p.getBasePositionAndOrientation(red_cube_id)
        red_cube_pos = list(pos)
        break

if red_cube_id is None:
    print("Error: Red cube not found!")
    p.disconnect()
    exit()

print(f"Found red cube (ID: {red_cube_id}) at position: {red_cube_pos}")

# 5. 机械臂设置
num_joints = p.getNumJoints(kukaId)
end_effector_link_index = 6  # KUKA iiwa end effector
controllable_joint_indices = [i for i in range(num_joints) if p.getJointInfo(kukaId, i)[2] != p.JOINT_FIXED]

# 初始化机械臂姿态 - 设置一个更好的初始姿态
init_joint_positions = [0, 0, 0, -1.57, 0, 1.57, 0]  # 根据KUKA的关节布局设置一个合适的初始姿态
for i, joint_idx in enumerate(controllable_joint_indices):
    if i < len(init_joint_positions):
        p.resetJointState(kukaId, joint_idx, init_joint_positions[i])

# 等待机械臂初始化完成
for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# 获取末端执行器的当前位置
ee_link_state = p.getLinkState(kukaId, end_effector_link_index)
ee_pos = ee_link_state[0]
print(f"End effector initial position: {ee_pos}")

# Define target poses relative to the red cube
gripper_orientation = p.getQuaternionFromEuler([0, -math.pi, 0])  # 指向下方

# 计算预抓取、抓取和提升位置
pre_grasp_pos = red_cube_pos[:]
pre_grasp_pos[2] += 0.15  # 15 cm above

grasp_pos = red_cube_pos[:]
grasp_pos[2] += 0.01  # 更靠近方块表面

lift_pos = grasp_pos[:]
lift_pos[2] += 0.2  # Lift 20 cm

# 6. 执行抓取流程
grasp_constraint_id = None

try:
    print("\n--- Moving to Pre-Grasp Position ---")
    print(f"Pre-grasp position: {pre_grasp_pos}")
    target_joints_pre_grasp = calculate_ik(kukaId, end_effector_link_index, pre_grasp_pos, gripper_orientation, controllable_joint_indices)
    if target_joints_pre_grasp:
        move_arm_to_pose(kukaId, controllable_joint_indices, target_joints_pre_grasp)
        time.sleep(1.0)  # 增加暂停时间
    else:
        print("Failed to calculate IK for pre-grasp position!")

    print("\n--- Moving to Grasp Position ---")
    print(f"Grasp position: {grasp_pos}")
    target_joints_grasp = calculate_ik(kukaId, end_effector_link_index, grasp_pos, gripper_orientation, controllable_joint_indices)
    if target_joints_grasp:
        move_arm_to_pose(kukaId, controllable_joint_indices, target_joints_grasp)
        time.sleep(1.0)  # 增加暂停时间
    else:
        print("Failed to calculate IK for grasp position!")

    # 检查末端执行器位置是否接近方块
    ee_link_state = p.getLinkState(kukaId, end_effector_link_index)
    ee_pos = ee_link_state[0]
    distance_to_cube = math.sqrt(sum([(ee_pos[i] - red_cube_pos[i])**2 for i in range(3)]))
    print(f"Distance from end effector to cube: {distance_to_cube}")

    print("\n--- Simulating Grasp (Creating Constraint) ---")
    # 使用方块的实际位置创建约束
    grasp_constraint_id = p.createConstraint(
        parentBodyUniqueId=kukaId,
        parentLinkIndex=end_effector_link_index,
        childBodyUniqueId=red_cube_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0.01],  # 末端执行器上的连接点
        childFramePosition=[0, 0, 0],      # 方块上的连接点
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=[0, 0, 0, 1])
    
    print(f"Constraint created with ID: {grasp_constraint_id}")
    
    # 给约束更多时间生效
    for _ in range(120):  # 增加模拟步数
        p.stepSimulation()
        time.sleep(1./240.)

    print("\n--- Moving to Lift Position ---")
    print(f"Lift position: {lift_pos}")
    target_joints_lift = calculate_ik(kukaId, end_effector_link_index, lift_pos, gripper_orientation, controllable_joint_indices)
    if target_joints_lift:
        move_arm_to_pose(kukaId, controllable_joint_indices, target_joints_lift)
        time.sleep(2.0)  # 延长最终姿势保持时间
    else:
        print("Failed to calculate IK for lift position!")

except Exception as e:
    print(f"An error occurred during the grasp sequence: {e}")

finally:
    # 7. 仿真结束
    print("\nSimulation finished. Press Ctrl+C or close the window to exit.")
    # Keep GUI open for a while
    try:
         while True:
              p.stepSimulation()
              time.sleep(1./240.)
    except KeyboardInterrupt:
         pass

    # Clean up constraint if created
    if grasp_constraint_id is not None:
        try:
             p.removeConstraint(grasp_constraint_id)
             print("Grasp constraint removed.")
        except p.error as e:
             print(f"Could not remove constraint: {e}")

    # 8. 断开连接
    p.disconnect()
    print("Disconnected from PyBullet.")
