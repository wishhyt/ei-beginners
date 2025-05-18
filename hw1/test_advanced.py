"""
PyBullet高级功能全面测试示例
此代码演示了PyBullet的主要功能，包括基础仿真、关节控制、碰撞检测、
逆运动学、约束、相机渲染、可变形体等高级特性
"""

import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import os
from datetime import datetime

# 用于保存图像和录制视频的目录
OUTPUT_DIR = "pybullet_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======== 1. 基础环境设置 ========
physicsClient = p.connect(p.GUI)  # 或 p.DIRECT 用于无界面模式
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)  # 禁用实时模式，使用stepSimulation手动推进

# 创建地面
planeId = p.loadURDF("plane.urdf")

# ======== 2. 加载各种类型的模型 ========
# 加载URDF模型
robot1Pos = [0, 0, 1]
robot1Orn = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("r2d2.urdf", robot1Pos, robot1Orn, 
                     useFixedBase=False, 
                     flags=p.URDF_USE_INERTIA_FROM_FILE)

# 加载其他形状
sphereId = p.loadURDF("sphere_small.urdf", [1, 0, 0.5])
boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
boxVisualId = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
boxMultiBodyId = p.createMultiBody(baseMass=1, 
                                   baseCollisionShapeIndex=boxId, 
                                   baseVisualShapeIndex=boxVisualId, 
                                   basePosition=[2, 0, 0.5])

# 尝试加载SDF/MJCF模型
try:
    sdfModels = p.loadSDF("stadium.sdf")
    mjcfModels = p.loadMJCF("mjcf/humanoid.xml")
    humanoidId = mjcfModels[0] if mjcfModels else None
except:
    print("SDF/MJCF模型加载失败，这可能需要额外的文件")
    humanoidId = None

# ======== 3. 动力学参数设置 ========
p.changeDynamics(robotId, -1, linearDamping=0.1, angularDamping=0.1)
for i in range(p.getNumJoints(robotId)):
    # 修改关节动力学属性
    p.changeDynamics(robotId, i, 
                    jointDamping=0.1, 
                    restitution=0.5, 
                    lateralFriction=0.5,
                    spinningFriction=0.1, 
                    rollingFriction=0.1)

# ======== 4. 关节控制演示 ========
# 获取关节信息
numJoints = p.getNumJoints(robotId)
print(f"机器人共有 {numJoints} 个关节")

# 创建关节参数调试滑块
joint_sliders = []
for i in range(numJoints):
    jointInfo = p.getJointInfo(robotId, i)
    jointName = jointInfo[1].decode('utf-8')
    jointType = jointInfo[2]
    
    # 只为可控制的关节创建滑块
    if jointType != p.JOINT_FIXED:
        lower = jointInfo[8]
        upper = jointInfo[9]
        # 避免无界限时的问题
        if lower > upper:
            lower, upper = -4, 4
        
        slider = p.addUserDebugParameter(
            jointName, lower, upper, 0)
        joint_sliders.append((i, slider))

# ======== 5. 约束创建示例 ========
# 点到点约束示例
p2p = p.createConstraint(
    parentBodyUniqueId=robotId,
    parentLinkIndex=-1,
    childBodyUniqueId=sphereId,
    childLinkIndex=-1,
    jointType=p.JOINT_POINT2POINT,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 1.2],
    childFramePosition=[0, 0, 0]
)

# 创建固定约束
fixed = p.createConstraint(
    parentBodyUniqueId=boxMultiBodyId,
    parentLinkIndex=-1,
    childBodyUniqueId=-1,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)

# ======== 6. 碰撞过滤设置 ========
# 设置碰撞过滤
p.setCollisionFilterPair(robotId, sphereId, -1, -1, 0)  # 禁用机器人和球体之间的碰撞

# ======== 7. 相机设置 ========
# 创建相机视图和投影矩阵
viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[3, 3, 2],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 0, 1]
)

projectionMatrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=1.0,
    nearVal=0.1,
    farVal=100
)

# ======== 8. 创建可调参数和调试信息 ========
# 添加调试线、文本和参数滑块
line1 = p.addUserDebugLine(
    lineFromXYZ=[0, 0, 0],
    lineToXYZ=[1, 0, 0],
    lineColorRGB=[1, 0, 0],
    lineWidth=3
)

text1 = p.addUserDebugText(
    text="PyBullet高级功能演示",
    textPosition=[0, 0, 2],
    textColorRGB=[0, 0, 1],
    textSize=1.5
)

forceScaling = p.addUserDebugParameter("外力大小", 0, 100, 10)
cameraDistance = p.addUserDebugParameter("相机距离", 1, 10, 3)

# ======== 9. 软体演示（如果支持） ========
try:
    bunnyId = None
    # 尝试加载可变形体
    if os.path.exists(pybullet_data.getDataPath() + "/bunny.obj"):
        bunnyId = p.loadSoftBody(
            pybullet_data.getDataPath() + "/bunny.obj", 
            mass=0.1, 
            useNeoHookean=1, 
            NeoHookeanMu=180, 
            NeoHookeanLambda=600, 
            NeoHookeanDamping=0.001, 
            useSelfCollision=1, 
            collisionMargin=0.006, 
            frictionCoeff=0.5, 
            basePosition=[0, 1, 1]
        )
    
    # 为软兔创建固定点
    if bunnyId is not None:
        p.createSoftBodyAnchor(bunnyId, 0, -1, -1)  # 固定第一个顶点
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)  # 软体参数
except:
    print("软体功能不可用或未找到兔子模型")

# ======== 10. 仿真主循环 ========
saved_state = -1  # 保存状态的ID
frame_count = 0
total_frames = 500  # 总仿真帧数
log_id = None  # 用于记录仿真数据的ID

try:
    # 开始记录
    log_id = p.startStateLogging(
        p.STATE_LOGGING_VIDEO_MP4, 
        os.path.join(OUTPUT_DIR, "pybullet_simulation.mp4")
    )
    
    for i in range(total_frames):
        # 更新相机视角
        cam_dist = p.readUserDebugParameter(cameraDistance)
        p.resetDebugVisualizerCamera(
            cameraDistance=cam_dist,
            cameraYaw=45 + i * 0.1,  # 相机自动旋转
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.5]
        )
        
        # 更新关节滑块控制
        for joint_idx, slider_idx in joint_sliders:
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=p.readUserDebugParameter(slider_idx),
                force=50
            )
        
        # 对球体施加周期性的外力
        if i % 100 == 0:
            force_mag = p.readUserDebugParameter(forceScaling)
            p.applyExternalForce(
                objectUniqueId=sphereId,
                linkIndex=-1,
                forceObj=[force_mag, 0, force_mag/2],
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME
            )

        # 保存和恢复状态示例
        if i == 200:
            # 保存状态
            saved_state = p.saveState()
            print("已保存状态 ID:", saved_state)
        
        if i == 300 and saved_state != -1:
            # 恢复状态
            p.restoreState(saved_state)
            print("已恢复到保存的状态")
        
        # 碰撞检测
        if i % 10 == 0:
            contacts = p.getContactPoints()
            if contacts:
                print(f"检测到 {len(contacts)} 个碰撞点")
        
        # 逆运动学示例 - 让机器人尝试追踪一个移动目标点
        if humanoidId is not None and i > 100:
            # 添加更详细的检查
            try:
                target_pos = [
                    1.5 * math.sin(i * 0.01), 
                    1.5 * math.cos(i * 0.01), 
                    1.0
                ]
                
                # 获取人形模型的关节信息确保末端执行器是有效的
                humanoid_joints = p.getNumJoints(humanoidId)
                if humanoid_joints > 0:
                    end_effector = 执行器索引  # 使用确定的关节索引而不是猜测
                    
                    joint_poses = p.calculateInverseKinematics(
                        humanoidId, 
                        end_effector, 
                        target_pos
                    )
                    
                    # 应用计算出的关节角度
                    for j in range(min(len(joint_poses), p.getNumJoints(humanoidId))):
                        p.setJointMotorControl2(
                            bodyUniqueId=humanoidId,
                            jointIndex=j,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=joint_poses[j],
                            force=100
                        )
            except Exception as e:
                print(f"IK计算错误: {e}")
        
        # 获取并保存相机图像
        if i % 50 == 0:
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=320,
                height=240,
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # 使用numpy保存图像
            try:
                from PIL import Image
                
                rgb_array = np.array(rgbImg)
                rgb_array = rgb_array[:, :, :3]  # 去除alpha通道
                
                img = Image.fromarray(rgb_array)
                img_path = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
                img.save(img_path)
                print(f"图像已保存到: {img_path}")
                
                # 保存深度图
                depth_array = np.array(depthImg)
                # 归一化深度图用于可视化
                depth_norm = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
                depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8))
                depth_path = os.path.join(OUTPUT_DIR, f"depth_{i:04d}.png")
                depth_img.save(depth_path)
            except ImportError:
                print("保存图像需要PIL库")
        
        # 步进仿真
        p.stepSimulation()
        time.sleep(1./240.)  # 模拟240Hz的更新频率
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"仿真进度: {frame_count}/{total_frames}")

finally:
    # 停止记录并清理
    if log_id is not None:
        p.stopStateLogging(log_id)
    
    # 断开连接
    p.disconnect()

print("仿真完成!")
