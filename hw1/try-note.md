# PyBullet 仿真流程使用教程

PyBullet 是一个用于机器人仿真和机器学习的 Python 模块。 PyBullet 构建仿真程序的基本流程。

**核心步骤：**

1.  **导入 PyBullet 及相关库**
2.  **连接到物理服务器**
3.  **设置仿真环境**
    * 设置重力
    * 加载模型 (URDF, SDF, MJCF等)
    * （可选）设置搜索路径以加载数据文件
4.  **运行仿真**
    * 步进仿真
    * 获取物体状态
5.  **（可选）控制机器人**
    * 获取关节信息
    * 设置关节电机控制
6.  **（可选）与仿真环境交互**
    * 应用外力/力矩
    * 碰撞检测
    * 渲染图像
7.  **结束仿真**
    * 断开与物理服务器的连接

## 1. 导入 PyBullet 及相关库

首先，你需要导入 `pybullet` 模块。通常，我们还会导入 `time` 模块来控制仿真速度，以及 `pybullet_data` 来方便地访问一些预置的 URDF 文件和环境。

```python
import pybullet as p
import time
import pybullet_data
```

## 2. 连接到物理服务器

PyBullet 使用客户端-服务器架构。你需要连接到一个物理仿真服务器。服务器可以在图形用户界面 (GUI) 模式下运行，也可以在无图形界面的直接 (DIRECT) 模式下运行。

```python
# 连接到带GUI的物理服务器
physicsClient = p.connect(p.GUI) 
# 或者连接到无图形界面的服务器
# physicsClient = p.connect(p.DIRECT) 
```

`connect` 函数返回一个 `physicsClientId`，在后续的多服务器场景中可能会用到。如果只连接一个服务器，大部分API会自动使用默认的客户端ID。

## 3. 设置仿真环境

### a. 设置附加搜索路径 (可选)

如果你想使用 PyBullet 自带的一些模型和数据（如平面、一些机器人模型等），可以添加 `pybullet_data` 的路径。

```python
p.setAdditionalSearchPath(pybullet_data.getDataPath())
```

### b. 设置重力

默认情况下，仿真世界中没有重力。你需要明确设置重力向量。

```python
# 设置重力，例如沿Z轴负方向，大小为 -9.81 (或 -10)
p.setGravity(0, 0, -9.81)
```

### c. 加载模型

你可以从 URDF (Universal Robot Description Format), SDF (Simulation Description Format), MJCF (MuJoCo XML Format) 等文件中加载机器人或其他物体模型。最常用的是 loadURDF：

```python
# 加载一个平面作为地面
planeId = p.loadURDF("plane.urdf")

# 加载一个机器人模型 (例如 R2D2)
startPos = [0, 0, 1]  # 初始位置 [x, y, z]
startOrientation = p.getQuaternionFromEuler([0, 0, 0]) # 初始姿态 (欧拉角转四元数)
robotId = p.loadURDF("r2d2.urdf", startPos, startOrientation) 
# 你也可以加载其他模型，例如："cube.urdf", "sphere2.urdf" 等
```

`loadURDF` 返回一个 `bodyUniqueId`，用于在后续操作中唯一标识该物体。

**其他加载函数：**

* `p.loadSDF("model.sdf")`
* `p.loadMJCF("model.xml")`

这些函数通常返回一个包含多个 `bodyUniqueId` 的列表，因为 SDF 和 MJCF 文件可以定义多个模型。

**重要提示：** `loadURDF` 加载的模型，其关节默认情况下电机是启用的，并且具有很高的摩擦力，这会阻止关节自由运动。你需要使用 `setJointMotorControl2` 来设置关节的控制模式和目标值，或者将最大力设置为0以禁用默认电机。

## 4. 运行仿真

### a. 步进仿真

通过调用 `p.stepSimulation()` 来使仿真向前推进一步。默认的时间步长是 1/240 秒。

```python
for i in range(1000):  # 例如，运行1000个仿真步
    p.stepSimulation()
    time.sleep(1./240.) # 按照仿真频率暂停，以便观察 (GUI模式下)
```

你可以使用 `p.setTimeStep(your_time_step)` 来更改时间步长，或使用 `p.setPhysicsEngineParameter` 来设置更详细的物理引擎参数。

### b. 获取物体状态

在仿真过程中，你可能需要获取物体的位置和姿态。

```python
# 获取基座 (base) 的位置和姿态
basePos, baseOrn = p.getBasePositionAndOrientation(robotId)
print(f"机器人位置: {basePos}, 机器人姿态 (四元数): {baseOrn}")

# 如果需要欧拉角
baseEuler = p.getEulerFromQuaternion(baseOrn)
print(f"机器人姿态 (欧拉角): {baseEuler}")
```

## 5. (可选) 控制机器人

### a. 获取关节信息

要控制机器人，首先需要了解其关节。

```python
numJoints = p.getNumJoints(robotId)
print(f"机器人共有 {numJoints} 个关节")

for i in range(numJoints):
    jointInfo = p.getJointInfo(robotId, i)
    print(f"关节 {i}: 名称={jointInfo[1].decode('utf-8')}, 类型={jointInfo[2]}")
    # jointInfo 包含更多信息，如关节限制、阻尼等
```

### b. 设置关节电机控制

使用 `p.setJointMotorControl2` (或其数组版本 `p.setJointMotorControlArray`) 来控制关节。控制模式：

* `p.POSITION_CONTROL: 位置控制模式`
* `p.VELOCITY_CONTROL: 速度控制模式`
* `p.TORQUE_CONTROL: 力矩控制模式`
* `p.PD_CONTROL: PD 控制模式`

```python
# 示例：将第一个关节 (jointIndex=0) 设置为速度控制模式
jointIndex = 0
targetVelocity = 1.0  # 目标速度 (弧度/秒 或 米/秒)
maxForce = 100       # 电机能施加的最大力或力矩

p.setJointMotorControl2(bodyUniqueId=robotId,
                        jointIndex=jointIndex,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=targetVelocity,
                        force=maxForce)

# 示例：禁用默认电机，允许关节自由运动 (例如，对于轮子)
# 假设 jointIndex_wheel 是一个轮子关节的索引
# p.setJointMotorControl2(bodyUniqueId=robotId,
#                         jointIndex=jointIndex_wheel,
#                         controlMode=p.VELOCITY_CONTROL,
#                         force=0) # 设置最大力为0
```

### c. 获取关节状态

可以查询关节的当前状态，如位置、速度、反作用力等。

```python
jointState = p.getJointState(robotId, jointIndex)
currentPosition = jointState[0]
currentVelocity = jointState[1]
appliedMotorTorque = jointState[3] # 在速度/位置控制模式下电机施加的力矩
print(f"关节 {jointIndex} 当前位置: {currentPosition}, 速度: {currentVelocity}, 电机力矩: {appliedMotorTorque}")
```

## 6. (可选) 与仿真环境交互

### a. 应用外力/力矩

可以对物体的基座或特定连杆施加外力或力矩。

```python
# 对 robotId 的基座 (-1 代表基座) 在世界坐标系施加一个力
linkIndex = -1 # 基座
force = [10, 0, 0] # 沿X轴的力
position = [0, 0, 0.1] # 力作用于基座质心上方0.1米处 (相对于基座的局部坐标)
# p.applyExternalForce(objectUniqueId=robotId,
#                      linkIndex=linkIndex,
#                      forceObj=force,
#                      posObj=position, # 注意：文档中posObj是世界坐标，但通常在LINK_FRAME下更有用
#                      flags=p.WORLD_FRAME) # 或者 p.LINK_FRAME

# 施加力矩
# torque = [0, 0, 1] # 绕Z轴的力矩
# p.applyExternalTorque(objectUniqueId=robotId,
#                       linkIndex=linkIndex,
#                       torqueObj=torque,
#                       flags=p.WORLD_FRAME)
```

注意： `applyExternalForce/Torque` 通常在 `setRealTimeSimulation(0)` 时使用，并且在每个 `stepSimulation()` 后力会被清除。

### b. 碰撞检测

```python
p.getContactPoints(): 获取上一步仿真中发生的接触点信息。
p.getClosestPoints(bodyA, bodyB, distance): 计算两个物体之间（或物体与特定点之间）的最近点，即使它们没有接触。
# 获取与 robotId 相关的所有接触点
contactPoints = p.getContactPoints(bodyA=robotId)
if contactPoints:
    for point in contactPoints:
        print(f"接触点: 物体A={point[1]}, 物体B={point[2]}, A上位置={point[5]}, B上法向={point[7]}, 距离={point[8]}, 法向力={point[9]}")
```

### c. 渲染图像 (合成相机)

PyBullet 可以从虚拟相机生成 RGB 图像、深度图和分割掩码。

```python
# 定义相机参数
viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[0, 3, 2],    # 相机眼睛位置
    cameraTargetPosition=[0, 0, 0], # 相机焦点
    cameraUpVector=[0, 0, 1])       # 相机向上向量

projectionMatrix = p.computeProjectionMatrixFOV(
    fov=60.0, # 视野角度
    aspect=1.0, # 宽高比
    nearVal=0.1, # 近平面距离
    farVal=100.0) # 远平面距离

# 获取图像
width, height, rgbImg, depthImg, segImg = p.getCameraImage(
    width=224,
    height=224,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix,
    renderer=p.ER_BULLET_HARDWARE_OPENGL # 或者 p.ER_TINY_RENDERER (CPU渲染)
)
```

## 7. 结束仿真

当仿真完成或不再需要时，断开与物理服务器的连接。

```python
p.disconnect()
```

这是一个基本的 PyBullet 仿真流程。PyBullet 提供了非常丰富的功能，包括但不限于：创建和修改约束 (Constraints)保存和恢复仿真状态记录仿真数据和视频可变形体和布料仿真逆动力学和逆运动学计算与强化学习环境 (Gym Envs) 集成