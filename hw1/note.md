### PyBullet 快速入门指南 - 逻辑梳理

这份指南旨在帮助用户快速上手 PyBullet 这个用于机器人仿真和机器学习的 Python 模块 。其逻辑结构大致遵循了使用一个物理仿真库的典型流程：

1.  **环境搭建与连接:**
    * **导入库:** `import pybullet as p` 。
    * **连接物理服务器:** 使用 `p.connect()` 连接到仿真环境。可以选择图形化界面 (`p.GUI`) 或非图形化模式 (`p.DIRECT`) 。还支持通过共享内存、UDP 或 TCP 连接到远程服务器 。
    * **环境设置:** 设置重力 `p.setGravity()` ，设置文件搜索路径 `p.setAdditionalSearchPath()` 以便加载资源 。

2.  **加载与创建仿真资源:**
    * **加载模型:** 使用 `p.loadURDF()`, `p.loadSDF()`, `p.loadMJCF()` 等函数从标准格式文件加载机器人或其他物体模型 。可以指定初始位置、方向、是否固定基座等参数 。
    * **程序化创建:** 可以不依赖文件，直接使用 `p.createCollisionShape()`, `p.createVisualShape()`, `p.createMultiBody()` 等函数以编程方式创建碰撞体、视觉模型和多体系统 。
    * **保存与恢复状态:** 使用 `p.saveState()`, `p.saveBullet()`, `p.restoreState()` 来保存和恢复仿真世界的精确状态，包括接触点信息，这对于需要确定性仿真的场景很重要 。`p.saveWorld()` 可将当前场景近似保存为 Python 脚本 。

3.  **仿真循环与控制:**
    * **步进仿真:** 调用 `p.stepSimulation()` 来推进仿真世界一个时间步长 。默认步长为 1/240 秒，可通过 `p.setTimeStep()` 或 `p.setPhysicsEngineParameter()` 修改 。
    * **实时仿真:** 可使用 `p.setRealTimeSimulation()` 让服务器自动按实时时钟步进仿真 。
    * **获取状态:** 使用 `p.getBasePositionAndOrientation()`, `p.getBaseVelocity()`, `p.getLinkState()`, `p.getJointState()` 等函数查询物体（基座、连杆、关节）的位置、姿态、速度、关节角度、力等状态信息。
    * **施加控制:**
        * **关节控制:** 使用 `p.setJointMotorControl2()` 或更高效的 `p.setJointMotorControlArray()` 来控制关节电机，支持位置控制、速度控制和力矩控制模式 。
        * **外力施加:** 使用 `p.applyExternalForce()`, `p.applyExternalTorque()` 对物体的基座或连杆施加外力/力矩 。
        * **重置状态:** 使用 `p.resetBasePositionAndOrientation()`, `p.resetBaseVelocity()`, `p.resetJointState()` 可以强制设置物体状态，通常在仿真开始时使用 。

4.  **物理属性与交互:**
    * **动力学属性:** 使用 `p.getDynamicsInfo()` 查询物体的质量、摩擦系数、恢复系数等，并可通过 `p.changeDynamics()` 进行修改 。
    * **碰撞检测:**
        * **接触点:** `p.getContactPoints()` 获取上一步仿真计算出的接触点信息（位置、法线、距离、法向力等）。
        * **最近点:** `p.getClosestPoints()` 计算两个物体间（即使没有接触）的最近点信息 。
        * **射线检测:** `p.rayTest()`, `p.rayTestBatch()` 从给定起点到终点发射射线，检测与物体的碰撞点 。
        * **重叠查询:** `p.getOverlappingObjects()` 查询与给定 AABB 包围盒重叠的物体 。`p.getAABB()` 获取物体的 AABB 。
        * **碰撞过滤:** `p.setCollisionFilterGroupMask()`, `p.setCollisionFilterPair()` 控制哪些物体之间会发生碰撞检测 。
    * **约束:** 使用 `p.createConstraint()` 创建约束（如固定、点对点、齿轮约束）来连接物体或物体与世界，以实现闭环结构或特殊连接 。

5.  **高级功能:**
    * **渲染与可视化:**
        * **相机图像:** 使用 `p.getCameraImage()` 获取合成相机的 RGB 图像、深度图和分割掩码图 。需要先设置好视图矩阵 (`p.computeViewMatrix()`) 和投影矩阵 (`p.computeProjectionMatrix()`) 。
        * **视觉调试:** 使用 `p.addUserDebugLine()`, `p.addUserDebugText()`, `p.addUserDebugParameter()` 在 3D 视图中添加调试线条、文字和可调参数滑块/按钮 。
        * **视觉外观:** `p.getVisualShapeData()` 获取视觉模型信息 。`p.changeVisualShape()`, `p.loadTexture()` 修改视觉外观（颜色、纹理等）。
    * **逆动力学/运动学:**
        * **逆动力学:** `p.calculateInverseDynamics()` 计算达到目标关节加速度所需的力矩 。
        * **雅可比/质量矩阵:** `p.calculateJacobian()`, `p.calculateMassMatrix()` 计算运动学雅可比矩阵和系统的质量矩阵 。
        * **逆运动学 (IK):** `p.calculateInverseKinematics()` 计算使末端执行器达到目标位姿（位置/方向）所需的关节角度 。
    * **强化学习 (RL):** PyBullet 与 OpenAI Gym 集成，提供了一系列预置的 RL 环境（如 Minitaur, KUKA 抓取, Humanoid 等）。支持使用 Stable Baselines, TensorFlow Agents 等库进行训练 。
    * **虚拟现实 (VR):** 支持连接到 VR 设备 (HTC Vive, Oculus Rift)，获取控制器事件 (`p.getVREvents()`)，设置 VR 相机状态 (`p.setVRCameraState()`) 。
    * **可变形体:** 支持模拟布料和软体，使用有限元 (FEM) 或基于位置的动力学 (PBD) 。`p.loadSoftBody()` 加载可变形体 ，`p.createSoftBodyAnchor()` 固定或连接可变形体的顶点 。
    * **插件:** 允许加载 C/C++ 编写的插件以扩展功能 (`p.loadPlugin()`, `p.executePluginCommand()`) 。

6.  **安装与支持:**
    * **安装:** 主要通过 `pip install pybullet` 安装 。也提供了从源码编译安装的方法 (premake/cmake) 。
    * **支持与资源:** 提供了论坛、GitHub Issue 跟踪器、引用信息和常见问题解答 (FAQ) 。

### 总结

PyBullet 是一个强大的 Python 物理仿真库，特别适合机器人学和机器学习研究 。掌握它主要分几步：

1.  **连接:** 用 `p.connect()` 启动仿真（可选带图形界面）。
2.  **加载:** 用 `p.loadURDF()` 等函数载入你的机器人或物体模型。
3.  **设置:** 用 `p.setGravity()` 等设置仿真环境参数。
4.  **仿真:** 在循环中调用 `p.stepSimulation()` 驱动仿真前进。
5.  **控制:** 用 `p.setJointMotorControl2()` 控制机器人关节运动（设置目标位置、速度或力矩）。
6.  **感知:** 用 `p.getBasePositionAndOrientation()`, `p.getJointState()`, `p.getContactPoints()`, `p.getCameraImage()` 等获取物体状态、接触信息或渲染图像。

**核心概念：** 你通过 Client-Server 模式与物理引擎交互 ，加载 URDF 等标准格式模型 ，在仿真循环中施加控制并获取反馈。

**进阶功能：** 它还支持碰撞检测查询 、逆运动学计算 、强化学习环境 、VR 交互 和软体仿真 。

**调试:** 可以使用 `p.addUserDebugLine/Text/Parameter` 在图形界面中添加辅助线、文字和参数滑块方便调试 。

**简单来说：** 连接 -> 加载模型 -> 循环 (控制 -> 步进 -> 获取状态) -> 断开连接。这份指南就是围绕这个流程，详细介绍了每个环节可用的 API 及其参数。希望这个梳理能帮你快速入门！