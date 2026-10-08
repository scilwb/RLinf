使用 PICO 采集双臂 YAM 示教数据
===============================

使用 PICO 双手柄和 3 台 RealSense RGB 相机采集双臂 YAM 示教数据。关节状态与动作均为 14 维绝对值，顺序为 ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``；夹爪用 ``0=关闭，1=打开``。

先用 fake vendor SDK 检查环境链路，再由操作员完成现场硬件检查，最后采集示教数据。短 SAC/MLP 测试单独验证训练链路，与现场验收区分记录。

准备设备
--------

准备两台配备 ``flexible_4310`` 夹爪的 YAM 从臂、3 台 RealSense 相机、PICO 头显与双手柄，以及 ZeroMQ PICO publisher。按 :doc:`Franka VR 配置 <franka_vr>` 准备 XRoboToolkit、PC Service 和 publisher，再安装 YAM 运行环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

安装脚本包含固定版本的 ``i2rt`` backend 和 PICO 传输依赖。启用运动前，核对已安装的机器人模型、工具中心点、关节限位、手柄到基座的 yaw 和运动缩放参数是否与真机一致。

.. warning::

   采集器在第一次 reset 时连接两条从臂 CAN 和全部 3 台相机。请先让双臂处于有支撑且已确认安全的姿态，清空运动范围，并安排操作员守在急停旁。示例默认不自动复位或泊车；结束会话前，操作员需确认关闭电机输出后双臂仍能安全稳定地受到支撑。

配置并采集
----------

配置两条从臂 CAN、3 个相机序列号、PICO publisher 地址，以及左右臂各自实测的操作者坐标系到机器人坐标系的 yaw。**启动 Ray 前** 导出这些值，供环境 worker 读取：

.. code-block:: bash

   export YAM_LEFT_FOLLOWER_CAN='<left-can-interface>'
   export YAM_RIGHT_FOLLOWER_CAN='<right-can-interface>'
   export YAM_TOP_CAMERA_SERIAL='<top-serial>'
   export YAM_LEFT_CAMERA_SERIAL='<left-serial>'
   export YAM_RIGHT_CAMERA_SERIAL='<right-serial>'
   export YAM_PICO_ZMQ_ADDR='tcp://<publisher-host>:5555'
   export YAM_LEFT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'
   export YAM_RIGHT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'

将占位符替换为本站确认过的值。示例配置位于 ``examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml``，将环境 worker 和硬件放在同一个 Ray 节点。新启动的单机环境可按下面的命令运行：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head
   bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

按住某侧手柄的 grip 可接管对应机械臂，松开即结束接管。示例将右菜单键映射为开始或结束一条录制，将左菜单键映射为丢弃当前录制。正式采集前，请在本站 PICO 设备上确认按键事件能送达。丢弃只会移除当前未完成的 episode；故障或超时的录制仍可能保存为未成功的 episode。

启动脚本默认在 ``logs/<timestamp>-realworld_dual_yam_collect_data_pico/`` 下保存日志，LeRobot 数据位于该目录的 ``collected_data/rank_*/`` 中。流式写入会保存已录制片段，包括未成功片段，并标记 ``is_success``。使用数据前，请检查该标记和三路 RGB 图像；任务是否成功仍由现场操作员判断。

无硬件检查
----------

安装完成后，用 fake ``i2rt`` 和 RealSense SDK 运行硬件检查工具。环境保持 ``is_dummy: false``，YAM backend、相机适配、反馈检查和资源关闭均经过与真机相同的接口：

.. code-block:: bash

   python -m toolkits.realworld_check.test_yam_env --mock --once

去掉 ``--once`` 可进入交互式 mock 会话。工具启动后读取 14 维反馈和三路 RGB，退出时关闭设备。

在 GPU 主机上运行与 CI 相同的短 SAC 测试：

.. code-block:: bash

   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh yam_mock_sac_mlp_reach

启动脚本在 driver 和 worker 进程中加载 SDK fakes。MLP 从随机权重开始，使用 14 维状态与动作，运行两个训练 epoch。``DualYamReachEnv-v1`` 按实测关节与左右臂各 6 维目标的距离评分；夹爪仍参与状态与动作，但不计入奖励。在 ``env/yam_reach.yaml`` 中配置 ``target_joint_qpos``、``reward_threshold``、``success_hold_steps`` 和 ``use_dense_reward``。预设目标仅用于 mock 测试；将该任务用于真机前，需要选择经现场确认的安全目标。采集任务的奖励保持原样。

训练日志和 loss 值作为软件 smoke test 的证据单独保留，不能用 mock 奖励或 loss 判断真机任务是否成功。

现场硬件检查
------------

按站点配置导出两条 CAN 和三个相机变量，再由操作员确认启动与关闭的安全条件。工具从 ``realworld_dual_yam_collect_data_pico.yaml`` 读取硬件配置，不使用 PICO。通过 ``--config`` 指定本站配置副本，并在 ``env.eval.override_cfg`` 中设置已核对的关节限位。

.. warning::

   打开 SDK 时可能校准夹爪并启用实测姿态保持，因此仅读取观测也不等于被动连接。整个检查期间应让双臂保持安全支撑，清空工作区，并安排操作员守在急停旁。关闭会释放电机输出；自动复位和泊车必须保持关闭。

先读取一次反馈并保存相机快照：

.. code-block:: bash

   python -m toolkits.realworld_check.test_yam_env \
       --confirm-hardware-ready --once --snapshot-dir /path/to/yam-check/frames

核对关节顺序、反馈是否有限，以及三路相机画面。操作员同意运动后，去掉 ``--once`` 并添加 ``--enable-motion``。交互命令包括 ``observe``、``joint left 0 0.01``、``gripper right 0.05`` 和 ``quit``。每次动作都需输入 ``MOVE``；关节增量最多 0.02 弧度，夹爪增量最多为开合范围的 0.05。runtime 故障会结束检查并关闭设备，不自动重试动作。

硬件配置、反馈、小幅动作、夹爪和关闭结果，以及日志与现场视频，应与 mock 测试证据分别保存。由操作员核对这些产物后，再确认现场检查是否完成。
