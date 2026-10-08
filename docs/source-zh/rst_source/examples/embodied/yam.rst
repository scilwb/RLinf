使用 PICO 采集双臂 YAM 示教数据
===============================

使用 PICO 双手柄和 3 台 RealSense RGB 相机采集双臂 YAM 示教数据。关节状态与动作均为 14 维绝对值，顺序为 ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``；夹爪用 ``0=关闭，1=打开``。

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

微调 Pi0.5
-----------

按 :doc:`OpenPI SFT 流程 <sft_openpi>` 使用 ``realworld_dual_yam_sft_openpi_pi05`` 配置。``pi05_yam`` 保留 14 维绝对关节目标，将模型 state/actions 补零至 32 维，预测窗口为 10 帧；``image``、``extra_view_image-0``、``extra_view_image-1`` 分别对应 OpenPI 的三路 RGB 槽位。

在训练机器上安装模型与环境组合：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env yam
   source .venv/bin/activate
   export REPO_PATH="$PWD"
   export PYTHONPATH="$REPO_PATH:$PYTHONPATH"

安装复用现有 OpenPI 安装逻辑和 YAM 依赖，对应 Docker 目标为 ``embodied-yam-openpi``。另行准备 Pi0.5 的 PyTorch checkpoint，以及整理完毕、包含 ``meta/info.json`` 的 LeRobot 数据集。先通过 ``toolkits/lerobot/merge_lerobot_datasets.py`` 合并选好的采集 shard；采集输出的父目录并不是数据集本身。

使用 YAM 数据计算归一化统计量，然后配置三个路径：

.. code-block:: bash

   export YAM_SFT_DATASET='/absolute/path/to/yam_dataset'
   export YAM_SFT_MODEL_PATH='/absolute/path/to/pi05_pytorch'
   python toolkits/lerobot/calculate_norm_stats.py --config-name pi05_yam --repo-id "$YAM_SFT_DATASET"
   export YAM_SFT_NORM_STATS="$YAM_SFT_DATASET/norm_stats.json"

使用绝对数据集路径时，统计工具把文件写入该数据集目录，数据至少需满足默认 32 帧的 batch。将示例 YAML 中的 ``data.train_data_paths``、``actor.model.model_path``、``actor.model.openpi_data.norm_stats_path`` 替换为对应实际路径，再启动训练：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_dual_yam_sft_openpi_pi05

两步训练 smoke test 复用现有 SFT e2e runner：

.. code-block:: bash

   bash tests/e2e_tests/sft/run_vla_sft.sh yam_sft_openpi_pi05

测试配置读取前面导出的三个路径，使用一张 actor GPU，通过现有 Ray/FSDP runner 检查 checkpoint 加载、前向、反向和优化器更新。请核对 loss 日志；完成两步更新不能证明 policy 质量。训练只读取录制文件，部署需另外进行推理和有现场操作员参与的安全验证。
