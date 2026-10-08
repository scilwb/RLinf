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

用录制数据进行微调
------------------

部署 policy 前，先使用已完成写盘的 YAM LeRobot 数据集微调 Pi0.5。SFT 只读取录制文件，只启动 actor worker。选择包含 ``meta/info.json`` 的数据集目录，而不是采集输出的 ``rank_*/`` 父目录。流式采集可能为每条 episode 生成一个 ``id_*/`` shard，因此训练前需要整理出包含目标 episode 的数据集，并核对任务标签、成功标记和故障片段。

使用现有准备工具合并选好的完整 shard：

.. code-block:: bash

   python toolkits/lerobot/merge_lerobot_datasets.py \
     --source-dir /absolute/path/to/selected_shards \
     --output-dir /absolute/path/to/yam_dataset --dry-run
   python toolkits/lerobot/merge_lerobot_datasets.py \
     --source-dir /absolute/path/to/selected_shards \
     --output-dir /absolute/path/to/yam_dataset

第一条命令只列出输入，不写文件。确认列表后，第二条命令合并所选目录下的全部 episode。工具不会自动筛选成功 episode，需先整理输入。

将模型与 YAM 依赖安装到独立环境：

.. code-block:: bash

   bash requirements/install.sh embodied --venv .venv-yam-openpi --model openpi --env yam
   source .venv-yam-openpi/bin/activate
   export PYTHONPATH="$PWD:$PYTHONPATH"
   export EMBODIED_PATH="$PWD/examples/embodiment"

这条命令显式安装 OpenPI，原来的纯采集安装命令保持不变。Docker 构建目标为 ``BUILD_TARGET=embodied-yam-openpi``。请另外准备 Pi0.5 的 PyTorch checkpoint；安装器下载模型运行所需的 assets，不会自动下载你的微调权重。

``pi05_yam`` 数据配置保留 14 维绝对关节目标和左臂在前的顺序，将 ``image``、``extra_view_image-0``、``extra_view_image-1`` 对应到 OpenPI 的主视角、左视角和右视角槽位。三路图像均为必需。模型输入的 state/actions 补零至 32 维，预测窗口为 10 帧；不进行 Aloha 的关节或夹爪转换，也不把绝对动作转换为相对增量。

使用 YAM 数据计算归一化统计量，不要直接使用其他机器人的统计量：

.. code-block:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_yam --repo-id /absolute/path/to/yam_dataset

传入绝对数据集路径时，该工具把 ``norm_stats.json`` 写入数据集目录。数据帧数需满足默认 32 个样本的统计 batch。通过 SFT 配置或命令行明确指定 checkpoint、数据集和统计文件路径。

先在 CPU 上检查数据，再到具备足够模型内存的训练机器检查前向、反向和优化器更新：

.. code-block:: bash

   python examples/sft/validate_openpi_sft.py \
     data.train_data_paths=/absolute/path/to/yam_dataset \
     actor.model.model_path=/absolute/path/to/pi05_pytorch \
     actor.model.openpi_data.norm_stats_path=/absolute/path/to/yam_dataset/norm_stats.json

   python examples/sft/validate_openpi_sft.py \
     data.train_data_paths=/absolute/path/to/yam_dataset \
     actor.model.model_path=/absolute/path/to/pi05_pytorch \
     actor.model.openpi_data.norm_stats_path=/absolute/path/to/yam_dataset/norm_stats.json \
     validation.mode=train validation.device=cuda validation.steps=2

数据检查确认三路 RGB 已启用、语言 token 存在、32 维模型张量数值有限，以及动作窗口为 10 帧。训练检查加载 checkpoint，检查 loss 为有限标量、梯度有限且非零，再在内存中执行 AdamW 更新；不会覆盖 checkpoint 文件。通过这些检查只能证明数据与优化链路兼容，不能证明 policy 质量或真机运行安全。

验证完成后，使用现有入口启动完整 SFT：

.. code-block:: bash

   python examples/sft/train_vla_sft.py \
     --config-name realworld_dual_yam_sft_openpi_pi05 \
     data.train_data_paths=/absolute/path/to/yam_dataset \
     actor.model.model_path=/absolute/path/to/pi05_pytorch \
     actor.model.openpi_data.norm_stats_path=/absolute/path/to/yam_dataset/norm_stats.json

该命令通过 YAM 配置启动现有 Ray/FSDP SFT runner。检查训练 loss 和保存的 checkpoint 后再安排推理验证；真机执行需另行完成有现场操作员参加的安全检查。
