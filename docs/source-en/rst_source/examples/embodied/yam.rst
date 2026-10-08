Collect Dual-Arm YAM Demonstrations with PICO
=============================================

Collect dual-arm YAM demonstrations with PICO controllers and three RealSense RGB cameras. The recipe records absolute joint position and action vectors in the order ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``; each vector has 14 values, and grippers use ``0=closed, 1=open``.

Prepare the station
-------------------

Use two YAM follower arms with ``flexible_4310`` grippers, three RealSense cameras, a PICO headset with two controllers, and a ZeroMQ PICO publisher. Follow :doc:`the Franka VR setup <franka_vr>` to prepare XRoboToolkit, its PC Service, and the publisher. Install the YAM runtime in a Python environment:

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

The installer includes the pinned ``i2rt`` backend and PICO transport dependencies. Check that the installed robot model, tool center point, joint limits, controller-to-base yaw, and motion scales match your hardware before enabling motion.

.. warning::

   The collector opens both follower CAN devices and all three cameras on its first reset. Keep the arms supported in a verified safe pose, clear the motion area, and have an operator at the emergency stop. Automatic reset and parking are disabled in this recipe. Before ending a session, the operator must confirm that both arms can safely remain supported when motor output is released.

Configure and collect
---------------------

Set the two follower CAN interfaces, three camera serial numbers, PICO publisher address, and measured operator-to-robot yaw for each arm. Export these values **before** starting Ray so its environment worker receives them:

.. code-block:: bash

   export YAM_LEFT_FOLLOWER_CAN='<left-can-interface>'
   export YAM_RIGHT_FOLLOWER_CAN='<right-can-interface>'
   export YAM_TOP_CAMERA_SERIAL='<top-serial>'
   export YAM_LEFT_CAMERA_SERIAL='<left-serial>'
   export YAM_RIGHT_CAMERA_SERIAL='<right-serial>'
   export YAM_PICO_ZMQ_ADDR='tcp://<publisher-host>:5555'
   export YAM_LEFT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'
   export YAM_RIGHT_OPERATOR_TO_ROBOT_YAW='<measured-radians>'

Replace every placeholder with a value checked at your station. The example configuration is ``examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml``. It assigns the environment worker and hardware to one Ray node. On a fresh single-node host, start Ray and launch collection:

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head
   bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

Hold a controller's grip to control its corresponding arm; release it to stop taking over. The recipe maps the right menu button to start/end an episode and the left menu button to discard the current recording. Confirm these button events on your PICO setup before collecting task data. Discard removes only the current unfinished episode; faulted or truncated recordings can still be saved as unsuccessful episodes.

The launcher writes logs under ``logs/<timestamp>-realworld_dual_yam_collect_data_pico/`` by default. LeRobot data is written under that run's ``collected_data/rank_*/`` directory. Streaming saves recorded episodes, including unsuccessful ones, with an ``is_success`` flag; inspect that flag and the three RGB views before using the dataset. A saved episode or an ``is_success`` flag does not replace the operator's task-result check.

Fine-tune from recorded data
------------------------------

Fine-tune Pi0.5 on a finalized YAM LeRobot dataset before deploying a policy.
The SFT recipe uses recorded files and starts only actor workers. Select one
curated dataset directory containing ``meta/info.json``; the collection run's
``rank_*/`` parent directory is not a dataset. Streaming collection can produce
one ``id_*/`` shard per episode, so prepare a curated dataset containing the
episodes you intend to train on. Check task labels and success/fault outcomes
before including recordings.

Merge the selected finalized shards with the existing preparation tool:

.. code-block:: bash

   python toolkits/lerobot/merge_lerobot_datasets.py \
     --source-dir /absolute/path/to/selected_shards \
     --output-dir /absolute/path/to/yam_dataset --dry-run
   python toolkits/lerobot/merge_lerobot_datasets.py \
     --source-dir /absolute/path/to/selected_shards \
     --output-dir /absolute/path/to/yam_dataset

The first command lists the inputs without writing. Review that list before
the second command merges all episodes under those selected input directories.
It does not filter successful episodes automatically; curate inputs first.

Install the model and YAM dependencies in a separate environment:

.. code-block:: bash

   bash requirements/install.sh embodied --venv .venv-yam-openpi --model openpi --env yam
   source .venv-yam-openpi/bin/activate
   export PYTHONPATH="$PWD:$PYTHONPATH"
   export EMBODIED_PATH="$PWD/examples/embodiment"

This explicitly adds OpenPI to the YAM environment. The collection-only command
remains unchanged. A container uses ``BUILD_TARGET=embodied-yam-openpi``.
Supply a Pi0.5 PyTorch checkpoint separately; the installer downloads model
runtime assets, not your fine-tuned policy weights.

The ``pi05_yam`` data configuration preserves absolute joint targets and the
left-arm-first 14-value order. It maps ``image``, ``extra_view_image-0`` and
``extra_view_image-1`` to OpenPI's base, left and right RGB slots. All three
views are required. Model state/actions are zero-padded to 32 values, and the
prediction horizon is 10 frames. It does not apply Aloha joint/gripper
conversions or convert absolute targets into deltas.

Compute YAM normalization statistics rather than using another robot's stats:

.. code-block:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_yam --repo-id /absolute/path/to/yam_dataset

With an absolute dataset path, this tool writes ``norm_stats.json`` into that
dataset directory. Use enough frames for the configured 32-sample statistics
batch. Keep the checkpoint, dataset, and stats paths explicit in the SFT config
or command-line overrides.

Validate data on CPU first, then run forward, backward and optimizer updates
on a training machine with sufficient model memory:

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

The data check verifies three enabled RGB views, language tokens, finite
32-value model tensors, and the 10-frame action window. Train mode loads the
checkpoint, checks a finite scalar loss and nonzero finite gradients, then
applies AdamW updates in memory. It does not overwrite checkpoint files.
Passing these checks establishes data/optimization compatibility, not policy
quality or safe robot behavior.

Start full SFT through the existing entry point after validation:

.. code-block:: bash

   python examples/sft/train_vla_sft.py \
     --config-name realworld_dual_yam_sft_openpi_pi05 \
     data.train_data_paths=/absolute/path/to/yam_dataset \
     actor.model.model_path=/absolute/path/to/pi05_pytorch \
     actor.model.openpi_data.norm_stats_path=/absolute/path/to/yam_dataset/norm_stats.json

This launches the existing Ray/FSDP SFT runner using the YAM recipe. Read
training loss and saved checkpoints before planning inference validation;
hardware execution requires a separate attended safety check.
