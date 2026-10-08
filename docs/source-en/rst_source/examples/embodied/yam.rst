Collect Dual-Arm YAM Demonstrations with PICO
=============================================

Collect dual-arm YAM demonstrations with PICO controllers and three RealSense RGB cameras. The recipe records absolute joint position and action vectors in the order ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``; each vector has 14 values, and grippers use ``0=closed, 1=open``.

Check the environment first with fake vendor SDKs, then use the attended hardware check before collecting demonstrations. A short SAC/MLP test exercises the training path independently of the on-site checks.

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

Check Without Hardware
----------------------

After installation, run the hardware tool against fake ``i2rt`` and RealSense SDKs. It uses ``is_dummy: false`` so the YAM backend, camera adapters, feedback checks, and cleanup run through the same interfaces as hardware:

.. code-block:: bash

   python -m toolkits.realworld_check.test_yam_env --mock --once

For an interactive mock session, omit ``--once``. The tool starts by observing the 14-D feedback and three RGB streams. It closes the devices on exit.

On a GPU host, run the same short SAC test as CI:

.. code-block:: bash

   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh yam_mock_sac_mlp_reach

The launcher installs SDK fakes in the driver and worker processes. The MLP starts from random weights, uses 14-D observations and actions, and runs two training epochs. ``DualYamReachEnv-v1`` scores the measured distance to two six-joint targets; grippers remain in the action and observation but do not contribute to the reward. Configure ``target_joint_qpos``, ``reward_threshold``, ``success_hold_steps``, and ``use_dense_reward`` in ``env/yam_reach.yaml``. The preset target is for the mock test; choose a verified safe target before using this task on hardware. The collection task's reward stays unchanged.

Keep the training log and loss values as software smoke-test evidence. Mock rewards and losses do not measure real-robot task success.

Check On-Site Hardware
----------------------

Export the five CAN and camera variables from the station configuration, then let the operator confirm startup and shutdown safety. The check reads hardware configuration from ``realworld_dual_yam_collect_data_pico.yaml``; PICO is not used by this tool. Pass ``--config`` to select a station-specific copy, including its verified joint limits in ``env.eval.override_cfg``.

.. warning::

   Opening the SDK can calibrate the grippers and enable measured-pose hold. Even an observation-only check is not a passive hardware connection. Keep both arms supported throughout the check, clear the workspace, and keep an operator at the emergency stop. Closing releases motor output. Automatic reset and parking must remain disabled.

Read feedback once and save camera snapshots:

.. code-block:: bash

   python -m toolkits.realworld_check.test_yam_env \
       --confirm-hardware-ready --once --snapshot-dir /path/to/yam-check/frames

Check the joint order, finite feedback, and all three camera views. After the operator approves motion, omit ``--once`` and add ``--enable-motion``. Use ``observe``, ``joint left 0 0.01``, ``gripper right 0.05``, and ``quit``. Each move requires typing ``MOVE``; joint increments are limited to 0.02 radians and gripper increments to 0.05 of the opening range. A runtime failure ends the check and closes the devices rather than retrying motion.

Record the hardware configuration, feedback, small-motion and gripper results, shutdown result, logs, and an on-site video separately from the mock test. Inspect these artifacts with the operator before treating hardware checks as complete.
