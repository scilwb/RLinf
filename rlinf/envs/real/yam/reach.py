# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dual-arm YAM joint reaching with reward from measured feedback."""

from typing import Any

import numpy as np

from rlinf.robotics import DualYamConfig, RobotInfo
from rlinf.scheduler import WorkerInfo

from .dual_yam_joint_env import DualYamJointEnv


class DualYamReachEnv(DualYamJointEnv):
    """Reach and hold two six-joint targets, excluding grippers from scoring."""

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: WorkerInfo | None = None,
        robot_info: RobotInfo[DualYamConfig] | None = None,
        env_idx: int = 0,
    ) -> None:
        values = dict(override_cfg)
        self.target_joint_qpos = np.asarray(
            values.pop("target_joint_qpos", None), dtype=np.float64
        )
        if self.target_joint_qpos.shape != (2, 6) or not np.all(
            np.isfinite(self.target_joint_qpos)
        ):
            raise ValueError(
                "target_joint_qpos must contain two finite six-joint targets"
            )
        self.reward_threshold = float(values.pop("reward_threshold", 0.05))
        if not np.isfinite(self.reward_threshold) or self.reward_threshold <= 0:
            raise ValueError("reward_threshold must be finite and positive")
        self.success_hold_steps = values.pop("success_hold_steps", 1)
        if (
            isinstance(self.success_hold_steps, bool)
            or not isinstance(self.success_hold_steps, int)
            or self.success_hold_steps < 1
        ):
            raise ValueError("success_hold_steps must be a positive integer")
        self.use_dense_reward = values.pop("use_dense_reward", False)
        if not isinstance(self.use_dense_reward, bool):
            raise TypeError("use_dense_reward must be a bool")
        values.setdefault("task_description", "reach a dual-arm joint configuration")
        super().__init__(values, worker_info, robot_info, env_idx)
        if np.any(self.target_joint_qpos < self.config.joint_limit_min) or np.any(
            self.target_joint_qpos > self.config.joint_limit_max
        ):
            raise ValueError("target_joint_qpos must be within configured joint limits")
        self._success_hold_counter = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset episode counters; physical reset follows the base env config."""
        self._success_hold_counter = 0
        return super().reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Score measured joint distance after the bounded command is applied."""
        observation, _, _, truncated, info = super().step(action)
        measured = observation["state"]["joint_position"].reshape(2, 7)[:, :6]
        distance = np.abs(measured - self.target_joint_qpos)
        reached = bool(np.all(distance < self.reward_threshold))
        self._success_hold_counter = self._success_hold_counter + 1 if reached else 0
        terminated = self._success_hold_counter >= self.success_hold_steps
        joint_distance = float(np.linalg.norm(distance))
        reward = -joint_distance if self.use_dense_reward else float(reached)
        info["joint_distance"] = joint_distance
        return observation, reward, terminated, truncated, info
