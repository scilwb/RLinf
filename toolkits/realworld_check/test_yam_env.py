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

"""Inspect YAM feedback and cameras, then optionally check small manual moves."""

import argparse
import contextlib
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.real.yam import DualYamJointEnv
from rlinf.robotics import DualYamConfig, RobotInfo
from rlinf.utils.logging import get_logger

_ROOT = Path(__file__).resolve().parents[2]
_HELP = "observe | joint <left/right> <0..5> <delta radians> | gripper <left/right> <delta opening> | quit"


def _load_station(path: Path) -> tuple[dict[str, Any], RobotInfo]:
    config = OmegaConf.load(path)
    stations = [
        station
        for group in config.cluster.node_groups
        if group.hardware.type == "DualYam"
        for station in group.hardware.configs
    ]
    if len(stations) != 1:
        raise ValueError(
            f"hardware check needs exactly one DualYam station, got {len(stations)}"
        )
    hardware = DualYamConfig(**OmegaConf.to_container(stations[0], resolve=True))
    base = OmegaConf.load(
        _ROOT / "examples/embodiment/config/env/realworld_dual_yam_joint.yaml"
    ).override_cfg
    settings = OmegaConf.to_container(
        OmegaConf.merge(base, config.env.eval.get("override_cfg", {})), resolve=True
    )
    if (
        settings["is_dummy"]
        or not settings.get("enforce_runtime_joint_limits", True)
        or settings.get("reset", {}).get("enabled", False)
        or settings.get("park_on_close", {}).get("enabled", False)
    ):
        raise ValueError(
            "hardware check requires SDK control, bounded joints, and disabled reset/parking"
        )
    return settings, RobotInfo(type="DualYam", model="DualYam", config=hardware)


def _action_for_command(command: str, measured: np.ndarray) -> np.ndarray:
    """Build one bounded manual target from the latest measured positions."""
    parts = shlex.split(command)
    if len(parts) not in (3, 4) or parts[1] not in ("left", "right"):
        raise ValueError(_HELP)
    offset = 0 if parts[1] == "left" else 7
    if parts[0] == "joint" and len(parts) == 4:
        joint = int(parts[2])
        if not 0 <= joint < 6:
            raise ValueError("joint index must be between 0 and 5")
        index, delta, limit = offset + joint, float(parts[3]), 0.02
    elif parts[0] == "gripper" and len(parts) == 3:
        index, delta, limit = offset + 6, float(parts[2]), 0.05
    else:
        raise ValueError(_HELP)
    if not np.isfinite(delta) or abs(delta) > limit:
        raise ValueError(f"delta must be finite and within [-{limit}, {limit}]")
    action = np.asarray(measured, dtype=np.float32).copy()
    action[index] += delta
    return action


def _report(env: DualYamJointEnv, snapshot_dir: Path | None) -> None:
    observation = env.observe()
    logger = get_logger()
    logger.info("joint_position=%s", observation["state"]["joint_position"].tolist())
    for name, frame in observation["frames"].items():
        logger.info("%s: shape=%s dtype=%s", name, frame.shape, frame.dtype)
        if snapshot_dir is not None:
            import cv2

            snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = snapshot_dir / f"{name}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise OSError(f"could not save camera snapshot to {path}")


def main() -> None:
    """Run an attended check without automatic reset or parking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="station YAML in the collection config format"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use fake vendor SDKs, not a dummy environment",
    )
    parser.add_argument(
        "--once", action="store_true", help="read feedback and frames once, then close"
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, help="save the latest RGB frame from each camera"
    )
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="allow individually confirmed small moves",
    )
    parser.add_argument(
        "--confirm-hardware-ready",
        action="store_true",
        help="operator confirms clear workspace, emergency stop, startup calibration, and supported torque-off",
    )
    args = parser.parse_args()
    if not args.mock and not args.confirm_hardware_ready:
        parser.error(
            "an on-site operator must pass --confirm-hardware-ready before opening hardware"
        )
    path = args.config or _ROOT / (
        "tests/e2e_tests/embodied/yam_mock_sac_mlp_reach.yaml"
        if args.mock
        else "examples/embodiment/config/realworld_dual_yam_collect_data_pico.yaml"
    )
    if args.mock:
        sys.path.insert(0, str(_ROOT / "tests"))
        from robot_mocks import mocked_sdks

        context = mocked_sdks()
    else:
        context = contextlib.nullcontext()
    with context:
        settings, robot_info = _load_station(path)
        env = DualYamJointEnv(settings, robot_info=robot_info)
        logger = get_logger()
        try:
            _report(env, args.snapshot_dir)
            while not args.once:
                command = input("yam> ").strip()
                if command == "quit":
                    break
                if command == "observe":
                    _report(env, args.snapshot_dir)
                    continue
                if not args.enable_motion:
                    logger.info("motion disabled; %s", _HELP)
                    continue
                try:
                    _action_for_command(command, env.get_joint_positions())
                except ValueError as error:
                    logger.warning("%s", error)
                    continue
                if input("Confirm clear workspace; type MOVE: ").strip() != "MOVE":
                    continue
                action = _action_for_command(command, env.get_joint_positions())
                _, _, _, _, info = env.step(action)
                if info["action_rejected"]:
                    raise RuntimeError(
                        f"YAM rejected the move: {info['action_rejected']}"
                    )
                logger.info(
                    "accepted_action=%s clipped=%s",
                    info["accepted_action"].tolist(),
                    info["action_clipped"],
                )
                _report(env, args.snapshot_dir)
        except (EOFError, KeyboardInterrupt):
            logger.info("ending attended check")
        finally:
            env.close()
            logger.info("YAM runtime and cameras closed")


if __name__ == "__main__":
    main()
