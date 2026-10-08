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

"""Validate an OpenPI SFT batch and optionally update a local checkpoint copy."""

import json
import math
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from rlinf.utils.logging import get_logger


def validate_batch(batch: tuple[Any, torch.Tensor], model_cfg: DictConfig) -> dict:
    """Check the model-facing state, action window, and three RGB slots."""
    observation, actions = batch
    if actions.ndim != 3:
        raise ValueError(
            f"Expected a batched action window with three axes, got {actions.shape}."
        )
    batch_size = actions.shape[0]
    model_dim = int(model_cfg.openpi.model_action_dim)
    horizon = int(model_cfg.openpi.action_horizon)
    if tuple(actions.shape) != (batch_size, horizon, model_dim):
        raise ValueError(
            f"Expected actions shaped (batch, {horizon}, {model_dim}), got {actions.shape}."
        )
    if tuple(observation.state.shape) != (batch_size, model_dim):
        raise ValueError(
            f"Expected state shaped (batch, {model_dim}), got {observation.state.shape}."
        )
    if not torch.isfinite(actions).all() or not torch.isfinite(observation.state).all():
        raise ValueError("SFT state and actions must be finite.")
    expected_keys = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    if (
        set(observation.images) != expected_keys
        or set(observation.image_masks) != expected_keys
    ):
        raise ValueError(
            f"Expected the three OpenPI RGB slots {sorted(expected_keys)}."
        )
    for key, image in observation.images.items():
        if tuple(image.shape) != (batch_size, 3, 224, 224):
            raise ValueError(
                f"Expected {key} shaped (batch, 3, 224, 224), got {image.shape}."
            )
        mask = torch.as_tensor(observation.image_masks[key])
        if (
            tuple(mask.shape) != (batch_size,)
            or not torch.isfinite(image).all()
            or not mask.all()
        ):
            raise ValueError(f"Expected a finite, enabled RGB image for {key}.")
    if (
        observation.tokenized_prompt is None
        or observation.tokenized_prompt_mask is None
    ):
        raise ValueError("SFT batches require a tokenized language prompt and mask.")
    if not torch.as_tensor(observation.tokenized_prompt_mask).any(dim=-1).all():
        raise ValueError("Each SFT sample requires at least one language token.")
    return {
        "batch_size": batch_size,
        "action_horizon": horizon,
        "model_action_dim": model_dim,
    }


def apply_validation_step(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad: float,
) -> dict:
    """Require a finite scalar loss and gradients before an optimizer update."""
    if not math.isfinite(clip_grad) or clip_grad <= 0:
        raise ValueError(
            f"Gradient clipping threshold must be positive, got {clip_grad}."
        )
    if loss.numel() != 1 or not torch.isfinite(loss).all():
        raise ValueError("SFT validation requires a finite scalar loss.")
    loss.backward()
    parameters = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    if not parameters or any(
        not torch.isfinite(parameter.grad).all() for parameter in parameters
    ):
        raise ValueError("SFT backward must produce finite gradients.")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters, clip_grad, error_if_nonfinite=True
    )
    if grad_norm.item() <= 0:
        raise ValueError("SFT backward produced only zero gradients.")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": loss.detach().item(), "grad_norm": grad_norm.item()}


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_dual_yam_sft_openpi_pi05",
)
def main(cfg: DictConfig) -> None:
    mode = str(cfg.validation.mode)
    steps = int(cfg.validation.steps)
    if mode not in ("data", "train") or steps < 1:
        raise ValueError(
            "validation.mode must be 'data' or 'train', and steps must be positive."
        )

    from rlinf.data.datasets.openpi import build_openpi_sft_dataloader
    from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root

    dataset_root = resolve_lerobot_dataset_root(str(cfg.data.train_data_paths))
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Expected a finalized LeRobot dataset at {dataset_root}."
        )

    loader, _ = build_openpi_sft_dataloader(
        cfg, world_size=1, rank=0, data_paths=cfg.data.train_data_paths
    )
    model = None
    optimizer = None
    if mode == "train":
        from rlinf.models import get_model

        torch.manual_seed(int(cfg.actor.seed))
        model = get_model(cfg.actor.model).to(torch.device(cfg.validation.device))
        model.train()
        optim_cfg = cfg.actor.optim
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(optim_cfg.lr),
            betas=(float(optim_cfg.adam_beta1), float(optim_cfg.adam_beta2)),
            eps=float(optim_cfg.adam_eps),
            weight_decay=float(optim_cfg.weight_decay),
        )
    iterator = iter(loader)
    logger = get_logger()
    for step in range(steps):
        batch = next(iterator)
        metrics = validate_batch(batch, cfg.actor.model)
        if model is not None:
            loss = model.sft_forward(batch)
            metrics.update(
                apply_validation_step(
                    loss, model, optimizer, float(cfg.actor.optim.clip_grad)
                )
            )
        logger.info(
            "OpenPI SFT validation: %s",
            json.dumps({"mode": mode, "step": step + 1, **metrics}),
        )


if __name__ == "__main__":
    main()
