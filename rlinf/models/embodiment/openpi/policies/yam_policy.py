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

"""Map absolute dual-YAM joint demonstrations to OpenPI tensors."""

import dataclasses

import numpy as np
from openpi import transforms


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an RGB image with three axes, got {image.shape}.")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating-point RGB images must be finite and in [0, 1].")
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 or floating-point RGB, got {image.dtype}.")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Keep left/right joint order and pad 14 absolute values to the model width."""

    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        if self.action_dim < 14:
            raise ValueError(
                f"YAM needs at least 14 model dimensions, got {self.action_dim}."
            )
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(
                f"Expected finite YAM state with shape (14,), got {state.shape}."
            )
        images = {
            "base_0_rgb": _parse_image(data["observation/image"]),
            "left_wrist_0_rgb": _parse_image(data["observation/left_image"]),
            "right_wrist_0_rgb": _parse_image(data["observation/right_image"]),
        }
        inputs = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": images,
            "image_mask": dict.fromkeys(images, np.True_),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[-1] != 14
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    f"Expected finite YAM actions with shape (horizon, 14), got {actions.shape}."
                )
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Return the first 14 absolute joint targets without changing arm order."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 14:
            raise ValueError(
                f"Expected model actions with at least 14 values, got {actions.shape}."
            )
        return {"actions": actions[:, :14]}
