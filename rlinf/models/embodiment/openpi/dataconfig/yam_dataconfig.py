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

"""OpenPI data configuration for the RLinf dual-YAM collection schema."""

import dataclasses
import pathlib

import openpi.models.model as model
from openpi import transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from rlinf.models.embodiment.openpi.policies.yam_policy import YamInputs, YamOutputs


@dataclasses.dataclass(frozen=True)
class LeRobotYamDataConfig(DataConfigFactory):
    """Read top/left/right RGB and absolute 14-dimensional state/actions."""

    default_prompt: str | None = None

    def create(
        self, assets_dirs: pathlib.Path, model_config: model.BaseModelConfig
    ) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/left_image": "extra_view_image-0",
                            "observation/right_image": "extra_view_image-1",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            data_transforms=transforms.Group(
                inputs=[YamInputs(action_dim=model_config.action_dim)],
                outputs=[YamOutputs()],
            ),
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(
                model_config
            ),
            use_quantile_norm=model_config.model_type == model.ModelType.PI05,
        )
