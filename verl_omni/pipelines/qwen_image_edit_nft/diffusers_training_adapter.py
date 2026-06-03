# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen-Image-Edit-Plus training-side adapter for DiffusionNFT algorithm.

NFT does not use reverse-SDE log-probs. Instead, it samples timesteps
independently and computes a forward-process matching loss. The training
adapter only needs to produce noise predictions for given noised latents.

Key difference from FlowGRPO:
  - `forward_and_sample_previous_step()` is NOT used by the NFT trainer.
    The DirectPreferenceRayTrainer calls `prepare_model_inputs()` to build inputs,
    then runs the model forward directly to get noise_pred for the NFT loss.
  - `build_scheduler()`, `set_timesteps()`, and `prepare_model_inputs()` (with
    condition-image latent concatenation) are reused identically from the
    FlowGRPO adapter.
"""

from typing import Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_edit_flow_grpo.diffusers_training_adapter import QwenImageEditPlus
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["QwenImageEditPlusNFT"]


@DiffusionModelBase.register("QwenImageEditPlusPipeline", algorithm="nft")
class QwenImageEditPlusNFT(QwenImageEditPlus):
    """Training adapter for Qwen-Image-Edit-Plus under the DiffusionNFT algorithm.

    Reuses the FlowGRPO adapter's scheduler setup and input preparation
    (including condition-image latent concatenation). The NFT training loop
    (DirectPreferenceRayTrainer) handles the per-timestep forward-process
    optimization independently:
      1. Calls `prepare_model_inputs()` to build transformer inputs (with
         condition image latents concatenated).
      2. Runs the model forward to get noise_pred (v-prediction).
      3. Slices noise_pred to target tokens only (excluding condition tokens).
      4. Computes the positive/negative MSE loss weighted by advantages.

    `forward_and_sample_previous_step()` is deliberately disabled since NFT
    does not perform reverse-SDE log-prob computation during training.
    """

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Not used by NFT — raises an error if called accidentally.

        NFT computes forward-process matching loss rather than reverse-SDE
        log-probabilities. The DirectPreferenceRayTrainer never calls this
        method; if you see this error, check that `algorithm.trainer_type`
        is set to `direct_preference` (not `policy_gradient`).
        """
        raise NotImplementedError(
            "QwenImageEditPlusNFT does not support forward_and_sample_previous_step(). "
            "DiffusionNFT uses forward-process matching loss and does not require "
            "reverse-SDE log-probabilities. Ensure algorithm.trainer_type='direct_preference'."
        )
