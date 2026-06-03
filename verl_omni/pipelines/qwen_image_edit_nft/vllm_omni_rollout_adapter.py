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
Qwen-Image-Edit-Plus rollout adapter for DiffusionNFT algorithm.

NFT only needs the final clean latents from the rollout — no per-step
log-probabilities are required during rollout. This adapter overrides
`forward()` to default `logprobs=False`, saving compute by skipping
the per-step log-prob calculation in the SDE scheduler.

The SDE noise injection is still used during rollout (controlled by
`algo.noise_level`) for exploration diversity, but only the final
denoised image is retained for NFT training.
"""

from typing import Any

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_edit_flow_grpo.vllm_omni_rollout_adapter import (
    QwenImageEditPlusPipelineWithLogProb,
)

__all__ = ["QwenImageEditPlusNFTPipelineWithLogProb"]


@VllmOmniPipelineBase.register("QwenImageEditPlusPipeline", algorithm="nft")
class QwenImageEditPlusNFTPipelineWithLogProb(QwenImageEditPlusPipelineWithLogProb):
    """Rollout pipeline for Qwen-Image-Edit-Plus under the NFT algorithm.

    Inherits from the FlowGRPO rollout adapter (which handles condition-image
    latent encoding and concatenation) but defaults `logprobs=False` to skip
    unnecessary per-step log-probability computation. The NFT training loop
    only uses `all_latents[:, -1]` (the final clean latent) — intermediate
    log-probs are never consumed.

    If `logprobs` is explicitly set to `True` via `sampling_params.extra_args`,
    that override is still respected (useful for debugging/analysis).
    """

    def forward(self, req: OmniDiffusionRequest, **kwargs: Any) -> DiffusionOutput:
        """Run rollout with log-prob computation disabled by default.

        NFT does not need per-step log-probs from the rollout. This override
        sets `logprobs=False` unless explicitly overridden in sampling_params.
        """
        # Default logprobs to False for NFT (saves compute in SDE scheduler)
        if "logprobs" not in kwargs:
            kwargs["logprobs"] = False
        # Respect explicit override from sampling_params.extra_args
        extra_args = req.sampling_params.extra_args
        if "logprobs" not in extra_args:
            extra_args["logprobs"] = False

        return super().forward(req, **kwargs)
