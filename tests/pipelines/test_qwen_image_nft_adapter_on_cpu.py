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
"""CPU tests for QwenImageNFT training adapter (T2I, no condition image).

Necessity: ``QwenImageNFT`` is the NFT training adapter for the standard
Qwen-Image text-to-image model. It must (a) be discoverable via the
``DiffusionModelBase`` registry under ``architecture="QwenImagePipeline" +
algorithm="nft"``, (b) inherit FlowGRPO's input preparation unchanged so the
trainer side only sees forward-process noise prediction, and (c) refuse to
serve ``forward_and_sample_previous_step`` (NFT has no reverse SDE).

The Qwen-Image-Edit NFT counterpart is covered separately in
``test_qwen_image_edit_adapter_on_cpu.py``.
"""

from unittest.mock import MagicMock

import pytest
import torch
from tensordict import NonTensorData, TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage
from verl_omni.pipelines.qwen_image_nft.diffusers_training_adapter import QwenImageNFT
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import (
    DiffusionPipelineConfig,
    DiffusionRolloutAlgoConfig,
)

# ---------------------------------------------------------------------------
# Tensor dimensions used throughout the tests
# ---------------------------------------------------------------------------
_B = 2
_N = 3
_LS = 16  # latent seq len
_LC = 8  # latent channels
_TS = 12  # text seq len
_TC = 64  # text channels


def _make_model_config(
    *,
    algorithm: str = "nft",
    true_cfg_scale: float = 1.0,
    noise_level: float = 0.7,
    sde_type: str = "sde",
) -> DiffusionModelConfig:
    """Build a minimal DiffusionModelConfig without triggering __post_init__."""
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", "QwenImagePipeline")
    object.__setattr__(cfg, "algorithm", algorithm)
    object.__setattr__(cfg, "external_lib", None)
    object.__setattr__(cfg, "pipeline", DiffusionPipelineConfig(true_cfg_scale=true_cfg_scale))
    object.__setattr__(cfg, "algo", DiffusionRolloutAlgoConfig(noise_level=noise_level, sde_type=sde_type))
    return cfg


def _batch_tensors():
    return {
        "all_latents": torch.randn(_B, _N + 1, _LS, _LC),
        "all_timesteps": torch.rand(_B, _N) * 1000,
        "prompt_embeds": torch.randn(_B, _TS, _TC),
        "prompt_embeds_mask": torch.ones(_B, _TS, dtype=torch.bool),
        "negative_prompt_embeds": torch.randn(_B, _TS, _TC),
        "negative_prompt_embeds_mask": torch.ones(_B, _TS, dtype=torch.bool),
    }


def _mock_module_no_guidance() -> MagicMock:
    """Build a MagicMock transformer that disables guidance_embeds.

    QwenImage's prepare_model_inputs reads ``module.config.guidance_embeds``;
    a default MagicMock returns a truthy attribute and routes into the
    ``torch.full([1], guidance_scale, ...)`` path, which then crashes when
    the config's ``guidance_scale`` defaults to ``None``. Pin the attribute
    so the no-guidance branch runs.
    """
    module = MagicMock()
    module.config.guidance_embeds = False
    return module


def _make_micro_batch_t2i() -> TensorDict:
    """T2I micro_batch — no condition image latents.

    QwenImage's ``prepare_model_inputs`` always rebuilds img_shapes from
    height/width/vae_scale_factor in the micro_batch (it doesn't consume a
    pre-built ``img_shapes`` like the edit adapter does), so we must provide
    those three non-tensor metadata fields.
    """
    td = TensorDict({}, batch_size=_B)
    # 64 / 8 // 2 = 4 → matches _LS // 4 in the assertions
    td["height"] = NonTensorData(64)
    td["width"] = NonTensorData(64)
    td["vae_scale_factor"] = NonTensorData(8)
    return td


# ===========================================================================
# 1. Registry
# ===========================================================================


class TestQwenImageNFTRegistry:
    def test_registered_for_qwen_image_pipeline_and_nft(self):
        cfg = _make_model_config(algorithm="nft")
        assert DiffusionModelBase.get_class(cfg) is QwenImageNFT

    def test_inherits_from_qwen_image_flow_grpo_adapter(self):
        # NFT shares scheduler setup, prepare_model_inputs, etc. with FlowGRPO —
        # the only behavioural difference is the disabled reverse-SDE step.
        assert issubclass(QwenImageNFT, QwenImage)

    def test_flow_grpo_and_nft_resolve_to_distinct_classes(self):
        flow_grpo_cfg = _make_model_config(algorithm="flow_grpo")
        nft_cfg = _make_model_config(algorithm="nft")
        flow_grpo_cls = DiffusionModelBase.get_class(flow_grpo_cfg)
        nft_cls = DiffusionModelBase.get_class(nft_cfg)
        # Even though NFT inherits from QwenImage, registry lookup must return
        # the more specific class for algorithm="nft".
        assert nft_cls is QwenImageNFT
        assert flow_grpo_cls is QwenImage
        assert flow_grpo_cls is not nft_cls


# ===========================================================================
# 2. Reuses FlowGRPO prepare_model_inputs unchanged
# ===========================================================================


class TestQwenImageNFTPrepareModelInputs:
    """NFT inherits FlowGRPO's prepare_model_inputs verbatim. We assert this
    behaviour rather than calling super() so a future override in the NFT
    adapter that breaks the contract gets caught here.
    """

    def test_routes_positive_prompt_into_model_inputs(self):
        tensors = _batch_tensors()
        micro_batch = _make_micro_batch_t2i()

        model_inputs, negative_model_inputs = QwenImageNFT.prepare_model_inputs(
            module=_mock_module_no_guidance(),
            model_config=_make_model_config(),
            latents=tensors["all_latents"],
            timesteps=tensors["all_timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=0,
        )

        torch.testing.assert_close(model_inputs["encoder_hidden_states"], tensors["prompt_embeds"])
        torch.testing.assert_close(negative_model_inputs["encoder_hidden_states"], tensors["negative_prompt_embeds"])

    def test_hidden_states_match_target_latent_shape(self):
        tensors = _batch_tensors()
        micro_batch = _make_micro_batch_t2i()

        model_inputs, _ = QwenImageNFT.prepare_model_inputs(
            module=_mock_module_no_guidance(),
            model_config=_make_model_config(),
            latents=tensors["all_latents"],
            timesteps=tensors["all_timesteps"],
            prompt_embeds=tensors["prompt_embeds"],
            prompt_embeds_mask=tensors["prompt_embeds_mask"],
            negative_prompt_embeds=tensors["negative_prompt_embeds"],
            negative_prompt_embeds_mask=tensors["negative_prompt_embeds_mask"],
            micro_batch=micro_batch,
            step=0,
        )

        # T2I has no condition image — hidden_states must be exactly the target latent.
        torch.testing.assert_close(model_inputs["hidden_states"], tensors["all_latents"][:, 0])


# ===========================================================================
# 3. forward_and_sample_previous_step is disabled
# ===========================================================================


class TestQwenImageNFTReverseSDEDisabled:
    def test_reverse_sde_call_raises_not_implemented(self):
        """NFT does not have a reverse-SDE log-prob path — calling this method
        means the wrong trainer routed the batch (likely PolicyGradientRayTrainer
        instead of DirectPreferenceRayTrainer).
        """
        with pytest.raises(NotImplementedError, match="forward_and_sample_previous_step"):
            QwenImageNFT.forward_and_sample_previous_step(
                module=MagicMock(),
                scheduler=MagicMock(),
                model_config=_make_model_config(),
                model_inputs={"hidden_states": torch.randn(_B, _LS, _LC)},
                negative_model_inputs=None,
                scheduler_inputs=None,
                step=0,
            )

    def test_error_message_points_at_trainer_type(self):
        """The error message must guide the user to fix ``algorithm.trainer_type``
        — that's the actual misconfiguration when this path fires.
        """
        with pytest.raises(NotImplementedError, match=r"trainer_type"):
            QwenImageNFT.forward_and_sample_previous_step(
                module=MagicMock(),
                scheduler=MagicMock(),
                model_config=_make_model_config(),
                model_inputs={"hidden_states": torch.randn(_B, _LS, _LC)},
                negative_model_inputs=None,
                scheduler_inputs=None,
                step=0,
            )
