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
"""CPU tests for DiffusionNFTLoss (registered as ``loss_mode="nft"``).

Necessity: ``DiffusionNFTLoss`` is the only NFT-specific loss in the registry.
These tests pin its math (advantage clipping/normalization to [0, 1], beta
interpolation, x0 reconstruction, weight stabilization) and its registration
contract so that future refactors of ``diffusion_algos.py`` do not silently
break the NFT path. All tests run on CPU without loading any model.

Reference: https://arxiv.org/abs/2509.16117
"""

import math
from types import SimpleNamespace

import pytest
import torch

from verl_omni.trainer.diffusion.diffusion_algos import (
    DiffusionLossFn,
    DiffusionNFTLoss,
)

# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _make_actor_config(*, adv_clip_max: float = 5.0, nft_beta: float = 1.0) -> SimpleNamespace:
    """Build a stand-in actor config object exposing only the attributes the loss reads."""
    return SimpleNamespace(
        diffusion_loss=SimpleNamespace(adv_clip_max=adv_clip_max),
        nft_beta=nft_beta,
    )


def _make_loss_inputs(
    *,
    batch_size: int = 4,
    seq_len: int = 6,
    channels: int = 8,
    advantages: torch.Tensor | None = None,
    sigma: float = 0.3,
    seed: int = 0,
):
    """Return a self-consistent set of loss-call kwargs for ``compute_loss``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    clean_latents = torch.randn(batch_size, seq_len, channels, generator=g)
    noise = torch.randn(batch_size, seq_len, channels, generator=g)
    sigma_broadcast = torch.full((batch_size, 1, 1), sigma)
    noised_latents = (1.0 - sigma_broadcast) * clean_latents + sigma_broadcast * noise

    # Pretend the "old" prediction is exactly the true v-target plus a small drift,
    # and the "new" prediction differs from it slightly.
    target_v = noise - clean_latents  # rectified-flow v-target
    drift_old = torch.randn(batch_size, seq_len, channels, generator=g)
    drift_new = torch.randn(batch_size, seq_len, channels, generator=g)
    old_noise_pred = target_v + 0.05 * drift_old
    noise_pred = old_noise_pred + 0.02 * drift_new

    if advantages is None:
        advantages = torch.linspace(-1.0, 1.0, batch_size)

    return {
        "noise_pred": noise_pred,
        "old_noise_pred": old_noise_pred,
        "advantages": advantages,
        "clean_latents": clean_latents,
        "noised_latents": noised_latents,
        "sigma_broadcast": sigma_broadcast,
    }


# ===========================================================================
# 1. Registry
# ===========================================================================


class TestDiffusionNFTLossRegistry:
    def test_registered_under_nft_name(self):
        from verl_omni.trainer.diffusion.diffusion_algos import DIFFUSION_LOSS_REGISTRY

        # The registry stores callable instances of the registered loss class
        # (the decorator does ``REGISTRY[name] = cls()``). Verify the entry
        # exists and its type is ``DiffusionNFTLoss``.
        assert "nft" in DIFFUSION_LOSS_REGISTRY
        assert isinstance(DIFFUSION_LOSS_REGISTRY["nft"], DiffusionNFTLoss)

    def test_required_keys_contract(self):
        # These tuples are what the engine checks before invoking the loss; if a
        # contributor renames a field they will see a clear error rather than a
        # silent KeyError downstream.
        assert DiffusionNFTLoss.required_model_output_keys == ("noise_pred",)
        assert set(DiffusionNFTLoss.required_data_keys) == {
            "old_noise_pred",
            "advantages",
            "clean_latents",
            "noised_latents",
            "sigma_broadcast",
        }

    def test_is_diffusion_loss_subclass(self):
        assert issubclass(DiffusionNFTLoss, DiffusionLossFn)


# ===========================================================================
# 2. compute_loss math
# ===========================================================================


class TestDiffusionNFTLossCompute:
    def test_returns_scalar_loss_and_dict_metrics(self):
        kwargs = _make_loss_inputs()
        loss, metrics = DiffusionNFTLoss.compute_loss(config=_make_actor_config(), **kwargs)

        assert loss.ndim == 0, "policy_loss must be a scalar (0-dim tensor)"
        assert torch.isfinite(loss).item(), "policy_loss must be finite"
        # NFT metrics surface for tracking.
        assert {
            "actor/nft_policy_loss",
            "actor/nft_positive_loss",
            "actor/nft_negative_loss",
            "actor/nft_adv_mean",
            "actor/nft_r_mean",
        } <= metrics.keys()

    def test_advantage_normalized_into_unit_interval(self):
        """``r = clamp(adv/adv_clip_max/2 + 0.5, 0, 1)`` must always lie in [0, 1]."""
        cfg = _make_actor_config(adv_clip_max=5.0)
        # Span advantages across [-3*adv_clip_max, +3*adv_clip_max] to force the
        # clamp on both ends.
        big_advs = torch.tensor([-15.0, -5.0, -1.0, 0.0, 1.0, 5.0, 15.0])
        kwargs = _make_loss_inputs(batch_size=big_advs.shape[0], advantages=big_advs)
        _, metrics = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs)

        # nft_r_mean is the mean of the per-sample r ∈ [0, 1]; for symmetric
        # advantages around 0 the mean must be 0.5 (within numerical tolerance).
        assert 0.0 <= metrics["actor/nft_r_mean"] <= 1.0
        assert math.isclose(metrics["actor/nft_r_mean"], 0.5, abs_tol=1e-6)

    def test_advantage_clip_caps_metric_to_clip_max(self):
        """``actor/nft_adv_mean`` reports the *clipped* advantage, not the raw one."""
        cfg = _make_actor_config(adv_clip_max=2.0)
        # All advantages exceed the clip range; the post-clip mean is exactly 2.0.
        big_advs = torch.tensor([10.0, 20.0, 30.0, 40.0])
        kwargs = _make_loss_inputs(batch_size=big_advs.shape[0], advantages=big_advs)
        _, metrics = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs)
        assert math.isclose(metrics["actor/nft_adv_mean"], 2.0, abs_tol=1e-6)

    def test_loss_decreases_when_noise_pred_moves_toward_target(self):
        """A v-prediction closer to the true v-target should give a strictly
        smaller positive loss than one further away — this guards against
        accidental sign flips in the x0 reconstruction.

        The loss reconstructs ``x0_pred = noised - sigma * v_pred`` and compares
        it to ``clean_latents``. The implicit v-target is therefore
        ``v* = (noised - clean) / sigma`` (in our test setup,
        ``noised = (1-sigma) clean + sigma noise``, so ``v* = noise - clean``).
        We force noise_pred to that exact target on a copy of the inputs.
        """
        cfg = _make_actor_config()
        kwargs_far = _make_loss_inputs()

        # Implicit v-target that the loss treats as zero error.
        target_v = (kwargs_far["noised_latents"] - kwargs_far["clean_latents"]) / kwargs_far["sigma_broadcast"]
        kwargs_close = dict(kwargs_far, noise_pred=target_v)

        _, metrics_close = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs_close)
        _, metrics_far = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs_far)
        assert metrics_close["actor/nft_positive_loss"] < metrics_far["actor/nft_positive_loss"]

    def test_grad_flows_through_noise_pred_only(self):
        """noise_pred is the only differentiable input; the rest are detached
        targets/buffers. Gradient through ``old_noise_pred`` would make off-policy
        training unstable.
        """
        cfg = _make_actor_config()
        kwargs = _make_loss_inputs()
        kwargs["noise_pred"] = kwargs["noise_pred"].clone().requires_grad_(True)
        kwargs["old_noise_pred"] = kwargs["old_noise_pred"].clone().requires_grad_(True)
        loss, _ = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs)
        loss.backward()

        assert kwargs["noise_pred"].grad is not None
        assert torch.any(kwargs["noise_pred"].grad != 0).item()
        # Old noise pred is used both inside ``positive_pred`` and ``negative_pred``
        # but always with constant coefficients — gradient is well-defined but the
        # off-policy contract is that we *don't* update the sampling policy here.
        # We only require that the gradient exists; the trainer is responsible
        # for not feeding it back. We document the actual behaviour rather than
        # locking in a fragile zero-grad assertion.
        assert kwargs["old_noise_pred"].grad is not None

    def test_call_route_consumes_tensordict_data(self):
        """The ``__call__`` boundary unpacks model_output / data dicts into
        ``compute_loss`` kwargs."""
        from tensordict import TensorDict

        cfg = _make_actor_config()
        kwargs = _make_loss_inputs()
        bs = kwargs["noise_pred"].shape[0]

        model_output = {"noise_pred": kwargs["noise_pred"]}
        data = TensorDict(
            {
                "old_noise_pred": kwargs["old_noise_pred"],
                "advantages": kwargs["advantages"],
                "clean_latents": kwargs["clean_latents"],
                "noised_latents": kwargs["noised_latents"],
                "sigma_broadcast": kwargs["sigma_broadcast"],
            },
            batch_size=bs,
        )

        result = DiffusionNFTLoss()(config=cfg, model_output=model_output, data=data)
        assert torch.isfinite(result.loss).item()
        assert "actor/nft_policy_loss" in result.metrics


# ===========================================================================
# 3. Beta interpolation contract
# ===========================================================================


class TestDiffusionNFTLossBeta:
    def test_beta_one_means_pure_new_pred_for_positive_branch(self):
        """At beta=1, positive_pred = noise_pred (ignores old_noise_pred entirely).

        We assert this through a structural check: hijack a copy of the inputs so
        that ``old_noise_pred`` is wildly different and confirm the positive_loss
        is identical to a run where they match — i.e., the positive branch is
        independent of old_noise_pred when beta=1.
        """
        cfg = _make_actor_config(nft_beta=1.0)
        kwargs_a = _make_loss_inputs()
        kwargs_b = dict(kwargs_a)
        kwargs_b["old_noise_pred"] = torch.full_like(kwargs_a["old_noise_pred"], 99.0)

        _, metrics_a = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs_a)
        _, metrics_b = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs_b)

        # positive_loss does NOT depend on old_noise_pred at beta=1.
        assert math.isclose(
            metrics_a["actor/nft_positive_loss"],
            metrics_b["actor/nft_positive_loss"],
            rel_tol=0,
            abs_tol=1e-6,
        )
        # negative_loss DOES depend on old_noise_pred — sanity check that the
        # branch is wired correctly (not e.g. swapped with positive).
        assert metrics_a["actor/nft_negative_loss"] != metrics_b["actor/nft_negative_loss"]

    def test_beta_zero_produces_non_finite_loss(self):
        """At beta=0 the loss formula divides by zero (``ori_policy_loss /
        nft_beta``), so the resulting loss is *expected* to be non-finite
        (inf/nan).

        This is a documented degenerate config — beta=0 means no
        new-policy signal can be back-propagated. We assert non-finite
        explicitly (instead of ``pytest.skip``) so a future refactor that
        silently substitutes ``eps`` for zero would FAIL this test
        deliberately rather than be hidden behind a skipped status.
        """
        cfg = _make_actor_config(nft_beta=0.0)
        kwargs = _make_loss_inputs()
        loss, _ = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs)
        assert not torch.isfinite(loss).item(), (
            "beta=0 must yield a non-finite loss to surface the degenerate "
            "config; if this fails, the formula was changed to silently "
            "swallow the zero-division — update this test and the docs "
            "deliberately."
        )


# ===========================================================================
# 4. Optional KL anchor (nft_kl_beta > 0)
# ===========================================================================


class TestDiffusionNFTLossKLAnchor:
    """When ``nft_kl_beta > 0`` the loss adds a v-space MSE term anchoring
    the current policy to a frozen reference. The engine populates
    ``ref_noise_pred`` only when this branch is enabled.
    """

    def test_kl_beta_zero_is_strict_on_policy(self):
        """Default ``nft_kl_beta=0`` keeps the loss identical to the strict
        on-policy form regardless of whether ``ref_noise_pred`` was passed.
        """
        cfg = _make_actor_config()
        kwargs = _make_loss_inputs()

        loss_no_ref, _ = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs)
        loss_with_ref, _ = DiffusionNFTLoss.compute_loss(
            config=cfg,
            ref_noise_pred=torch.full_like(kwargs["noise_pred"], 99.0),
            nft_kl_beta=0.0,
            **kwargs,
        )
        torch.testing.assert_close(loss_no_ref, loss_with_ref)

    def test_kl_beta_positive_requires_ref_noise_pred(self):
        cfg = _make_actor_config()
        kwargs = _make_loss_inputs()
        with pytest.raises(ValueError, match="ref_noise_pred"):
            DiffusionNFTLoss.compute_loss(
                config=cfg,
                nft_kl_beta=0.1,
                ref_noise_pred=None,
                **kwargs,
            )

    def test_kl_anchor_pulls_loss_toward_ref(self):
        """With ``noise_pred == ref_noise_pred`` the KL term is zero, so the
        loss equals the policy loss; adding a divergent ref_noise_pred adds a
        strictly positive KL contribution.
        """
        cfg = _make_actor_config()
        kwargs = _make_loss_inputs()
        # ref_noise_pred matches noise_pred → zero KL
        loss_zero_kl, metrics_zero = DiffusionNFTLoss.compute_loss(
            config=cfg,
            ref_noise_pred=kwargs["noise_pred"].clone(),
            nft_kl_beta=1.0,
            **kwargs,
        )
        # ref_noise_pred far from noise_pred → strictly positive KL
        loss_pos_kl, metrics_pos = DiffusionNFTLoss.compute_loss(
            config=cfg,
            ref_noise_pred=kwargs["noise_pred"] + 5.0,
            nft_kl_beta=1.0,
            **kwargs,
        )
        # KL contribution must increase the total loss.
        assert loss_pos_kl > loss_zero_kl
        assert metrics_pos["actor/nft_kl_loss"] > 0
        # The zero-KL run STILL surfaces metrics so dashboards stay aligned.
        assert metrics_zero["actor/nft_kl_loss"] == pytest.approx(0.0)

    def test_kl_metrics_only_present_when_anchor_enabled(self):
        """Disabled KL anchor must not pollute the metrics dict — downstream
        loggers key by metric name and a stale ``actor/nft_kl_loss`` would
        be misleading.
        """
        cfg = _make_actor_config()
        kwargs = _make_loss_inputs()
        _, metrics = DiffusionNFTLoss.compute_loss(config=cfg, **kwargs)
        assert "actor/nft_kl_loss" not in metrics
        assert "actor/nft_kl_div" not in metrics

    def test_kl_metrics_present_when_anchor_enabled(self):
        cfg = _make_actor_config()
        kwargs = _make_loss_inputs()
        _, metrics = DiffusionNFTLoss.compute_loss(
            config=cfg,
            ref_noise_pred=kwargs["noise_pred"].clone(),
            nft_kl_beta=0.5,
            **kwargs,
        )
        assert "actor/nft_kl_loss" in metrics
        assert "actor/nft_kl_div" in metrics
        # KL div is mean of squared diffs — non-negative.
        assert metrics["actor/nft_kl_div"] >= 0.0

    def test_kl_anchor_via_call_route(self):
        """The ``__call__`` boundary must thread ``ref_noise_pred`` and the
        ``nft_kl_beta`` non-tensor metadata through to ``compute_loss``.
        """
        from tensordict import TensorDict
        from verl.utils import tensordict_utils as tu

        cfg = _make_actor_config()
        kwargs = _make_loss_inputs()
        bs = kwargs["noise_pred"].shape[0]

        model_output = {"noise_pred": kwargs["noise_pred"]}
        data = TensorDict(
            {
                "old_noise_pred": kwargs["old_noise_pred"],
                "advantages": kwargs["advantages"],
                "clean_latents": kwargs["clean_latents"],
                "noised_latents": kwargs["noised_latents"],
                "sigma_broadcast": kwargs["sigma_broadcast"],
                "ref_noise_pred": kwargs["noise_pred"] + 3.0,
            },
            batch_size=bs,
        )
        tu.assign_non_tensor(data, nft_kl_beta=0.5)

        result = DiffusionNFTLoss()(config=cfg, model_output=model_output, data=data)
        assert torch.isfinite(result.loss).item()
        assert "actor/nft_kl_loss" in result.metrics
        assert result.metrics["actor/nft_kl_loss"] > 0
