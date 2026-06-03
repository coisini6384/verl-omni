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
"""CPU tests for DiffusionNFT timestep-sampling utilities.

Necessity: ``flow_match_sigma`` / ``to_broadcast_tensor`` / ``TimeSampler``
are pure-Python helpers used by both the trainer (``_fit_nft_online``) and the
engine (``_forward_backward_batch_nft``). Bugs in their range/shape contracts
silently produce sigmas outside [0, 1] or non-broadcasting tensors that crash
downstream FSDP forwards. These tests pin the contract without GPU.
"""

import pytest
import torch

from verl_omni.trainer.diffusion.nft_utils import (
    TimeSampler,
    flow_match_sigma,
    to_broadcast_tensor,
)

# ===========================================================================
# 1. flow_match_sigma
# ===========================================================================


class TestFlowMatchSigma:
    def test_zero_timestep_gives_zero_sigma(self):
        t = torch.zeros(4)
        torch.testing.assert_close(flow_match_sigma(t), torch.zeros(4))

    def test_max_timestep_gives_unit_sigma(self):
        # Scheduler-scale t lives in [0, 1000]; t=1000 must map to sigma=1.0.
        t = torch.full((3,), 1000.0)
        torch.testing.assert_close(flow_match_sigma(t), torch.ones(3))

    def test_linear_in_range(self):
        t = torch.tensor([0.0, 250.0, 500.0, 750.0, 1000.0])
        torch.testing.assert_close(flow_match_sigma(t), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))

    def test_preserves_shape(self):
        t = torch.rand(2, 5) * 1000.0
        sigma = flow_match_sigma(t)
        assert sigma.shape == t.shape


# ===========================================================================
# 2. to_broadcast_tensor
# ===========================================================================


class TestToBroadcastTensor:
    def test_reshape_to_match_3d_reference(self):
        sigma = torch.linspace(0.1, 0.9, steps=4)  # shape (4,)
        ref = torch.zeros(4, 6, 8)  # (B, S, C)
        out = to_broadcast_tensor(sigma, ref)

        assert out.shape == (4, 1, 1)
        # broadcasting against ref must succeed and produce a (4, 6, 8) tensor.
        broadcast_result = sigma.view(-1, 1, 1) * torch.ones_like(ref)
        torch.testing.assert_close(out * torch.ones_like(ref), broadcast_result)

    def test_reshape_to_match_4d_reference(self):
        sigma = torch.zeros(2)
        ref = torch.zeros(2, 3, 4, 5)
        out = to_broadcast_tensor(sigma, ref)
        assert out.shape == (2, 1, 1, 1)

    def test_reshape_to_match_2d_reference(self):
        sigma = torch.zeros(7)
        ref = torch.zeros(7, 11)
        out = to_broadcast_tensor(sigma, ref)
        assert out.shape == (7, 1)


# ===========================================================================
# 3. TimeSampler.uniform
# ===========================================================================


class TestTimeSamplerUniform:
    def test_shape_is_num_steps_by_batch(self):
        t = TimeSampler.uniform(batch_size=4, num_timesteps=3)
        assert t.shape == (3, 4)

    def test_within_scaled_range(self):
        # range fraction 0.0..0.9 along the denoise axis of length 1000.
        t = TimeSampler.uniform(batch_size=64, num_timesteps=10, timestep_range=(0.0, 0.9))
        assert torch.all(t >= 0.0).item()
        assert torch.all(t <= 900.0).item()

    def test_clamped_to_explicit_range(self):
        t = TimeSampler.uniform(batch_size=32, num_timesteps=5, timestep_range=(0.2, 0.5))
        assert torch.all(t >= 200.0).item()
        assert torch.all(t <= 500.0).item()


# ===========================================================================
# 4. TimeSampler.logit_normal_shifted
# ===========================================================================


class TestTimeSamplerLogitNormal:
    def test_shape_is_num_steps_by_batch(self):
        t = TimeSampler.logit_normal_shifted(batch_size=4, num_timesteps=3)
        assert t.shape == (3, 4)

    def test_within_range(self):
        t = TimeSampler.logit_normal_shifted(batch_size=128, num_timesteps=8, timestep_range=(0.0, 0.9), time_shift=3.0)
        # Values must remain inside the requested fraction of [0, 1000].
        # Allow tiny float error at boundaries.
        assert torch.all(t >= -1e-3).item()
        assert torch.all(t <= 900.0 + 1e-3).item()

    def test_stratified_covers_all_bins(self):
        """Stratified mode draws one value per bin [k/N, (k+1)/N]; with N bins
        and B samples per bin, every bin must contribute at least one sample.
        """
        t = TimeSampler.logit_normal_shifted(batch_size=64, num_timesteps=10, time_shift=3.0, stratified=True)
        # Stratified sampling guarantees row k draws from bin [k/N, (k+1)/N];
        # rows must therefore have non-overlapping ranges (modulo logit-normal
        # transform, which is monotonic). We sanity-check row means are
        # monotonically non-decreasing.
        row_means = t.mean(dim=1)
        # The transform is monotonic, so means should be in increasing order.
        for i in range(1, row_means.shape[0]):
            assert row_means[i] >= row_means[i - 1] - 1e-3

    def test_unstratified_does_not_partition_bins(self):
        # Sanity check: when stratified=False, samples are i.i.d. — row means
        # should not be monotonically ordered (with high probability across
        # multiple seeds).
        torch.manual_seed(123)
        t = TimeSampler.logit_normal_shifted(batch_size=8, num_timesteps=10, time_shift=3.0, stratified=False)
        row_means = t.mean(dim=1)
        # At least one inversion in the order is overwhelmingly likely.
        diffs = row_means[1:] - row_means[:-1]
        assert torch.any(diffs < 0).item(), (
            "Unstratified sampling produced perfectly monotonic row means — "
            "the implementation may have stratified by accident."
        )


# ===========================================================================
# 5. TimeSampler.discrete
# ===========================================================================


class TestTimeSamplerDiscrete:
    def test_shape_is_num_steps_by_batch(self):
        scheduler_timesteps = torch.linspace(999.0, 0.0, steps=50)
        t = TimeSampler.discrete(
            batch_size=4,
            num_train_timesteps=3,
            scheduler_timesteps=scheduler_timesteps,
        )
        assert t.shape == (3, 4)

    def test_only_picks_from_scheduler_grid(self):
        scheduler_timesteps = torch.tensor([900.0, 600.0, 300.0, 100.0])
        t = TimeSampler.discrete(
            batch_size=10,
            num_train_timesteps=5,
            scheduler_timesteps=scheduler_timesteps,
            timestep_range=(0.0, 1.0),
        )
        # Every sampled value must literally appear in the scheduler grid.
        unique_in_t = set(t.flatten().tolist())
        unique_grid = set(scheduler_timesteps.tolist())
        assert unique_in_t <= unique_grid

    def test_force_init_pins_first_timestep_to_grid_head(self):
        scheduler_timesteps = torch.tensor([1000.0, 500.0, 100.0])
        t = TimeSampler.discrete(
            batch_size=3,
            num_train_timesteps=4,
            scheduler_timesteps=scheduler_timesteps,
            force_init=True,
        )
        # Row 0 must be exactly scheduler_timesteps[0] across the whole batch.
        torch.testing.assert_close(t[0], torch.full((3,), 1000.0))

    def test_include_init_replaces_one_row_with_grid_head(self):
        scheduler_timesteps = torch.tensor([1000.0, 100.0])
        # Use range that excludes index 0 to make the test unambiguous.
        t = TimeSampler.discrete(
            batch_size=4,
            num_train_timesteps=6,
            scheduler_timesteps=scheduler_timesteps,
            timestep_range=(0.5, 1.0),
            include_init=True,
            force_init=False,
        )
        # Exactly one row should be all 1000.0 (the grid head); the others should
        # come from the second half of the grid (index 1 = 100.0).
        rows_at_head = ((t - 1000.0).abs() < 1e-6).all(dim=1)
        assert rows_at_head.sum().item() == 1


# ===========================================================================
# 6. Engine call-site wiring (regression for the BLOCKER bug fix)
# ===========================================================================


class TestTimeSamplerEngineWiring:
    """Pin the call shape that ``PPODiffusersFSDPEngine._forward_backward_batch_nft``
    uses for its per-step timestep sampling.

    Each NFT training step samples exactly ONE timestep — the engine
    therefore calls ``TimeSampler.<strategy>(batch_size=B, num_timesteps=1, …)``
    and indexes ``[0]`` to get a (B,) tensor that aligns with
    ``micro_batch["all_timesteps"][:, 0]``. A regression that drops the
    required ``num_timesteps`` / ``num_train_timesteps`` keyword would
    fail with ``TypeError`` deep inside FSDP forward; these unit tests
    catch it on CPU.
    """

    BATCH_SIZE = 4

    def test_discrete_one_timestep_per_step(self):
        scheduler_timesteps = torch.linspace(999.0, 0.0, steps=50)
        t = TimeSampler.discrete(
            batch_size=self.BATCH_SIZE,
            num_train_timesteps=1,
            scheduler_timesteps=scheduler_timesteps,
            timestep_range=(0.0, 0.9),
        )[0]
        assert t.shape == (self.BATCH_SIZE,)
        assert set(t.tolist()) <= set(scheduler_timesteps.tolist())

    def test_logit_normal_one_timestep_per_step(self):
        t = TimeSampler.logit_normal_shifted(
            batch_size=self.BATCH_SIZE,
            num_timesteps=1,
            timestep_range=(0.0, 0.9),
            time_shift=3.0,
        )[0]
        assert t.shape == (self.BATCH_SIZE,)
        assert torch.all(t >= -1e-3).item()
        assert torch.all(t <= 900.0 + 1e-3).item()

    def test_uniform_one_timestep_per_step(self):
        t = TimeSampler.uniform(
            batch_size=self.BATCH_SIZE,
            num_timesteps=1,
            timestep_range=(0.0, 0.9),
        )[0]
        assert t.shape == (self.BATCH_SIZE,)
        assert torch.all(t >= 0.0).item()
        assert torch.all(t <= 900.0).item()

    def test_discrete_requires_num_train_timesteps_kwarg(self):
        """Forgetting ``num_train_timesteps`` must raise a clear ``TypeError``
        — pins the parameter contract that the NFT engine path depends on.
        Earlier drafts of the engine path silently dropped this kwarg and
        only failed at runtime under FSDP forward; this test catches the
        regression on CPU.
        """
        scheduler_timesteps = torch.linspace(999.0, 0.0, steps=10)
        with pytest.raises(TypeError, match="num_train_timesteps"):
            TimeSampler.discrete(
                batch_size=self.BATCH_SIZE,
                scheduler_timesteps=scheduler_timesteps,
                timestep_range=(0.0, 0.9),
            )

    def test_uniform_requires_num_timesteps_kwarg(self):
        with pytest.raises(TypeError, match="num_timesteps"):
            TimeSampler.uniform(
                batch_size=self.BATCH_SIZE,
                timestep_range=(0.0, 0.9),
            )

    def test_logit_normal_requires_num_timesteps_kwarg(self):
        with pytest.raises(TypeError, match="num_timesteps"):
            TimeSampler.logit_normal_shifted(
                batch_size=self.BATCH_SIZE,
                timestep_range=(0.0, 0.9),
                time_shift=3.0,
            )


# ===========================================================================
# 7. Discrete strategy variants (engine-path string mapping)
# ===========================================================================


class TestDiscreteStrategyVariants:
    """The engine resolves three string aliases into different
    ``(include_init, force_init)`` tuples for ``TimeSampler.discrete``::

        discrete            → (True,  False)   one row may be the grid head
        discrete_with_init  → (True,  True)    every row is the grid head
        discrete_wo_init    → (False, False)   never use the grid head

    Test the underlying ``TimeSampler.discrete`` invariants for each
    tuple so the engine's mapping remains semantically correct.
    """

    def test_discrete_with_init_pins_first_row_to_grid_head(self):
        scheduler_timesteps = torch.tensor([1000.0, 500.0, 100.0])
        # When force_init=True every batch row in the first-output row
        # must equal the grid head.
        t = TimeSampler.discrete(
            batch_size=4,
            num_train_timesteps=2,
            scheduler_timesteps=scheduler_timesteps,
            timestep_range=(0.0, 1.0),
            include_init=True,
            force_init=True,
        )
        torch.testing.assert_close(t[0], torch.full((4,), 1000.0))

    def test_discrete_wo_init_avoids_grid_head_when_range_excludes(self):
        # Restrict range to indices [1..end) so the grid head is excluded.
        scheduler_timesteps = torch.tensor([1000.0, 500.0, 250.0, 100.0])
        t = TimeSampler.discrete(
            batch_size=8,
            num_train_timesteps=4,
            scheduler_timesteps=scheduler_timesteps,
            timestep_range=(0.25, 1.0),
            include_init=False,
            force_init=False,
        )
        # Grid head 1000.0 must not appear at all.
        assert (t == 1000.0).any().item() is False

    def test_discrete_default_includes_init_in_some_row(self):
        scheduler_timesteps = torch.tensor([1000.0, 500.0])
        torch.manual_seed(42)
        # Default ``discrete`` (include_init=True, force_init=False) — at
        # least one row across many trials should land on the grid head.
        t = TimeSampler.discrete(
            batch_size=4,
            num_train_timesteps=8,
            scheduler_timesteps=scheduler_timesteps,
            timestep_range=(0.5, 1.0),
            include_init=True,
            force_init=False,
        )
        # Range starts at 0.5 so only grid index 1 (=500.0) is in the
        # valid window; ``include_init`` injects 1000.0 into one row.
        rows_at_head = ((t - 1000.0).abs() < 1e-6).all(dim=1)
        assert rows_at_head.sum().item() == 1
