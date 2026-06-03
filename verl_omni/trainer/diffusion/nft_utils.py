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
"""DiffusionNFT utilities: timestep sampling and noise schedule helpers.

Reference: https://arxiv.org/abs/2509.16117
Adapted from third_party/flow-factory/src/flow_factory/utils/noise_schedule.py
"""

import torch


def flow_match_sigma(t: torch.Tensor) -> torch.Tensor:
    """Convert scheduler-scale timestep t in [0, 1000] to sigma = t / 1000."""
    return t / 1000.0


def to_broadcast_tensor(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Reshape a 1-D tensor t of shape (B,) to broadcast with reference of shape (B, ...)."""
    return t.view(-1, *([1] * (reference.ndim - 1)))


class TimeSampler:
    """Collection of timestep sampling strategies for DiffusionNFT training."""

    @staticmethod
    def uniform(
        batch_size: int,
        num_timesteps: int,
        timestep_range: tuple[float, float] = (0.0, 0.9),
        time_shift: float = 3.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Sample uniform continuous timesteps.

        Args:
            batch_size: Number of samples per timestep.
            num_timesteps: Number of timesteps to sample.
            timestep_range: Fraction range [start, end] along denoise axis.
            time_shift: Not used for uniform, included for interface consistency.
            device: Target device.

        Returns:
            Tensor of shape (num_timesteps, batch_size) in scheduler scale [0, 1000].
        """
        t_start = timestep_range[0] * 1000.0
        t_end = timestep_range[1] * 1000.0
        # Sample uniformly in the range
        t = torch.rand(num_timesteps, batch_size, device=device) * (t_end - t_start) + t_start
        return t

    @staticmethod
    def logit_normal_shifted(
        batch_size: int,
        num_timesteps: int,
        timestep_range: tuple[float, float] = (0.0, 0.9),
        time_shift: float = 3.0,
        device: torch.device = None,
        stratified: bool = True,
    ) -> torch.Tensor:
        """Sample timesteps from a logit-normal distribution with shift.

        Args:
            batch_size: Number of samples per timestep.
            num_timesteps: Number of timesteps to sample.
            timestep_range: Fraction range [start, end] along denoise axis.
            time_shift: Shift parameter for the logit-normal distribution.
            device: Target device.
            stratified: Whether to use stratified sampling.

        Returns:
            Tensor of shape (num_timesteps, batch_size) in scheduler scale [0, 1000].
        """
        if stratified:
            # Stratified sampling: divide [0, 1] into num_timesteps bins
            u = torch.rand(num_timesteps, batch_size, device=device)
            bins = torch.linspace(0, 1, num_timesteps + 1, device=device)
            u = u * (bins[1:] - bins[:-1]).unsqueeze(1) + bins[:-1].unsqueeze(1)
        else:
            u = torch.rand(num_timesteps, batch_size, device=device)

        # Apply logit-normal transform with shift
        # logit(u) = log(u / (1 - u)), then apply time_shift
        u = u.clamp(1e-6, 1 - 1e-6)
        t_normalized = torch.sigmoid(torch.log(u / (1 - u)) / time_shift)

        # Map to timestep range
        t_start = timestep_range[0] * 1000.0
        t_end = timestep_range[1] * 1000.0
        t = t_normalized * (t_end - t_start) + t_start
        return t

    @staticmethod
    def discrete(
        batch_size: int,
        num_train_timesteps: int,
        scheduler_timesteps: torch.Tensor,
        timestep_range: tuple[float, float] = (0.0, 0.9),
        include_init: bool = True,
        force_init: bool = False,
    ) -> torch.Tensor:
        """Sample discrete timesteps from the scheduler's timestep grid.

        Args:
            batch_size: Number of samples per timestep.
            num_train_timesteps: Number of timesteps to sample.
            scheduler_timesteps: The scheduler's full timestep tensor.
            timestep_range: Fraction range [start, end] of scheduler steps to use.
            include_init: Whether to include the first scheduler timestep.
            force_init: Whether the first sampled timestep must be the first scheduler timestep.

        Returns:
            Tensor of shape (num_train_timesteps, batch_size) in scheduler scale.
        """
        device = scheduler_timesteps.device
        total_steps = len(scheduler_timesteps)

        # Compute valid range of indices
        start_idx = int(timestep_range[0] * total_steps)
        end_idx = int(timestep_range[1] * total_steps)
        end_idx = min(end_idx, total_steps)
        valid_timesteps = scheduler_timesteps[start_idx:end_idx]

        if len(valid_timesteps) == 0:
            valid_timesteps = scheduler_timesteps

        # Sample indices
        indices = torch.randint(0, len(valid_timesteps), (num_train_timesteps, batch_size), device=device)
        sampled = valid_timesteps[indices]

        if force_init and len(sampled) > 0:
            sampled[0] = scheduler_timesteps[0]
        elif include_init and len(sampled) > 0:
            # Randomly replace one timestep with the init timestep
            replace_idx = torch.randint(0, num_train_timesteps, (1,)).item()
            sampled[replace_idx] = scheduler_timesteps[0]

        return sampled
