# DiffusionNFT

Last updated: 05/27/2026.

DiffusionNFT ([paper](https://arxiv.org/abs/2509.16117)) is an online diffusion reinforcement learning algorithm that uses a **forward-process matching** objective instead of the reverse-SDE log-probability approach used by FlowGRPO. It enables reward-driven fine-tuning of flow matching models without collecting per-step log-probabilities during rollout, making the rollout phase significantly cheaper.

Two core technical contributions make this possible:

1. **Forward-Process Training**: Rather than training on reverse-SDE trajectories, NFT takes only the final denoised image from rollout, independently samples timesteps during training, noises the clean latent via the flow-matching forward process `x_t = (1-σ)x_1 + σε`, and trains the model to predict the velocity field `v_θ(x_t, t)` with reward-weighted objectives. This decouples rollout cost from training depth.

2. **Positive/Negative Decomposition**: The loss decomposes into a *positive prediction* (steered toward the clean image proportional to advantage) and a *negative prediction* (pushed away from the clean image proportional to disadvantage), interpolated by the `nft_beta` parameter. This gives the algorithm a contrastive character: high-reward images attract the model and low-reward images repel it, simultaneously.

## Key Components

- **Flow Matching Backbone**: operates on continuous-time flow matching models (e.g., Qwen-Image, Qwen-Image-Edit-Plus) rather than discrete-token LLMs.
- **Final-Latent Rollout**: generates a group of independent images per prompt. Only the final clean latent `x_1` is retained from each trajectory — no intermediate states or log-probs are required.
- **Independent Timestep Sampling**: during training, timesteps are sampled fresh for each training batch according to `nft_time_sampling_strategy` and `nft_timestep_range`. This is different from FlowGRPO which reuses the timesteps visited during rollout.
- **v-Prediction Matching Loss**: the per-sample loss is computed in image space (`x_0` reconstruction MSE) rather than in velocity space, weighted by reward-derived `r ∈ [0, 1]`.
- **No Critic**: like GRPO for LLMs, no separate value network is trained; advantages are computed from group-relative rewards.
- **Image Reward Models**: rewards are assigned by external reward models (e.g., OCR, aesthetic score, VLM judge) or rule-based scorers.

## Key Differences: FlowGRPO vs. DiffusionNFT

| Dimension | FlowGRPO | DiffusionNFT |
|---|---|---|
| **Model type** | Flow matching / diffusion model | Flow matching / diffusion model |
| **Rollout data** | Per-step latents + log-probs within SDE window | Only final clean latent (last denoising step) |
| **Training signal** | Importance ratio (old vs. new log-prob) | Forward-process v-prediction matching |
| **Loss type** | Clipped PPO objective | Positive/negative weighted MSE |
| **Timestep handling** | Uses rollout timesteps (SDE window) | Independently samples timesteps during training |
| **Trainer type** | `policy_gradient` | `direct_preference` |
| **Advantage estimator** | `flow_grpo` | `nft` (same group-normalized computation) |
| **Loss mode** | `flow_grpo` | `nft` |
| **SDE window required** | Yes (`sde_window_size`, `sde_window_range`) | No (replaced by `nft_num_train_timesteps`) |
| **Off-policy support** | Not applicable | Optional EMA sampling via `nft_off_policy` |
| **Rollout memory** | Higher (stores all trajectory steps) | Lower (stores only final latents) |

## Configuration

Diffusion training uses dedicated diffusion config blocks. In `verl_omni/trainer/config/diffusion_trainer.yaml`,
the main sections are:

- `algorithm`: diffusion-specific advantage computation, NFT training parameters
- `actor_rollout_ref.actor`: optimization and diffusion loss settings
- `actor_rollout_ref.rollout`: rollout backend, sampling controls
- `actor_rollout_ref.model`: model path plus diffusion-model / LoRA settings
- `reward`: reward manager, reward model, and custom reward function

### Core Parameters

#### Algorithm

- `algorithm.trainer_type`: Must be set to `direct_preference` for DiffusionNFT.

- `algorithm.adv_estimator`: Set to `nft`. Uses the same group-normalized advantage
  computation as `flow_grpo` — mean-subtracted, std-normalized per group.

- `algorithm.nft_beta`: Interpolation weight between the current policy's v-prediction
  and the sampling policy's v-prediction. At `1.0` (default), the positive prediction
  is entirely the current policy; lower values blend in the sampling policy's prediction.

- `algorithm.nft_num_train_timesteps`: Number of timesteps sampled per training batch.
  When set to `0` (default), the value is derived automatically as
  `num_inference_steps × (range_max - range_min)`. For a 50-step scheduler with
  `nft_timestep_range=[0.0, 0.9]`, this yields 45 training timesteps.

- `algorithm.nft_time_sampling_strategy`: How to sample training timesteps. One of:
  - `discrete`: samples uniformly from the scheduler's discrete timestep grid
    *with* one row pinned to the grid head (`include_init=True`,
    `force_init=False`). Default — recommended for most runs.
  - `discrete_with_init`: every batch row uses the grid head
    (`force_init=True`). Useful when the scheduler's first step is the most
    informative, or when ablating the contribution of the very first
    denoising step.
  - `discrete_wo_init`: never uses the grid head (`include_init=False`,
    `force_init=False`). Useful when the grid head is known to be
    out-of-distribution for training.
  - `uniform`: samples continuously and uniformly over `nft_timestep_range`.
  - `logit_normal`: samples with a logit-normal distribution shifted by `nft_time_shift`
    (concentrates more samples toward the middle of the denoising trajectory).

- `algorithm.nft_time_shift`: Shift parameter for `logit_normal` sampling. Larger
  values push more samples toward the midpoint of the denoising trajectory.

- `algorithm.nft_timestep_range`: A `[low, high]` fraction of the denoising axis from
  which training timesteps are sampled. `[0.0, 0.9]` excludes the last 10% (pure-noise
  end) which tends to be less informative for image quality.

- `algorithm.nft_adv_clip_range`: A `[min, max]` range for clipping raw advantages
  before mapping them to `r ∈ [0, 1]`. Symmetric ranges like `[-5.0, 5.0]` are typical.

- `algorithm.nft_kl_beta`: KL divergence penalty coefficient. When greater than `0`,
  adds `nft_kl_beta × mean((v_θ - v_ref)²)` to the loss in v-space.
  `v_ref` comes from the LoRA-disabled forward of the same module
  (semantically identical to flow-factory's reference-EMA / `use_ref_parameters`),
  so this path **requires `actor_rollout_ref.model.lora_rank > 0`** —
  otherwise `disable_adapter()` is a no-op and the KL term is silently zero.
  The engine raises `RuntimeError` if you try to enable `nft_kl_beta > 0`
  (or `nft_off_policy=True`) without LoRA.
  Reasonable values: `0.001`–`0.1`. Larger pulls the policy more strongly
  toward the base model and hurts reward; `0.0` (default) disables the term
  entirely (the metrics dict won't expose `actor/nft_kl_loss` either).

- `algorithm.nft_off_policy`: When `True`, the sampling policy for
  `old_noise_pred` is the LoRA-disabled base model rather than the current
  policy under `torch.no_grad()`. This makes the positive/negative
  interpolation truly off-policy (matches the paper's off-policy setup
  and flow-factory's `off_policy=True` EMA mode). When combined with
  `nft_kl_beta > 0` the engine reuses a single reference forward for
  both purposes, so enabling both flags only costs ONE extra forward
  per training step. Same LoRA requirement as `nft_kl_beta`. Default
  `False` (strict on-policy).

#### Actor / Loss

- `actor_rollout_ref.actor.diffusion_loss.loss_mode`: Set to `nft`.

- `actor_rollout_ref.actor.diffusion_loss.adv_clip_max`: Maximum absolute advantage
  used before computing the policy loss. Should match `algorithm.nft_adv_clip_range`
  (half the symmetric range, e.g., `5.0`).

- `actor_rollout_ref.actor.nft_beta`: Per-actor copy of the beta interpolation weight.
  Must match `algorithm.nft_beta`.

#### Rollout / Sampling

NFT rollout does not require SDE window parameters because it only collects the final
clean latent. The rollout section still controls image resolution and inference steps:

- `actor_rollout_ref.rollout.name`: Selects the rollout backend. Currently supports
  `vllm_omni`.

- `actor_rollout_ref.rollout.n`: Number of sampled images per prompt. This is the NFT
  group size and must be greater than `1` for group-relative advantage computation.

- `actor_rollout_ref.rollout.pipeline.num_inference_steps`: Number of denoising steps
  used for rollout generation during training. NFT uses all steps for the full
  denoising trajectory, though only the final latent is retained.

- `actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps`: Number of
  denoising steps used during validation / evaluation. Can be set higher than training
  for better image quality during evaluation.

- `actor_rollout_ref.rollout.pipeline.true_cfg_scale`: True classifier-free guidance
  scale during rollout. Used in Qwen-Image and Qwen-Image-Edit-Plus.

- `actor_rollout_ref.rollout.algo.noise_level`: SDE noise level. For NFT, this can be
  set to `0.0` (pure ODE rollout) since no SDE trajectory is needed during training.
  A positive value is still valid and introduces diversity.

#### Model

- `actor_rollout_ref.model.path`: Base diffusion model path or Hugging Face Hub ID.

- `actor_rollout_ref.model.algorithm`: Set to `nft` to select NFT-specific pipeline
  adapters (both the diffusers training adapter and the vllm-omni rollout adapter).

- `actor_rollout_ref.model.tokenizer_path`: Optional tokenizer path if not located
  under the model path.

#### Batch Size

DiffusionNFT uses three nested batch-size parameters that operate at different stages
of the training loop. They address different concerns (RL sample diversity, multi-epoch
reuse, and GPU memory) and must be understood together.

**Step 1 — Rollout (`data.train_batch_size`)**

`data.train_batch_size` is the number of **unique prompts** drawn from the dataset per
training step. Before rollout, each prompt is replicated `actor_rollout_ref.rollout.n`
times so that the rollout engine generates `n` independent images per prompt. The
in-memory batch after rollout therefore holds `train_batch_size × n` image samples.
Group-normalized advantage computation runs over this **full** batch — it needs all `n`
images for every prompt to compute group-relative rewards before any splitting occurs.

**Step 2 — Actor Update (`actor_rollout_ref.actor.ppo_mini_batch_size`)**

`ppo_mini_batch_size` controls how the full post-rollout batch is sliced for actor
gradient updates. **Important:** this value is specified in **prompts**, not image
samples. The trainer internally scales it by `rollout.n` to get the actual mini-batch
size in samples:

```
effective mini-batch = ppo_mini_batch_size × rollout.n  (image samples)
number of mini-batches per epoch = train_batch_size / ppo_mini_batch_size
```

All `n` images belonging to the same prompt are kept in the same mini-batch. This is
not optional: advantages are computed globally before this split, but the gradient
update for each image depends on its advantage within its group. Scattering a prompt's
images across different mini-batches would break that correspondence.
`ppo_mini_batch_size` must divide `train_batch_size` evenly.

**Step 3 — FSDP Sharding and Gradient Accumulation
(`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`)**

Each mini-batch is distributed across GPUs by FSDP data parallelism, so each GPU
receives `(ppo_mini_batch_size × n) / n_gpus` image samples. That per-GPU shard is
then **chunked into micro-batches** of `ppo_micro_batch_size_per_gpu` for the actual
forward/backward passes, with gradients accumulated across chunks before the optimizer
step.

For DiffusionNFT, each micro-batch additionally loops over `nft_num_train_timesteps`
independently sampled timesteps, so the total gradient accumulation steps per GPU per
mini-batch is:

```
gradient_accumulation_steps = (per_gpu_samples / ppo_micro_batch_size_per_gpu)
                              × nft_num_train_timesteps
```

`ppo_micro_batch_size_per_gpu` must satisfy: `(ppo_mini_batch_size × n) / n_gpus`
is divisible by `ppo_micro_batch_size_per_gpu`.

**Concrete Walkthrough** (reference OCR script, 4 GPUs, `nft_num_train_timesteps=5`):

```
data.train_batch_size              = 32    # 32 prompts loaded
actor_rollout_ref.rollout.n        = 16    # 16 images generated per prompt
  → post-rollout batch             = 512   # advantage computed over all 512

ppo_mini_batch_size (config)       = 16    # in prompts
  → effective mini-batch           = 16 × 16 = 256 samples
  → mini-batches per epoch         = 512 / 256 = 2 actor gradient steps

FSDP shards 256 samples across 4 GPUs:
  → per-GPU samples                = 256 / 4 = 64

ppo_micro_batch_size_per_gpu       = 16
  → micro-batches per GPU          = 64 / 16 = 4
  → gradient_accumulation_steps    = 4 × 5 (nft_num_train_timesteps) = 20
```

#### Reward

- `reward.reward_manager.name`: Selects the reward manager.

- `reward.custom_reward_function.path` and
  `reward.custom_reward_function.name`: Register the task-specific reward
  post-processing function such as `compute_score_ocr`.

For an end-to-end OCR training walkthrough, including dataset preparation and
the full runnable command, see `docs/start/nft_quickstart.md`.

## Loss Formula

Given:
- `v_new`: current policy v-prediction at timestep `t`
- `v_old`: sampling-policy (or EMA) v-prediction at timestep `t`
- `A_raw`: raw group-normalized advantage
- `β = nft_beta`
- `σ`: sigma at timestep `t`  (`σ = t / 1000.0` for flow-matching schedulers)

The normalized reward weight `r ∈ [0, 1]` is obtained by clipping and rescaling:

$$r = \text{clamp}\!\left(\frac{A_\text{raw}}{A_\text{clip}} \cdot \frac{1}{2} + \frac{1}{2},\ 0,\ 1\right)$$

The positive and negative velocity predictions are formed by interpolation:

$$v_{\text{pos}} = \beta \cdot v_\text{new} + (1 - \beta) \cdot v_\text{old}$$
$$v_{\text{neg}} = (1 + \beta) \cdot v_\text{old} - \beta \cdot v_\text{new}$$

These are converted to clean-image (`x_0`) reconstructions:

$$\hat{x}_{0}^{+} = x_t - \sigma \cdot v_\text{pos}, \qquad \hat{x}_{0}^{-} = x_t - \sigma \cdot v_\text{neg}$$

The per-sample loss is:

$$\mathcal{L} = \frac{1}{\beta}\left[r \cdot \text{MSE}(\hat{x}_{0}^{+},\, x_1) + (1-r) \cdot \text{MSE}(\hat{x}_{0}^{-},\, x_1)\right]$$

where `x_1` is the clean latent from rollout and MSE uses per-sample adaptive weighting
for numerical stability. Averaged across all sampled timesteps and mini-batches.

When `nft_kl_beta > 0`, an optional KL penalty term is added:

$$\mathcal{L}_\text{total} = \mathcal{L} + \beta_\text{KL} \cdot \|v_\theta - v_\text{ref}\|^2$$

## Supported Models

| Model | Architecture String | Algorithm Key | Status |
|-------|-------------------|---------------|--------|
| Qwen-Image | `QwenImagePipeline` | `nft` | ✅ Integrated |
| Qwen-Image-Edit-Plus (2511) | `QwenImageEditPlusPipeline` | `nft` | ✅ Integrated |

## Reference Examples

Standard LoRA training with OCR reward (Qwen-Image, 4 GPUs):

```bash
bash examples/nft_trainer/run_qwen_image_ocr_nft_lora.sh
```

LoRA training for image editing tasks (Qwen-Image-Edit-Plus, 4 GPUs):

```bash
bash examples/nft_trainer/run_qwen_image_edit_nft_lora.sh
```

For a step-by-step guide to the OCR training workflow, see
{doc}`../start/nft_quickstart`.

## Variants

### Rule-Based Reward Training: JPEG Incompressibility

DiffusionNFT also supports rule-based rewards that score images directly without a
VLM reward model, reusing the default `VisualRewardManager` from
`verl_omni/trainer/config/reward/reward.yaml`.

`verl_omni/utils/reward_score/jpeg_compressibility.py` rewards images that are harder
to JPEG-compress (richer texture, more complex content). No extra dependencies or
reward model process are required.

Minimal dataset row:

```python
{
    "data_source": "jpeg_compressibility",
    "prompt": [{"role": "user", "content": "<your prompt>"}],
    "reward_model": {"ground_truth": ""},  # required by schema, ignored by scorer
}
```

Config changes relative to the OCR example — **remove** these lines:

```bash
reward.reward_model.enable=True
reward.reward_model.model_path=...
reward.reward_model.rollout.name=...
reward.reward_model.rollout.tensor_model_parallel_size=...
reward.custom_reward_function.path=...
reward.custom_reward_function.name=...
```

Keep all actor/rollout settings unchanged. The e2e smoke test (`tests/special_e2e/run_nft_qwen_image.sh`)
uses this rule-based reward to validate the full NFT pipeline without a reward model.

### KL Regularization

To add KL divergence regularization against a frozen reference policy, enable:

```bash
algorithm.nft_kl_beta=0.001 \
```

This requires a reference model to be loaded. The KL term is computed as the squared
L2 distance between the current policy's v-prediction and the reference policy's
v-prediction, averaged over sampled timesteps.

### Off-Policy (EMA) Sampling

When `algorithm.nft_off_policy=True`, the sampling policy for computing `v_old` uses
an Exponential Moving Average (EMA) of the trained weights rather than the current
online weights. This can improve training stability at the cost of slightly stale
reference predictions.

### Image Editing (Qwen-Image-Edit-Plus)

For image editing tasks, use the `QwenImageEditPlusPipeline` adapter with
`actor_rollout_ref.model.algorithm=nft`. The condition image is latent-concatenated
with the noised target, and norm-preserving CFG is applied during rollout:

```bash
bash examples/nft_trainer/run_qwen_image_edit_nft_lora.sh
```

This adapter handles the additional `condition_image` field in the parquet data and
adjusts the vae encode path to produce concatenated latents for the model.

## Citation

```bibtex
@article{diffusionnft2025,
  title={DiffusionNFT: Online Diffusion Reinforcement with Forward Process Matching},
  author={},
  journal={arXiv preprint arXiv:2509.16117},
  year={2025}
}
```
