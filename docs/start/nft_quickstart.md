(nft_quickstart)=
# Quickstart: DiffusionNFT training on Qwen-Image OCR dataset

Last updated: 05/27/2026.

Post-train a diffusion image generation model with DiffusionNFT.

## Introduction

In this example, we post-train a `Qwen-Image` policy with DiffusionNFT for OCR-style image generation tasks. The rollout uses `vllm-omni` for multimodal generation and collects only the final denoised image from each trajectory — no per-step log-probabilities are required. The reward is computed by a visual generative reward model (*Qwen3-VL-8B-Instruct* in this example) that compares OCR text extracted from generated images against the dataset ground truth.

DiffusionNFT differs from FlowGRPO in that:

- **Rollout is cheaper**: only the final clean latent is retained per trajectory.
- **Training uses forward-process matching**: timesteps are independently sampled during training (no SDE window).
- **`trainer_type: direct_preference`**: the NFT loop bypasses the PPO importance-ratio path entirely.

## Prerequisite

- Install VeRL-Omni and its dependencies following the {doc}`installation guide <install>`. Also install the NFT-specific reward dependency:

```bash
pip install Levenshtein
```

- Use a machine with `4` GPUs for the provided example script.
- Run the commands below from the repository root.

## Dataset Introduction

We use the OCR dataset from the original Flow-GRPO repository: [dataset/ocr](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr). Each sample asks the model to generate an image containing specific text, and the reward model scores the generated image by reading the rendered text and comparing it with the reference OCR string.

The raw dataset is a plain-text file (`train.txt` / `test.txt`) where each line is one generation prompt. The OCR target — the text the model must render in the image — is enclosed in double quotes within the prompt. A few representative samples:

```text
A close-up of a medicine bottle with a clear, red warning label that reads "Take With Food" prominently displayed, set against a neutral background.
A close-up of a robot's chest panel, with a digital display blinking "System Override Active" in red, set against a dimly lit industrial background.
A detailed textbook diagram labeled "Photosynthesis Process", viewed under a high-powered microscope, showcasing the intricate cellular structures and chemical reactions involved.
An ancient, leather-bound wizard's spellbook lies open, revealing a worn, yellowed page. A delicate bookmark rests precisely on "Page 666", casting a subtle glow that illuminates the arcane text.
An astronaut's boot print on the Martian surface, clearly reading "First Steps", surrounded by the red, dusty terrain under a pale, distant sky.
```

The preprocessing script converts the raw dataset into parquet files that contain:

- the multimodal prompt used for image generation,
- a negative prompt for true CFG sampling,
- OCR ground truth stored under `reward_model.ground_truth`,
- auxiliary metadata such as split and sample index.

## Step 1: Prepare the Dataset

Set the `WORKSPACE` environment variable to any writable directory you prefer (defaults to `$HOME` if unset):

```bash
export WORKSPACE=${WORKSPACE:-$HOME}
```

Obtain the raw OCR dataset from the original Flow-GRPO repository and place it under `$WORKSPACE/data/ocr`. Then preprocess it into `train.parquet` and `test.parquet`:

```bash
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr
```

The command above writes:

- `$WORKSPACE/data/ocr/train.parquet`
- `$WORKSPACE/data/ocr/test.parquet`

These parquet files are the inputs consumed by the NFT training script.

### Preparing a Custom Dataset

To train on your own OCR-style data, create `train.txt` and `test.txt` following the same one-prompt-per-line convention. Each prompt must contain the target OCR string enclosed in double quotes — the preprocessing script extracts the text between the first pair of quotes as the ground truth. For example:

```text
A vintage storefront sign above the door reads "Open 24 Hours" in bold neon letters.
A handwritten sticky note on a refrigerator says "Buy milk" in blue ink.
```

Place the files in `$WORKSPACE/data/ocr/` (or any directory you prefer) and run the same preprocessing command, adjusting `--input_dir` and `--output_dir` as needed. For datasets with a different ground-truth extraction scheme, modify `extract_solution` and the `process_fn` function in `examples/flowgrpo_trainer/data_process/qwenimage_ocr.py` to match your format, then re-run the script.

## Step 2: Obtain Models for RL Training

In this example, we train `Qwen/Qwen-Image` with LoRA and use `Qwen/Qwen3-VL-8B-Instruct` as the OCR reward model.

**Policy model (Qwen-Image):** the script uses the Hugging Face Hub ID `Qwen/Qwen-Image` directly — no manual download is required. Hugging Face will cache the weights automatically on first run. To use a local copy instead, edit the `model_name` variable in the script directly.

**Reward model (Qwen3-VL-8B-Instruct):** the script defaults to the Hugging Face Hub ID `Qwen/Qwen3-VL-8B-Instruct`, so no manual download is required. To use a local copy instead, edit the `reward_model_name` variable in the script directly.

The run script exposes the following environment variable:

```bash
WORKSPACE              # base directory for data (default: $HOME)
```

## Step 3: Perform DiffusionNFT Training

The provided example script launches `python3 -m verl_omni.trainer.main_diffusion` with the NFT-specific config needed for this OCR task:

- `algorithm.trainer_type=direct_preference`
- `algorithm.adv_estimator=nft`
- `actor_rollout_ref.model.algorithm=nft`
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=nft`
- `reward.custom_reward_function.name=compute_score_ocr`
- LoRA fine-tuning on `Qwen-Image`
- a single-node, `4`-GPU layout

Run the training script:

```bash
bash examples/nft_trainer/run_qwen_image_ocr_nft_lora.sh
```

Optional KL regularization (adds a penalty against the LoRA-disabled
reference policy's v-prediction — requires LoRA training):

```bash
algorithm.nft_kl_beta=0.001
```

Optional off-policy sampling (uses the LoRA-disabled base model as the
sampling policy for `old_noise_pred`, matching the paper's off-policy
setup — requires LoRA training):

```bash
algorithm.nft_off_policy=True
```

The two flags can be combined; in that case the engine reuses a single
reference forward for both, so enabling both only costs one extra forward
per training step. See
[`examples/nft_trainer/run_qwen_image_edit_nft_kl_anchor_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/nft_trainer/run_qwen_image_edit_nft_kl_anchor_lora.sh)
for an end-to-end Qwen-Image-Edit example with both enabled.

The script uses `$WORKSPACE` (default: `$HOME`) as the base directory. Override any path via the environment variables described in Step 2, or set `WORKSPACE` to point to a volume with enough free space before launching.

You are expected to see training, validation, actor, and reward metrics logged through the configured backends. By default, checkpoints are saved under:

```bash
checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

## FAQ: Tuning OOM-Related Parameters

| OOM location | First parameter to tune | What it changes |
| --- | --- | --- |
| Rollout generation OOM | Increase `ROLLOUT_TP` | Sets `actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP` and reduces `actor_rollout_ref.rollout.agent.num_workers` to `NUM_GPUS / ROLLOUT_TP`. This shards the rollout model and lowers rollout request concurrency. |
| Reward-model OOM | Increase `REWARD_TP` | Sets `reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP` and reduces `reward.num_workers` to `NUM_GPUS / REWARD_TP`. This shards the reward model and lowers reward request concurrency. |
| Actor loss forward/backward OOM | Decrease `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | Splits each actor mini-batch into smaller per-GPU chunks and accumulates gradients across chunks. For NFT this also interacts with `nft_num_train_timesteps` — total gradient accumulation steps per GPU per mini-batch is `(per_gpu_samples / ppo_micro_batch_size_per_gpu) × nft_num_train_timesteps`. |
| Reference log-prob OOM | Decrease `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | Splits reference-policy v-prediction inference into smaller per-GPU chunks. In LoRA runs where the reference is served by the actor with adapters disabled, the actor forward path is used instead. |
| High memory from many train timesteps | Decrease `algorithm.nft_num_train_timesteps` | Reduces the number of independent timesteps sampled per training batch, lowering gradient accumulation depth and peak activation memory. |

If rollout OOM persists after increasing `ROLLOUT_TP`, reduce memory-heavy rollout settings such as `actor_rollout_ref.rollout.n`, image `height` / `width`, or `actor_rollout_ref.rollout.pipeline.max_sequence_length`. If reward-model OOM persists after increasing `REWARD_TP`, consider placing the reward model on its own resource pool via `reward.reward_model.enable_resource_pool=True`.

## Wandb Logging

The provided script already enables:

```bash
trainer.logger='["console", "wandb"]' \
trainer.project_name=diffusion_nft \
trainer.experiment_name=qwen_image_ocr_nft_lora
```

Set your W&B credentials before launching if you want remote tracking:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

You can also override `trainer.project_name` and `trainer.experiment_name` from the command line to organize runs under your own project names.

## Further Reading

For the algorithm background, detailed configuration notes, loss formula, batch size
walkthrough, and image editing variant, see:

- {doc}`../algo/diffusionnft`
