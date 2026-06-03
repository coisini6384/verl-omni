# Qwen-Image-Edit-2511 NFT LoRA RL on ShareGPT-4o-Image-Mini, vllm_omni rollout
#
# Sibling of ``examples/flowgrpo_trainer/run_qwen_image_edit_sharegpt4o_image_mini_lora.sh``
# but trained with DiffusionNFT (https://arxiv.org/abs/2509.16117) instead
# of FlowGRPO. Reuses the same:
#   * dataset (sharegpt4o_image_mini_qwen_image_edit, the .parquet pair),
#   * Qwen-Image-Edit-2511 backbone + LoRA target modules,
#   * Qwen3-VL-8B-Instruct GRM judge as the reward model,
#   * ``compute_score_image_edit`` custom reward function.
#
# Differences vs the FlowGRPO sibling:
#   - ``algorithm.trainer_type=direct_preference`` selects
#     ``DirectPreferenceRayTrainer._fit_nft_online`` instead of FlowGRPO's
#     policy-gradient loop.
#   - ``algorithm.adv_estimator=nft`` — same group-normalized computation
#     as ``flow_grpo``, applied at sample level (no per-step expansion).
#   - ``actor.diffusion_loss.loss_mode=nft`` — registers the
#     ``DiffusionNFTLoss`` positive/negative weighted MSE.
#   - NFT does not use a reverse-SDE window; ``rollout.algo.sde_window_*``
#     keys are dropped.
#   - ``rollout.algo.noise_level=0.0`` — pure-flow rollout (no extra
#     stochasticity injected during sampling); only the final clean
#     latent is consumed by the trainer.
#   - ``actor_rollout_ref.model.algorithm=nft`` so the
#     ``DiffusionModelBase`` registry returns ``QwenImageEditPlusNFT``.
#
# Optional toggles (default on-policy without KL anchor):
#   NFT_KL_BETA=0.05   — turn on the v-space KL anchor (requires LoRA, default).
#   NFT_OFF_POLICY=true — produce ``old_noise_pred`` from the LoRA-disabled
#                         base model instead of the current policy under
#                         ``torch.no_grad()``. Same LoRA requirement.
#
# When both are set the engine reuses a single reference forward per
# training step, so enabling them together costs only ONE extra forward.
set -x

export RAY_DEDUP_LOGS=0

model_name=${MODEL_PATH:-Qwen/Qwen-Image-Edit-2511}
reward_model_name=${REWARD_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
reward_function_path=${REWARD_FUNCTION_PATH:-verl_omni/utils/reward_score/genrm_image_edit.py}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-4}
ACTOR_SP=${ACTOR_SP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
REWARD_TP=${REWARD_TP:-4}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}

# NFT-specific knobs. Defaults match strict on-policy behaviour so this
# script reproduces the simplest published NFT setup; flip the two
# toggles below to study off-policy / KL-anchor variants.
NFT_BETA=${NFT_BETA:-1.0}
NFT_NUM_TRAIN_TIMESTEPS=${NFT_NUM_TRAIN_TIMESTEPS:-5}
NFT_TIME_SAMPLING_STRATEGY=${NFT_TIME_SAMPLING_STRATEGY:-discrete}
NFT_TIMESTEP_RANGE=${NFT_TIMESTEP_RANGE:-"[0.0,0.9]"}
NFT_ADV_CLIP_RANGE=${NFT_ADV_CLIP_RANGE:-"[-5.0,5.0]"}
NFT_KL_BETA=${NFT_KL_BETA:-0.0}
NFT_OFF_POLICY=${NFT_OFF_POLICY:-False}

ENGINE=vllm_omni
REWARD_ENGINE=vllm

# Optional reproducibility (yaml defaults are null / unseeded):
#   data.seed=42
#   actor_rollout_ref.rollout.seed=42

script_path=$(readlink -f "$0")
script_name=$(basename "$script_path" .sh)
repo_root=$(dirname "$script_path")
while [[ "$repo_root" != "/" && ! -f "$repo_root/LICENSE" ]]; do
    repo_root=$(dirname "$repo_root")
done
if [[ ! -f "$repo_root/LICENSE" ]]; then
    echo "Unable to locate repo root from $script_path: no LICENSE found" >&2
    exit 1
fi

# Set WORKSPACE to any writable directory; defaults to the repository root.
WORKSPACE=${WORKSPACE:-$repo_root}
train_path=${TRAIN_FILES:-$WORKSPACE/data/sharegpt4o_image_mini_qwen_image_edit/train.parquet}
test_path=${VAL_FILES:-$WORKSPACE/data/sharegpt4o_image_mini_qwen_image_edit/test.parquet}

output_dir=$repo_root/outputs/$script_name
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_images
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "Logging to $log_file"

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    algorithm.trainer_type=direct_preference \
    algorithm.adv_estimator=nft \
    algorithm.nft_beta=$NFT_BETA \
    algorithm.nft_num_train_timesteps=$NFT_NUM_TRAIN_TIMESTEPS \
    algorithm.nft_time_sampling_strategy=$NFT_TIME_SAMPLING_STRATEGY \
    algorithm.nft_timestep_range=$NFT_TIMESTEP_RANGE \
    algorithm.nft_adv_clip_range=$NFT_ADV_CLIP_RANGE \
    algorithm.nft_kl_beta=$NFT_KL_BETA \
    algorithm.nft_off_policy=$NFT_OFF_POLICY \
    actor_rollout_ref.model.algorithm=nft \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=nft \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.nft_beta=$NFT_BETA \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$ACTOR_SP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_image_edit \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name=qwen_image_edit_sharegpt4o_image_mini_nft_lora \
    trainer.default_local_dir=$checkpoint_dir \
    +trainer.rollout_data_dir=$rollout_data_dir \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
