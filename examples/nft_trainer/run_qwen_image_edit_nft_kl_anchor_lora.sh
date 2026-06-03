# Qwen-Image-Edit NFT LoRA training with KL anchor and off-policy sampling.
#
# This example demonstrates the two NFT enhancement flags introduced after
# the initial DiffusionNFT integration:
#
#   * ``algorithm.nft_off_policy=true``
#       Generates ``old_noise_pred`` from the LoRA-disabled base model
#       (sampling-policy ≠ training-policy). This matches the "off-policy"
#       mode in flow-factory's DiffusionNFTTrainer (which uses an EMA
#       wrapper for the same purpose) and is the configuration the
#       paper studies for off-policy correction.
#
#   * ``algorithm.nft_kl_beta=0.05``
#       Adds a v-space KL anchor::
#
#           kl_loss = nft_kl_beta * mean((noise_pred - ref_noise_pred) ** 2)
#
#       to keep the LoRA-trained policy from drifting too far from the
#       frozen base model. ``ref_noise_pred`` reuses the same off-policy
#       reference forward, so enabling both flags only costs ONE extra
#       forward per training step (not two).
#
# REQUIREMENTS
#   * LoRA training (``actor_rollout_ref.model.lora_rank > 0``). The two
#     flags rely on ``DiffusersFSDPEngine.disable_adapter()`` to produce
#     the reference forward — turning off the LoRA adapter only yields a
#     useful reference when LoRA is actually active. The engine raises
#     RuntimeError otherwise.
#
# Tune nft_kl_beta in [0.001, 0.1]. Larger values pull more strongly
# toward the base model and hurt reward; smaller values drift toward the
# unanchored policy. ``0.0`` recovers the strict on-policy behaviour
# (equivalent to ``run_qwen_image_edit_nft_lora.sh``).
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

train_path=$WORKSPACE/data/sharegpt4o_image_mini_qwen_image_edit/train.parquet
test_path=$WORKSPACE/data/sharegpt4o_image_mini_qwen_image_edit/test.parquet

model_name=Qwen/Qwen-Image-Edit-2511
reward_model_name=Qwen/Qwen3-VL-8B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_image_edit.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=4
ROLLOUT_TP=1
REWARD_TP=4

ENGINE=vllm_omni
REWARD_ENGINE=vllm

# KL anchor strength. Set to 0.0 to recover strict on-policy NFT.
NFT_KL_BETA=${NFT_KL_BETA:-0.05}
# Off-policy sampling. Set to false to recover strict on-policy NFT.
NFT_OFF_POLICY=${NFT_OFF_POLICY:-true}

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    algorithm.trainer_type=direct_preference \
    algorithm.adv_estimator=nft \
    algorithm.nft_beta=1.0 \
    algorithm.nft_off_policy=$NFT_OFF_POLICY \
    algorithm.nft_num_train_timesteps=5 \
    algorithm.nft_time_sampling_strategy=discrete \
    algorithm.nft_timestep_range="[0.0,0.9]" \
    algorithm.nft_adv_clip_range="[-5.0,5.0]" \
    algorithm.nft_kl_beta=$NFT_KL_BETA \
    actor_rollout_ref.model.algorithm=nft \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=nft \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.nft_beta=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=512 \
    actor_rollout_ref.rollout.pipeline.height=1024 \
    actor_rollout_ref.rollout.pipeline.width=1024 \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_edit \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name=qwen_image_edit_nft_kl_anchor_lora \
    trainer.log_val_generations=4 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=10 \
    "$@"
