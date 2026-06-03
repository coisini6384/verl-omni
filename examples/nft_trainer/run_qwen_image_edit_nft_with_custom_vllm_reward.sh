#!/bin/bash
# examples/nft_trainer/run_qwen_image_edit_nft_with_custom_vllm_reward.sh
#
# NFT 训练示例：使用自定义 vLLM 奖励函数
#
# 使用方法：
#   1. 启动 vLLM 服务：
#      python -m vllm.entrypoints.openai.api_server \
#          --model Qwen/Qwen-VL-7B-Chat \
#          --tensor-parallel-size 2 \
#          --port 8000
#
#   2. 运行训练：
#      bash examples/nft_trainer/run_qwen_image_edit_nft_with_custom_vllm_reward.sh

set -x

# ============================================================================
# 配置
# ============================================================================

WORKSPACE=${WORKSPACE:-$HOME}

train_path=$WORKSPACE/data/image_edit/train.parquet
test_path=$WORKSPACE/data/image_edit/test.parquet

# 模型配置
model_name=Qwen/Qwen-Image-Edit-2511

# 奖励函数配置 - 自定义 vLLM 奖励函数
# 路径：verl_omni/utils/reward_score/vllm_quality_reward.py
# 函数名：compute_score_vllm_quality（async 函数）
reward_function_path=verl_omni/utils/reward_score/vllm_quality_reward.py
reward_function_name=compute_score_vllm_quality

# vLLM 服务配置
# 假设在 127.0.0.1:8000 运行 vLLM 推理服务
VLLM_HOST=127.0.0.1
VLLM_PORT=8000
REWARD_ROUTER_ADDRESS="${VLLM_HOST}:${VLLM_PORT}"

# GPU 配置
NUM_GPUS_ACTOR_ROLLOUT_REWARD=4
ROLLOUT_TP=2
REWARD_TP=1

ENGINE=vllm_omni
REWARD_ENGINE=vllm

# ============================================================================
# 检查必要的文件和服务
# ============================================================================

if [ ! -f "$reward_function_path" ]; then
    echo "错误：找不到奖励函数文件 $reward_function_path"
    echo "请确保文件存在或修改 reward_function_path 变量"
    exit 1
fi

# 检查 vLLM 服务是否运行
echo "检查 vLLM 服务..."
if ! curl -s "http://${REWARD_ROUTER_ADDRESS}/v1/models" > /dev/null; then
    echo "警告：无法连接到 vLLM 服务 http://${REWARD_ROUTER_ADDRESS}"
    echo "请先启动 vLLM 服务："
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "      --model Qwen/Qwen-VL-7B-Chat \\"
    echo "      --tensor-parallel-size 2 \\"
    echo "      --port 8000"
    echo ""
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================================================
# 开始训练
# ============================================================================

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    \
    # ========== 算法配置 ==========
    algorithm.trainer_type=direct_preference \
    algorithm.adv_estimator=nft \
    algorithm.nft_beta=1.0 \
    algorithm.nft_num_train_timesteps=5 \
    algorithm.nft_time_sampling_strategy=discrete \
    algorithm.nft_timestep_range="[0.0,0.9]" \
    algorithm.nft_adv_clip_range="[-5.0,5.0]" \
    algorithm.nft_kl_beta=0.0 \
    \
    # ========== 模型配置 ==========
    actor_rollout_ref.model.algorithm=nft \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','net.0.proj','net.2']" \
    \
    # ========== 训练优化器配置 ==========
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=nft \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.nft_beta=1.0 \
    \
    # ========== FSDP 配置 ==========
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    \
    # ========== 生成（Rollout）配置 ==========
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
    \
    # ========== 自定义奖励函数配置 ==========
    # 这是关键部分：指定自定义的 vLLM 奖励函数
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=$reward_function_name \
    reward.num_workers=4 \
    \
    # ========== 奖励模型配置（可选）==========
    # 如果自定义函数需要模型服务，启用以下配置
    # reward.reward_model.enable=True \
    # reward.reward_model.model_path="Qwen/Qwen-VL-7B-Chat" \
    # reward.reward_model.rollout.name=$REWARD_ENGINE \
    # reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    \
    # ========== 日志和检查点配置 ==========
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name=qwen_image_edit_nft_custom_vllm_reward \
    trainer.log_val_generations=4 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 \
    "$@"
