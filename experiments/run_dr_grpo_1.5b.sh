#!/bin/bash
# Qwen2.5-1.5B DR-GRPO Training for DGX Spark (GB10)
# Algorithm: DR-GRPO (Dr. GRPO - no std scaling, no length normalization)
# Differences from GRPO:
#   1. No std scaling (only subtract mean)
#   2. No length normalization - longer sequences get more gradient
# See QWEN_1.5B_CONFIG_REPORT.md for details
set -x

export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-1.5B
export DATA_DIR=/workspace/data/countdown

# Weights & Biases API key (set this before running)
# export WANDB_API_KEY="your-api-key-here"

/workspace/.venv.linux-aarch64/bin/python -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=64 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=16 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
  algorithm.adv_estimator=dr_grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=50 \
  trainer.project_name=TinyZero-GB10 \
  trainer.experiment_name=dr_grpo-qwen2.5-1.5b \
  trainer.total_epochs=15 2>&1 | tee /workspace/experiments/logs/dr_grpo_1.5b.log
