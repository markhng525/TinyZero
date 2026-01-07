#!/bin/bash
# Quick GRPO test for GB10 - Conservative batch sizes for fast iteration
set -x

export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-1.5B
export DATA_DIR=/workspace/data/countdown
export ROLLOUT_TP_SIZE=1

# SGLang env vars for GB10
export SGLANG_KERNEL_DISABLE=1
export SGLANG_ATTENTION_BACKEND=triton
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

# Use venv Python directly to avoid uv reinstalling CPU torch
/workspace/.venv.linux-aarch64/bin/python -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=64 \
  data.val_batch_size=64 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.project_name=TinyZero-GB10-QuickTest \
  trainer.experiment_name=grpo-quick-test \
  trainer.total_epochs=2 2>&1 | tee /workspace/experiments/logs/grpo_quick_test.log
