#!/bin/bash
# GRPO baseline experiment for countdown task (with length normalization)
# Uses SGLang rollout backend for GB10 compatibility
set -x

export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-1.5B
export DATA_DIR=/workspace/data/countdown
export ROLLOUT_TP_SIZE=1

# SGLang env vars for GB10 compatibility
export SGLANG_KERNEL_DISABLE=1
export SGLANG_ATTENTION_BACKEND=triton
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

uv run python -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=256 \
  data.max_prompt_length=256 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.project_name=TinyZero-Ablation \
  trainer.experiment_name=grpo-qwen2.5-1.5b-countdown \
  trainer.total_epochs=15 2>&1 | tee /workspace/experiments/logs/grpo_experiment.log
