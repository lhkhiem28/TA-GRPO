export WANDB_API_KEY="e9e462d100f1ec04aab88e70a25510f0f99d43a1"
export ACCELERATE_LOG_LEVEL=info

project_name="TA-GRPO"
experiment_name="Qwen2.5-1.5B-MATH-GRPO"

train_files="['../TA-GRPO-datasets/MATH/train.parquet']"
val_files="['../TA-GRPO-datasets/MATH/test.parquet']"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.critic_warmup=0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.test_freq=57 \
    trainer.save_freq=57 $@

PYTHONUNBUFFERED=1 python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/${project_name}/${experiment_name}/global_step_57/actor \
    --target_dir checkpoints/${project_name}/${experiment_name}/global_step_57/actor/huggingface
python test.py --repo_id lhkhiem28/${experiment_name} --folder_path checkpoints/${project_name}/${experiment_name}/global_step_57/actor/huggingface

dos2unix eval.sh; bash eval.sh lhkhiem28/${experiment_name}