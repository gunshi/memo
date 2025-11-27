#!/bin/bash
#SBATCH --job-name=ddppo
#SBATCH --output=exp_data/logs/ddppo_%x_%j.out
#SBATCH --error=exp_data/logs/ddppo_%x_%j.err
#SBATCH --gpus a40:16
#SBATCH --nodes 2
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 8
#SBATCH --partition=kira-lab,overcap
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --qos=short

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
# export CUDA_LAUNCH_BLOCKING=1
# export HYDRA_FULL_ERROR=1
# export HABITAT_ENV_DEBUG=1

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.95,max_split_size_mb:4096"

set -x
#use 50 as default seed
SEED=${1:-50}
echo $SEED

source  /srv/flash1/kyadav32/mambaforge/etc/profile.d/conda.sh
conda activate memorymaze_mt2

JOB_ID="feb1_rmt_slack3_lr4e-4_128_256sumseg_oldmodelrope_4096steps_newdata_fullaccum"

DATA_DIR="/srv/flash1/kyadav32/datasets/memory_maze"
WB_ENTITY="causalsampling"
PROJECT_NAME="habitat_transformers"
EXP_GROUPS="training_config"

srun python habitat_transformers/run.py --config-name baseline/smallObjNav_transformers_base_autocomp \
        habitat_baselines.wb.entity=$WB_ENTITY \
        habitat_baselines.wb.run_name=$JOB_ID \
        habitat_baselines.wb.project_name=$PROJECT_NAME \
        habitat_baselines.wb.group=$EXP_GROUPS \
        habitat_baselines.checkpoint_folder=$DATA_DIR/checkpoints/"$PROJECT_NAME"_$JOB_ID/ \
        habitat_baselines.video_dir=$DATA_DIR/vids/"$PROJECT_NAME"_$JOB_ID/ \
        habitat_baselines.log_file=$DATA_DIR/logs/"$PROJECT_NAME"_$JOB_ID.log \
        habitat_baselines.tensorboard_dir=$DATA_DIR/tb/"$PROJECT_NAME"_$JOB_ID/ \
        habitat_baselines.writer_type=wb \
        habitat_baselines.torch_gpu_id=0 \
        habitat_baselines.rl.policy.main_agent.transformer_config.inter_episodes_attention=True \
        habitat_baselines.rl.policy.main_agent.transformer_config.add_sequence_idx_embed=False \
        habitat_baselines.rl.policy.main_agent.transformer_config.reset_position_index=False \
        habitat_baselines.rl.policy.main_agent.transformer_config.sequence_embed_type=linear \
        habitat_baselines.rl.policy.main_agent.transformer_config.position_embed_type=rope \
        habitat_baselines.rl.ppo.shift_scene_every=-1 \
        habitat_baselines.rl.ppo.lr=4e-4 \
        habitat_baselines.rl.ppo.update_stale_kv=True \
        habitat_baselines.rl.ppo.update_stale_values=True \
        habitat_baselines.rl.policy.main_agent.transformer_config.segment_length=256 \
        habitat_baselines.rl.policy.main_agent.transformer_config.summary_length=128 \
        habitat.task.slack_reward=-0.003 \
        habitat_baselines.rl.ppo.storage_low_precision=False \
        habitat_baselines.rl.ppo.num_steps=4096 \
        'habitat.dataset.data_path="data/datasets/hssd_11100_samples_37_scenes_30objmax/combined_episodes.json.gz"' \
        habitat_baselines.rl.policy.main_agent.transformer_config.model_name="rmtllamarl" \
        habitat_baselines.rollout_storage_name="RMTTransformerRolloutStorage" \

        # to evaluate 
        # habitat_baselines.eval_ckpt_path_dir=ckpt.n.pth \
        # habitat_baselines.evaluate=True \
        # habitat_baselines.load_resume_state_config=False \
        
        # to not use gradient acccumulation
        # habitat_baselines.rl.ppo.grad_accum_mini_batches=1 \
        # habitat_baselines.rl.ppo.num_mini_batch=2 \

        # to make videos during eval
        # habitat_baselines.eval.video_option=["disk"]
