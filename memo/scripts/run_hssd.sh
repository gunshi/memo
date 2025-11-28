#!/bin/bash
#SBATCH --job-name=ddppo
#SBATCH --output=exp_data/logs/ddppo_%x_%j.out
#SBATCH --error=exp_data/logs/ddppo_%x_%j.err
#SBATCH --gpus a40:16
#SBATCH --nodes 2
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 8
#SBATCH --signal=USR1@90
#SBATCH --requeue

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

conda activate memo

JOB_ID="FCT_testrun_$SEED"
WB_ENTITY="causalsampling"
PROJECT_NAME="memo"
EXP_GROUPS="training_config"
DATA_DIR="data/results/$JOB_ID/"

mkdir -p $DATA_DIR

srun python memo/run.py --config-name baseline/smallObjNav_transformers_base \
        habitat_baselines.wb.entity=$WB_ENTITY \
        habitat_baselines.wb.run_name=$JOB_ID \
        habitat_baselines.wb.project_name=$PROJECT_NAME \
        habitat_baselines.wb.group=$EXP_GROUPS \
        habitat_baselines.checkpoint_folder=$DATA_DIR/checkpoints/ \
        habitat_baselines.video_dir=$DATA_DIR/vids/ \
        habitat_baselines.log_file=$DATA_DIR/log_file.log \
        habitat_baselines.tensorboard_dir=$DATA_DIR/tb/ \
        habitat_baselines.writer_type=wb \
        habitat_baselines.torch_gpu_id=0 \
        habitat_baselines.rl.policy.main_agent.transformer_config.inter_episodes_attention=True \
        habitat_baselines.rl.policy.main_agent.transformer_config.add_sequence_idx_embed=True \
        habitat_baselines.rl.policy.main_agent.transformer_config.reset_position_index=True \
        habitat_baselines.rl.policy.main_agent.transformer_config.sequence_embed_type=rope \
        habitat_baselines.rl.policy.main_agent.transformer_config.position_embed_type=linear \
        habitat_baselines.rl.ppo.shift_scene_every=-1 \
        habitat_baselines.rl.ppo.lr=4e-4 \
        habitat.task.slack_reward=-0.003 \
        habitat_baselines.rl.ppo.storage_low_precision=False \
        habitat_baselines.rl.ppo.num_steps=4096 \
        'habitat.dataset.data_path="data/datasets/ExtObjNav_HSSD_Diverse/train/combined_episodes.json.gz"' \
        habitat.seed=$SEED \