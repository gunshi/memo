#!/bin/bash
#SBATCH --job-name=ddppo-rl2-eval-hab
#SBATCH --output=slurm_logs/eval/ddppo_rl2_hab_%x_%j.out
#SBATCH --error=slurm_logs/eval/ddppo_rl2_hab_%x_%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --exclude=baymax,xaea-12,heistotron,gundam,cyborg
#SBATCH --partition=cvmlp-lab
#SBATCH --qos=long
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

set -x

JOB_ID="small_objnav_ppo_rl2_lstm_256_hidden_4096_switch_256_update_done_not_during_rollout_fixed_evaluator_habevaluator_all_episodes_100_ckpt_10_evals_per_ep"
TRAIN_JOB_ID="small_objnav_ppo_rl2_lstm_256_hidden_4096_switch_256_update_done_not_during_rollout_fixed"

EXP_DATA="exp_data"
split="val"
DATA_PATH="/coc/testnvme/aelawady3/src/local_dev/habitat-transformers/habitat_transformers/data/datasets/hssd109_1eps_clean/rearrange/${split}/rearrange_ep_dataset.json.gz"
WB_ENTITY="gchhablani"
PROJECT_NAME="transformers_agent"
CKPT=100

source /srv/flash1/gchhablani3/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate hab_mt

srun python run.py --config-name lstm/smallObjNav_rl2 \
        habitat_baselines.evaluate=True \
        habitat.dataset.split=val \
        habitat.dataset.data_path=${DATA_PATH} \
        habitat_baselines.wb.entity=$WB_ENTITY \
        habitat_baselines.wb.run_name=$JOB_ID \
        habitat_baselines.wb.project_name=$PROJECT_NAME \
        habitat_baselines.checkpoint_folder=$EXP_DATA/new_checkpoints/$TRAIN_JOB_ID/ \
        habitat_baselines.eval_ckpt_path_dir=$EXP_DATA/new_checkpoints/$TRAIN_JOB_ID/ckpt.${CKPT}.pth \
        habitat_baselines.video_dir=$EXP_DATA/vids/$JOB_ID/ \
        habitat_baselines.log_file=$EXP_DATA/logs/$JOB_ID.log \
        habitat_baselines.tensorboard_dir=$EXP_DATA/tb/$JOB_ID/ \
        habitat_baselines.writer_type=wb habitat_baselines.torch_gpu_id=0 \
        habitat_baselines.rl.ppo.num_steps=256 \
        habitat_baselines.trainer_name="rl2" \
        habitat_baselines.rollout_storage_name="RL2RolloutStorage" \
        habitat_baselines.call_after_update_env=True \
        habitat_baselines.rl.ppo.shift_scene_every=4096 \
        habitat_baselines.rl.ppo.hidden_size=256 \
        habitat_baselines.load_resume_state_config=False \
        habitat_baselines.eval.use_ckpt_config=False \
        habitat_baselines.test_episode_count=-1 \
        habitat_baselines.eval.video_option="[]" \
        habitat_baselines.eval.evals_per_ep=10 \
        habitat_baselines.num_environments=12 \
        habitat_baselines.vector_env_factory._target_="habitat_baselines.common.HabitatVectorEnvFactory" \
        habitat.env_task="GymHabitatEnv" \
        habitat.simulator.type="RearrangeSim-v0" \
        habitat.dataset.type="RearrangeDataset-v0" \
        habitat.task.target_sampling_strategy="object_type" \
        habitat.environment.iterator_options.max_scene_repeat_episodes=-1 \
        habitat.environment.iterator_options.max_scene_repeat_steps=-1 \
        habitat.environment.iterator_options.shuffle=False \
        habitat.task.fix_target_same_episode=False \
        habitat.task.max_num_start_pos=-1 \
        
