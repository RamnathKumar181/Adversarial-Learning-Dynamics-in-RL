#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=ate_ppo_mt1
#SBATCH --output=../logs/ate_ppo_mt1_%a.out
#SBATCH --error=../logs/ate_ppo_mt1_%a.err
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -W ignore -m src.main --exp_name ate_ppo_mt1_handle-pull --algo ate_ppo --train --env mt1 --mt1_env_name 'handle-pull-v2'
