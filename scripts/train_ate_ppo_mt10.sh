#!/bin/bash
#SBATCH --job-name=ate_ppo_mt10
#SBATCH --output=../logs/ate_ppo_mt10.out
#SBATCH --error=../logs/ate_ppo_mt10.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name ate_ppo_mt10_limit_encoder_opt --algo ate_ppo --train --env mt10
