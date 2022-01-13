#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=test_ate_ppo_mt5
#SBATCH --output=../logs/test_ate_ppo_mt5.out
#SBATCH --error=../logs/test_ate_ppo_mt5.err
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name test_ate_ppo_mt5 --algo ate_ppo --env mt5
