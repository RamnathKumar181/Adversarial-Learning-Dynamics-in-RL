#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=sweep_ate_ppo_mt5
#SBATCH --output=../logs/sweep_ate_ppo_mt5.out
#SBATCH --error=../logs/sweep_ate_ppo_mt5.err
#SBATCH --gres=gpu:titanrtx:16gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -W ignore -m src.main --exp_name sweep_ate_ppo_mt5 --algo ate_ppo --train --env mt5 --policy_ent_coeff~'loguniform(1e-5, 1.0)'
