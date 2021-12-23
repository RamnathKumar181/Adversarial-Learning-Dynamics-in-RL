#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=te_ppo_point
#SBATCH --output=../logs/te_ppo_point.out
#SBATCH --error=../logs/te_ppo_point.err
#SBATCH --gres=gpu:titanrtx:16gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=0-12:00:00

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name te_ppo_point --train --policy_optimizer_lr 1e-4
