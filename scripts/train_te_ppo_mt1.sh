#!/bin/bash
#SBATCH --job-name=te_ppo_mt1
#SBATCH --output=../logs/te_ppo_mt1.out
#SBATCH --error=../logs/te_ppo_mt1.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name te_ppo_mt1 --train --env mt1 --epochs 600 --policy_optimizer_lr 1e-3 --inference_optimizer_lr 1e-3
