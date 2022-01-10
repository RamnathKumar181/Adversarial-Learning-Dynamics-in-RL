#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=te_ppo_navigation
#SBATCH --output=../logs/te_ppo_navigation.out
#SBATCH --error=../logs/te_ppo_navigation.err
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name te_ppo_navigation --algo te_ppo --train --env navigation
