#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=ppo_point_embed
#SBATCH --output=../logs/ppo_point_embed_og.out
#SBATCH --error=../logs/ppo_point_embed_og.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G


source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name te_ppo_pointenv --train --plot
