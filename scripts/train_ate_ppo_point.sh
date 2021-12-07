#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=ate_ppo_point
#SBATCH --output=../logs/ate_ppo_point.out
#SBATCH --error=../logs/ate_ppo_point.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G


source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name ate_ppo_point --epochs 10 --train --plot
