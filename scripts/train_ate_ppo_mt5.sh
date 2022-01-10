#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=ate_ppo_mt5
#SBATCH --output=../logs/ate_ppo_mt5.out
#SBATCH --error=../logs/ate_ppo_mt5.err
#SBATCH --gres=gpu:titanrtx:16gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -W ignore -m src.main --exp_name ate_ppo_mt5_titan --algo ate_ppo --train --env mt5 --snapshot_dir ./config/test/
