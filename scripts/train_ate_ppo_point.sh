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

export MUJOCO_GL=osmesa
export MJLIB_PATH=/cvmfs/ai.mila.quebec/apps/x86_64/common/mujoco/2.0/bin/libmujoco200.so
export MJKEY_PATH=/cvmfs/config.mila.quebec/etc/mujoco/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200/
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt

cd ..
python -m src.main --exp_name ate_ppo_point --algo ate_ppo --epochs 2 --train --plot --latent_length 4
