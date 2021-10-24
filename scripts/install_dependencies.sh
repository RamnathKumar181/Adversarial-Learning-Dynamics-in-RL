#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=install_dependencies
#SBATCH --output=../logs/install_dependencies.out
#SBATCH --error=../logs/install_dependencies.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G


source ../env/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
pip install -r requirements.txt
