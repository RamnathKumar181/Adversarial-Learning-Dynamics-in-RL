#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=maml_half_cheetah
#SBATCH --output=../logs/maml_half_cheetah.out
#SBATCH --error=../logs/maml_half_cheetah.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G


source ../env/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py


cd .. && python -m src.main --train --exp_name maml_cheetah --config configs/maml/halfcheetah-vel.yaml --output-folder maml-halfcheetah-vel --seed 1 --num-workers 8
