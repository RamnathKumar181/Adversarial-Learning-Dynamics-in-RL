#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=test_ate_ppo_point
#SBATCH --output=../logs/test_ate_ppo_point.out
#SBATCH --error=../logs/test_ate_ppo_point.err
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name test_ate_ppo_point --algo ate_ppo --causal


# Plotted ace for task: 0 with goal: [0. 3.]
# Importance for task 0: [1.         0.         0.44247897 0.7678517 ]
# Plotted ace for task: 1 with goal: [3.0000000e+00 1.8369701e-16]
# Importance for task 1: [1.         0.         0.80243822 0.82104762]
# Plotted ace for task: 2 with goal: [ 3.6739403e-16 -3.0000000e+00]
# Importance for task 2: [1.         0.         0.26947691 0.80458654]
# Plotted ace for task: 3 with goal: [-3.0000000e+00 -5.5109107e-16]
# Importance for task 3: [0.70655665 0.         1.         0.82527847]
