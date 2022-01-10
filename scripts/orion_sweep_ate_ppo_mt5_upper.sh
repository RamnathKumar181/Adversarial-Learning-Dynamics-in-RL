#!/bin/bash
#SBATCH --job-name=sweep_ate_ppo_mt5_upper
#SBATCH --output=../logs/sweep_ate_ppo_mt5_upper.out
#SBATCH --error=../logs/sweep_ate_ppo_mt5_upper.err
#SBATCH --gres=gpu:titanrtx:16gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ../../TS_Sweep/
orion hunt -n sweep_ate_ppo_mt5_upper_final --exp-max-trials 100 python sweep.py --exp_name sweep_ate_ppo_mt5_upper --algo ate_ppo --train --env mt5 --policy_ent_coeff~'loguniform(1e-3, 1.0)' --encoder_ent_coeff~'loguniform(1.0, 1e5)' --inference_ce_coeff_ent_coeff~'loguniform(1e-3, 1.0)' --embedding_init_std~'loguniform(1e-2, 1.0)' --embedding_min_std~'loguniform(1e-7, 1e-3)' --embedding_max_std~'loguniform(1e-3, 1.0)' --policy_init_std~'loguniform(1e-2, 1.0)' --policy_min_std~'loguniform(1e-7, 1e-2)' --policy_max_std~'loguniform(1e-2, 1)'
