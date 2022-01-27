#!/bin/bash

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    python -W ignore -m src.main --exp_name te_ppo_mt1 --algo te_ppo --train --env mt1 --mt1_env_name 'faucet-open-v2'
fi

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    python -W ignore -m src.main --exp_name te_ppo_mt1 --algo te_ppo --train --env mt1 --mt1_env_name 'push-back-v2'
fi

if [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    python -W ignore -m src.main --exp_name te_ppo_mt1 --algo te_ppo --train --env mt1 --mt1_env_name 'coffee-button-v2'
fi

if [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    python -W ignore -m src.main --exp_name te_ppo_mt1 --algo te_ppo --train --env mt1 --mt1_env_name 'push-wall-v2'
fi
