#!/bin/bash

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -W ignore -m src.main --exp_name te_ppo_mt5 --train --env mt5 --seed $SLURM_ARRAY_TASK_ID
