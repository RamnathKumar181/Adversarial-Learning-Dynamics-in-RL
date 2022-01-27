#!/bin/bash

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name te_ppo_navigation --algo te_ppo --train --env navigation --seed $SLURM_ARRAY_TASK_ID
