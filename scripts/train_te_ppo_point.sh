#!/bin/bash

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py


cd ..
python -m src.main --exp_name te_ppo_point --train --policy_optimizer_lr 1e-4
