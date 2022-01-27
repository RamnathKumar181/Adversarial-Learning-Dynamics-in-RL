#!/bin/bash

source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name test_ate_ppo_navigation --env navigation --algo ate_ppo --vis --causal
