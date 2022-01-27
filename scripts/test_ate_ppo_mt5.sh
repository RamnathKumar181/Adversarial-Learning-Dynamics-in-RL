#!/bin/bash
source ../venv/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py

cd ..
python -m src.main --exp_name test_ate_ppo_mt5 --algo ate_ppo --env mt5 --causal
