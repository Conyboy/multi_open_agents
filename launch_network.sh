#!/bin/bash
set -e

# Create env
# conda create -n openagents python=3.12 -y
conda activate openagents

# Install
# pip install -r requirements.txt
# pip install openagents

# Initialize network
openagents init ./network

# Start network in background
openagents network start ./my_ml_network > ./logs/network.log  &
sleep 5

# Start agents (in background)
python agents/agent_a_load_analyze.py > ./logs/agent_a.log &
python agents/agent_b_feature_select.py > ./logs/agent_a.log  &
python agents/agent_c_model_tune.py > ./logs/agent_a.log  &
python agents/agent_d_train_eval.py > ./logs/agent_a.log  &
python agents/agent_e_analyze_result.py > ./logs/agent_a.log  &

# Start studio
openagents studio -s

# Wait
wait
