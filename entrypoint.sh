#!/bin/bash
set -e

echo "Initializing OpenAgents network..."
openagents init ./my_ml_network

echo "Starting OpenAgents network in background..."
openagents network start ./my_ml_network &
sleep 5

echo "Starting all agents in background..."
python agents/agent_a_load_analyze.py &
python agents/agent_b_feature_select.py &
python agents/agent_c_model_tune.py &
python agents/agent_d_train_eval.py &
python agents/agent_e_analyze_result.py &

echo "Starting OpenAgents Studio..."
openagents studio -s

# Keep container alive
wait
