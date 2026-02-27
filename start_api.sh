#!/usr/bin/env bash
TRL_PATH="/cluster/home/fraluca/negotio2"
export PYTHONPATH="$TRL_PATH:$PYTHONPATH"
export PYTHONPATH="$TRL_PATH/trl:$PYTHONPATH"

echo "RUN_DIR=$RUN_DIR"
echo "MODEL_NAME=$MODEL_NAME"

echo "RUN_DIR=$RUN_DIR NODE=$NODE M0=$MODEL0_PORT M1=$MODEL1_PORT MODEL=$MODEL_NAME" | tee -a "$RUN_DIR/setup.log"


# Start API1 (responder)
CUDA_VISIBLE_DEVICES=0 python "$TRL_PATH/trl/trl/scripts/vllm_serve_opp.py" \
     --model "$MODEL_NAME" \
     --tensor_parallel_size 1 \
     --host "$NODE" \
     --port $MODEL1_PORT \
     --max_model_len 4096 \
     --gpu_memory_utilization 0.8 > "$RUN_DIR/process1.out" 2> "$RUN_DIR/process1.err" &

API1_PID=$!
echo "API1 started with PID: $API1_PID" | tee -a "$RUN_DIR/setup.log"

# Start API0 (Initiator) simultaneously
CUDA_VISIBLE_DEVICES=1 python "$TRL_PATH/trl/trl/scripts/vllm_serve_base.py" \
     --model "$MODEL_NAME" \
     --tensor_parallel_size 1 \
     --host "$NODE" \
     --port $MODEL0_PORT \
     --gpu_memory_utilization 0.8 \
     --partner_host "$NODE" \
     --partner_port $MODEL1_PORT \
     --conversation_turns 5 > "$RUN_DIR/process0.out" 2> "$RUN_DIR/process0.err" &

API0_PID=$!
echo "API0 started with PID: $API0_PID" | tee -a "$RUN_DIR/setup.log"

# Now wait for both to be ready
echo "Waiting for both APIs to be ready..." | tee -a "$RUN_DIR/setup.log"

MAX_TRIES=20  # Increased to allow more time for both to start
COUNTER=0

while [ $COUNTER -lt $MAX_TRIES ]; do
    sleep 30
    
    # Check if processes are still running
    API0_RUNNING=0
    API1_RUNNING=0
    
    if ps -p $API0_PID > /dev/null; then
        API0_RUNNING=1
    else
        echo "Error: API0 process failed!" | tee -a "$RUN_DIR/setup.log"
        break
    fi
    
    if ps -p $API1_PID > /dev/null; then
        API1_RUNNING=1
    else
        echo "Error: API1 process failed!" | tee -a "$RUN_DIR/setup.log"
        break
    fi
    
    # Check health status
    API0_READY=0
    API1_READY=0
    
    # Validate that the response is actually from our vLLM server (JSON with "status"),
    # not from another service (e.g., Nvidia GPU Exporter returning HTML on the same port)
    if curl -s --max-time 5 --noproxy "*" http://$NODE:$MODEL0_PORT/health/ 2>/dev/null | grep -q '"status"'; then
        API0_READY=1
        echo "API0 is ready!" | tee -a "$RUN_DIR/setup.log"
    fi
    
    if curl -s --max-time 5 --noproxy "*" http://$NODE:$MODEL1_PORT/health/ 2>/dev/null | grep -q '"status"'; then
        API1_READY=1
        echo "API1 is ready!" | tee -a "$RUN_DIR/setup.log"
    fi
    
    # If both are ready, we can proceed
    if [ $API0_READY -eq 1 ] && [ $API1_READY -eq 1 ]; then
        echo "Both APIs are ready!" | tee -a "$RUN_DIR/setup.log"
        break
    fi
    
    echo "APIs not fully ready yet, waiting another 30 seconds..." | tee -a "$RUN_DIR/setup.log"
    COUNTER=$((COUNTER+1))
    
    if [ $COUNTER -eq $MAX_TRIES ]; then
        echo "Timeout waiting for APIs. Continuing anyway as long as processes are running..." | tee -a "$RUN_DIR/setup.log"
    fi
done

# Try the health check with the actual hostname
echo "Trying health check with hostname: $NODE" | tee -a "$RUN_DIR/setup.log"

curl -v --noproxy "*" http://$(hostname):$MODEL0_PORT/health/
curl -v --noproxy "*" http://$(hostname):$MODEL1_PORT/health/

echo "Running API communication test..." | tee -a "$RUN_DIR/setup.log"
mkdir -p "$RUN_DIR/tests"
python test_apis.py "$NODE:$MODEL0_PORT" "$NODE:$MODEL1_PORT" > "$RUN_DIR/tests/test_api.out" 2> "$RUN_DIR/test_api.err" &
TEST_RESULT=$?

export ACCELERATE_CONFIG="/cluster/home/fraluca/.cache/huggingface/accelerate/default_config.yaml"

export VLLM_SERVER_HOST="$NODE"
export VLLM_SERVER_PORT="$MODEL0_PORT"
