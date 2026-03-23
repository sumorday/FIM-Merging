#!/bin/bash
# run_eval.sh - Run eval one benchmark at a time, auto-retry on crash
#
# Usage:
#   bash run_eval.sh <model_path> <output_dir> <log_file> <gpu_id>
#
# Example:
#   bash run_eval.sh /home/t640/models/l2s_7b_fim_ties \
#                   /home/t640/models/l2s_7b_fim_ties/math_eval \
#                   ~/fim_7b_ties_eval.log \
#                   0

MODEL_PATH=$1
OUTPUT_DIR=$2
LOG_FILE=$3
GPU_ID=${4:-0}
MAX_RETRIES=3
EVAL_DIR=~/Long-to-Short-via-Model-Merging/Qwen2.5-Math-main/evaluation

DATASETS=("gsm8k" "math" "minerva_math" "olympiadbench" "college_math" "aime24")

echo "========================================" | tee -a $LOG_FILE
echo "Starting eval for: $MODEL_PATH" | tee -a $LOG_FILE
echo "GPU: $GPU_ID" | tee -a $LOG_FILE
echo "Time: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

for DATASET in "${DATASETS[@]}"; do
    # Check if already completed (result file exists and has acc)
    RESULT_FILE="$OUTPUT_DIR/$DATASET/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"
    if [ -f "$RESULT_FILE" ]; then
        echo "[SKIP] $DATASET already completed, skipping." | tee -a $LOG_FILE
        continue
    fi

    echo "" | tee -a $LOG_FILE
    echo "--------------------" | tee -a $LOG_FILE
    echo "[START] $DATASET at $(date)" | tee -a $LOG_FILE

    SUCCESS=0
    for ATTEMPT in $(seq 1 $MAX_RETRIES); do
        echo "[ATTEMPT $ATTEMPT/$MAX_RETRIES] Running $DATASET..." | tee -a $LOG_FILE

        CUDA_VISIBLE_DEVICES=$GPU_ID python $EVAL_DIR/math_eval.py \
            --model_name_or_path $MODEL_PATH \
            --data_name $DATASET \
            --output_dir $OUTPUT_DIR \
            --split test \
            --prompt_type qwen25-math-cot \
            --num_test_sample -1 \
            --seed 0 \
            --temperature 0 \
            --n_sampling 1 \
            --max_tokens_per_call 10240 \
            --top_p 1 \
            --num_shots 0 \
            --data_dir /home/t640/Long-to-Short-via-Model-Merging/Qwen2.5-Math-main/evaluation/data/ --data_dir /home/t640/Long-to-Short-via-Model-Merging/Qwen2.5-Math-main/evaluation/data/ --use_vllm \
            --save_outputs \
            --overwrite 2>&1 | tee -a $LOG_FILE

        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[DONE] $DATASET completed successfully." | tee -a $LOG_FILE
            SUCCESS=1
            break
        else
            echo "[FAILED] $DATASET attempt $ATTEMPT failed (exit code $EXIT_CODE)." | tee -a $LOG_FILE
            if [ $ATTEMPT -lt $MAX_RETRIES ]; then
                echo "[RETRY] Waiting 10s before retry..." | tee -a $LOG_FILE
                sleep 10
            fi
        fi
    done

    if [ $SUCCESS -eq 0 ]; then
        echo "[SKIP] $DATASET failed after $MAX_RETRIES attempts, moving to next." | tee -a $LOG_FILE
    fi
done

echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "All benchmarks finished at $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# Print summary
echo "" | tee -a $LOG_FILE
echo "===== FINAL RESULTS =====" | tee -a $LOG_FILE
grep "'acc'" $LOG_FILE | tail -20 | tee -a /dev/null
