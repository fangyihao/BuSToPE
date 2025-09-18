#!/bin/bash
set -x

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="log/videomme_$timestamp.log"
> "$LOG_FILE"

exec > "$LOG_FILE" 2>&1

export CKPT_DIR=../LLaMA-Factory/checkpoints/models--Qwen--Qwen2-VL-2B-Instruct/
export DECORD_EOF_RETRY_MAX=20480
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list='7'
IFS=',' read -ra GPULIST <<< "$gpu_list"
export CHUNKS=${#GPULIST[@]}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export MIN_PIXELS_FACTOR=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

#CONTEXT_LENGTHS=(8192 16384 32768 64000)
CONTEXT_LENGTHS=(1024 2048 4096)

MODEL_LIST=(
    "m_rope m_rope 1.0"
    "videorope videorope 2.0"
    "tad_rope tad_rope 1.0"
)



for MODEL_ENTRY in "${MODEL_LIST[@]}"; do

    CKPT=$(echo "$MODEL_ENTRY" | awk '{print $1}')
    WHICH_ROPE=$(echo "$MODEL_ENTRY" | awk '{print $2}')
    SCALE_FACTOR=$(echo "$MODEL_ENTRY" | awk '{print $3}')

    for context_length in "${CONTEXT_LENGTHS[@]}"; do
        OUTPUT_FOLDER="results/video_mme/${CKPT}-${context_length}-${MIN_PIXELS_FACTOR}tokens"
        
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 model_videomme_qwen2_vl.py \
                --model-path ${CKPT_DIR}/${CKPT} \
                --max_new_tokens 128 \
                --Eval_Video_root /data/Video-MME/data \
                --Eval_QA_root /data/Video-MME/videomme/test-00000-of-00001.tsv \
                --chat_conversation_output_folder $OUTPUT_FOLDER \
                --context_length $context_length \
                --num-chunks $CHUNKS \
                --chunk-idx $IDX \
                --which_rope $WHICH_ROPE \
                --scale_factor $SCALE_FACTOR \
                --min_pixels $(( MIN_PIXELS_FACTOR * 28 * 28 )) &
        done
        wait
        python3 check_videomme.py $OUTPUT_FOLDER
    done
done
