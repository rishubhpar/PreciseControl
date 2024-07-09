#!/bin/bash

export PYTHONPATH="${PWD}"

SD_PATH=$1  # /Your/Path/To/sd-v1-4-full-ema.ckpt
LOG_DIR_NAME=${2:-"id_name"}
EMBEDDING_PATH=${3:-logs/wt_mapper_70k_sd2_idloss/checkpoints/embeddings_gs-139999.pt}
CONFIG_FILE=${4:-configs/stable-diffusion/aigc_id_for_lora.yaml}


# Usage Example:
# ./start_train.sh ./weights/sd-v1-4-full-ema.ckpt

python main_id_embed.py --base "${CONFIG_FILE}" \
               -t \
               --actual_resume "${SD_PATH}" \
               -n "${LOG_DIR_NAME}" \
               --gpus 0, --no-test True --embedding_manager_ckpt "${EMBEDDING_PATH}" \