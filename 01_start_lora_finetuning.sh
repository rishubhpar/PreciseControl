#!/bin/bash

export PYTHONPATH="${PWD}"

SD_PATH=$1  # /Your/Path/To/sd-v1-4-full-ema.ckpt
LOG_DIR_NAME=${2:-"celebbasis"}
EMBEDDING_PATH=${3:-/mnt/data/rishubh/sachi/CelebBasis_pstar_sd2/logs/wt_interpolation_sd2_idloss_finetune/checkpoints/embeddings_gs-149999.pt}
CONFIG_FILE=${4:-configs/stable-diffusion/aigc_id_for_lora_all.yaml}


# Usage Example:
# ./start_train.sh ./weights/sd-v1-4-full-ema.ckpt

python main_id_embed.py --base "${CONFIG_FILE}" \
               -t \
               --actual_resume "${SD_PATH}" \
               -n "${LOG_DIR_NAME}" \
               --gpus 0, --no-test True --embedding_manager_ckpt "${EMBEDDING_PATH}" \