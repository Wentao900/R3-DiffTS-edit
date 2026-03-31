#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cuda:0}
GUIDE_W=${GUIDE_W:-1.0}
NSAMPLE=${NSAMPLE:-15}
RAG_TOPK=${RAG_TOPK:-3}
DATA_PATH=${DATA_PATH:-Economy/Economy.csv}
CONFIG=${CONFIG:-economy_36_12.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}

python -u exe_forecasting.py \
  --root_path "${ROOT_PATH}" \
  --device "${DEVICE}" \
  --data_path "${DATA_PATH}" \
  --config "${CONFIG}" \
  --seq_len "${SEQ_LEN}" \
  --pred_len "${PRED_LEN}" \
  --text_len "${TEXT_LEN}" \
  --freq "${FREQ}" \
  --guide_w "${GUIDE_W}" \
  --nsample "${NSAMPLE}" \
  --use_rag_cot \
  --rag_only \
  --rag_topk "${RAG_TOPK}" \
  "${@}"
