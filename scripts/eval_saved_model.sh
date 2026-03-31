#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cuda:0}
NSAMPLE=${NSAMPLE:-3}
DATA_PATH=${DATA_PATH:-Economy/Economy.csv}
CONFIG=${CONFIG:-economy_36_12.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
MODEL_DIR=${MODEL_DIR:-}
MODEL_FILE=${MODEL_FILE:-}

if [[ -z "${MODEL_DIR}" && -z "${MODEL_FILE}" ]]; then
  echo "Missing model location. Set MODEL_DIR to a checkpoint directory or MODEL_FILE to a model.pth path." >&2
  exit 1
fi

MODEL_PATH="${MODEL_FILE}"
if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="${MODEL_DIR}"
fi

if [[ -d "${MODEL_PATH}" ]]; then
  if [[ ! -f "${MODEL_PATH}/model.pth" ]]; then
    echo "Checkpoint directory exists but model file is missing: ${MODEL_PATH}/model.pth" >&2
    exit 1
  fi
elif [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

python -u exe_forecasting.py \
  --root_path "${ROOT_PATH}" \
  --device "${DEVICE}" \
  --data_path "${DATA_PATH}" \
  --config "${CONFIG}" \
  --seq_len "${SEQ_LEN}" \
  --pred_len "${PRED_LEN}" \
  --text_len "${TEXT_LEN}" \
  --freq "${FREQ}" \
  --nsample "${NSAMPLE}" \
  --modelfolder "__external__" \
  --model_path "${MODEL_PATH}" \
  "${@}"
