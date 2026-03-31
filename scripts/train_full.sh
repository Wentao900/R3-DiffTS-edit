#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cuda:0}
GUIDE_W=${GUIDE_W:-1.0}
NSAMPLE=${NSAMPLE:-15}
RAG_TOPK=${RAG_TOPK:-3}
RAG_STAGE1_TOPK=${RAG_STAGE1_TOPK:-9}
RAG_STAGE2_TOPK=${RAG_STAGE2_TOPK:-3}
MULTI_RES_BANDS=${MULTI_RES_BANDS:-"3 6 12"}
MULTI_RES_WEIGHT=${MULTI_RES_WEIGHT:-0.1}
ROUTER_HIDDEN=${ROUTER_HIDDEN:-32}
ROUTER_ENTROPY=${ROUTER_ENTROPY:-0.001}
ROUTER_TEACHER=${ROUTER_TEACHER:-0.1}
ROUTER_WARMUP=${ROUTER_WARMUP:-400}
TREND_CFG_POWER=${TREND_CFG_POWER:-1.0}
TREND_TIME_FLOOR=${TREND_TIME_FLOOR:-0.30}
TREND_STRENGTH_SCALE=${TREND_STRENGTH_SCALE:-0.35}
TREND_VOLATILITY_SCALE=${TREND_VOLATILITY_SCALE:-1.0}
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
  --use_two_stage_rag \
  --trend_cfg \
  --rag_topk "${RAG_TOPK}" \
  --rag_stage1_topk "${RAG_STAGE1_TOPK}" \
  --rag_stage2_topk "${RAG_STAGE2_TOPK}" \
  --trend_cfg_power "${TREND_CFG_POWER}" \
  --trend_time_floor "${TREND_TIME_FLOOR}" \
  --trend_strength_scale "${TREND_STRENGTH_SCALE}" \
  --trend_volatility_scale "${TREND_VOLATILITY_SCALE}" \
  --multi_res_band_boundaries ${MULTI_RES_BANDS} \
  --multi_res_loss_weight "${MULTI_RES_WEIGHT}" \
  --use_scale_router \
  --scale_router_hidden_dim "${ROUTER_HIDDEN}" \
  --scale_router_entropy_weight "${ROUTER_ENTROPY}" \
  --scale_router_teacher_weight "${ROUTER_TEACHER}" \
  --scale_router_warmup_steps "${ROUTER_WARMUP}" \
  "${@}"
