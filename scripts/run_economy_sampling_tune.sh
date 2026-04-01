#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cuda:0}
DATA_PATH=${DATA_PATH:-Economy/Economy.csv}
CONFIG=${CONFIG:-economy_36_12_tune.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
NSAMPLE=${NSAMPLE:-15}
REPEATS=${REPEATS:-1}

SAMPLE_STEPS_LIST=${SAMPLE_STEPS_LIST:-"10 20 30"}
TREND_CFG_POWER_LIST=${TREND_CFG_POWER_LIST:-"0.8 1.0 1.2"}
TREND_TIME_FLOOR_LIST=${TREND_TIME_FLOOR_LIST:-"0.2 0.3 0.4"}

RAG_TOPK=${RAG_TOPK:-3}
RAG_STAGE1_TOPK=${RAG_STAGE1_TOPK:-9}
RAG_STAGE2_TOPK=${RAG_STAGE2_TOPK:-3}
MULTI_RES_BANDS=${MULTI_RES_BANDS:-"3 6 12"}
MULTI_RES_WEIGHT=${MULTI_RES_WEIGHT:-0.1}
ROUTER_HIDDEN=${ROUTER_HIDDEN:-32}
ROUTER_ENTROPY=${ROUTER_ENTROPY:-0.001}
ROUTER_TEACHER=${ROUTER_TEACHER:-0.1}
ROUTER_WARMUP=${ROUTER_WARMUP:-400}
TREND_STRENGTH_SCALE=${TREND_STRENGTH_SCALE:-0.35}
TREND_VOLATILITY_SCALE=${TREND_VOLATILITY_SCALE:-1.0}

for repeat_id in $(seq 1 "${REPEATS}"); do
  for sample_steps in ${SAMPLE_STEPS_LIST}; do
    for trend_cfg_power in ${TREND_CFG_POWER_LIST}; do
      for trend_time_floor in ${TREND_TIME_FLOOR_LIST}; do
        echo "===== repeat=${repeat_id} sample_steps=${sample_steps} trend_cfg_power=${trend_cfg_power} trend_time_floor=${trend_time_floor} ====="
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
          --use_rag_cot \
          --use_two_stage_rag \
          --trend_cfg \
          --rag_topk "${RAG_TOPK}" \
          --rag_stage1_topk "${RAG_STAGE1_TOPK}" \
          --rag_stage2_topk "${RAG_STAGE2_TOPK}" \
          --trend_cfg_power "${trend_cfg_power}" \
          --trend_time_floor "${trend_time_floor}" \
          --trend_strength_scale "${TREND_STRENGTH_SCALE}" \
          --trend_volatility_scale "${TREND_VOLATILITY_SCALE}" \
          --sample_steps "${sample_steps}" \
          --multi_res_band_boundaries ${MULTI_RES_BANDS} \
          --multi_res_loss_weight "${MULTI_RES_WEIGHT}" \
          --use_scale_router \
          --scale_router_hidden_dim "${ROUTER_HIDDEN}" \
          --scale_router_entropy_weight "${ROUTER_ENTROPY}" \
          --scale_router_teacher_weight "${ROUTER_TEACHER}" \
          --scale_router_warmup_steps "${ROUTER_WARMUP}"
      done
    done
  done
done
