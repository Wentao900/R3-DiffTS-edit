#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cuda:0}
GUIDE_W=${GUIDE_W:-1.0}
NSAMPLE=${NSAMPLE:-15}
RAG_TOPK=${RAG_TOPK:-3}
RAG_STAGE1_TOPK=${RAG_STAGE1_TOPK:-9}
RAG_STAGE2_TOPK=${RAG_STAGE2_TOPK:-3}
TREND_CFG_POWER=${TREND_CFG_POWER:-1.0}
TREND_TIME_FLOOR=${TREND_TIME_FLOOR:-0.30}
TREND_STRENGTH_SCALE=${TREND_STRENGTH_SCALE:-0.35}
TREND_VOLATILITY_SCALE=${TREND_VOLATILITY_SCALE:-1.0}

run_case() {
  local data_path="$1"
  local config="$2"
  local seq_len="$3"
  local pred_len="$4"
  local text_len="$5"
  local freq="$6"

  python -u exe_forecasting.py \
    --root_path "${ROOT_PATH}" \
    --device "${DEVICE}" \
    --data_path "${data_path}" \
    --config "${config}" \
    --seq_len "${seq_len}" \
    --pred_len "${pred_len}" \
    --text_len "${text_len}" \
    --freq "${freq}" \
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
    --trend_volatility_scale "${TREND_VOLATILITY_SCALE}"
}

run_case "Traffic/Traffic.csv" "traffic_36_6.yaml" 36 6 36 m
run_case "Traffic/Traffic.csv" "traffic_36_12.yaml" 36 12 36 m
run_case "Traffic/Traffic.csv" "traffic_36_18.yaml" 36 18 36 m

run_case "SocialGood/SocialGood.csv" "socialgood_36_6.yaml" 36 6 36 m
run_case "SocialGood/SocialGood.csv" "socialgood_36_12.yaml" 36 12 36 m
run_case "SocialGood/SocialGood.csv" "socialgood_36_18.yaml" 36 18 36 m

run_case "Health_US/Health_US.csv" "health_96_12.yaml" 96 12 36 w
run_case "Health_US/Health_US.csv" "health_96_24.yaml" 96 24 36 w
run_case "Health_US/Health_US.csv" "health_96_48.yaml" 96 48 36 w

run_case "Environment/Environment.csv" "environment_336_48.yaml" 336 48 36 d
run_case "Environment/Environment.csv" "environment_336_96.yaml" 336 96 36 d
run_case "Environment/Environment.csv" "environment_336_192.yaml" 336 192 36 d

run_case "Energy/Energy.csv" "energy_96_12.yaml" 96 12 36 w
run_case "Energy/Energy.csv" "energy_96_24.yaml" 96 24 36 w
run_case "Energy/Energy.csv" "energy_96_48.yaml" 96 48 36 w

run_case "Economy/Economy.csv" "economy_36_6.yaml" 36 6 36 m
run_case "Economy/Economy.csv" "economy_36_12.yaml" 36 12 36 m
run_case "Economy/Economy.csv" "economy_36_18.yaml" 36 18 36 m

run_case "Climate/Climate.csv" "climate_96_12.yaml" 96 12 36 w
run_case "Climate/Climate.csv" "climate_96_24.yaml" 96 24 36 w
run_case "Climate/Climate.csv" "climate_96_48.yaml" 96 48 36 w

run_case "Agriculture/Agriculture.csv" "agriculture_36_6.yaml" 36 6 36 m
run_case "Agriculture/Agriculture.csv" "agriculture_36_12.yaml" 36 12 36 m
run_case "Agriculture/Agriculture.csv" "agriculture_36_18.yaml" 36 18 36 m
