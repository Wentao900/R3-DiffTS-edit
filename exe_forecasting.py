import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import random

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils.utils import train, evaluate

parser = argparse.ArgumentParser(description="MCD-TSF")
parser.add_argument("--config", type=str, default="economy_36_18.yaml")
parser.add_argument("--datatype", type=str, default="multimodal")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=15)
parser.add_argument("--data", type=str, default="custom")
parser.add_argument("--embed", type=str, default="timeF")
parser.add_argument('--root_path', type=str, default='Time-MMD-main', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Economy/Economy.csv', help='data file')
parser.add_argument('--seq_len', type=int, default=36, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=18, help='prediction sequence length')
parser.add_argument('--text_len', type=int, default=36, help='context length in time series freq')
parser.add_argument('--use_rag_cot', action='store_true', help='enable retrieval + CoT text augmentation')
parser.add_argument('--rag_only', action='store_true', help='enable retrieval evidence only, without CoT reasoning text')
parser.add_argument('--cot_only', action='store_true', help='use CoT text augmentation without retrieval evidence')
parser.add_argument('--rag_topk', type=int, default=3, help='number of retrieved evidence snippets')
parser.add_argument('--use_two_stage_rag', action='store_true', help='enable two-stage retrieval for RAG')
parser.add_argument('--rag_stage1_topk', type=int, default=6, help='top-k in stage-1 retrieval')
parser.add_argument('--rag_stage2_topk', type=int, default=3, help='top-k in stage-2 retrieval')
parser.add_argument('--two_stage_gate', action='store_true', default=True, help='enable safe fallback for two-stage retrieval')
parser.add_argument('--trend_slope_eps', type=float, default=1.0e-3, help='trend slope threshold used in retrieval/trend fallback')
parser.add_argument('--cot_model', type=str, default=None, help='optional local text-generation model for CoT')
parser.add_argument('--cot_max_new_tokens', type=int, default=96, help='max generated CoT tokens')
parser.add_argument('--cot_temperature', type=float, default=0.7, help='CoT generation temperature')
parser.add_argument('--cot_cache_size', type=int, default=512, help='cache size for generated CoT text')
parser.add_argument('--cot_device', type=str, default=None, help='device for CoT generation, e.g. cpu or cuda:0')
parser.add_argument('--guide_w', type=float, default=-1, help='fixed CFG weight; negative keeps the original sweep behavior')
parser.add_argument('--trend_cfg', action='store_true', help='enable trend-aware guidance modulation during sampling')
parser.add_argument('--trend_cfg_power', type=float, default=1.0, help='time schedule exponent for trend-aware CFG')
parser.add_argument('--trend_strength_scale', type=float, default=0.35, help='trend strength gain scale')
parser.add_argument('--trend_volatility_scale', type=float, default=1.0, help='trend volatility suppression scale')
parser.add_argument('--trend_time_floor', type=float, default=0.30, help='minimum time schedule floor for trend-aware CFG')
parser.add_argument('--trend_cfg_random', action='store_true', help='replace parsed trend prior with random trend vectors')
parser.add_argument('--use_min_snr', action='store_true', help='enable Min-SNR weighting for diffusion training')
parser.add_argument('--min_snr_gamma', type=float, default=5.0, help='gamma for Min-SNR timestep weighting')
parser.add_argument('--multi_res_band_boundaries', nargs='*', type=int, default=None, help='prediction band boundaries for multi-resolution loss')
parser.add_argument('--multi_res_loss_weight', type=float, default=-1, help='weight for multi-resolution auxiliary loss')
parser.add_argument('--use_scale_router', action='store_true', help='enable scale router for band weighting')
parser.add_argument('--scale_router_hidden_dim', type=int, default=-1, help='hidden size of scale router')
parser.add_argument('--scale_router_entropy_weight', type=float, default=-1, help='entropy regularizer for scale router')
parser.add_argument('--scale_router_teacher_weight', type=float, default=-1, help='teacher warmup mix weight for scale router')
parser.add_argument('--scale_router_warmup_steps', type=int, default=-1, help='warmup steps for scale router teacher mixing')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--freq', type=str, default='m', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--attn_drop', type=float, default=0.)
parser.add_argument('--init', type=str, default='None')
parser.add_argument('--valid_interval', type=int, default=1)
parser.add_argument('--time_weight', type=float, default=0.1)
parser.add_argument('--c_mask_prob', type=float, default=-1)
parser.add_argument('--beta_end', type=float, default=-1)
parser.add_argument('--lr', type=float, default=-1)
parser.add_argument('--save_attn', type=bool, default=False)
parser.add_argument('--save_token', type=bool, default=False)


args = parser.parse_args()
print(args)

if args.cot_only:
    args.use_rag_cot = True
if args.rag_only:
    args.use_rag_cot = True

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.text_len == 0:
    args.text_len = args.seq_len

timestep_dim_dict = {
    'd': 3,
    'w': 2,
    'm': 1
}
context_dim_dict = {
    'bert': 768,
    'llama': 4096,
    'gpt2': 768
}
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)
eval_cfg = config.setdefault("eval", {})
guide_w_values = eval_cfg.get(
    "guide_w_values",
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0],
)
if args.guide_w < 0:
    args.guide_w = eval_cfg.get("guide_w", args.guide_w)
if args.embed == 'timeF':
    if config["model"]["timestep_branch"] or config["model"]["timestep_emb_cat"]:
        config["model"]["timestep_dim"] = timestep_dim_dict[args.freq]
    else:
        config["model"]["timestep_dim"] = 0
else:
    config["model"]["timestep_dim"] = 4
config["model"]["context_dim"] = context_dim_dict[config["model"]["llm"]] if config["model"]["with_texts"] else 0
config["model"]["use_rag_cot"] = args.use_rag_cot
config["model"]["rag_only"] = args.rag_only
config["model"]["cot_only"] = args.cot_only
config["model"]["rag_topk"] = args.rag_topk
config["model"]["use_two_stage_rag"] = args.use_two_stage_rag
config["model"]["rag_stage1_topk"] = args.rag_stage1_topk
config["model"]["rag_stage2_topk"] = args.rag_stage2_topk
config["model"]["two_stage_gate"] = args.two_stage_gate
config["model"]["trend_slope_eps"] = args.trend_slope_eps
config["model"]["cot_model"] = args.cot_model
config["model"]["cot_max_new_tokens"] = args.cot_max_new_tokens
config["model"]["cot_temperature"] = args.cot_temperature
config["model"]["cot_cache_size"] = args.cot_cache_size
config["model"]["cot_device"] = args.cot_device
config["diffusion"]["trend_cfg"] = args.trend_cfg
config["diffusion"]["trend_cfg_power"] = args.trend_cfg_power
config["diffusion"]["trend_strength_scale"] = args.trend_strength_scale
config["diffusion"]["trend_volatility_scale"] = args.trend_volatility_scale
config["diffusion"]["trend_time_floor"] = args.trend_time_floor
config["diffusion"]["trend_cfg_random"] = args.trend_cfg_random
config["diffusion"]["use_min_snr"] = args.use_min_snr
config["diffusion"]["min_snr_gamma"] = args.min_snr_gamma
config["train"].setdefault("multi_res_band_boundaries", [])
config["train"].setdefault("multi_res_loss_weight", 0.0)
config["train"].setdefault("multi_res_use_huber", True)
config["train"].setdefault("multi_res_huber_delta", 1.0)
config["train"].setdefault("use_scale_router", False)
config["train"].setdefault("scale_router_hidden_dim", 32)
config["train"].setdefault("scale_router_dropout", 0.1)
config["train"].setdefault("scale_router_entropy_weight", 1.0e-3)
config["train"].setdefault("scale_router_teacher_weight", 0.1)
config["train"].setdefault("scale_router_warmup_steps", 400)
config["eval"]["guide_w"] = args.guide_w
config["eval"]["guide_w_values"] = guide_w_values
if args.multi_res_band_boundaries is not None:
    config["train"]["multi_res_band_boundaries"] = args.multi_res_band_boundaries
if args.multi_res_loss_weight >= 0:
    config["train"]["multi_res_loss_weight"] = args.multi_res_loss_weight
if args.use_scale_router:
    config["train"]["use_scale_router"] = True
if args.scale_router_hidden_dim > 0:
    config["train"]["scale_router_hidden_dim"] = args.scale_router_hidden_dim
if args.scale_router_entropy_weight >= 0:
    config["train"]["scale_router_entropy_weight"] = args.scale_router_entropy_weight
if args.scale_router_teacher_weight >= 0:
    config["train"]["scale_router_teacher_weight"] = args.scale_router_teacher_weight
if args.scale_router_warmup_steps >= 0:
    config["train"]["scale_router_warmup_steps"] = args.scale_router_warmup_steps

if args.datatype == 'electricity':
    target_dim = 370
    args.seq_len = 168
    args.pred_len = 24
else:
    target_dim = 1

config["model"]["is_unconditional"] = args.unconditional
config["model"]["lookback_len"] = args.seq_len
config["model"]["pred_len"] = args.pred_len
config["model"]["domain"] = args.data_path.split('/')[0]
config["model"]["text_len"] = args.text_len
config["model"]["save_attn"] = args.save_attn
config["model"]["save_token"] = args.save_token
config["diffusion"]["dropout"] = args.dropout
config["diffusion"]["attn_drop"] = args.attn_drop
config["diffusion"]["time_weight"] = args.time_weight

if args.c_mask_prob > 0:
    config["diffusion"]["c_mask_prob"] = args.c_mask_prob

if args.beta_end > 0:
    config["diffusion"]["beta_end"] = args.beta_end

if args.lr > 0:
    config["train"]["lr"] = args.lr

args.batch_size = config["train"]["batch_size"]

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.data_path.split('/')[0] + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config_results.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.datatype,
    device= args.device,
    batch_size=config["train"]["batch_size"],
    args=args
)

model = CSDI_Forecasting(config, args.device, target_dim, window_lens=[args.seq_len, args.pred_len]).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        valid_epoch_interval=args.valid_interval
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
model.target_dim = target_dim
if config["diffusion"]["cfg"]:
    if args.guide_w >= 0:
        evaluate(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            window_lens=[args.seq_len, args.pred_len],
            guide_w=args.guide_w,
            save_attn=args.save_attn,
            save_token=args.save_token
        )
    else:
        best_mse = 10e10
        for guide_w in guide_w_values:
            mse = evaluate(
                model,
                test_loader,
                nsample=args.nsample,
                scaler=scaler,
                mean_scaler=mean_scaler,
                foldername=foldername,
                window_lens=[args.seq_len, args.pred_len],
                guide_w=guide_w,
                save_attn=args.save_attn,
                save_token=args.save_token
            )
            best_mse = min(best_mse, mse)
else:
    evaluate(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            window_lens=[args.seq_len, args.pred_len],
            save_attn=args.save_attn,
            save_token=args.save_token
        )
