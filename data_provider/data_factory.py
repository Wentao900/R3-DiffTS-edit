from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
import torch
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = False if flag == 'test' else True
    batch_size = args.batch_size
    freq = args.freq
    extra_kwargs = {}
    if args.data == "custom":
        extra_kwargs = {
            "use_rag_cot": getattr(args, "use_rag_cot", False),
            "rag_topk": getattr(args, "rag_topk", 3),
            "cot_model": getattr(args, "cot_model", None),
            "cot_max_new_tokens": getattr(args, "cot_max_new_tokens", 96),
            "cot_temperature": getattr(args, "cot_temperature", 0.7),
            "cot_cache_size": getattr(args, "cot_cache_size", 512),
            "cot_device": getattr(args, "cot_device", None),
            "rag_only": getattr(args, "rag_only", False),
            "rag_use_retrieval": not getattr(args, "cot_only", False),
            "use_two_stage_rag": getattr(args, "use_two_stage_rag", False),
            "rag_stage1_topk": getattr(args, "rag_stage1_topk", 6),
            "rag_stage2_topk": getattr(args, "rag_stage2_topk", 3),
            "two_stage_gate": getattr(args, "two_stage_gate", True),
            "trend_slope_eps": getattr(args, "trend_slope_eps", 1.0e-3),
        }
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        text_len=args.text_len,
        **extra_kwargs,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
