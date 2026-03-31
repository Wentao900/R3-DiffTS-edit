import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_CSDI
from utils.prepare4llm import get_llm

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=25):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class CSDI_series_decomp(nn.Module):
    def __init__(self, lookback_len, pred_len, kernel_size=25):
        super(CSDI_series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.lookback_len = lookback_len
        self.pred_len = pred_len

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lookback = x[:, :self.lookback_len, :]

        moving_mean = self.moving_avg(lookback)
        res = lookback - moving_mean
        
        moving_mean = moving_mean.permute(0, 2, 1)
        res = res.permute(0, 2, 1)

        moving_mean = nn.functional.pad(moving_mean, (0, self.pred_len), "constant", 0)
        res = nn.functional.pad(res, (0, self.pred_len), "constant", 0)
        return res, moving_mean


class ScaleRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    

class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, window_lens):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.timestep_branch = config["model"]["timestep_branch"]
        self.timestep_emb_cat = config["model"]["timestep_emb_cat"]
        self.with_texts = config["model"]["with_texts"]
        self.noise_esti = config["diffusion"]["noise_esti"]
        self.relative_size_emb_cat = config["model"]["relative_size_emb_cat"]
        self.decomp = config["model"]["decomp"]
        self.ddim = config["diffusion"]["ddim"]
        self.sample_steps = config["diffusion"]["sample_steps"]
        self.sample_method = config["diffusion"]["sample_method"]

        self.lookback_len = config["model"]["lookback_len"]
        self.pred_len = config["model"]["pred_len"]
        self.diff_channels = config["diffusion"]["channels"]
        self.cfg = config["diffusion"]["cfg"]
        self.c_mask_prob = config["diffusion"]["c_mask_prob"]
        self.context_dim = config["model"]["context_dim"]
        self.llm = config["model"]["llm"]
        self.domain = config["model"]["domain"]
        self.save_attn = config["model"]["save_attn"]
        self.save_token = config["model"]["save_token"]
        self.refine_init_mode = config["model"].get("refine_init_mode", "none")
        self.refine_init_noise_scale = float(config["model"].get("refine_init_noise_scale", 1.0))
        self.trend_cfg = config["diffusion"].get("trend_cfg", False)
        self.trend_cfg_power = config["diffusion"].get("trend_cfg_power", 1.0)
        self.trend_strength_scale = config["diffusion"].get("trend_strength_scale", 0.35)
        self.trend_volatility_scale = config["diffusion"].get("trend_volatility_scale", 1.0)
        self.trend_time_floor = config["diffusion"].get("trend_time_floor", 0.30)
        self.trend_cfg_random = config["diffusion"].get("trend_cfg_random", False)

        train_cfg = config.get("train", {})
        self.multi_res_band_boundaries = [int(x) for x in train_cfg.get("multi_res_band_boundaries", [])]
        self.multi_res_loss_weight = float(train_cfg.get("multi_res_loss_weight", 0.0))
        self.multi_res_use_huber = bool(train_cfg.get("multi_res_use_huber", True))
        self.multi_res_huber_delta = float(train_cfg.get("multi_res_huber_delta", 1.0))
        self.use_scale_router = bool(train_cfg.get("use_scale_router", False))
        self.scale_router_entropy_weight = float(train_cfg.get("scale_router_entropy_weight", 1.0e-3))
        self.scale_router_teacher_weight = float(train_cfg.get("scale_router_teacher_weight", 0.1))
        self.scale_router_warmup_steps = int(train_cfg.get("scale_router_warmup_steps", 400))
        self.router_step = 0

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1 
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
            
        if self.decomp:
            self.decomposition = CSDI_series_decomp(self.lookback_len, self.pred_len, kernel_size=25)
        
        if self.timestep_emb_cat:
            self.timestep_emb = nn.Sequential(nn.Linear(config["model"]["timestep_dim"], self.diff_channels//8), 
                                      nn.LayerNorm(self.diff_channels//8),
                                      nn.ReLU(),
                                      nn.Linear(self.diff_channels//8, self.diff_channels//4), 
                                      nn.LayerNorm(self.diff_channels//4),
                                      nn.ReLU())
        
        if self.relative_size_emb_cat:
            self.relative_size_emb = nn.Sequential(nn.Linear(self.lookback_len, self.lookback_len), 
                                                   nn.LayerNorm(self.lookback_len),
                                                   nn.ReLU(),
                                                   nn.Linear(self.lookback_len, self.diff_channels),
                                                   nn.LayerNorm(self.diff_channels),
                                                   nn.ReLU(),)

        if self.with_texts:
            self.text_encoder, self.tokenizer = get_llm(self.llm, config["model"]["llm_layers"])
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            if self.llm != 'bert':
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    pad_token = '[PAD]'
                    self.tokenizer.add_special_tokens({'pad_token': pad_token})
                    self.tokenizer.pad_token = pad_token

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["decomp"] = self.decomp
        config_diff["lookback_len"] = self.lookback_len
        config_diff["pred_len"] = self.pred_len
        config_diff["with_timestep"] = True if self.timestep_emb_cat else False
        config_diff["context_dim"] = self.context_dim
        config_diff["with_texts"] = self.with_texts
        config_diff["time_weight"] = config["diffusion"]["time_weight"]
        config_diff["save_attn"] = config["model"]["save_attn"]

        input_dim = 1 if self.is_unconditional == True else 2
        mode_num = 1

        if self.decomp:
            self.diffmodel_trend = diff_CSDI(config_diff, input_dim, mode_num=mode_num)
            self.diffmodel_sesonal = diff_CSDI(config_diff, input_dim, mode_num=mode_num)
        else:
            self.diffmodel = diff_CSDI(config_diff, input_dim, mode_num=mode_num)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        self.band_slices = self._build_band_slices(self.multi_res_band_boundaries, self.pred_len)
        self.scale_router = None
        if self.use_scale_router and len(self.band_slices) > 0:
            hidden_dim = int(train_cfg.get("scale_router_hidden_dim", 32))
            dropout = float(train_cfg.get("scale_router_dropout", 0.1))
            self.scale_router = ScaleRouter(input_dim=8, hidden_dim=hidden_dim, output_dim=len(self.band_slices), dropout=dropout)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else: 
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) 
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K, emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        side_info = side_info.permute(0, 3, 2, 1) 

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1) 
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def _build_band_slices(self, boundaries, pred_len):
        if not boundaries:
            return []
        valid_boundaries = sorted({min(max(int(boundary), 1), pred_len) for boundary in boundaries})
        if valid_boundaries[-1] != pred_len:
            valid_boundaries.append(pred_len)
        slices = []
        start = 0
        for end in valid_boundaries:
            if end > start:
                slices.append((start, end))
                start = end
        return slices

    def prepare_trend_prior(self, trend_prior):
        if trend_prior is None:
            return None
        if self.trend_cfg_random:
            direction = torch.randint(-1, 2, (trend_prior.shape[0], 1), device=trend_prior.device).float()
            strength = torch.empty((trend_prior.shape[0], 1), device=trend_prior.device).uniform_(0.5, 1.5)
            volatility = torch.empty((trend_prior.shape[0], 1), device=trend_prior.device).uniform_(0.0, 1.0)
            return torch.cat([direction, strength, volatility], dim=1)
        return trend_prior

    def get_effective_guide_weight(self, base_guide_w, trend_prior, step_idx, batch_size):
        guide = torch.full((batch_size,), float(base_guide_w), device=self.device)
        if (not self.trend_cfg) or trend_prior is None:
            return guide
        trend_prior = self.prepare_trend_prior(trend_prior)
        total_steps = max(self.sample_steps - 1, 1)
        progress = 1.0 - (float(step_idx) / total_steps)
        time_scale = self.trend_time_floor + (1.0 - self.trend_time_floor) * (progress ** self.trend_cfg_power)
        strength_scale = 1.0 + self.trend_strength_scale * (trend_prior[:, 1] - 1.0)
        volatility_scale = 1.0 / (1.0 + self.trend_volatility_scale * trend_prior[:, 2])
        return guide * time_scale * strength_scale * volatility_scale

    def build_router_features(self, history, text_mask=None, trend_prior=None):
        slope = history[:, :, -1] - history[:, :, 0]
        volatility = history.std(dim=-1)
        if history.shape[-1] >= 4:
            mid = history.shape[-1] // 2
            left_slope = history[:, :, mid - 1] - history[:, :, 0]
            right_slope = history[:, :, -1] - history[:, :, mid]
            acceleration = right_slope - left_slope
        else:
            acceleration = torch.zeros_like(slope)

        stats = torch.stack(
            [
                slope.mean(dim=1),
                slope.abs().mean(dim=1),
                volatility.mean(dim=1),
                acceleration.mean(dim=1),
            ],
            dim=1,
        )
        if trend_prior is None:
            trend_prior = torch.zeros((history.shape[0], 3), device=history.device)
        if text_mask is None:
            text_mask = torch.zeros((history.shape[0],), device=history.device)
        return torch.cat([stats, trend_prior.float(), text_mask.float().unsqueeze(1)], dim=1)

    def compute_multi_res_loss(self, predicted, target, target_mask, history, text_mask=None, trend_prior=None, is_train=1):
        future_pred = predicted[:, :, self.lookback_len:]
        future_target = target[:, :, self.lookback_len:]
        future_mask = target_mask[:, :, self.lookback_len:]

        band_losses = []
        for start, end in self.band_slices:
            band_pred = future_pred[:, :, start:end]
            band_target = future_target[:, :, start:end]
            band_mask = future_mask[:, :, start:end]
            if self.multi_res_use_huber:
                band_error = F.huber_loss(band_pred, band_target, delta=self.multi_res_huber_delta, reduction="none")
            else:
                band_error = (band_pred - band_target) ** 2
            denom = band_mask.sum(dim=(1, 2)).clamp_min(1.0)
            band_loss = (band_error * band_mask).sum(dim=(1, 2)) / denom
            band_losses.append(band_loss)

        band_losses = torch.stack(band_losses, dim=1)
        weights = torch.full_like(band_losses, 1.0 / band_losses.shape[1])
        regularizer = torch.tensor(0.0, device=predicted.device)

        if self.scale_router is not None:
            features = self.build_router_features(history, text_mask=text_mask, trend_prior=trend_prior)
            logits = self.scale_router(features)
            weights = torch.softmax(logits, dim=-1)
            if is_train == 1 and self.router_step < self.scale_router_warmup_steps and self.scale_router_teacher_weight > 0:
                teacher = torch.softmax(band_losses.detach(), dim=-1)
                weights = (1.0 - self.scale_router_teacher_weight) * weights + self.scale_router_teacher_weight * teacher
            entropy = -(weights * torch.log(weights.clamp_min(1.0e-8))).sum(dim=-1).mean()
            regularizer = -self.scale_router_entropy_weight * entropy

        return (weights * band_losses).sum(dim=-1).mean() + regularizer

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, context=None, text_mask=None, trend_prior=None
    ):
        loss_sum = 0
        for t in range(self.num_steps): 
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, text_mask=text_mask, trend_prior=trend_prior
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, context=None, text_mask=None, trend_prior=None, set_t=-1
    ):  
        
        B, K, L = observed_data.shape
        if not self.noise_esti:
            means = torch.sum(observed_data*cond_mask, dim=2, keepdim=True) / torch.sum(cond_mask, dim=2, keepdim=True)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / (torch.sum(cond_mask, dim=2, keepdim=True) - 1) + 1e-5)
            observed_data = (observed_data - means) / stdev

        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device) 
        current_alpha = self.alpha_torch[t]  
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask) 

        if self.cfg:
            cfg_mask = torch.bernoulli(torch.ones((B, )) - self.c_mask_prob).to(self.device) 
        else:
            cfg_mask = None

        if self.decomp:
            predicted_seasonal, _ = self.diffmodel_sesonal(total_input[0], side_info, t, cfg_mask, timestep_emb, size_emb)
            predicted_trend = self.diffmodel_trend(total_input[1], side_info, t, cfg_mask, timestep_emb, size_emb)
            predicted, _ = predicted_seasonal + predicted_trend
        else:
            if self.save_attn:
                predicted, _ = self.diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context) 
            else:
                predicted = self.diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context) 

        if self.timestep_branch and timesteps is not None:
            predicted_from_timestep = self.timestep_pred(timesteps)
            predicted = 0.9 * predicted + 0.1 * predicted_from_timestep

        target_mask = observed_mask - cond_mask
        if self.noise_esti:
            residual = (noise - predicted) * target_mask 
        else:
            residual = (observed_data - predicted) * target_mask 
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        if (not self.noise_esti) and self.multi_res_loss_weight > 0 and len(self.band_slices) > 0:
            aux_loss = self.compute_multi_res_loss(
                predicted=predicted,
                target=observed_data,
                target_mask=target_mask,
                history=observed_data[:, :, :self.lookback_len],
                text_mask=text_mask,
                trend_prior=trend_prior,
                is_train=is_train,
            )
            loss = loss + self.multi_res_loss_weight * aux_loss
            if is_train == 1:
                self.router_step += 1
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  
        else:
            cond_obs = cond_mask * observed_data
            noisy_target = noisy_data.unsqueeze(1) 
            if self.decomp:
                res, moving_mean = self.decomposition(cond_obs) 
                res, moving_mean = res.unsqueeze(1), moving_mean.unsqueeze(1) 
                res_input = torch.cat([res, noisy_target], dim=1)  
                moving_mean_input = torch.cat([moving_mean, noisy_target], dim=1) 
                total_input = [res_input, moving_mean_input]
            else:
                cond_obs = cond_obs.unsqueeze(1) 
                total_input = torch.cat([cond_obs, noisy_target], dim=1) 

        return total_input

    def build_refinement_base(self, observed_data, cond_mask):
        if self.refine_init_mode == "none":
            return None

        history = observed_data[:, :, :self.lookback_len]
        if self.refine_init_mode == "persistence":
            future_base = history[:, :, -1:].repeat(1, 1, self.pred_len)
        elif self.refine_init_mode == "linear":
            last_value = history[:, :, -1:]
            slope = (history[:, :, -1:] - history[:, :, :1]) / max(self.lookback_len - 1, 1)
            future_steps = torch.arange(1, self.pred_len + 1, device=observed_data.device, dtype=observed_data.dtype).view(1, 1, -1)
            future_base = last_value + slope * future_steps
        else:
            return None

        full_base = observed_data.clone()
        full_base[:, :, self.lookback_len:] = future_base
        return cond_mask * observed_data + (1.0 - cond_mask) * full_base

    def impute(self, observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=None, timestep_emb=None, size_emb=None, context=None, trend_prior=None):
        B, K, L = observed_data.shape
        if self.ddim:
            if self.sample_method == 'linear':
                a = self.num_steps // self.sample_steps
                time_steps = np.asarray(list(range(0, self.num_steps, a)))
            elif self.sample_method == "quad":
                time_steps = (np.linspace(0, np.sqrt(self.num_steps * 0.8), self.sample_steps) ** 2).astype(np.int)
            else:
                raise NotImplementedError(f"sampling method {self.sample_method} is not implemented!")
            time_steps = time_steps + 1
            time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        else:
            self.sample_steps = self.num_steps
        if not self.noise_esti:
            means = torch.sum(observed_data*cond_mask, dim=2, keepdim=True) / torch.sum(cond_mask, dim=2, keepdim=True)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / (torch.sum(cond_mask, dim=2, keepdim=True) - 1) + 1e-5)
            observed_data = (observed_data - means) / stdev
        
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        if self.cfg:
            side_info = side_info.repeat(2, 1, 1, 1)
            if timestep_emb is not None:
                timestep_emb = timestep_emb.repeat(2, 1, 1, 1)
            if context is not None:
                context = context.repeat(2, 1, 1)
            cfg_mask = torch.zeros((2*B, )).to(self.device) 
            cfg_mask[:B] = 1.
        else:
            cfg_mask = None

        for i in range(n_samples):
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            base_sample = self.build_refinement_base(observed_data, cond_mask)
            if base_sample is None:
                current_sample = torch.randn_like(observed_data)
            else:
                init_noise = torch.randn_like(observed_data) * self.refine_init_noise_scale
                current_sample = base_sample + (1.0 - cond_mask) * init_noise
            for t in range(self.sample_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1) 
                else:
                    if self.decomp:
                        cond_obs = cond_mask * observed_data
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1) # (B, 1, K, L)
                        res, moving_mean = self.decomposition(cond_obs) # (B, K, L), (B, K, L)
                        res, moving_mean = res.unsqueeze(1), moving_mean.unsqueeze(1) # (B, 1, K, L), (B, 1, K, L)
                        res_input = torch.cat([res, noisy_target], dim=1)  # (B,2,K,L)
                        moving_mean_input = torch.cat([moving_mean, noisy_target], dim=1)  # (B,2,K,L)
                        if self.cfg:
                            res_input = res_input.repeat(2, 1, 1, 1) # (2*B, 2, K, L)
                            moving_mean_input = moving_mean_input.repeat(2, 1, 1, 1) # (2*B, 2, K, L)
                        predicted_seasonal = self.diffmodel_sesonal(res_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)
                        predicted_trend = self.diffmodel_trend(moving_mean_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)
                        predicted = predicted_seasonal + predicted_trend # (2*B, K, L)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1) # (B, 1, K, L)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1) # (B, 1, K, L)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
                        if self.cfg:
                            diff_input = diff_input.repeat(2, 1, 1, 1) # (2*B, 2, K, L)
                        if self.save_attn:
                            predicted, attn = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)
                        else:
                            predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), cfg_mask, timestep_emb, size_emb, context) # (2*B, K, L)
                if self.cfg:
                    predicted_cond, predicted_uncond = predicted[:B], predicted[B:]
                    effective_guide = self.get_effective_guide_weight(guide_w, trend_prior, t, B)
                    predicted = predicted_uncond + effective_guide[:, None, None] * (predicted_cond - predicted_uncond)

                if self.noise_esti:
                    # noise prediction
                    if not self.ddim:
                        coeff1 = 1 / self.alpha_hat[t] ** 0.5
                        coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                        current_sample = coeff1 * (current_sample - coeff2 * predicted) # (B, K, L)
                        if t > 0:
                            noise = torch.randn_like(current_sample)
                            sigma = (
                                (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                            current_sample += sigma * noise
                    else:
                        tau, tau_prev = time_steps[t], time_steps_prev[t]
                        current_sample = (
                            torch.sqrt(self.alpha[tau_prev] / self.alpha[tau]) * current_sample +
                            (torch.sqrt(1 - self.alpha[tau_prev]) - torch.sqrt(
                                (self.alpha[tau_prev] * (1 - self.alpha[tau])) / self.alpha[tau])) * predicted
                        )
                else:
                    if not self.ddim:
                        if t > 1:
                            # data prediction
                            coeff1 = (self.alpha_hat[t] ** 0.5 * (1 - self.alpha[t-1])) / (1 - self.alpha[t])
                            coeff2 = (self.alpha[t-1] ** 0.5 * self.beta[t]) / (1 - self.alpha[t])
                            current_sample = coeff1 * current_sample + coeff2 * predicted # (B, K, L)
                            
                            if t > 2:
                                noise = torch.randn_like(current_sample)
                                sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                                ) ** 0.5
                                current_sample += sigma * noise
                    else:
                        tau, tau_prev = time_steps[t], time_steps_prev[t]
                        aaa_ = (1-self.alpha[tau_prev])/(1-self.alpha[tau]) ** 0.5
                        current_sample = (
                            aaa_ * current_sample +
                            ((self.alpha[tau_prev])**0.5 - (self.alpha[tau])**0.5 * aaa_) * predicted
                        )

            imputed_samples[:, i] = current_sample.detach()
            if self.timestep_branch and timesteps is not None:
                predicted_from_timestep = self.timestep_pred(timesteps)
                imputed_samples[:, i] = 0.9 * imputed_samples[:, i] + 0.1 * predicted_from_timestep.detach()
            if not self.noise_esti:
                imputed_samples[:, i] = imputed_samples[:, i] * stdev + means
        if self.save_attn:
            return imputed_samples, attn 
        else:
            return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)): 
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim, window_lens):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device, window_lens)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]
        

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        text_mask = batch["text_mark"].to(self.device).int()
        trend_prior = batch["trend_prior"].to(self.device).float() if "trend_prior" in batch else None
        if self.timestep_emb_cat or self.timestep_branch:
            timesteps = batch["timesteps"].to(self.device).float()
            timesteps = timesteps.permute(0, 2, 1)
        else:
            timesteps = None
        texts = batch["texts"] if self.with_texts else None

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data, 
            observed_mask,
            observed_tp, 
            gt_mask,
            for_pattern_mask, 
            cut_length,
            feature_id,
            timesteps, 
            texts,
            text_mask, 
            trend_prior,
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask
    
    def get_timestep_info(self, timesteps):
        timestep_emb = self.timestep_emb(timesteps.transpose(1, 2)).transpose(1, 2)
        timestep_emb = timestep_emb.unsqueeze(2).expand(-1, -1, self.target_dim, -1) 
        return timestep_emb
    
    def get_relative_size_info(self, observed_data):
        B, K, L = observed_data.shape

        size_emb = observed_data[:, :, :self.lookback_len].clone().unsqueeze(3).expand(-1, -1, -1, self.lookback_len) - \
            observed_data[:, :, :self.lookback_len].clone().unsqueeze(2).expand(-1, -1, self.lookback_len, -1) 
        size_emb = self.relative_size_emb(size_emb)
        size_emb = size_emb.permute(0, 3, 1, 2)
        size_emb = torch.cat([size_emb, torch.zeros((B, self.diff_channels, K, self.pred_len)).to(observed_data.device)], dim=-1) 
        return size_emb
    
    def get_text_info(self, text, text_mask):
        token_input = self.tokenizer(text,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors='pt',
                                     ).to(self.device)
        context = self.text_encoder(**token_input).last_hidden_state
        context = context * text_mask.unsqueeze(1).unsqueeze(1)
        context = context.permute(0, 2, 1) 
        if self.save_token:
            tokens_str = self.tokenizer.batch_decode(token_input['input_ids'])
            return context, tokens_str
        else:
            return context

    def get_side_info(self, observed_tp, cond_mask, feature_id=None, timesteps=None, texts=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) 
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1) 

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            ) 
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else: 
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1) 

        side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        side_info = side_info.permute(0, 3, 2, 1) 

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1) 
            side_info = torch.cat([side_info, side_mask], dim=1) 
    

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask, 
            observed_tp,
            gt_mask, 
            _,
            _,
            feature_id,
            timesteps, 
            texts,
            text_mask, 
            trend_prior,
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id, timesteps, texts)

        if self.timestep_emb_cat:
            timestep_emb = self.get_timestep_info(timesteps)
        else:
            timestep_emb = None

        if self.relative_size_emb_cat:
            size_emb = self.get_relative_size_info(observed_data)
        else:
            size_emb = None

        if self.with_texts:
            if self.save_token:
                context, _ = self.get_text_info(texts, text_mask)
            else:
                context = self.get_text_info(texts, text_mask)
        else:
            context = None

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(
            observed_data,
            cond_mask,
            observed_mask,
            side_info,
            is_train,
            timesteps=timesteps,
            timestep_emb=timestep_emb,
            size_emb=size_emb,
            context=context,
            text_mask=text_mask,
            trend_prior=trend_prior,
        )

    def evaluate(self, batch, n_samples, guide_w):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
            timesteps,
            texts,
            text_mask,
            trend_prior,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask, timesteps=timesteps, texts=texts)

            if self.timestep_emb_cat:
                timestep_emb = self.get_timestep_info(timesteps)
            else:
                timestep_emb = None

            if self.relative_size_emb_cat:
                size_emb = self.get_relative_size_info(observed_data)
            else:
                size_emb = None

            if self.with_texts:
                if self.save_token:
                    context, tokens = self.get_text_info(texts, text_mask)
                else:
                    context = self.get_text_info(texts, text_mask)
            else:
                context = None
            if self.save_attn:
                samples, attn = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, trend_prior=trend_prior)
            else:
                samples = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, trend_prior=trend_prior)

        if self.save_attn:
            if self.save_token:
                return samples, observed_data, target_mask, observed_mask, observed_tp, attn, tokens
            else:
                return samples, observed_data, target_mask, observed_mask, observed_tp, attn
        else:
            return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
