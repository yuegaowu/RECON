import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.base_model.base_model import BaseModel


def _linear(in_features: int, out_features: int, init_scale: float = 1.0) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_normal_(layer.weight, gain=math.sqrt(init_scale))
    nn.init.zeros_(layer.bias)
    return layer


def _masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.masked_fill(~mask.unsqueeze(-1), -1e4)
    out = x.max(dim=dim).values
    valid = mask.any(dim=dim, keepdim=True)
    return torch.where(valid, out, torch.zeros_like(out))


def _sinusoidal_pe(pos: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        return torch.zeros(*pos.shape[:-1], dim, device=pos.device, dtype=pos.dtype)
    freq_dim = dim // 4
    div = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, freq_dim, device=pos.device, dtype=pos.dtype)
        / max(freq_dim, 1)
    )
    x = 2 * math.pi * pos[..., 0:1] * div
    y = 2 * math.pi * pos[..., 1:2] * div
    return torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)


class PolylineSubGraph(nn.Module):
    """VectorNet-style point/polyline encoder used by SeNeVA."""

    def __init__(self, in_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input = nn.Sequential(_linear(in_dim, hidden_size), nn.SiLU(), _linear(hidden_size, hidden_size))
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    _linear(hidden_size * 2, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    _linear(hidden_size, hidden_size),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, P, F], mask: [B, L, P]
        out = self.input(x.float())
        for layer in self.layers:
            pooled = _masked_max(out, mask, dim=2).unsqueeze(2).expand_as(out)
            out = layer(torch.cat([out, pooled], dim=-1))
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return self.norm(_masked_max(out, mask, dim=2))


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim: int, key_dim: int = None, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        key_dim = query_dim if key_dim is None else key_dim
        self.q_norm = nn.LayerNorm(query_dim)
        self.kv_norm = nn.LayerNorm(key_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=key_dim,
            vdim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(_linear(query_dim, query_dim * 4), nn.SiLU(), _linear(query_dim * 4, query_dim))
        self.ffn_norm = nn.LayerNorm(query_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        key_value = query if key_value is None else key_value
        out, _ = self.attn(
            self.q_norm(query),
            self.kv_norm(key_value),
            self.kv_norm(key_value),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + out
        return query + self.ffn(self.ffn_norm(query))


class SenevaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.motion_net = PolylineSubGraph(config.num_agent_feature, config.motion_hidden_size, config.num_subgraph_layers)
        self.map_net = PolylineSubGraph(config.num_map_feature, config.map_hidden_size, config.num_subgraph_layers)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "a2m": CrossAttentionBlock(config.map_hidden_size, config.motion_hidden_size, config.encoder_num_heads, config.dropout),
                        "m2m": CrossAttentionBlock(config.map_hidden_size, None, config.encoder_num_heads, config.dropout),
                        "m2a": CrossAttentionBlock(config.motion_hidden_size, config.map_hidden_size, config.encoder_num_heads, config.dropout),
                        "a2a": CrossAttentionBlock(config.motion_hidden_size, None, config.encoder_num_heads, config.dropout),
                    }
                )
                for _ in range(config.num_global_layers)
            ]
        )
        self.out_feature = config.motion_hidden_size

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        obj = inputs["obj_trajs"]
        obj_mask = inputs["obj_trajs_mask"].bool()
        roads = inputs["map_polylines"]
        road_mask = inputs["map_polylines_mask"].bool()
        track_index = inputs["track_index_to_predict"].long()

        agent_valid = obj_mask.any(dim=-1)
        map_valid = road_mask.any(dim=-1)
        # Avoid all-masked attention rows. Empty padded tokens have zero features,
        # so unmasking one placeholder token is neutral but prevents MHA NaNs.
        agent_valid[:, 0] = True
        map_valid[:, 0] = True
        agent_feat = self.motion_net(obj, obj_mask)
        map_feat = self.map_net(roads, road_mask)

        agent_pos = inputs["obj_trajs_last_pos"][..., 0:2]
        map_pos = inputs["map_polylines_center"][..., 0:2]
        agent_feat = agent_feat + _sinusoidal_pe(agent_pos, agent_feat.size(-1))
        map_feat = map_feat + _sinusoidal_pe(map_pos, map_feat.size(-1))

        # MultiheadAttention expects True for padded tokens.
        agent_pad = ~agent_valid
        map_pad = ~map_valid
        for layer in self.layers:
            map_feat = layer["a2m"](map_feat, agent_feat, key_padding_mask=agent_pad)
            map_feat = layer["m2m"](map_feat, key_padding_mask=map_pad)
            agent_feat = layer["m2a"](agent_feat, map_feat, key_padding_mask=map_pad)
            agent_feat = layer["a2a"](agent_feat, key_padding_mask=agent_pad)
            agent_feat = agent_feat.masked_fill(agent_pad.unsqueeze(-1), 0.0)
            map_feat = map_feat.masked_fill(map_pad.unsqueeze(-1), 0.0)

        bidx = torch.arange(agent_feat.size(0), device=agent_feat.device)
        return agent_feat[bidx, track_index]


class GenerativeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_components: int, horizon: int, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_components = num_components
        self.horizon = horizon
        self.output_dim = output_dim
        self.input_head = nn.Sequential(_linear(input_dim, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        self.lstms = nn.ModuleList(
            [nn.LSTM(hidden_size + input_dim, hidden_size, batch_first=True, num_layers=2) for _ in range(num_components)]
        )
        self.mean_head = _linear(hidden_size, hidden_size, init_scale=1e-10)
        self.vars_head = nn.Sequential(_linear(hidden_size, hidden_size, init_scale=1e-10), nn.Softplus())
        self.emission_mean = _linear(input_dim + hidden_size, output_dim, init_scale=1e-10)
        self.emission_vars = nn.Parameter(torch.zeros(output_dim), requires_grad=True)

    def forward_s(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (x.size(0), self.num_components, self.horizon, self.hidden_size)
        s_mean = x.new_zeros(shape)
        s_vars = x.new_ones(shape)
        init = self.input_head(x)
        for k, lstm in enumerate(self.lstms):
            s_mean[:, k, 0] = self.mean_head(init)
            s_vars[:, k, 0] = self.vars_head(init) + torch.finfo(x.dtype).eps
            inp = torch.cat([s_mean[:, k, 0], x], dim=-1)
            hx = None
            for t in range(1, self.horizon):
                out, hx = lstm(inp.unsqueeze(1), hx)
                inp_h = out.squeeze(1)
                s_mean[:, k, t] = self.mean_head(inp_h)
                s_vars[:, k, t] = self.vars_head(inp_h) + torch.finfo(x.dtype).eps
                inp = torch.cat([s_mean[:, k, t], x], dim=-1)
        return s_mean, s_vars

    def forward_y(self, s: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if s.ndim == 4:
            x_exp = x[:, None, None, :].expand(-1, s.size(1), s.size(2), -1)
        else:
            x_exp = x[:, None, :].expand(-1, s.size(1), -1)
        y_mean = self.emission_mean(torch.cat([s, x_exp], dim=-1))
        y_vars = self.emission_vars.square() + 0.5 / math.pi
        return y_mean, y_vars.view(*([1] * (y_mean.ndim - 1)), -1).expand_as(y_mean)

    def predictive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_mean, s_vars = self.forward_s(x)
        y_mean, y_vars = self.forward_y(s_mean, x)
        return y_mean.cumsum(dim=-2), y_vars.cumsum(dim=-2)


class InferenceNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, horizon: int, num_components: int, output_dim: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.num_components = num_components
        self.output_dim = output_dim
        self.smooth_lstm = nn.LSTM(input_dim + output_dim, hidden_size, batch_first=True, bidirectional=True)
        self.mean_head = _linear(2 * hidden_size, hidden_size, init_scale=1e-10)
        self.vars_head = nn.Sequential(_linear(2 * hidden_size, hidden_size, init_scale=1e-10), nn.Softplus())
        self.z_proxy = nn.Sequential(
            _linear(input_dim, 4 * hidden_size),
            nn.SiLU(),
            _linear(4 * hidden_size, 2 * hidden_size),
            nn.SiLU(),
            _linear(2 * hidden_size, num_components),
        )

    def forward_s(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        out, _ = self.smooth_lstm(inp)
        return self.mean_head(out), self.vars_head(out) + torch.finfo(x.dtype).eps

    def forward_z(self, x: torch.Tensor) -> torch.Tensor:
        return self.z_proxy(x)

    def evaluate_z(
        self,
        prior_mean: torch.Tensor,
        prior_vars: torch.Tensor,
        posterior_mean: torch.Tensor,
        posterior_vars: torch.Tensor,
    ) -> torch.Tensor:
        logits = -0.5 * torch.sum(
            (prior_mean - posterior_mean).pow(2).div(prior_vars).sum(dim=-1)
            + posterior_vars.div(prior_vars).sum(dim=-1)
            + prior_vars.div(posterior_vars).log().sum(dim=-1),
            dim=-1,
        )
        logits = logits - logits.max(dim=-1, keepdim=True).values
        return F.softmax(logits, dim=-1)


class VariationalMixtureDecoder(nn.Module):
    def __init__(self, config, input_dim: int):
        super().__init__()
        self.num_components = config.num_components
        self.horizon = config.future_len
        self.output_dim = config.output_dim
        self.kl_weight = config.kl_weight
        self.proxy_loss_weight = config.proxy_loss_weight
        self.prior_trajectory_weight = config.get("prior_trajectory_weight", 0.0)
        self.prior_endpoint_weight = config.get("prior_endpoint_weight", 0.0)
        self.proxy_target = config.get("proxy_target", "nearest_prior")
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.generative_net = GenerativeNet(input_dim, config.decoder_hidden_size, config.num_components, config.future_len, config.output_dim)
        self.inference_net = InferenceNet(input_dim, config.decoder_hidden_size, config.future_len, config.num_components, config.output_dim)

    def predict(self, x: torch.Tensor, num_modes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(self.inference_net.forward_z(x), dim=-1)
        y_mean, _ = self.generative_net.predictive(x)
        if num_modes < self.num_components:
            top_probs, top_idx = probs.topk(num_modes, dim=-1)
            bidx = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
            y_mean = y_mean[bidx, top_idx]
            probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return y_mean, probs

    def fit(self, x: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        q_s_mean, q_s_vars = self.inference_net.forward_s(x, y)
        p_s_mean, p_s_vars = self.generative_net.forward_s(x)
        y_mean, y_vars = self.generative_net.forward_y(q_s_mean, x)
        prior_y_mean, _ = self.generative_net.forward_y(p_s_mean, x)

        # emission 预测逐帧位移 Δ;此处积分(cumsum)回绝对位置后再算重建 NLL,使训练 loss
        # 与 predictive(:201)/评测指标所用的「累积绝对位置」口径一致(方差同步 cumsum)。
        # 旧实现把 NLL 算在 Δ 空间(target=y_diff),只压逐帧位移、不惩罚累积漂移:长 horizon
        # (Waymo 8s/80 帧)下每步微小偏置沿 80 帧累积成十几米端点误差,表现为「train loss 低
        # 但 minADE/minFDE 极大、miss_rate≈0.9」。改到绝对位置空间后,低 loss 才等价于轨迹好。
        y_mean = y_mean.cumsum(dim=-2)
        y_vars = y_vars.cumsum(dim=-2)
        prior_y_mean = prior_y_mean.cumsum(dim=-2)
        rec = 0.5 * (torch.log(2 * math.pi * y_vars) + (y - y_mean).pow(2).div(y_vars)).sum(dim=-1)
        rec = (rec * y_mask.float()).sum(dim=-1) / y_mask.float().sum(dim=-1).clamp_min(1.0)

        # The posterior reconstruction path sees the ground-truth future, while
        # inference uses only the free-running prior. On an 80-frame horizon,
        # a tiny per-step latent mismatch can accumulate into a large endpoint
        # error even when latent KL is nearly zero. Directly supervise the prior
        # trajectories with a best-of-K metric-aligned objective.
        prior_dist = torch.linalg.vector_norm(prior_y_mean - y[:, None], dim=-1)
        prior_ade_per_mode = (
            (prior_dist * y_mask[:, None].float()).sum(dim=-1)
            / y_mask[:, None].float().sum(dim=-1).clamp_min(1.0)
        )
        final_valid_idx = (
            torch.arange(y_mask.size(-1), device=y_mask.device)
            .view(1, -1)
            .expand_as(y_mask)
            .masked_fill(~y_mask, 0)
            .max(dim=-1)
            .values
        )
        gather_idx = final_valid_idx[:, None, None].expand(-1, self.num_components, 1)
        prior_fde_per_mode = prior_dist.gather(dim=-1, index=gather_idx).squeeze(-1)
        nearest_prior = (prior_ade_per_mode + prior_fde_per_mode).argmin(dim=-1)
        bidx = torch.arange(x.size(0), device=x.device)
        prior_ade = prior_ade_per_mode[bidx, nearest_prior]
        prior_fde = prior_fde_per_mode[bidx, nearest_prior]

        q_s_mean_exp = q_s_mean.unsqueeze(1).expand_as(p_s_mean)
        q_s_vars_exp = q_s_vars.unsqueeze(1).expand_as(p_s_vars)
        with torch.no_grad():
            q_z = self.inference_net.evaluate_z(p_s_mean, p_s_vars, q_s_mean_exp, q_s_vars_exp)

        kl_s = 0.5 * (
            p_s_vars.div(q_s_vars_exp).log().sum(dim=-1)
            + q_s_vars_exp.div(p_s_vars).sum(dim=-1)
            + (p_s_mean - q_s_mean_exp).pow(2).div(p_s_vars).sum(dim=-1)
            - p_s_mean.size(-1)
        ).sum(dim=-1)
        kl_s = (q_z * kl_s).sum(dim=-1)

        if self.proxy_target == "variational":
            target = q_z.argmax(dim=-1)
        elif self.proxy_target == "nearest_prior":
            target = nearest_prior.detach()
        else:
            raise ValueError(f"Unsupported proxy_target: {self.proxy_target}")
        ce = F.cross_entropy(self.inference_net.forward_z(x), target, reduction="none")
        pt = torch.exp(-ce)
        proxy = self.focal_alpha * (1 - pt).pow(self.focal_gamma) * ce
        loss = (
            rec.mean()
            + self.kl_weight * kl_s.mean()
            + self.proxy_loss_weight * proxy.mean()
            + self.prior_trajectory_weight * prior_ade.mean()
            + self.prior_endpoint_weight * prior_fde.mean()
        )
        endpoint_dist = torch.cdist(prior_y_mean[:, :, -1], prior_y_mean[:, :, -1])
        off_diagonal = ~torch.eye(
            self.num_components, dtype=torch.bool, device=endpoint_dist.device
        )
        return {
            "loss": loss,
            "reconstruction_loss": rec.mean().detach(),
            "kl_loss_s": kl_s.mean().detach(),
            "z_proxy_loss": proxy.mean().detach(),
            "prior_minade_loss": prior_ade.mean().detach(),
            "prior_minfde_loss": prior_fde.mean().detach(),
            "mode_endpoint_spread": endpoint_dist[:, off_diagonal].mean().detach(),
        }


class SenevaModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_modes = config.num_modes
        self.encoder = SenevaEncoder(config)
        self.decoder = VariationalMixtureDecoder(config, self.encoder.out_feature)

    def forward(self, batch):
        inputs = batch["input_dict"]
        x = self.encoder(inputs)
        pred_trajs, pred_probs = self.decoder.predict(x, self.num_modes)
        output = {
            "predicted_trajectory": pred_trajs,
            "predicted_probability": pred_probs,
            "dataset_name": inputs["dataset_name"],
        }

        y = inputs["center_gt_trajs"][..., : self.config.output_dim]
        y_mask = inputs["center_gt_trajs_mask"].bool()
        losses = self.decoder.fit(x, y, y_mask)
        self._latest_loss_terms = losses
        return output, losses["loss"]

    def _log_loss_terms(self, status: str, batch_size: int):
        for name, value in self._latest_loss_terms.items():
            if name == "loss":
                continue
            self.log(
                f"{status}/{name}",
                value,
                on_step=(status == "train"),
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )

    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        batch_size = len(batch["input_dict"]["center_objects_id"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self._log_loss_terms("train", batch_size)
        self.log_info(batch, batch_idx, prediction, status="train")
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        batch_size = len(batch["input_dict"]["center_objects_id"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self._log_loss_terms("val", batch_size)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=tuple(self.config.betas),
            weight_decay=self.config.weight_decay,
        )
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / max(float(self.config.warmup_steps), 1.0)),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(self.config.max_steps), 1),
            eta_min=float(self.config.min_lr),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[int(self.config.warmup_steps)],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
