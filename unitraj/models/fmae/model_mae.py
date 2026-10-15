from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block


class ModelMAE(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        actor_mask_ratio: float = 0.5,
        lane_mask_ratio: float = 0.5,
        history_steps: int = 50,
        future_steps: int = 60,
        lane_points: int = 20,
        loss_weight: List[float] = [1.0, 1.0, 0.35],
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.actor_mask_ratio = actor_mask_ratio
        self.lane_mask_ratio = lane_mask_ratio
        self.loss_weight = loss_weight
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.lane_points = lane_points

        # hist_embed kernel_size=[3,3,3]:原默认 [3,3,5] 是按 Argoverse2 的 50 帧历史设计的;
        # Waymo past_len=11 经两次 stride-2 下采样后最深一级长度只剩 3,NATTEN 要求
        # 序列长度 >= kernel×dilation,kernel=5 会崩(同 model_forecast.py 的修复)。
        # future_embed 走 future_len=80:80→40→20,kernel=5 OK,保留默认 [3,3,5]。
        self.hist_embed = AgentEmbeddingLayer(
            4, 32, kernel_size=[3, 3, 3], drop_path_rate=drop_path
        )
        self.future_embed = AgentEmbeddingLayer(3, 32, drop_path_rate=drop_path)
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.lane_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.future_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.history_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.future_pred = nn.Linear(embed_dim, future_steps * 2)
        self.history_pred = nn.Linear(embed_dim, history_steps * 2)
        self.lane_pred = nn.Linear(embed_dim, lane_points * 2)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)
        nn.init.normal_(self.future_mask_token, std=0.02)
        nn.init.normal_(self.lane_mask_token, std=0.02)
        nn.init.normal_(self.history_mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def agent_random_masking(
        hist_tokens, fut_tokens, mask_ratio, future_padding_mask, num_actors
    ):
        pred_masks = ~future_padding_mask.all(-1)  # [B, A]
        fut_num_tokens = pred_masks.sum(-1)  # [B]

        len_keeps = (fut_num_tokens * (1 - mask_ratio)).int()
        hist_masked_tokens, fut_masked_tokens = [], []
        hist_keep_ids_list, fut_keep_ids_list = [], []
        hist_key_padding_mask, fut_key_padding_mask = [], []

        device = hist_tokens.device
        agent_ids = torch.arange(hist_tokens.shape[1], device=device)
        for i, (fut_num_token, len_keep, future_pred_mask) in enumerate(
            zip(fut_num_tokens, len_keeps, pred_masks)
        ):
            pred_agent_ids = agent_ids[future_pred_mask]
            noise = torch.rand(fut_num_token, device=device)
            ids_shuffle = torch.argsort(noise)
            fut_ids_keep = ids_shuffle[:len_keep]
            fut_ids_keep = pred_agent_ids[fut_ids_keep]
            fut_keep_ids_list.append(fut_ids_keep)

            hist_keep_mask = torch.zeros_like(agent_ids).bool()
            hist_keep_mask[: num_actors[i]] = True
            hist_keep_mask[fut_ids_keep] = False
            hist_ids_keep = agent_ids[hist_keep_mask]
            hist_keep_ids_list.append(hist_ids_keep)

            fut_masked_tokens.append(fut_tokens[i, fut_ids_keep])
            hist_masked_tokens.append(hist_tokens[i, hist_ids_keep])

            fut_key_padding_mask.append(torch.zeros(len_keep, device=device))
            hist_key_padding_mask.append(torch.zeros(len(hist_ids_keep), device=device))

        fut_masked_tokens = pad_sequence(fut_masked_tokens, batch_first=True)
        hist_masked_tokens = pad_sequence(hist_masked_tokens, batch_first=True)
        fut_key_padding_mask = pad_sequence(
            fut_key_padding_mask, batch_first=True, padding_value=True
        )
        hist_key_padding_mask = pad_sequence(
            hist_key_padding_mask, batch_first=True, padding_value=True
        )

        return (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        )

    @staticmethod
    def lane_random_masking(x, future_mask_ratio, key_padding_mask):
        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int()

        x_masked, new_key_padding_mask, ids_keep_list = [], [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)

            ids_keep = ids_shuffle[:len_keep]
            ids_keep_list.append(ids_keep)
            x_masked.append(x[i, ids_keep])
            new_key_padding_mask.append(torch.zeros(len_keep, device=x.device))

        x_masked = pad_sequence(x_masked, batch_first=True)
        new_key_padding_mask = pad_sequence(
            new_key_padding_mask, batch_first=True, padding_value=True
        )

        return x_masked, new_key_padding_mask, ids_keep_list

    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :self.history_steps]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )
        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat = self.hist_embed(hist_feat.permute(0, 2, 1).contiguous())
        hist_feat = hist_feat.view(B, N, hist_feat.shape[-1])

        future_padding_mask = data["x_padding_mask"][:, :, self.history_steps:]
        future_feat = torch.cat([data["y"], ~future_padding_mask[..., None]], dim=-1)
        B, N, L, D = future_feat.shape
        future_feat = future_feat.view(B * N, L, D)
        future_feat = self.future_embed(future_feat.permute(0, 2, 1).contiguous())
        future_feat = future_feat.view(B, N, future_feat.shape[-1])

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)
        B, M, L, D = lane_feat.shape
        lane_feat = self.lane_embed(lane_feat.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]

        hist_feat += actor_type_embed
        lane_feat += self.lane_type_embed
        future_feat += actor_type_embed

        x_centers = torch.cat(
            [data["x_centers"], data["x_centers"], data["lane_centers"]], dim=1
        )
        angles = torch.cat(
            [
                data["x_angles"][..., self.history_steps - 1],
                data["x_angles"][..., self.history_steps - 1],
                data["lane_angles"],
            ],
            dim=1,
        )
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)

        pos_embed = self.pos_embed(pos_feat)
        hist_feat += pos_embed[:, :N]
        lane_feat += pos_embed[:, -M:]
        future_feat += pos_embed[:, N : N + N]

        (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        ) = self.agent_random_masking(
            hist_feat,
            future_feat,
            self.actor_mask_ratio,
            future_padding_mask,
            data["num_actors"],
        )

        lane_mask_ratio = self.lane_mask_ratio
        (
            lane_masked_tokens,
            lane_key_padding_mask,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            lane_feat, lane_mask_ratio, data["lane_key_padding_mask"]
        )

        x = torch.cat(
            [hist_masked_tokens, fut_masked_tokens, lane_masked_tokens], dim=1
        )
        key_padding_mask = torch.cat(
            [hist_key_padding_mask, fut_key_padding_mask, lane_key_padding_mask],
            dim=1,
        )

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # decoding
        x_decoder = self.decoder_embed(x)
        Nh, Nf, Nl = (
            hist_masked_tokens.shape[1],
            fut_masked_tokens.shape[1],
            lane_masked_tokens.shape[1],
        )
        assert x_decoder.shape[1] == Nh + Nf + Nl
        hist_tokens = x_decoder[:, :Nh]
        fut_tokens = x_decoder[:, Nh : Nh + Nf]
        lane_tokens = x_decoder[:, -Nl:]

        decoder_hist_token = self.history_mask_token.repeat(B, N, 1)
        hist_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(hist_keep_ids_list):
            decoder_hist_token[i, idx] = hist_tokens[i, : len(idx)]
            hist_pred_mask[i, idx] = False

        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        future_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(fut_keep_ids_list):
            decoder_fut_token[i, idx] = fut_tokens[i, : len(idx)]
            future_pred_mask[i, idx] = False

        decoder_lane_token = self.lane_mask_token.repeat(B, M, 1)
        lane_pred_mask = ~data["lane_key_padding_mask"]
        for i, idx in enumerate(lane_ids_keep_list):
            decoder_lane_token[i, idx] = lane_tokens[i, : len(idx)]
            lane_pred_mask[i, idx] = False

        x_decoder = torch.cat(
            [decoder_hist_token, decoder_fut_token, decoder_lane_token], dim=1
        )
        x_decoder = x_decoder + self.decoder_pos_embed(pos_feat)
        decoder_key_padding_mask = torch.cat(
            [
                data["x_key_padding_mask"],
                future_padding_mask.all(-1),
                data["lane_key_padding_mask"],
            ],
            dim=1,
        )

        for blk in self.decoder_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=decoder_key_padding_mask)

        x_decoder = self.decoder_norm(x_decoder)
        hist_token = x_decoder[:, :N].reshape(-1, self.embed_dim)
        future_token = x_decoder[:, N : 2 * N].reshape(-1, self.embed_dim)
        lane_token = x_decoder[:, -M:]

        # lane pred loss
        lane_pred = self.lane_pred(lane_token).view(B, M, self.lane_points, 2)
        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = self.history_pred(hist_token).view(-1, self.history_steps, 2)
        x = (data["x_positions"] - data["x_centers"].unsqueeze(-2)).view(-1, self.history_steps, 2)
        x_reg_mask = ~data["x_padding_mask"][:, :, :self.history_steps]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.history_steps)
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x[x_reg_mask])

        # future pred loss
        y_hat = self.future_pred(future_token).view(-1, self.future_steps, 2)  # B*N, 120
        y = data["y"].view(-1, self.future_steps, 2)
        reg_mask = ~data["x_padding_mask"][:, :, self.history_steps:]
        reg_mask[~future_pred_mask] = False
        reg_mask = reg_mask.view(-1, self.future_steps)
        future_loss = F.l1_loss(y_hat[reg_mask], y[reg_mask])

        # ── 单模态预测指标(米,和 MTR/EMP 的 ADE/FDE 同坐标系同量纲、可直接对比)──
        # MAE 单模态 → minADE6/brier_fde 等多模态指标无意义,不计算。
        # 只在「被掩码且未来有效」的 agent-step(reg_mask)上评估 —— 这才是真正在做预测的部分;
        # 未掩码 agent 的未来已喂给 future_embed,算上去等于作弊。
        with torch.no_grad():
            err = torch.linalg.norm(y_hat.detach() - y, dim=-1)          # (B*N, T)
            m = reg_mask                                                  # (B*N, T) bool
            agent_valid = m.any(dim=-1)                                   # (B*N,)
            step_cnt = m.sum(dim=-1).clamp(min=1)
            ade_a = (err * m).sum(dim=-1) / step_cnt                      # per-agent ADE
            ar = torch.arange(self.future_steps, device=err.device)
            last_idx = (ar[None, :] * m.long()).argmax(dim=-1)            # 末端有效步
            fde_a = err.gather(1, last_idx[:, None]).squeeze(1)           # per-agent FDE
            types = data["x_attr"][..., 2].reshape(-1)                    # 0veh1ped2cyc3pad

            def _agg(sel):
                if sel.sum() == 0:
                    nan = err.new_tensor(float("nan"))
                    return nan, nan, nan
                return (ade_a[sel].mean(),
                        fde_a[sel].mean(),
                        (fde_a[sel] > 2.0).float().mean())

            fm = {}
            fm["minADE"], fm["minFDE"], fm["miss_rate"] = _agg(agent_valid)
            for tid, tname in [(0, "vehicle"), (1, "pedestrian"), (2, "cyclist")]:
                a, f, mr = _agg(agent_valid & (types == tid))
                fm[f"{tname}_minADE"], fm[f"{tname}_minFDE"], fm[f"{tname}_miss_rate"] = a, f, mr

        loss = (
            self.loss_weight[0] * future_loss
            + self.loss_weight[1] * hist_loss
            + self.loss_weight[2] * lane_pred_loss
        )

        out = {
            "loss": loss,
            "hist_loss": hist_loss.item(),
            "future_loss": future_loss.item(),
            "lane_pred_loss": lane_pred_loss.item(),
            "metrics": fm,
        }

        #for UniTraj integration--------------------
        out["pi"] = torch.ones((B, 1))
        out["y_hat"] =  y_hat.view(1, B, N, self.future_steps, 2)#y_hat.view(B, N, self.future_steps, 2)[:,:1,:,:]
        #------------------------------------------

        if not self.training:
            out["x_hat"] = x_hat.view(B, N, self.history_steps, 2)
            out["y_hat"] = y_hat.view(1, B, N, self.future_steps, 2)
            out["lane_hat"] = lane_pred.view(B, M, self.lane_points, 2)
            out["lane_keep_ids"] = lane_ids_keep_list
            out["hist_keep_ids"] = hist_keep_ids_list
            out["fut_keep_ids"] = fut_keep_ids_list

        return out
