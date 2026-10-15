import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_len=100):
        super().__init__()
        d_model = 256
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 直接按 (1, max_len, d_model) 构造，不要用 (max_len,1,d).permute(1,0,2)：
        # permute 后第 0 维 size=1，is_contiguous() 会忽略 size-1 维而误报 True，
        # 导致 .contiguous()/.clone() 都成 no-op、strides 仍重叠 (256,256,1)，
        # DDP 的 broadcast_coalesced 同步 buffer 时会报
        # "refer to a single memory location"（单卡走 DDP 也触发）。
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(-2)])


class AgentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)
        # Project the data-provided agent type one-hot directly (native state
        # cols 8:11). Equivalent to the old nn.Embedding(4) on argmax(one-hot)
        # but without the redundant argmax round-trip; all-zero (no type) -> 0.
        self.type_proj = nn.Linear(3, 256, bias=False)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        return traj[:, -1] + self.type_proj(inputs[:, -1, 8:11])


class LaneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        # Project the data-provided 20-dim map type one-hot directly (native lane
        # cols 10:30). Replaces the old nn.Embedding(4) on argmax(one-hot). The
        # official left/right/traffic-light/interpolating/stop-sign type embeddings
        # are dropped: UniTraj never populates them (their inputs were constant
        # zero), so they only added dead, always-index-0 parameters.
        self.self_type = nn.Linear(20, 64, bias=False)
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256))
        self.position_encode = PositionalEncoding(max_len=100)

    def forward(self, inputs):
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[..., 6:9])
        speed_limit = self.speed_limit(inputs[..., 9:10])
        lane_attr = self.self_type(inputs[..., 10:30])
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
        return self.position_encode(self.pointnet(lane_embedding))


class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

    def forward(self, inputs):
        return self.point_net(inputs)


class FutureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256))
        # Project the agent type one-hot directly (native state cols 8:11).
        self.type_proj = nn.Linear(3, 256, bias=False)

    def state_process(self, trajs, current_states):
        modes = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, modes, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        velocity = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-3)).unsqueeze(-1)
        size = current_states[:, :, :, None, 5:8].expand(-1, -1, -1, trajs.shape[3], -1)
        return torch.cat([trajs, theta, velocity, size], dim=-1)

    def forward(self, trajs, current_states):
        trajs = self.mlp(self.state_process(trajs, current_states).detach())
        actor_type = self.type_proj(current_states[:, :, None, 8:11])
        return torch.max(trajs, dim=-2).values + actor_type


class GMMPredictor(nn.Module):
    def __init__(self, future_len):
        super().__init__()
        self.future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(512, future_len * 4),
        )
        self.score = nn.Sequential(
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, inputs):
        batch, modes, _ = inputs.shape
        gaussian = self.gaussian(inputs).view(batch, modes, self.future_len, 4)
        return gaussian, self.score(inputs).squeeze(-1)


class SelfTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        dim, heads, dropout = 256, 8, 0.1
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(
            inputs, inputs, inputs, key_padding_mask=mask, need_weights=False
        )
        attention_output = self.norm_1(attention_output + inputs)
        return self.norm_2(self.ffn(attention_output) + attention_output)


class CrossTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        dim, heads, dropout = 256, 8, 0.1
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(
            query, key, value, key_padding_mask=mask, need_weights=False
        )
        attention_output = self.norm_1(attention_output)
        return self.norm_2(self.ffn(attention_output) + attention_output)


class InitialDecoder(nn.Module):
    def __init__(self, modalities, neighbors, future_len):
        super().__init__()
        self.modalities = modalities
        self.multi_modal_query_embedding = nn.Embedding(modalities, 256)
        self.agent_query_embedding = nn.Embedding(neighbors + 1, 256)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor(future_len)
        self.register_buffer("modal", torch.arange(modalities).long())
        self.register_buffer("agent", torch.arange(neighbors + 1).long())

    def forward(self, actor_id, current_state, encoding, mask):
        modal_query = self.multi_modal_query_embedding(self.modal)
        actor_query = self.agent_query_embedding(self.agent[actor_id])
        query = encoding[:, None, actor_id] + modal_query + actor_query[None, :]
        query_content = self.query_encoder(query, encoding, encoding, mask)
        predictions, scores = self.predictor(query_content)
        predictions[..., :2] += current_state[:, None, None, :2]
        return query_content, predictions, scores


class InteractionDecoder(nn.Module):
    def __init__(self, future_encoder, future_len):
        super().__init__()
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor(future_len)

    def forward(self, actor_id, current_states, actors, scores, last_content, encoding, mask):
        num_actors = actors.shape[1]
        multi_futures = self.future_encoder(actors[..., :2], current_states)
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)
        interaction = self.interaction_encoder(futures, mask[:, :num_actors])
        encoding = torch.cat([interaction, encoding], dim=1)
        mask = torch.cat([mask[:, :num_actors], mask], dim=1).clone()
        mask[:, actor_id] = True
        query = last_content + multi_futures[:, actor_id]
        query_content = self.query_encoder(query, encoding, encoding, mask)
        trajectories, scores = self.decoder(query_content)
        trajectories[..., :2] += current_states[:, actor_id, None, None, :2]
        return query_content, trajectories, scores
