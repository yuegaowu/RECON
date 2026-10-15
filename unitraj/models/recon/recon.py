"""RECON implementation shared by the full model and its configured ablations."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math
from pathlib import Path

from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.wayformer.wayformer_utils import (
    PerceiverEncoder, PerceiverDecoder, TrainableQueryProvider,
)


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def masked_mean(x, mask, dim, keepdim=False, eps=1e-6):
    mask = mask.to(dtype=x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)


class SparseDirectedGraphLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.SELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, node_mask):
        q = self.q_proj(nodes)
        k = self.k_proj(nodes)
        v = self.v_proj(nodes)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(nodes.shape[-1])
        key_mask = ~node_mask.bool()
        logits = logits.masked_fill(key_mask[:, None, :], -1e4)
        attn = F.softmax(logits, dim=-1)
        attn = attn * node_mask[:, None, :].to(attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        msg = torch.matmul(attn, v)
        nodes = self.norm1(nodes + self.dropout(self.out_proj(msg)))
        nodes = self.norm2(nodes + self.dropout(self.ffn(nodes)))
        valid_rows = node_mask.to(attn.dtype)
        entropy = -(attn.clamp_min(1e-8).log() * attn).sum(dim=-1)
        entropy = (entropy * valid_rows).sum() / valid_rows.sum().clamp_min(1.0)
        return nodes, entropy


class RECON(BaseModel):
    """RECON: normal-driving references for candidate confidence learning.

    Wayformer supplies the trajectory backbone. The added scoring branch uses
    fixed normal prototypes, scene interactions and trajectory risk features.
    Method YAMLs select the component ablations and scoring probes.
    """

    def __init__(self, config):
        super(RECON, self).__init__(config)
        self.config = config
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.fisher_information = None
        self.map_attr = config['num_map_feature']
        self.k_attr = config['num_agent_feature']
        self.d_k = config['hidden_size']
        self._M = config['max_num_agents']  # num agents without the ego-agent
        self.c = config['num_modes']
        self.T = config['future_len']
        self.L_enc = config['num_encoder_layers']
        self.dropout = config['dropout']
        self.num_heads = config['tx_num_heads']
        self.L_dec = config['num_decoder_layers']
        self.tx_hidden_size = config['tx_hidden_size']
        self.use_map_img = config['use_map_image']
        self.use_map_lanes = config['use_map_lanes']
        self.past_T = config['past_len']
        self.max_points_per_lane = config['max_points_per_lane']
        self.max_num_roads = config['max_num_roads']
        self.num_queries_enc = config['num_queries_enc']
        self.num_queries_dec = config['num_queries_dec']
        if self.tx_hidden_size % self.d_k != 0:
            raise ValueError('tx_hidden_size must be divisible by hidden_size')
        widening_factor = self.tx_hidden_size // self.d_k

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))
        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        self.perceiver_encoder = PerceiverEncoder(self.num_queries_enc, self.d_k,
                                                  num_cross_attention_heads=self.num_heads,
                                                  num_cross_attention_qk_channels=self.d_k,
                                                  num_cross_attention_v_channels=self.d_k,
                                                  num_self_attention_heads=self.num_heads,
                                                  num_self_attention_qk_channels=self.d_k,
                                                  num_self_attention_v_channels=self.d_k,
                                                  num_self_attention_blocks=self.L_enc,
                                                  cross_attention_widening_factor=widening_factor,
                                                  self_attention_widening_factor=widening_factor,
                                                  dropout=self.dropout)

        output_query_provider = TrainableQueryProvider(
            num_queries=config['num_queries_dec'],
            num_query_channels=self.d_k,
            init_scale=0.1,
        )

        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )

        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )

        # self.map_positional_embedding = nn.parameter.Parameter(
        #     torch.zeros((1, self.max_points_per_lane * self.max_num_roads, self.d_k)), requires_grad=True
        # )

        self.perceiver_decoder = PerceiverDecoder(
            output_query_provider,
            self.d_k,
            num_cross_attention_heads=self.num_heads,
            num_cross_attention_layers=self.L_dec,
            num_cross_attention_qk_channels=self.d_k,
            num_cross_attention_v_channels=self.d_k,
            cross_attention_widening_factor=widening_factor,
            dropout=self.dropout,
        )

        self.prob_predictor = nn.Sequential(init_(nn.Linear(self.d_k, 1)))

        self.output_model = nn.Sequential(init_(nn.Linear(self.d_k, 5 * self.T)))

        self.causal_v1 = bool(config.get('causal_v1_enabled', False))
        self.causal_v2 = bool(config.get('causal_v2_enabled', False))
        # scorefix 打分路解锁开关(默认 False → 老结构完全不变,现有 ckpt/评测零影响)。
        self.clean_norm_v2 = bool(config.get('clean_norm_v2', False))
        if self.causal_v1 or self.causal_v2:
            self.graph_max_agents = int(config.get('graph_max_agents', 16))
            self.graph_max_map_nodes = int(config.get('graph_max_map_nodes', 64))
            self.graph_risk_tokens = int(config.get('graph_risk_tokens', 4))
            self.graph_type_embedding = nn.Parameter(torch.zeros(1, 3, self.d_k))
            self.risk_token_encoder = nn.Sequential(
                init_(nn.Linear(6, self.d_k)),
                nn.SELU(inplace=True),
                init_(nn.Linear(self.d_k, self.d_k)),
            )
            self.causal_graph = SparseDirectedGraphLayer(self.d_k, self.tx_hidden_size, self.dropout)
            self.graph_q = init_(nn.Linear(self.d_k, self.d_k))
            self.graph_k = init_(nn.Linear(self.d_k, self.d_k))
            self.graph_v = init_(nn.Linear(self.d_k, self.d_k))
            self.graph_out = init_(nn.Linear(self.d_k, self.d_k))
            if self.causal_v2:
                self.prototype_conditioning = str(config.get('prototype_conditioning', 'dictionary'))
                if self.prototype_conditioning not in ('dictionary', 'mlp', 'none'):
                    raise ValueError('prototype_conditioning must be dictionary, mlp or none')
                if self.prototype_conditioning in ('mlp', 'none'):
                    if float(config.get('confounder_subtract_scale', 1.0)) != 0.0:
                        raise ValueError('Dictionary-free control requires subtraction scale zero')
                    if float(config.get('prototype_entropy_weight', 0.0)) != 0.0:
                        raise ValueError('Dictionary-free control has no prototype entropy regularizer')
                else:
                    prototypes = self._load_frozen_prototypes(config)
                    self.register_buffer('confounder_prototypes', prototypes,
                        persistent=bool(config.get('confounder_save_in_ckpt', True)))
                self.prototype_value_proj = init_(nn.Linear(self.d_k, self.d_k))
                self.prototype_value_norm = nn.LayerNorm(self.d_k)
                self.prototype_residual_norm = nn.LayerNorm(self.d_k)
                fusion_dim = self.d_k * 4 + 6
                prob_fusion_dim = self.d_k * 5 + 6
                if self.clean_norm_v2:
                    # scorefix:clean 基底与 confounder 各 LayerNorm 到同空间(减法占比 2.4%→~25%);
                    #   prob_input 删去原始 mode_emb → 少一个 d_k 块。
                    self.clean_base_norm = nn.LayerNorm(self.d_k)
                    self.clean_conf_norm = nn.LayerNorm(self.d_k)
                    prob_fusion_dim = self.d_k * 4 + 6
            else:
                self.confounder_prototypes = nn.Parameter(
                    torch.randn(int(config.get('confounder_num_prototypes', 16)), self.d_k) * 0.02
                )
                fusion_dim = self.d_k * 3 + 6
                prob_fusion_dim = self.d_k * 3 + 6
            self.confounder_q = init_(nn.Linear(self.d_k, self.d_k))
            self.confounder_gate = nn.Sequential(
                init_(nn.Linear(fusion_dim, self.d_k)),
                nn.SELU(inplace=True),
                init_(nn.Linear(self.d_k, self.d_k)),
                nn.Sigmoid(),
            )
            self.mode_intervention = nn.Sequential(
                init_(nn.Linear(fusion_dim, self.d_k)),
                nn.SELU(inplace=True),
                nn.Dropout(self.dropout),
                init_(nn.Linear(self.d_k, self.d_k)),
            )
            _deconf_head = nn.Sequential(
                init_(nn.Linear(prob_fusion_dim, self.d_k)),
                nn.SELU(inplace=True),
                nn.Dropout(self.dropout),
                init_(nn.Linear(self.d_k, 1)),
            )
            if self.clean_norm_v2:
                # 改名 → warm-start(strict=False)时旧 ckpt 的 deconfounded_prob_predictor.* 成 unexpected、
                #   本层成 missing 干净 reinit,避免形状不符(prob_fusion_dim 变了)导致 load 报错。
                self.deconf_prob_v2 = _deconf_head
            else:
                self.deconfounded_prob_predictor = _deconf_head

            if self.causal_v2 and self.prototype_conditioning == 'none':
                if self.clean_norm_v2:
                    raise ValueError('NoProto is a deployment-architecture ablation, not a probe')
                # Construct in the original order before removing these modules:
                # retained layers keep the same seeded initial tensors as P/F.
                # No dictionary is loaded; no prototype/difference/gate path remains.
                for name in ('prototype_value_proj', 'prototype_value_norm',
                             'prototype_residual_norm', 'confounder_q', 'confounder_gate'):
                    delattr(self, name)

        self.selu = nn.SELU(inplace=True)

        self.criterion = Criterion(self.config)

        self.fisher_information = None
        self.optimal_params = None

    def _resolve_unitraj_path(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        unitraj_root = Path(__file__).resolve().parents[2]
        return unitraj_root / path

    def _load_frozen_prototypes(self, config):
        proto_path = config.get('confounder_init_path', None)
        if proto_path is None:
            raise ValueError('causal_v2_enabled=True requires confounder_init_path')
        proto_path = self._resolve_unitraj_path(proto_path)
        data = torch.load(proto_path, map_location='cpu', weights_only=False)
        prototypes = data['prototypes'] if isinstance(data, dict) and 'prototypes' in data else data
        prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
        if prototypes.dim() != 2:
            raise ValueError(f'confounder prototypes must be [K,D], got {tuple(prototypes.shape)} from {proto_path}')
        if prototypes.shape[-1] != self.d_k:
            raise ValueError(f'prototype dim {prototypes.shape[-1]} != hidden_size {self.d_k}')
        max_k = int(config.get('confounder_num_prototypes', prototypes.shape[0]))
        prototypes = prototypes[:max_k].contiguous()
        if bool(config.get('prototype_l2_normalize', True)):
            prototypes = F.normalize(prototypes, dim=-1)
        print(f'[RECON] loaded frozen confounder prototypes: {tuple(prototypes.shape)} from {proto_path}')
        return prototypes

    def _select_topk_nodes(self, nodes, mask, score, k):
        k = min(k, nodes.shape[1])
        masked_score = score.masked_fill(~mask.bool(), -1e9)
        idx = torch.topk(masked_score, k=k, dim=1).indices
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
        return torch.gather(nodes, 1, gather_idx), torch.gather(mask, 1, idx), idx

    def _build_causal_graph(self, agents_tensor, agent_valid, agents_emb_4d, roads, road_pts_feats_4d):
        B, _, num_agents, _ = agents_tensor.shape
        agent_nodes = masked_mean(agents_emb_4d, agent_valid, dim=1)
        agent_node_mask = agent_valid.any(dim=1)
        agent_node_mask[:, 0] = True

        map_valid = roads[:, :self.max_num_roads, :, -1].bool()
        map_nodes = masked_mean(road_pts_feats_4d, map_valid, dim=2)
        map_node_mask = map_valid.any(dim=-1)

        cur = agents_tensor[:, -1, :, :]
        agent_pos = cur[..., 0:2]
        agent_vel = cur[..., 25:27] if cur.shape[-1] >= 27 else torch.zeros_like(agent_pos)
        ego_pos = agent_pos[:, 0:1]
        ego_vel = agent_vel[:, 0:1]
        rel = agent_pos - ego_pos
        rel_vel = agent_vel - ego_vel
        dist = rel.norm(dim=-1).clamp_min(1e-3)
        closing = -(rel * rel_vel).sum(dim=-1) / dist
        ttc = torch.where(closing > 0.1, dist / closing.clamp_min(0.1), torch.full_like(dist, 1e3))
        agent_score = (closing.clamp_min(0.0) / dist.clamp_min(1.0)) + (1.0 / ttc.clamp(0.5, 30.0))
        agent_score[:, 0] = 1e6
        agent_nodes, agent_node_mask, agent_idx = self._select_topk_nodes(
            agent_nodes, agent_node_mask, agent_score, self.graph_max_agents
        )

        map_xy = masked_mean(roads[:, :self.max_num_roads, :, 0:2], map_valid, dim=2)
        map_dist = (map_xy - ego_pos).norm(dim=-1)
        map_score = -map_dist
        map_nodes, map_node_mask, _ = self._select_topk_nodes(
            map_nodes, map_node_mask, map_score, self.graph_max_map_nodes
        )

        valid_neighbors = agent_node_mask.clone()
        valid_neighbors[:, 0] = False
        chosen_pos = torch.gather(agent_pos, 1, agent_idx.unsqueeze(-1).expand(-1, -1, 2))
        chosen_vel = torch.gather(agent_vel, 1, agent_idx.unsqueeze(-1).expand(-1, -1, 2))
        chosen_rel = chosen_pos - ego_pos
        chosen_rel_vel = chosen_vel - ego_vel
        chosen_dist = chosen_rel.norm(dim=-1).clamp_min(1e-3)
        chosen_closing = -(chosen_rel * chosen_rel_vel).sum(dim=-1) / chosen_dist
        chosen_ttc = torch.where(
            chosen_closing > 0.1,
            chosen_dist / chosen_closing.clamp_min(0.1),
            torch.full_like(chosen_dist, 30.0),
        )
        risk_base = torch.stack([
            (chosen_dist / 80.0).clamp(0.0, 2.0),
            (chosen_closing / 40.0).clamp(-1.0, 1.0),
            (chosen_ttc / 8.0).clamp(0.0, 4.0),
            chosen_rel[..., 0] / 80.0,
            chosen_rel[..., 1] / 80.0,
            chosen_rel_vel.norm(dim=-1) / 40.0,
        ], dim=-1)
        risk_score = (chosen_closing.clamp_min(0.0) / chosen_dist.clamp_min(1.0)) + (1.0 / chosen_ttc.clamp(0.5, 30.0))
        risk_score = risk_score.masked_fill(~valid_neighbors, -1e9)
        risk_k = min(self.graph_risk_tokens, risk_base.shape[1])
        risk_idx = torch.topk(risk_score, k=risk_k, dim=1).indices
        risk_features = torch.gather(risk_base, 1, risk_idx.unsqueeze(-1).expand(-1, -1, risk_base.shape[-1]))
        risk_mask = torch.gather(valid_neighbors, 1, risk_idx)
        risk_nodes = self.risk_token_encoder(risk_features)

        agent_nodes = agent_nodes + self.graph_type_embedding[:, 0:1]
        map_nodes = map_nodes + self.graph_type_embedding[:, 1:2]
        risk_nodes = risk_nodes + self.graph_type_embedding[:, 2:3]
        graph_nodes = torch.cat([agent_nodes, map_nodes, risk_nodes], dim=1)
        graph_mask = torch.cat([agent_node_mask, map_node_mask, risk_mask], dim=1)
        graph_nodes, graph_entropy = self.causal_graph(graph_nodes, graph_mask)
        graph_info = {
            'graph_nodes': graph_nodes,
            'graph_mask': graph_mask,
            'graph_entropy': graph_entropy,
            'agent_pos': agent_pos,
            'agent_vel': agent_vel,
            'agent_mask': agent_node_mask.new_zeros(B, num_agents).scatter(1, agent_idx, agent_node_mask),
        }
        return graph_info

    def _mode_risk_features(self, pred_xy, graph_info):
        B, c, T, _ = pred_xy.shape
        agent_pos = graph_info['agent_pos'][:, 1:]
        agent_vel = graph_info['agent_vel'][:, 1:]
        agent_mask = graph_info['agent_mask'][:, 1:].bool()
        if agent_pos.shape[1] == 0:
            return pred_xy.new_zeros(B, c, 6)
        t = torch.arange(1, T + 1, device=pred_xy.device, dtype=pred_xy.dtype) * float(self.config.get('kf_dt', 0.1))
        agent_future = agent_pos[:, None, :, None, :] + agent_vel[:, None, :, None, :] * t[None, None, None, :, None]
        rel = pred_xy[:, :, None, :, :] - agent_future
        dist = rel.norm(dim=-1).masked_fill(~agent_mask[:, None, :, None], 1e6)
        min_dist_t = dist.amin(dim=2)
        min_dist = min_dist_t.amin(dim=-1)
        mean_min_dist = min_dist_t.mean(dim=-1)
        min_idx = min_dist_t.argmin(dim=-1).to(pred_xy.dtype)
        endpoint = pred_xy[:, :, -1]
        endpoint_norm = endpoint.norm(dim=-1)
        disp = pred_xy[:, :, 1:] - pred_xy[:, :, :-1]
        speed = disp.norm(dim=-1) / float(self.config.get('kf_dt', 0.1))
        mean_speed = speed.mean(dim=-1)
        max_speed = speed.amax(dim=-1)
        return torch.stack([
            (min_dist / 50.0).clamp(0.0, 4.0),
            (mean_min_dist / 50.0).clamp(0.0, 4.0),
            (min_idx / max(float(T - 1), 1.0)).clamp(0.0, 1.0),
            (endpoint_norm / 120.0).clamp(0.0, 4.0),
            (mean_speed / 40.0).clamp(0.0, 2.0),
            (max_speed / 40.0).clamp(0.0, 2.0),
        ], dim=-1)

    def _causal_mode_intervention(self, mode_emb, pred_trajs, graph_info):
        graph_nodes = graph_info['graph_nodes']
        graph_mask = graph_info['graph_mask']
        q = self.graph_q(mode_emb)
        k = self.graph_k(graph_nodes)
        v = self.graph_v(graph_nodes)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        logits = logits.masked_fill(~graph_mask[:, None, :].bool(), -1e4)
        attn = F.softmax(logits, dim=-1)
        graph_context = self.graph_out(torch.matmul(attn, v))

        no_proto = self.causal_v2 and self.prototype_conditioning == 'none'
        cq = None if no_proto else self.confounder_q(mode_emb)
        if no_proto:
            confounder = torch.zeros_like(mode_emb)
            proto_attn = None
        elif self.causal_v2 and self.prototype_conditioning == 'mlp':
            # Capacity control: both learned projections and LN remain active,
            # without a dictionary, attention, or attention entropy penalty.
            confounder = self.prototype_value_norm(self.prototype_value_proj(F.selu(cq)))
            proto_attn = None
        else:
            proto = self.confounder_prototypes[None].expand(mode_emb.shape[0], -1, -1)
            if self.causal_v2 and bool(self.config.get('prototype_l2_normalize', True)):
                cq_for_attn = F.normalize(cq, dim=-1)
                proto_for_attn = F.normalize(proto, dim=-1)
            else:
                cq_for_attn = cq
                proto_for_attn = proto
            temperature = float(self.config.get('prototype_attention_temperature', 1.0))
            conf_logits = torch.matmul(cq_for_attn, proto_for_attn.transpose(-1, -2)) / max(temperature, 1e-3)
            proto_attn = F.softmax(conf_logits, dim=-1)
            if getattr(self, '_cap_proto', None) is not None:
                self._cap_proto.append(proto_attn.detach().float().cpu())
            confounder = torch.matmul(proto_attn, proto)
            if self.causal_v2:
                confounder = self.prototype_value_norm(self.prototype_value_proj(confounder))

        risk_features = self._mode_risk_features(pred_trajs[..., :2], graph_info)
        if self.causal_v2:
            residual_hint = (torch.zeros_like(mode_emb) if no_proto else
                             self.prototype_residual_norm(mode_emb - confounder))
            fusion = torch.cat([mode_emb, graph_context, confounder, residual_hint, risk_features], dim=-1)
        else:
            residual_hint = None
            fusion = torch.cat([mode_emb, graph_context, confounder, risk_features], dim=-1)
        gate = torch.zeros_like(mode_emb) if no_proto else self.confounder_gate(fusion)
        scale = float(self.config.get('confounder_subtract_scale', 1.0))
        if self.causal_v2 and self.clean_norm_v2:
            # scorefix 打分路:基底 mode_emb 与 confounder 各归一化到同空间 → 减法量级可比(~25% 而非 2.4%);
            #   prob_input 不含原始 mode_emb → 打分头必须依赖含减法的 clean(堵掉旁路);去杂头用 deconf_prob_v2。
            base = self.clean_base_norm(mode_emb)
            conf_n = self.clean_conf_norm(confounder)
            clean = base + self.mode_intervention(fusion) - scale * gate * conf_n
            prob_input = torch.cat([clean, graph_context, confounder, residual_hint, risk_features], dim=-1)
            proto_entropy = (-(proto_attn.clamp_min(1e-8).log() * proto_attn).sum(dim=-1).mean()
                             if proto_attn is not None else None)
            prob_logits = self.deconf_prob_v2(prob_input).squeeze(-1)
            return clean, prob_logits, risk_features, proto_entropy
        clean = mode_emb + self.mode_intervention(fusion) - scale * gate * confounder
        if self.causal_v2:
            prob_input = torch.cat([clean, mode_emb, graph_context, confounder, residual_hint, risk_features], dim=-1)
            proto_entropy = (-(proto_attn.clamp_min(1e-8).log() * proto_attn).sum(dim=-1).mean()
                             if proto_attn is not None else None)
        else:
            prob_input = torch.cat([clean, graph_context, confounder, risk_features], dim=-1)
            proto_entropy = None
        prob_logits = self.deconfounded_prob_predictor(prob_input).squeeze(-1)
        return clean, prob_logits, risk_features, proto_entropy

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.num_queries_dec, 1).view(ego.shape[0] * self.num_queries_dec,
                                                                                   -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def _forward(self, inputs):
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes or
                      [B, 3, 128, 128] image representing the road network if self.use_map_img or
                      [B, 1, 1] if self.use_map_lanes and self.use_map_img are False.
        :return:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''
        ego_in, agents_in, roads = inputs['ego_in'], inputs['agents_in'], inputs['roads']

        B = ego_in.size(0)
        num_agents = agents_in.shape[2] + 1
        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks_agents, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.selu(self.agents_dynamic_encoder(agents_tensor))
        agents_emb = agents_emb + self.agents_positional_embedding[:, :, :num_agents] + self.temporal_positional_embedding
        agents_emb_flat = agents_emb.view(B, -1, self.d_k)
        road_pts_feats = self.selu(self.road_pts_lin(roads[:, :self.max_num_roads, :, :self.map_attr]))
        road_pts_feats_flat = road_pts_feats.view(B, -1, self.d_k)  # + self.map_positional_embedding
        mixed_input_features = torch.concat([agents_emb_flat, road_pts_feats_flat], dim=1)
        opps_masks_roads = (1.0 - roads[:, :self.max_num_roads, :, -1]).to(torch.bool)
        mixed_input_masks = torch.concat([opps_masks_agents.view(B, -1), opps_masks_roads.view(B, -1)], dim=1)
        # Process through Wayformer's encoder

        context = self.perceiver_encoder(mixed_input_features, mixed_input_masks)

        # Wazformer-Ego Decoding

        out_seq = self.perceiver_decoder(context)

        mode_emb = out_seq[:, :self.c]
        out_dists = self.output_model(mode_emb).reshape(B, self.c, self.T, -1)

        # Mode prediction

        mode_probs = self.prob_predictor(mode_emb).reshape(B, self.c)
        graph_entropy = None
        proto_entropy = None
        if self.causal_v1 or self.causal_v2:
            graph_info = self._build_causal_graph(agents_tensor, ~opps_masks_agents, agents_emb, roads, road_pts_feats)
            clean_mode_emb, causal_logits, risk_features, proto_entropy = self._causal_mode_intervention(
                mode_emb, out_dists, graph_info
            )
            residual = float(self.config.get('causal_logit_residual_weight', 1.0))
            mode_probs = residual * mode_probs + causal_logits
            graph_entropy = graph_info['graph_entropy']
            mode_emb = clean_mode_emb

        # return  [c, T, B, 5], [B, c]
        output = {}
        output['predicted_probability'] = mode_probs  # #[B, c]
        output['predicted_trajectory'] = out_dists  # [B, c, T, 5] to be able to parallelize code
        output['scene_emb'] = out_seq[:, :self.num_queries_dec].reshape(B, -1)
        if graph_entropy is not None:
            output['graph_sparsity_loss'] = graph_entropy
        if proto_entropy is not None:
            output['prototype_entropy_loss'] = proto_entropy
        return output

    @torch.no_grad()
    def _current_velocity(self, ego_feats, ego_mask):
        """从 ego 历史(前 1.1s)取「当前真实速度 v0」:最后一个有效帧的 (vx,vy)。
        ego_feats: [B, T_hist, >=27] obj_trajs 特征(velocity 在 dims 25:27);
        ego_mask:  [B, T_hist] 历史有效掩码。返回 v0: [B,2](agent-centric,与预测同系)。
        """
        v_hist = ego_feats[..., 25:27]                         # [B,T,2]
        m = ego_mask.bool()                                    # [B,T]
        B, T = m.shape
        ar = torch.arange(T, device=m.device).unsqueeze(0).expand(B, T)
        last_idx = torch.where(m, ar, torch.full_like(ar, -1)).argmax(dim=1).clamp_min(0)  # 最后有效帧
        return v_hist[torch.arange(B, device=m.device), last_idx]   # [B,2]

    @torch.no_grad()
    def _kinematic_feasible_mask(self, pred_xy, v0):
        """历史锚定的运动学可行性判据(后处理,无梯度,仅评测调用)。
        pred_xy: [B, c, T, 2] agent-centric 预测均值(当前位置=原点);
        v0:      [B, 2] 由前 1.1s 历史得到的「当前真实速度」。
        判一条轨迹是否物理可达 = 从观测到的 v0 出发,沿轨迹所需的加/减速度、横向加速度是否
          超出车辆物理极限。**关键(历史锚定)**:把 v0 作为进入轨迹前的速度,使「第一步从真实
          速度到预测速度的过渡加速度」也被检查——能抓出与车当前速度矛盾的模态(如 30m/s 的车
          预测成瞬间静止/倒车),这是固定阈值法做不到的。
        ⚠ 必须降采样平滑(kf_smooth_step):10Hz 二阶差分=噪声(中位加速度~14>3g),2Hz 后~2 才真实。
        """
        cfg = self.config
        step = int(cfg.get('kf_smooth_step', 5))
        dt = float(cfg.get('kf_dt', 0.1)) * step
        # 从 step-1 起降采样:取 t=0.5,1.0,...,8.0(对齐 dt 网格),使原点(t=0)→首点正好 dt,
        # 否则 ::step 取 t=0.1,0.6,... 与原点间隔仅 0.1s 却按 dt 除 → 首速被低估 step 倍而误判。
        p = pred_xy[:, :, step - 1::step, :]                   # [B,c,L,2] 降采样平滑(0.5s 对齐)
        B, c, L, _ = p.shape
        origin = torch.zeros(B, c, 1, 2, device=p.device, dtype=p.dtype)  # 当前位置=原点
        pos = torch.cat([origin, p], dim=2)                    # [B,c,L+1,2]
        vel = (pos[:, :, 1:] - pos[:, :, :-1]) / dt            # [B,c,L,2] 平滑速度
        speed = vel.norm(dim=-1)                               # [B,c,L]
        # 历史锚定:把观测 v0 作为轨迹前一刻速度 → acc[0] 即「v0→首段速度」的过渡加速度
        v0e = v0[:, None, None, :].expand(B, c, 1, 2)
        vel_full = torch.cat([v0e, vel], dim=2)                # [B,c,L+1,2]
        acc = (vel_full[:, :, 1:] - vel_full[:, :, :-1]) / dt  # [B,c,L,2] 含 v0→首段过渡
        acc_mag = acc.norm(dim=-1)                             # [B,c,L]
        yaw = torch.atan2(vel[..., 1], vel[..., 0])            # [B,c,L]
        dyaw = yaw[:, :, 1:] - yaw[:, :, :-1]
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))   # wrap 到 [-pi,pi]
        yaw_rate = dyaw / dt                                   # [B,c,L-1]
        lat_acc = speed[:, :, 1:] * yaw_rate.abs()             # [B,c,L-1] 横向加速度
        slow = speed[:, :, 1:] < float(cfg.get('kf_min_speed_for_yaw', 2.0))
        lat_acc = torch.where(slow, torch.zeros_like(lat_acc), lat_acc)  # 低速免查横向
        bad = (speed.amax(-1) > float(cfg.get('kf_max_speed', 40.0))) \
            | (acc_mag.amax(-1) > float(cfg.get('kf_max_accel', 9.0))) \
            | (lat_acc.amax(-1) > float(cfg.get('kf_max_lat_accel', 8.0)))
        return ~bad                                            # [B,c] feasible

    def _apply_kinematic_filter(self, probs, pred_trajs, v0):
        """probs:[B,c] softmax 后概率;pred_trajs:[B,c,T,5];v0:[B,2]。判废不可达模态并重归一化。"""
        feas = self._kinematic_feasible_mask(pred_trajs[..., :2], v0)  # [B,c]
        if bool(self.config.get('kf_keep_if_all_bad', True)):
            feas = feas | (~feas.any(dim=-1, keepdim=True))           # 全坏则全保留
        probs = probs * feas.to(probs.dtype)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    def forward(self, batch):
        model_input = {}
        inputs = batch['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *agents_in.shape[
                                                                                                       -2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
                                                                                                       agents_mask.shape[
                                                                                                           -1])).squeeze(
            1)
        ego_feats = ego_in                                    # [B,T,29] 留作运动学过滤取 v0(velocity@25:27)
        agents_in = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)
        roads = torch.cat([inputs['map_polylines'], inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)
        model_input['ego_in'] = ego_in
        model_input['agents_in'] = agents_in
        model_input['roads'] = roads
        output = self._forward(model_input)

        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
        loss = self.criterion(output, ground_truth, inputs['center_gt_final_valid_idx'])
        output['dataset_name'] = inputs['dataset_name']
        probs = F.softmax(output['predicted_probability'], dim=-1)
        # 运动学过滤:仅评测启用(not self.training),训练零开销;基于前 1.1s 历史的物理可达性
        if (not self.training) and bool(self.config.get('kinematic_filter', False)):
            v0 = self._current_velocity(ego_feats, ego_mask)
            probs = self._apply_kinematic_filter(probs, output['predicted_trajectory'], v0)
        output['predicted_probability'] = probs
        return output, loss

    def configure_optimizers(self):
        # V3-sharp 决定性实验:score_head_only=True 时冻结几何(主干/encoder/decoder/output_model/
        # prototypes),只训「打分路径」(prob_predictor + causal 打分分支),可配大 LR。
        # 目的:隔离检验「给定 V2 产出的 mode_emb,打分头到底能不能把 6 条按精度重排」——
        # 若大 LR 只训打分头仍推不动 top1_is_best,才是真·信息天花板;否则就是之前 LR 太怂。
        # 几何冻结 → minFDE6 物理上不会变差(护栏绝对安全)。默认关,不影响其它模型/常规训练。
        if self.config.get('score_head_only', False):
            score_prefixes = (
                'prob_predictor', 'deconfounded_prob_predictor', 'mode_intervention',
                'confounder_q', 'confounder_gate', 'causal_graph',
                'graph_q', 'graph_k', 'graph_v', 'graph_out', 'risk_token_encoder',
                'prototype_value_proj', 'prototype_value_norm', 'prototype_residual_norm',
                # scorefix(clean_norm_v2):新打分路模块也归入可训练打分头。
                'deconf_prob_v2', 'clean_base_norm', 'clean_conf_norm',
            )
            train_params, n_train, n_freeze = [], 0, 0
            for name, p in self.named_parameters():
                if any(name.startswith(pre) for pre in score_prefixes):
                    p.requires_grad_(True); train_params.append(p); n_train += p.numel()
                else:
                    p.requires_grad_(False); n_freeze += p.numel()
            print(f'[V3-sharp score_head_only] trainable params={n_train:,} '
                  f'({len(train_params)} tensors) | frozen={n_freeze:,}')
            params_for_opt = train_params
        else:
            params_for_opt = self.parameters()

        optimizer = optim.AdamW(
            params_for_opt,
            lr=self.config['learning_rate'],
            eps=self.config['optimizer_eps'],
            weight_decay=self.config['weight_decay'],
        )
        if self.config.get('scheduler', 'onecycle') == 'cosine_step':
            # Full-budget fine-tuning: one scheduler update per optimizer update.
            total_updates = int(self.trainer.estimated_stepping_batches)
            assert total_updates > 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_updates, eta_min=float(self.config['cosine_min_lr']))
            return {'optimizer': optimizer, 'lr_scheduler': {
                'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['onecycle_max_lr'],
            steps_per_epoch=self.config['onecycle_steps_per_epoch'],
            epochs=self.config['onecycle_epochs'],
            pct_start=self.config['onecycle_pct_start'],
            div_factor=self.config['onecycle_div_factor'],
            final_div_factor=self.config['onecycle_final_div_factor'],
        )

        return [optimizer], [scheduler]


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config
        pass

    def forward(self, out, gt, center_gt_final_valid_idx):
        cls_w = float(self.config.get('cls_loss_weight', 1.0))   # 硬 CE 权重;=0 则完全交给排序蒸馏
        loss = self.nll_loss_gmm_direct(out['predicted_probability'], out['predicted_trajectory'], gt,
                                        center_gt_final_valid_idx, cls_weight=cls_w)
        # ── V3 置信度排序蒸馏(默认关:rank_loss_weight=0 → 与 V2 逐 bit 一致)──────────────
        # 目标:让 argmax(score) 那条 = 精度最优那条(top1_is_best↑),且不牺牲多样性。
        # 仅作用于 score;误差软标签 detach → 不对轨迹回归回传(A 方案:梯度经打分头进共享主干,
        # 不直接拉轨迹几何,多样性仍由 reg-WTA + best6 守)。逐样本 z-score 去绝对量级 → 一套 τ 城区/高速自适应。
        rank_w = float(self.config.get('rank_loss_weight', 0.0))
        if rank_w > 0.0:
            loss = loss + rank_w * self.rank_distill_loss(
                out['predicted_probability'], out['predicted_trajectory'], gt, center_gt_final_valid_idx)
        ade_w = float(self.config.get('best6_ade_weight', 0.0))
        fde_w = float(self.config.get('best6_fde_weight', 0.0))
        if ade_w > 0.0 or fde_w > 0.0:
            best_ade, best_fde = self.best_of_k_loss(out['predicted_trajectory'], gt, center_gt_final_valid_idx)
            loss = loss + ade_w * best_ade + fde_w * best_fde
        sparse_w = float(self.config.get('graph_sparsity_weight', 0.0))
        if sparse_w > 0.0 and 'graph_sparsity_loss' in out:
            loss = loss + sparse_w * out['graph_sparsity_loss']
        proto_w = float(self.config.get('prototype_entropy_weight', 0.0))
        if proto_w > 0.0 and 'prototype_entropy_loss' in out:
            loss = loss + proto_w * out['prototype_entropy_loss']
        return loss

    def best_of_k_loss(self, pred_trajs, gt_trajs, center_gt_final_valid_idx):
        gt_valid_mask = gt_trajs[..., -1].type_as(pred_trajs)
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
        valid_count = gt_valid_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ade = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) / valid_count
        best_ade = ade.min(dim=-1).values.mean()

        final_idx = center_gt_final_valid_idx.long().clamp(min=0, max=pred_trajs.shape[2] - 1)
        batch_idx = torch.arange(pred_trajs.shape[0], device=pred_trajs.device)
        pred_final = pred_trajs[batch_idx, :, final_idx, 0:2]
        gt_final = gt_trajs[batch_idx, final_idx, 0:2]
        fde = (pred_final - gt_final[:, None]).norm(dim=-1)
        best_fde = fde.min(dim=-1).values.mean()
        return best_ade, best_fde

    def rank_distill_loss(self, pred_scores, pred_trajs, gt_trajs, center_gt_final_valid_idx):
        """V3 置信度排序蒸馏:CE(scores, softmax(-zscore(err)/τ))。
        用真实误差把「6 条谁更准」的完整次序蒸馏进打分头 → argmax(score) 朝最优模态走。
        err 由轨迹与 GT 算出后 detach(纯标签,不回传轨迹几何);逐样本归一化让 τ 跨域一致
        (高速 6 条 FDE 挤成一簇、城区散开,z-score 后同量纲 → 一套 τ 通吃)。"""
        tau = float(self.config.get('rank_temperature', 1.0))
        err_metric = str(self.config.get('rank_err', 'ade'))
        norm = str(self.config.get('rank_norm', 'zscore'))
        gt_valid_mask = gt_trajs[..., -1].type_as(pred_trajs)                        # [B,T]
        dist = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)    # [B,K,T]
        if err_metric == 'fde':
            final_idx = center_gt_final_valid_idx.long().clamp(min=0, max=pred_trajs.shape[2] - 1)
            batch_idx = torch.arange(pred_trajs.shape[0], device=pred_trajs.device)
            err = dist[batch_idx, :, final_idx]                                      # [B,K]
        else:  # 'ade' —— 全时程,变道/刹车沿整条渐渐拉开,比终点 FDE 更判别(高速尤甚)
            valid = gt_valid_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)           # [B,1]
            err = (dist * gt_valid_mask[:, None, :]).sum(dim=-1) / valid             # [B,K]
        err = err.detach()                                                          # A 方案:标签不回传轨迹
        if norm == 'zscore':
            mu = err.mean(dim=-1, keepdim=True)
            sd = err.std(dim=-1, keepdim=True).clamp_min(1e-6)
            err = (err - mu) / sd
        target = torch.softmax(-err / tau, dim=-1)                                  # [B,K] 软标签:越准分越高
        logp = F.log_softmax(pred_scores, dim=-1)                                   # [B,K]
        return -(target * logp).sum(dim=-1).mean()

    def nll_loss_gmm_direct(self, pred_scores, pred_trajs, gt_trajs, center_gt_final_valid_idx,
                            pre_nearest_mode_idxs=None, cls_weight=1.0,
                            timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0),
                            rho_limit=0.5):
        """
        GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
        Written by Shaoshuai Shi

        Args:
            pred_scores (batch_size, num_modes):
            pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
            gt_trajs (batch_size, num_timestamps, 3):
            timestamp_loss_weight (num_timestamps):
        """
        if use_square_gmm:
            assert pred_trajs.shape[-1] == 3
        else:
            assert pred_trajs.shape[-1] == 5

        batch_size = pred_trajs.shape[0]

        gt_valid_mask = gt_trajs[..., -1]

        if pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = pre_nearest_mode_idxs
        else:
            distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

            nearest_mode_idxs = distance.argmin(dim=-1)
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
        res_trajs = gt_trajs[..., :2] - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        if use_square_gmm:
            log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                std1 * std2))  # (batch_size, num_timestamps)

        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

        loss_cls = (F.cross_entropy(input=pred_scores, target=nearest_mode_idxs, reduction='none'))

        return (reg_loss + cls_weight * loss_cls).mean()
