from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.base_model.base_model import BaseModel
from .modules import (
    AgentEncoder,
    CrosswalkEncoder,
    FutureEncoder,
    InitialDecoder,
    InteractionDecoder,
    LaneEncoder,
)


class GameFormerInputAdapter(nn.Module):
    """Convert UniTraj's common tensors to the official GameFormer representation."""

    LANE_TYPES = (1, 2, 3)
    CROSSWALK_TYPES = (18, 19)

    def __init__(self, config):
        super().__init__()
        self.num_lane_elements = int(config.gameformer_num_lane_elements)
        self.num_crosswalk_elements = int(config.gameformer_num_crosswalk_elements)
        self.lane_points = int(config.gameformer_lane_points)
        self.crosswalk_points = int(config.gameformer_crosswalk_points)

    @staticmethod
    def _impute_history(states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        states = states.clone()
        for t in range(states.size(-2) - 2, -1, -1):
            missing = ~valid[..., t]
            if missing.any():
                states[..., t, :] = torch.where(
                    missing.unsqueeze(-1), states[..., t + 1, :], states[..., t, :]
                )
                states[..., t, 0:2] = torch.where(
                    missing.unsqueeze(-1),
                    states[..., t + 1, 0:2] - states[..., t + 1, 3:5] * 0.1,
                    states[..., t, 0:2],
                )
        return states

    def _agent_states(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obj = inputs["obj_trajs"].float()
        mask = inputs["obj_trajs_mask"].bool()
        heading = torch.atan2(obj[..., 23], obj[..., 24]).unsqueeze(-1)
        # Carry the data-provided agent type one-hot (cols 6:9) straight through;
        # the encoders project it (nn.Linear) instead of re-deriving an id.
        type_onehot = obj[..., 6:9]
        states = torch.cat(
            [obj[..., 0:2], heading, obj[..., 25:27], obj[..., 3:6], type_onehot],
            dim=-1,
        )
        states = self._impute_history(states, mask)
        actor_valid = mask[..., -1]

        current_xy = states[:, :, -1, :2]
        dist = torch.linalg.norm(current_xy, dim=-1)
        partner_score = dist.masked_fill(~actor_valid, float("inf"))
        partner_score[:, 0] = float("inf")
        partner_idx = partner_score.argmin(dim=-1)
        has_partner = torch.isfinite(partner_score.gather(1, partner_idx[:, None]).squeeze(1))
        partner_idx = torch.where(has_partner, partner_idx, torch.zeros_like(partner_idx))

        batch_idx = torch.arange(obj.size(0), device=obj.device)
        partner = states[batch_idx, partner_idx]
        remaining_mask = actor_valid.clone()
        remaining_mask[:, 0] = False
        remaining_mask[batch_idx, partner_idx] = False
        remaining_dist = dist.masked_fill(~remaining_mask, float("inf"))
        remaining_idx = remaining_dist.argsort(dim=-1)
        remaining = states.gather(
            1, remaining_idx[:, :, None, None].expand(-1, -1, states.size(2), states.size(3))
        )
        remaining_valid = remaining_mask.gather(1, remaining_idx)
        remaining = remaining.masked_fill(~remaining_valid[:, :, None, None], 0.0)

        # The common pool contains 32 actors including the center target.
        # Keep one partner plus at most 30 additional background actors.
        neighbors = torch.cat([partner[:, None], remaining[:, : states.size(1) - 2]], dim=1)
        partner = partner.masked_fill(~has_partner[:, None, None], 0.0)
        neighbors[:, 0] = partner
        return states[:, 0], neighbors, partner_idx, has_partner

    def _select_map(
        self,
        polylines: torch.Tensor,
        point_mask: torch.Tensor,
        actor_xy: torch.Tensor,
        accepted_types: Tuple[int, ...],
        count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        map_type = polylines[..., 9:].argmax(dim=-1)
        feature_type = map_type[:, :, 0]
        type_valid = torch.zeros_like(feature_type, dtype=torch.bool)
        for accepted in accepted_types:
            type_valid |= feature_type == accepted
        feature_valid = point_mask.any(dim=-1) & type_valid
        centers = (polylines[..., :2] * point_mask[..., None]).sum(dim=-2)
        centers = centers / point_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        distance = torch.linalg.norm(centers - actor_xy[:, None], dim=-1)
        distance = distance.masked_fill(~feature_valid, float("inf"))
        indices = distance.argsort(dim=-1)[:, :count]
        selected = polylines.gather(
            1, indices[:, :, None, None].expand(-1, -1, polylines.size(2), polylines.size(3))
        )
        selected_mask = point_mask.gather(
            1, indices[:, :, None].expand(-1, -1, point_mask.size(2))
        )
        selected_valid = feature_valid.gather(1, indices)
        return selected, selected_mask & selected_valid[:, :, None]

    @staticmethod
    def _pad_points(
        tensor: torch.Tensor, mask: torch.Tensor, target_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = tensor[..., :target_points, :]
        mask = mask[..., :target_points]
        if tensor.size(-2) < target_points:
            pad = target_points - tensor.size(-2)
            tensor = F.pad(tensor, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad))
        return tensor, mask

    def _maps_for_actor(
        self, inputs: Dict[str, torch.Tensor], actor_xy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        roads = inputs["map_polylines"].float()
        road_mask = inputs["map_polylines_mask"].bool()
        lanes, lane_mask = self._select_map(
            roads, road_mask, actor_xy, self.LANE_TYPES, self.num_lane_elements
        )
        crosswalks, crosswalk_mask = self._select_map(
            roads, road_mask, actor_xy, self.CROSSWALK_TYPES, self.num_crosswalk_elements
        )

        lane_native = lanes.new_zeros(*lanes.shape[:-1], 30)
        lane_native[..., 0:2] = lanes[..., 0:2]
        lane_native[..., 2] = torch.atan2(lanes[..., 4], lanes[..., 3])
        # Carry the data-provided 20-dim map type one-hot (lanes cols 9:29) directly;
        # LaneEncoder.self_type projects it with nn.Linear (no argmax / id clamp).
        lane_native[..., 10:30] = lanes[..., 9:29]
        lane_native = lane_native.masked_fill(~lane_mask[..., None], 0.0)

        crosswalk_native = crosswalks.new_zeros(*crosswalks.shape[:-1], 3)
        crosswalk_native[..., 0:2] = crosswalks[..., 0:2]
        crosswalk_native[..., 2] = torch.atan2(crosswalks[..., 4], crosswalks[..., 3])
        crosswalk_native = crosswalk_native.masked_fill(~crosswalk_mask[..., None], 0.0)
        lane_native, lane_mask = self._pad_points(
            lane_native, lane_mask, self.lane_points
        )
        crosswalk_native, crosswalk_mask = self._pad_points(
            crosswalk_native, crosswalk_mask, self.crosswalk_points
        )
        return lane_native, lane_mask, crosswalk_native, crosswalk_mask

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ego, neighbors, partner_idx, has_partner = self._agent_states(inputs)
        partner = neighbors[:, 0]
        ego_maps = self._maps_for_actor(inputs, ego[:, -1, :2])
        partner_maps = self._maps_for_actor(inputs, partner[:, -1, :2])
        return {
            "ego_state": ego,
            "neighbors_state": neighbors,
            "map_lanes": torch.stack([ego_maps[0], partner_maps[0]], dim=1),
            "map_lanes_mask": torch.stack([ego_maps[1], partner_maps[1]], dim=1),
            "map_crosswalks": torch.stack([ego_maps[2], partner_maps[2]], dim=1),
            "map_crosswalks_mask": torch.stack([ego_maps[3], partner_maps[3]], dim=1),
            "partner_idx": partner_idx,
            "has_partner": has_partner,
        }


class Encoder(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6, map_segment_stride=10):
        super().__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.neighbors = neighbors_to_predict
        self.map_segment_stride = map_segment_stride
        self.agent_encoder = AgentEncoder()
        self.ego_encoder = AgentEncoder()
        self.lane_encoder = LaneEncoder()
        self.crosswalk_encoder = CrosswalkEncoder()
        attention_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            activation=F.gelu,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            attention_layer, layers, enable_nested_tensor=False
        )

    def segment_map(self, map_encoding, point_mask):
        stride = self.map_segment_stride
        batch, elements, points, dim = map_encoding.shape
        if points % stride != 0:
            raise ValueError(f"GameFormer map points ({points}) must be divisible by stride ({stride})")
        encoded = F.max_pool2d(
            map_encoding.permute(0, 3, 1, 2), kernel_size=(1, stride)
        )
        encoded = encoded.permute(0, 2, 3, 1).reshape(batch, -1, dim)
        valid = point_mask.reshape(batch, elements, points // stride, stride).any(dim=-1)
        return encoded, ~valid.reshape(batch, -1)

    def forward(self, inputs):
        ego = inputs["ego_state"]
        neighbors = inputs["neighbors_state"]
        actors = torch.cat([ego.unsqueeze(1), neighbors], dim=1)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([self.ego_encoder(ego)] + encoded_neighbors, dim=1)
        actors_mask = actors[:, :, -1, 8:11].sum(-1) == 0

        encoded_map_lanes = self.lane_encoder(inputs["map_lanes"])
        encoded_crosswalks = self.crosswalk_encoder(inputs["map_crosswalks"])
        encodings, masks = [], []
        num_predicted = self.neighbors + 1
        for i in range(num_predicted):
            lanes, lanes_mask = self.segment_map(
                encoded_map_lanes[:, i], inputs["map_lanes_mask"][:, i]
            )
            crosswalks, crosswalks_mask = self.segment_map(
                encoded_crosswalks[:, i], inputs["map_crosswalks_mask"][:, i]
            )
            fusion_input = torch.cat([encoded_actors, lanes, crosswalks], dim=1)
            mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
            all_masked = mask.all(dim=-1)
            mask[all_masked, 0] = False
            encodings.append(self.fusion_encoder(fusion_input, src_key_padding_mask=mask))
            masks.append(mask)
        return {
            "actors": actors,
            "encodings": torch.stack(encodings, dim=1),
            "masks": torch.stack(masks, dim=1),
        }


class Decoder(nn.Module):
    def __init__(self, modalities, future_len, neighbors_to_predict, levels=3):
        super().__init__()
        self.levels = levels
        self.neighbors = neighbors_to_predict
        future_encoder = FutureEncoder()
        self.initial_stage = InitialDecoder(modalities, neighbors_to_predict, future_len)
        self.interaction_stage = nn.ModuleList(
            [InteractionDecoder(future_encoder, future_len) for _ in range(levels)]
        )

    def forward(self, encoder_inputs):
        outputs = {}
        num_predicted = self.neighbors + 1
        current_states = encoder_inputs["actors"][:, :, -1]
        encodings, masks = encoder_inputs["encodings"], encoder_inputs["masks"]
        results = [
            self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i])
            for i in range(num_predicted)
        ]
        last_content = torch.stack([result[0] for result in results], dim=1)
        last_level = torch.stack([result[1] for result in results], dim=1)
        last_scores = torch.stack([result[2] for result in results], dim=1)
        outputs["level_0_interactions"] = last_level
        outputs["level_0_scores"] = last_scores

        for level, interaction_decoder in enumerate(self.interaction_stage, start=1):
            results = [
                interaction_decoder(
                    i,
                    current_states[:, :num_predicted],
                    last_level,
                    last_scores,
                    last_content[:, i],
                    encodings[:, i],
                    masks[:, i],
                )
                for i in range(num_predicted)
            ]
            last_content = torch.stack([result[0] for result in results], dim=1)
            last_level = torch.stack([result[1] for result in results], dim=1)
            last_scores = torch.stack([result[2] for result in results], dim=1)
            outputs[f"level_{level}_interactions"] = last_level
            outputs[f"level_{level}_scores"] = last_scores
        return outputs


class GameFormerCore(nn.Module):
    """Official interaction-prediction architecture with configurable horizons."""

    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            neighbors_to_predict=1,
            layers=int(config.encoder_layers),
            map_segment_stride=int(config.map_segment_stride),
        )
        self.decoder = Decoder(
            modalities=int(config.num_modes),
            future_len=int(config.future_len),
            neighbors_to_predict=1,
            levels=int(config.decoder_levels),
        )

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


class GameFormerModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.adapter = GameFormerInputAdapter(config)
        self.gameformer = GameFormerCore(config)

    @staticmethod
    def _joint_gmm_loss(trajectories, scores, ground_truth, valid_mask):
        metric_indices = [idx for idx in (29, 49, 79) if idx < trajectories.size(-2)]
        distance = torch.linalg.norm(
            trajectories[..., :2] - ground_truth[:, :, None, :, :2], dim=-1
        )
        valid = valid_mask[:, :, None].float()
        mean_distance = (distance * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        if metric_indices:
            endpoint_distance = (
                distance[..., metric_indices] * valid[..., metric_indices]
            ).sum(dim=-1)
        else:
            endpoint_distance = 0.0
        best_mode = (mean_distance + endpoint_distance).mean(dim=1).argmin(dim=-1)

        batch, agents = trajectories.shape[:2]
        batch_idx = torch.arange(batch, device=trajectories.device)[:, None]
        agent_idx = torch.arange(agents, device=trajectories.device)[None, :]
        best = trajectories[batch_idx, agent_idx, best_mode[:, None]]
        dx = best[..., 0] - ground_truth[..., 0]
        dy = best[..., 1] - ground_truth[..., 1]
        log_std_x = best[..., 2].clamp(0, 5)
        log_std_y = best[..., 3].clamp(0, 5)
        nll = log_std_x + log_std_y + 0.5 * (
            dx.square() / torch.exp(2 * log_std_x)
            + dy.square() / torch.exp(2 * log_std_y)
        )
        nll = (nll * valid_mask.float()).sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1)
        if metric_indices:
            nll = nll + (
                nll.new_tensor(0.0)
                + (
                    (
                        log_std_x[..., metric_indices]
                        + log_std_y[..., metric_indices]
                        + 0.5
                        * (
                            dx[..., metric_indices].square()
                            / torch.exp(2 * log_std_x[..., metric_indices])
                            + dy[..., metric_indices].square()
                            / torch.exp(2 * log_std_y[..., metric_indices])
                        )
                    )
                    * valid_mask[..., metric_indices].float()
                ).sum(dim=-1)
            )
        agent_valid = valid_mask.any(dim=-1)
        regression = (nll * agent_valid.float()).sum() / agent_valid.sum().clamp_min(1)

        joint_logits = (scores * agent_valid.unsqueeze(-1).float()).sum(dim=1)
        classification = F.cross_entropy(
            joint_logits, best_mode, label_smoothing=0.2
        )
        return regression + 2.0 * classification

    def _targets(self, inputs, partner_idx, has_partner):
        future = inputs["obj_trajs_future_state"][..., :2]
        future_mask = inputs["obj_trajs_future_mask"].bool()
        batch_idx = torch.arange(future.size(0), device=future.device)
        target = torch.stack([future[:, 0], future[batch_idx, partner_idx]], dim=1)
        target_mask = torch.stack(
            [future_mask[:, 0], future_mask[batch_idx, partner_idx]], dim=1
        )
        partner_current_valid = (
            inputs["obj_trajs_mask"][batch_idx, partner_idx, -1].bool() & has_partner
        )
        target_mask[:, 1] &= partner_current_valid[:, None]
        return target, target_mask

    def forward(self, batch):
        inputs = batch["input_dict"]
        native_inputs = self.adapter(inputs)
        levels = self.gameformer(native_inputs)
        target, target_mask = self._targets(
            inputs, native_inputs["partner_idx"], native_inputs["has_partner"]
        )
        loss = target.new_zeros(())
        for level in range(int(self.config.decoder_levels) + 1):
            loss = loss + self._joint_gmm_loss(
                levels[f"level_{level}_interactions"],
                levels[f"level_{level}_scores"],
                target,
                target_mask,
            )

        final_trajectories = levels[f"level_{self.config.decoder_levels}_interactions"]
        final_scores = levels[f"level_{self.config.decoder_levels}_scores"]
        score_weights = torch.stack(
            [
                torch.ones_like(native_inputs["has_partner"], dtype=final_scores.dtype),
                native_inputs["has_partner"].to(final_scores.dtype),
            ],
            dim=1,
        )
        joint_probability = F.softmax(
            (final_scores * score_weights.unsqueeze(-1)).sum(dim=1), dim=-1
        )
        output = {
            "predicted_trajectory": final_trajectories[:, 0, :, :, :2],
            "predicted_probability": joint_probability,
            "dataset_name": inputs["dataset_name"],
        }
        return output, loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        batch_size = len(batch["input_dict"]["center_objects_id"])
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.config.learning_rate_sched),
            gamma=float(self.config.lr_decay),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
