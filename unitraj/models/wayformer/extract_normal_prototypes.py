#!/usr/bin/env python3
"""Extract and cluster normal-driving Wayformer mode embeddings.

This builds the frozen normal driving reference dictionary used by RECON.  It intentionally uses the original Wayformer implementation
and a pretrained Wayformer checkpoint, not the experimental RECON-LP weights.

Outputs are written under artifacts/prototypes by default:
  - normal_mode_prototypes_k{K}.pt: cluster centers and metadata
  - normal_mode_samples_k{K}.npz: sampled embeddings and explanatory stats
  - normal_mode_clusters_k{K}.csv: per-cluster human-readable summary
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


UNITRAJ_ROOT = Path(__file__).resolve().parents[2]
if str(UNITRAJ_ROOT) not in sys.path:
    sys.path.insert(0, str(UNITRAJ_ROOT))

from datasets import build_dataset  # noqa: E402
from models.wayformer.wayformer import Wayformer  # noqa: E402


TYPE_NAME = {0: "UNSET", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Your trained Wayformer checkpoint.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--method-config", default="configs/method/wayformer.yaml")
    parser.add_argument("--output-dir", default="artifacts/prototypes")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--data-path", default=None, help="Override train/val data path with one ScenarioNet directory.")
    parser.add_argument("--max-scenes", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=3, help="Scene batch size before center-object flattening.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-agents-per-batch", type=int, default=100)
    parser.add_argument("--max-mode-samples", type=int, default=300000)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument("--kmeans-chunk", type=int, default=65536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-samples", action="store_true", default=True)
    parser.add_argument("--no-save-samples", action="store_false", dest="save_samples")
    parser.add_argument("--save-context", action="store_true", default=True)
    parser.add_argument("--no-save-context", action="store_false", dest="save_context")
    parser.add_argument("--context-agent-topk", type=int, default=32, help="Save nearest K non-target agents; -1 saves all.")
    parser.add_argument("--context-map-topk", type=int, default=128, help="Save nearest K map polylines; -1 saves all.")
    parser.add_argument("--save-pred-trajs", action="store_true", default=True)
    parser.add_argument("--no-save-pred-trajs", action="store_false", dest="save_pred_trajs")
    parser.add_argument("--save-decoder-tokens", action="store_true", default=True)
    parser.add_argument("--no-save-decoder-tokens", action="store_false", dest="save_decoder_tokens")
    parser.add_argument("--save-encoder-context", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return UNITRAJ_ROOT / path


def build_cfg(args):
    base = OmegaConf.load(resolve_path(args.config))
    method = OmegaConf.load(resolve_path(args.method_config))
    cfg = OmegaConf.merge(base, method)
    cfg.method = method
    cfg.method.model_name = "wayformer"
    cfg.model_name = "wayformer"
    cfg.onthefly = True
    cfg.use_cache = False
    cfg.store_data_in_memory = False
    cfg.load_num_workers = args.num_workers
    cfg.max_agents_per_batch = args.max_agents_per_batch
    cfg.train_batch_size = args.batch_size
    cfg.eval_batch_size = args.batch_size
    cfg.method.train_batch_size = args.batch_size
    cfg.method.eval_batch_size = args.batch_size
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.data_path is not None:
        if args.split == "train":
            cfg.train_data_path = [args.data_path]
        else:
            cfg.val_data_path = [args.data_path]
    if args.split == "train":
        cfg.max_data_num = [args.max_scenes]
        cfg.max_val_data_num = [None]
    else:
        cfg.max_data_num = [None]
        cfg.max_val_data_num = [args.max_scenes]
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wayformer(cfg, ckpt_path, device):
    model = Wayformer(OmegaConf.to_container(cfg, resolve=True))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[load] missing head: {missing[:8]}")
        if unexpected:
            print(f"[load] unexpected head: {unexpected[:8]}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_wayformer_internals(model, batch, device):
    inputs = batch["input_dict"]
    agents_in = inputs["obj_trajs"].to(device)
    agents_mask = inputs["obj_trajs_mask"].to(device)
    roads = inputs["map_polylines"].to(device)
    roads_mask = inputs["map_polylines_mask"].to(device)
    track_idx = inputs["track_index_to_predict"].to(device)

    ego_in = torch.gather(
        agents_in,
        1,
        track_idx.view(-1, 1, 1, 1).repeat(1, 1, *agents_in.shape[-2:]),
    ).squeeze(1)
    ego_mask = torch.gather(
        agents_mask,
        1,
        track_idx.view(-1, 1, 1).repeat(1, 1, agents_mask.shape[-1]),
    ).squeeze(1)

    agents_with_mask = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1).transpose(1, 2)
    ego_with_mask = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)
    roads_with_mask = torch.cat([roads, roads_mask.unsqueeze(-1)], dim=-1)

    bsz = ego_with_mask.size(0)
    num_agents = agents_with_mask.shape[2] + 1
    ego_tensor, other_tensor, opps_masks_agents, _ = model.process_observations(ego_with_mask, agents_with_mask)
    agents_tensor = torch.cat((ego_tensor.unsqueeze(2), other_tensor), dim=2)
    agents_emb = model.selu(model.agents_dynamic_encoder(agents_tensor))
    agents_emb = (
        agents_emb
        + model.agents_positional_embedding[:, :, :num_agents]
        + model.temporal_positional_embedding
    ).view(bsz, -1, model.d_k)

    road_pts_feats = model.selu(
        model.road_pts_lin(roads_with_mask[:, :model.max_num_roads, :, :model.map_attr]).view(bsz, -1, model.d_k)
    )
    mixed_input_features = torch.cat([agents_emb, road_pts_feats], dim=1)
    opps_masks_roads = (1.0 - roads_with_mask[:, :model.max_num_roads, :, -1]).to(torch.bool)
    mixed_input_masks = torch.cat([opps_masks_agents.view(bsz, -1), opps_masks_roads.view(bsz, -1)], dim=1)

    context = model.perceiver_encoder(mixed_input_features, mixed_input_masks)
    out_seq = model.perceiver_decoder(context)
    mode_emb = out_seq[:, :model.c]
    pred_trajs = model.output_model(mode_emb).reshape(bsz, model.c, model.T, -1)
    logits = model.prob_predictor(mode_emb).reshape(bsz, model.c)
    probs = F.softmax(logits, dim=-1)
    scene_pooled = out_seq[:, :model.num_queries_dec].mean(dim=1)
    return {
        "mode_emb": mode_emb,
        "scene_pooled": scene_pooled,
        "pred_trajs": pred_trajs,
        "logits": logits,
        "probs": probs,
        "decoder_tokens": out_seq,
        "encoder_context": context,
    }


def maneuver_labels(endpoint, disp):
    labels = []
    for (x, y), d in zip(endpoint, disp):
        if d < 2.0:
            labels.append("stop")
        elif abs(y) < max(2.0, 0.12 * d):
            labels.append("straight")
        elif y > 0:
            labels.append("left_bias")
        else:
            labels.append("right_bias")
    return np.asarray(labels, dtype=object)


def batch_stats(batch, outputs):
    inputs = batch["input_dict"]
    pred_xy = outputs["pred_trajs"][..., :2]
    bsz, modes, steps, _ = pred_xy.shape
    device = pred_xy.device
    dt = 0.1

    endpoint = pred_xy[:, :, -1].detach().cpu().float().numpy()
    disp = torch.linalg.norm(pred_xy[:, :, -1], dim=-1)
    speed = torch.linalg.norm(pred_xy[:, :, 1:] - pred_xy[:, :, :-1], dim=-1) / dt
    mean_speed = speed.mean(dim=-1)
    max_speed = speed.amax(dim=-1)
    lateral_abs = pred_xy[:, :, -1, 1].abs()

    first_delta = pred_xy[:, :, 0] - torch.zeros_like(pred_xy[:, :, 0])
    last_delta = pred_xy[:, :, -1] - pred_xy[:, :, -2]
    heading0 = torch.atan2(first_delta[..., 1], first_delta[..., 0])
    heading1 = torch.atan2(last_delta[..., 1], last_delta[..., 0])
    heading_change = torch.atan2(torch.sin(heading1 - heading0), torch.cos(heading1 - heading0)).abs()

    gt = inputs["center_gt_trajs"][..., :2].to(device)
    gt_mask = inputs["center_gt_trajs_mask"].to(device).to(pred_xy.dtype)
    dist = torch.linalg.norm(pred_xy - gt[:, None], dim=-1)
    valid_count = gt_mask.sum(dim=-1).clamp_min(1.0)
    ade = (dist * gt_mask[:, None]).sum(dim=-1) / valid_count[:, None]
    final_idx = inputs["center_gt_final_valid_idx"].to(device).long().clamp(0, steps - 1)
    batch_idx = torch.arange(bsz, device=device)
    pred_final = pred_xy[batch_idx, :, final_idx]
    gt_final = gt[batch_idx, final_idx]
    fde = torch.linalg.norm(pred_final - gt_final[:, None], dim=-1)
    nearest = ade.argmin(dim=-1)

    obj = inputs["obj_trajs"].to(device)
    obj_mask = inputs["obj_trajs_mask"].to(device)
    tr_idx = inputs["track_index_to_predict"].to(device)
    cur = obj[:, :, -1]
    cur_mask = obj_mask[:, :, -1].bool()
    ego = torch.gather(cur, 1, tr_idx.view(-1, 1, 1).repeat(1, 1, cur.shape[-1])).squeeze(1)
    ego_pos = ego[:, 0:2]
    ego_vel = ego[:, 25:27] if ego.shape[-1] >= 27 else torch.zeros_like(ego_pos)
    pos = cur[..., 0:2]
    vel = cur[..., 25:27] if cur.shape[-1] >= 27 else torch.zeros_like(pos)
    rel = pos - ego_pos[:, None]
    rel_vel = vel - ego_vel[:, None]
    rel_dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-3)
    closing = -(rel * rel_vel).sum(dim=-1) / rel_dist
    ego_onehot = torch.zeros_like(cur_mask)
    ego_onehot.scatter_(1, tr_idx.view(-1, 1), True)
    nb_mask = cur_mask & (~ego_onehot)
    min_agent_dist = rel_dist.masked_fill(~nb_mask, 1e6).amin(dim=-1)
    max_closing = closing.masked_fill(~nb_mask, -1e6).amax(dim=-1)
    ttc = torch.where(closing > 0.1, rel_dist / closing.clamp_min(0.1), torch.full_like(rel_dist, 1e3))
    min_ttc = ttc.masked_fill(~nb_mask, 1e6).amin(dim=-1)

    map_polys = inputs["map_polylines"].to(device)
    map_mask = inputs["map_polylines_mask"].to(device).bool()
    map_xy_sum = (map_polys[..., 0:2] * map_mask[..., None].to(map_polys.dtype)).sum(dim=2)
    map_count = map_mask.sum(dim=2).clamp_min(1).to(map_polys.dtype)
    map_center = map_xy_sum / map_count[..., None]
    map_node_mask = map_mask.any(dim=-1)
    min_map_dist = torch.linalg.norm(map_center, dim=-1).masked_fill(~map_node_mask, 1e6).amin(dim=-1)

    mode_idx = torch.arange(modes, device=device)[None, :].repeat(bsz, 1)
    is_nearest = mode_idx == nearest[:, None]
    return {
        "endpoint": endpoint.reshape(-1, 2),
        "disp": disp.detach().cpu().numpy().reshape(-1),
        "mean_speed": mean_speed.detach().cpu().numpy().reshape(-1),
        "max_speed": max_speed.detach().cpu().numpy().reshape(-1),
        "lateral_abs": lateral_abs.detach().cpu().numpy().reshape(-1),
        "heading_change": heading_change.detach().cpu().numpy().reshape(-1),
        "logit": outputs["logits"].detach().cpu().numpy().reshape(-1),
        "prob": outputs["probs"].detach().cpu().numpy().reshape(-1),
        "ade": ade.detach().cpu().numpy().reshape(-1),
        "fde": fde.detach().cpu().numpy().reshape(-1),
        "is_nearest_gt": is_nearest.detach().cpu().numpy().reshape(-1).astype(np.int8),
        "mode_index": mode_idx.detach().cpu().numpy().reshape(-1).astype(np.int16),
        "ego_speed": torch.linalg.norm(ego_vel, dim=-1).detach().cpu().numpy().repeat(modes),
        "min_agent_dist": min_agent_dist.detach().cpu().numpy().repeat(modes),
        "max_closing": max_closing.detach().cpu().numpy().repeat(modes),
        "min_ttc": min_ttc.detach().cpu().numpy().repeat(modes),
        "min_map_dist": min_map_dist.detach().cpu().numpy().repeat(modes),
    }


def repeat_metadata(batch, modes):
    inputs = batch["input_dict"]
    bsz = len(inputs["center_objects_id"])
    def rep(values):
        return np.asarray([str(v) for v in values for _ in range(modes)], dtype=object)
    obj_type = np.asarray(inputs["center_objects_type"]).reshape(bsz)
    return {
        "scenario_id": rep(inputs["scenario_id"]),
        "object_id": rep(inputs["center_objects_id"]),
        "dataset_name": rep(inputs["dataset_name"]),
        "object_type": obj_type.repeat(modes).astype(np.int16),
    }


def _append_arrays(store, item):
    for k, v in item.items():
        store.setdefault(k, []).append(v)


def _slice_mode_item(item, keep_modes):
    return {k: v[:keep_modes] for k, v in item.items()}


def _to_numpy(x, dtype=None):
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _gather_topk(values, indices):
    # values: [B, N, ...], indices: [B, K]
    view_shape = [indices.shape[0], indices.shape[1]] + [1] * (values.dim() - 2)
    expand_shape = list(indices.shape) + list(values.shape[2:])
    idx = indices.view(*view_shape).expand(*expand_shape)
    return torch.gather(values, 1, idx)


def _topk_context_indices(dist, valid_mask, topk):
    num_nodes = dist.shape[1]
    if topk < 0 or topk > num_nodes:
        topk = num_nodes
    score = dist.masked_fill(~valid_mask, 1e6)
    return torch.topk(score, k=topk, dim=1, largest=False).indices


def center_context(batch, outputs, args, center_start_id, keep_centers):
    inputs = batch["input_dict"]
    device = outputs["mode_emb"].device
    obj = inputs["obj_trajs"][:keep_centers].to(device)
    obj_mask = inputs["obj_trajs_mask"][:keep_centers].to(device).bool()
    map_polys = inputs["map_polylines"][:keep_centers].to(device)
    map_mask = inputs["map_polylines_mask"][:keep_centers].to(device).bool()
    track_idx = inputs["track_index_to_predict"][:keep_centers].to(device)
    gt = inputs["center_gt_trajs"][:keep_centers].to(device)
    gt_mask = inputs["center_gt_trajs_mask"][:keep_centers].to(device).bool()
    final_idx = inputs["center_gt_final_valid_idx"][:keep_centers].to(device).long()

    bsz, num_agents, hist_len, feat_dim = obj.shape
    batch_idx = torch.arange(bsz, device=device)
    ego_hist = obj[batch_idx, track_idx]
    ego_hist_mask = obj_mask[batch_idx, track_idx]
    cur = obj[:, :, -1]
    cur_mask = obj_mask[:, :, -1]
    ego_cur = ego_hist[:, -1]
    ego_pos = ego_cur[:, 0:2]
    ego_vel = ego_cur[:, 25:27] if feat_dim >= 27 else torch.zeros_like(ego_pos)

    pos = cur[..., 0:2]
    vel = cur[..., 25:27] if feat_dim >= 27 else torch.zeros_like(pos)
    rel = pos - ego_pos[:, None]
    rel_vel = vel - ego_vel[:, None]
    agent_dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-3)
    closing = -(rel * rel_vel).sum(dim=-1) / agent_dist
    ttc = torch.where(closing > 0.1, agent_dist / closing.clamp_min(0.1), torch.full_like(agent_dist, 1e3))
    ego_onehot = torch.zeros_like(cur_mask)
    ego_onehot.scatter_(1, track_idx.view(-1, 1), True)
    agent_valid = cur_mask & (~ego_onehot)
    agent_idx = _topk_context_indices(agent_dist, agent_valid, args.context_agent_topk)

    sel_agents = _gather_topk(obj, agent_idx)
    sel_agent_mask = _gather_topk(obj_mask.float(), agent_idx).bool()
    sel_agent_dist = torch.gather(agent_dist, 1, agent_idx)
    sel_agent_closing = torch.gather(closing, 1, agent_idx)
    sel_agent_ttc = torch.gather(ttc, 1, agent_idx)
    sel_agent_cur = _gather_topk(cur, agent_idx)

    map_xy_sum = (map_polys[..., 0:2] * map_mask[..., None].to(map_polys.dtype)).sum(dim=2)
    map_count = map_mask.sum(dim=2).clamp_min(1).to(map_polys.dtype)
    map_center = map_xy_sum / map_count[..., None]
    map_valid = map_mask.any(dim=-1)
    map_dist = torch.linalg.norm(map_center, dim=-1)
    map_idx = _topk_context_indices(map_dist, map_valid, args.context_map_topk)
    sel_map = _gather_topk(map_polys, map_idx)
    sel_map_mask = _gather_topk(map_mask.float(), map_idx).bool()
    sel_map_center = _gather_topk(map_center, map_idx)
    sel_map_dist = torch.gather(map_dist, 1, map_idx)

    center_ids = np.arange(center_start_id, center_start_id + keep_centers, dtype=np.int64)
    context = {
        "center_sample_id": center_ids,
        "track_index_to_predict": _to_numpy(track_idx, np.int32),
        "center_ego_hist": _to_numpy(ego_hist, np.float16),
        "center_ego_hist_mask": _to_numpy(ego_hist_mask),
        "center_gt_trajs": _to_numpy(gt, np.float16),
        "center_gt_trajs_mask": _to_numpy(gt_mask),
        "center_gt_final_valid_idx": _to_numpy(final_idx, np.int32),
        "context_agent_index": _to_numpy(agent_idx, np.int16),
        "context_agent_trajs": _to_numpy(sel_agents, np.float16),
        "context_agent_trajs_mask": _to_numpy(sel_agent_mask),
        "context_agent_current": _to_numpy(sel_agent_cur, np.float16),
        "context_agent_dist": _to_numpy(sel_agent_dist, np.float16),
        "context_agent_closing": _to_numpy(sel_agent_closing, np.float16),
        "context_agent_ttc": _to_numpy(torch.clamp(sel_agent_ttc, 0, 1e3), np.float16),
        "context_map_index": _to_numpy(map_idx, np.int16),
        "context_map_polylines": _to_numpy(sel_map, np.float16),
        "context_map_polylines_mask": _to_numpy(sel_map_mask),
        "context_map_center": _to_numpy(sel_map_center, np.float16),
        "context_map_dist": _to_numpy(sel_map_dist, np.float16),
    }
    if args.save_pred_trajs:
        context["pred_trajs"] = _to_numpy(outputs["pred_trajs"][:keep_centers], np.float16)
    if args.save_decoder_tokens:
        context["decoder_tokens"] = _to_numpy(outputs["decoder_tokens"][:keep_centers], np.float16)
    if args.save_encoder_context:
        context["encoder_context"] = _to_numpy(outputs["encoder_context"][:keep_centers], np.float16)
    return context


def append_sample(store, item, max_samples, rng):
    n_new = item["mode_emb"].shape[0]
    n_cur = store.get("_count", 0)
    keep = n_new
    if n_cur + n_new > max_samples:
        keep = max(0, max_samples - n_cur)
    if keep <= 0:
        return False
    if keep < n_new:
        idx = rng.choice(n_new, size=keep, replace=False)
        item = {k: v[idx] for k, v in item.items()}
    for k, v in item.items():
        store.setdefault(k, []).append(v)
    store["_count"] = n_cur + keep
    return store["_count"] < max_samples


def collect_embeddings(args, cfg, model, device):
    dataset = build_dataset(cfg, val=(args.split == "val"))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    store = {}
    context_store = {}
    center_offset = 0
    processed_modes = 0
    sampled_modes = 0
    for batch_idx, batch in enumerate(loader):
        outputs = extract_wayformer_internals(model, batch, device)
        bsz, modes, hidden = outputs["mode_emb"].shape
        remaining_centers = (args.max_mode_samples - sampled_modes) // modes
        keep_centers = min(bsz, max(0, remaining_centers))
        if keep_centers <= 0:
            break
        keep_modes = keep_centers * modes
        stats = batch_stats(batch, outputs)
        meta = repeat_metadata(batch, modes)
        mode_emb = outputs["mode_emb"].detach().cpu().float().reshape(-1, hidden).numpy()
        scene_pooled = outputs["scene_pooled"].detach().cpu().float().numpy().repeat(modes, axis=0)
        item = {
            "mode_emb": mode_emb,
            "scene_pooled": scene_pooled,
            **stats,
            **meta,
        }
        item["maneuver"] = maneuver_labels(item["endpoint"], item["disp"])
        item["center_sample_id"] = np.arange(center_offset, center_offset + bsz, dtype=np.int64).repeat(modes)
        _append_arrays(store, _slice_mode_item(item, keep_modes))
        if args.save_context:
            _append_arrays(context_store, center_context(batch, outputs, args, center_offset, keep_centers))
        center_offset += keep_centers
        sampled_modes += keep_modes
        processed_modes += mode_emb.shape[0]
        if batch_idx % 20 == 0:
            print(f"[extract] batch={batch_idx} sampled={sampled_modes} seen_modes={processed_modes}")
        if sampled_modes >= args.max_mode_samples:
            break
    arrays = {k: np.concatenate(v, axis=0) for k, v in store.items()}
    context_arrays = {k: np.concatenate(v, axis=0) for k, v in context_store.items()} if context_store else None
    print(f"[extract] final samples={arrays['mode_emb'].shape[0]} feature_dim={arrays['mode_emb'].shape[1]}")
    if context_arrays is not None:
        print(f"[extract] final centers={context_arrays['center_sample_id'].shape[0]} context_keys={len(context_arrays)}")
    return arrays, context_arrays


def kmeans_torch(features, k, iters, chunk, device, normalize=True, seed=42):
    x = torch.from_numpy(features).float()
    if normalize:
        x = F.normalize(x, dim=-1)
    n, dim = x.shape
    g = torch.Generator().manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    centers = x[init_idx].clone().to(device)
    x_dev = x.to(device)
    for it in range(iters):
        sums = torch.zeros(k, dim, device=device)
        counts = torch.zeros(k, device=device)
        inertia = 0.0
        for start in range(0, n, chunk):
            xb = x_dev[start:start + chunk]
            dist = torch.cdist(xb, centers)
            assign = dist.argmin(dim=1)
            inertia += dist[torch.arange(assign.numel(), device=device), assign].pow(2).sum().item()
            sums.index_add_(0, assign, xb)
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.float))
        empty = counts == 0
        if empty.any():
            repl = torch.randperm(n, generator=g)[:int(empty.sum())].to(device)
            sums[empty] = x_dev[repl]
            counts[empty] = 1.0
        centers = sums / counts[:, None].clamp_min(1.0)
        if normalize:
            centers = F.normalize(centers, dim=-1)
        print(f"[kmeans] iter={it + 1:02d}/{iters} inertia={inertia / n:.6f} min_count={int(counts.min().item())}")

    assignments = []
    for start in range(0, n, chunk):
        xb = x_dev[start:start + chunk]
        assignments.append(torch.cdist(xb, centers).argmin(dim=1).cpu())
    assignments = torch.cat(assignments).numpy().astype(np.int32)
    return centers.detach().cpu(), assignments


def cluster_summary(arrays, assignments, k):
    rows = []
    maneuvers = arrays["maneuver"]
    obj_types = arrays["object_type"]
    for cid in range(k):
        idx = np.flatnonzero(assignments == cid)
        if len(idx) == 0:
            continue
        man_vals, man_counts = np.unique(maneuvers[idx], return_counts=True)
        dominant_maneuver = str(man_vals[man_counts.argmax()])
        type_vals, type_counts = np.unique(obj_types[idx], return_counts=True)
        dominant_type = TYPE_NAME.get(int(type_vals[type_counts.argmax()]), str(type_vals[type_counts.argmax()]))
        rows.append({
            "cluster": cid,
            "count": int(len(idx)),
            "dominant_maneuver": dominant_maneuver,
            "dominant_object_type": dominant_type,
            "endpoint_x_mean": float(arrays["endpoint"][idx, 0].mean()),
            "endpoint_y_mean": float(arrays["endpoint"][idx, 1].mean()),
            "endpoint_y_std": float(arrays["endpoint"][idx, 1].std()),
            "disp_mean": float(arrays["disp"][idx].mean()),
            "mean_speed_mean": float(arrays["mean_speed"][idx].mean()),
            "max_speed_mean": float(arrays["max_speed"][idx].mean()),
            "old_prob_mean": float(arrays["prob"][idx].mean()),
            "old_logit_mean": float(arrays["logit"][idx].mean()),
            "ade_mean": float(arrays["ade"][idx].mean()),
            "fde_mean": float(arrays["fde"][idx].mean()),
            "nearest_gt_rate": float(arrays["is_nearest_gt"][idx].mean()),
            "ego_speed_mean": float(arrays["ego_speed"][idx].mean()),
            "min_agent_dist_mean": float(arrays["min_agent_dist"][idx].mean()),
            "max_closing_mean": float(arrays["max_closing"][idx].mean()),
            "min_ttc_mean": float(np.clip(arrays["min_ttc"][idx], 0, 100).mean()),
            "min_map_dist_mean": float(arrays["min_map_dist"][idx].mean()),
        })
    rows.sort(key=lambda r: r["cluster"])
    return rows


def save_outputs(args, cfg, arrays, context_arrays, centers, assignments, rows):
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"normal_mode_prototypes_k{args.clusters}"
    proto_path = out_dir / f"{prefix}.pt"
    sample_path = out_dir / f"normal_mode_samples_k{args.clusters}.npz"
    context_path = out_dir / f"normal_mode_context_k{args.clusters}.npz"
    csv_path = out_dir / f"normal_mode_clusters_k{args.clusters}.csv"
    meta_path = out_dir / f"normal_mode_metadata_k{args.clusters}.json"

    torch.save({
        "prototypes": centers.float(),
        "feature_type": "wayformer_decoder_mode_emb",
        "source_ckpt": str(resolve_path(args.ckpt)),
        "source_split": args.split,
        "num_clusters": args.clusters,
        "num_samples": int(arrays["mode_emb"].shape[0]),
        "normalize": bool(args.normalize),
        "assignments": torch.from_numpy(assignments),
        "cluster_summary": rows,
    }, proto_path)

    if args.save_samples:
        save_dict = {}
        for k, v in arrays.items():
            if v.dtype == object:
                save_dict[k] = v.astype(str)
            elif k in ("mode_emb", "scene_pooled"):
                save_dict[k] = v.astype(np.float16)
            else:
                save_dict[k] = v
        save_dict["cluster"] = assignments
        np.savez_compressed(sample_path, **save_dict)
    if args.save_context and context_arrays is not None:
        np.savez_compressed(context_path, **context_arrays)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["cluster", "count"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "source_ckpt": str(resolve_path(args.ckpt)),
        "config": str(resolve_path(args.config)),
        "method_config": str(resolve_path(args.method_config)),
        "split": args.split,
        "max_scenes": args.max_scenes,
        "max_mode_samples": args.max_mode_samples,
        "clusters": args.clusters,
        "kmeans_iters": args.kmeans_iters,
        "normalize": args.normalize,
        "save_samples": args.save_samples,
        "save_context": args.save_context,
        "context_agent_topk": args.context_agent_topk,
        "context_map_topk": args.context_map_topk,
        "save_pred_trajs": args.save_pred_trajs,
        "save_decoder_tokens": args.save_decoder_tokens,
        "save_encoder_context": args.save_encoder_context,
        "data_root": str(cfg.data_root),
        "train_data_path": list(cfg.train_data_path),
        "val_data_path": list(cfg.val_data_path),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[save] prototypes -> {proto_path}")
    print(f"[save] samples    -> {sample_path if args.save_samples else '(disabled)'}")
    print(f"[save] context    -> {context_path if args.save_context else '(disabled)'}")
    print(f"[save] summary    -> {csv_path}")
    print(f"[save] metadata   -> {meta_path}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.chdir(UNITRAJ_ROOT)
    device = torch.device(args.device)
    cfg = build_cfg(args)
    model = load_wayformer(cfg, resolve_path(args.ckpt), device)
    arrays, context_arrays = collect_embeddings(args, cfg, model, device)
    centers, assignments = kmeans_torch(
        arrays["mode_emb"], args.clusters, args.kmeans_iters, args.kmeans_chunk,
        device, normalize=args.normalize, seed=args.seed,
    )
    rows = cluster_summary(arrays, assignments, args.clusters)
    save_outputs(args, cfg, arrays, context_arrays, centers, assignments, rows)


if __name__ == "__main__":
    main()
