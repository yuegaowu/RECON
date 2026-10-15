"""Generate / auto-bootstrap MTR intention-point anchors (k-means of GT future endpoints).

MTR's decoder is anchor-based: NUM_ANCHORS per-type "intention points" act as query
anchors, in the agent-centric local frame (origin = predicted object at frame
past_len-1, x-axis along its heading). They must reflect the *training set's* endpoint
distribution, so Waymo anchors (turns/intersections) mismatch a highway set like highD.

The endpoint we cluster is *exactly* the quantity the loss matches against:
center_gt_trajs[last_valid, 0:2]  (see models/mtr/MTR.py:662-667). We obtain it by
reusing the real MTRDataset, so the coordinate frame is identical to training.

Two entry points:
  * CLI (standalone, regenerate explicitly):
        python tools/gen_intention_points.py method=MTR data_root=../data/highd \
            prediction_targets=all "max_data_num=[null,null]" \
            "+out_file=models/mtr/cluster_64_center_dict_highd_8s.pkl"
  * ensure_intention_points(cfg, build_dataset_fn): called from train.py before the
    model is built; if INTENTION_POINTS_FILE is 'auto'/null/missing, it clusters the
    training set, writes the pkl, persists the path back into MTR.yaml, and returns it.
"""
import os
import pickle
import re
import sys
import time

import numpy as np

NUM_ANCHORS = 64
TYPES = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
# center_objects_type is an int code (see models/mtr/MTR.py:23), not a string.
TYPE_DICT = {0: 'UNSET', 1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'CYCLIST'}
FALLBACK_PKL = 'models/mtr/cluster_64_center_dict_8s.pkl'  # Waymo anchors for sparse types
MTR_YAML = 'configs/method/MTR.yaml'
# Values that mean "please auto-generate" rather than "use this concrete file".
AUTO_SENTINELS = {'', 'auto', 'none', 'null'}


def kmeans(points, k, iters=200, seed=0):
    """Plain numpy k-means with k-means++ init. points: (N,2) -> centers: (k,2)."""
    rng = np.random.default_rng(seed)
    n = len(points)
    if n <= k:
        pad = points[rng.integers(0, n, size=k - n)] if n < k else points[:0]
        return np.concatenate([points, pad], axis=0)[:k]
    centers = [points[rng.integers(0, n)]]
    d2 = np.sum((points - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        probs = d2 / d2.sum()
        centers.append(points[rng.choice(n, p=probs)])
        d2 = np.minimum(d2, np.sum((points - centers[-1]) ** 2, axis=1))
    centers = np.stack(centers)
    for _ in range(iters):
        dist = np.linalg.norm(points[:, None, :] - centers[None], axis=2)
        assign = dist.argmin(axis=1)
        new = np.stack([
            points[assign == j].mean(axis=0) if np.any(assign == j) else centers[j]
            for j in range(k)
        ])
        if np.allclose(new, centers, atol=1e-4):
            centers = new
            break
        centers = new
    return centers


def collect_endpoints(dataset, log=print):
    """Iterate the dataset, return {type: (N,2) array of local-frame GT endpoints}."""
    buckets = {t: [] for t in TYPES}
    n_scenes = len(dataset)
    for i in range(n_scenes):
        for rec in dataset[i]:  # list of per-center-object dicts
            traj = np.asarray(rec['center_gt_trajs'])           # (T,4) x,y,vx,vy
            mask = np.asarray(rec['center_gt_trajs_mask']) > 0   # (T,)
            if not mask.any():
                continue
            last = np.where(mask)[0][-1]
            code = int(np.asarray(rec['center_objects_type']).reshape(-1)[0])
            ctype = TYPE_DICT.get(code, 'UNSET')
            if ctype in buckets:
                buckets[ctype].append(traj[last, 0:2])
        if (i + 1) % 200 == 0:
            tot = sum(len(v) for v in buckets.values())
            log(f'  scene {i + 1}/{n_scenes}  endpoints so far: {tot}')
    return {t: (np.stack(v) if v else np.zeros((0, 2))) for t, v in buckets.items()}


def cluster_to_dict(buckets, num_anchors=NUM_ANCHORS, fallback_pkl=FALLBACK_PKL, log=print):
    """buckets {type:(N,2)} -> {type:(num_anchors,2) float32}; sparse types keep fallback."""
    fallback = None  # Load only if a category needs fallback anchors.
    result = {}
    for t in TYPES:
        pts = buckets.get(t, np.zeros((0, 2)))
        if len(pts) >= num_anchors:
            log(f'[{t}] {len(pts)} endpoints -> k-means {num_anchors}')
            c = kmeans(pts, num_anchors).astype(np.float32)
            log(f'   x[{c[:,0].min():.1f},{c[:,0].max():.1f}] y[{c[:,1].min():.1f},{c[:,1].max():.1f}]')
            result[t] = c
        else:
            log(f'[{t}] only {len(pts)} endpoints (<{num_anchors}) -> keep fallback Waymo anchors')
            if fallback is None:
                if not fallback_pkl or not os.path.isfile(fallback_pkl):
                    raise ValueError(f'{t} has {len(pts)} endpoints; {num_anchors} are required. '
                                     f'Provide normal training data with enough examples per type '
                                     f'or your own compatible fallback anchors at {fallback_pkl}.')
                with open(fallback_pkl, 'rb') as f:
                    fallback = pickle.load(f)
            result[t] = np.asarray(fallback[t], dtype=np.float32)
    return result


def generate_to_file(dataset, out_file, log=print):
    """Cluster `dataset`'s GT endpoints and write the anchor pkl to out_file."""
    buckets = collect_endpoints(dataset, log=log)
    result = cluster_to_dict(buckets, log=log)
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(result, f)
    log(f'Wrote {out_file}')
    return out_file


# ─────────────────────────── train-time auto-bootstrap ───────────────────────────

def _dataset_tag(data_root):
    return re.sub(r'[^A-Za-z0-9]+', '_', os.path.basename(os.path.realpath(str(data_root)))).strip('_')


def _resolve_out_path(cfg, current):
    """Decide where the anchor file should live.
    - concrete missing path (ends .pkl)  -> generate exactly there
    - auto/null                          -> models/mtr/cluster_64_center_dict_<tag>_<H>s.pkl
    """
    if current and str(current).strip().lower() not in AUTO_SENTINELS and str(current).endswith('.pkl'):
        return str(current)
    tag = _dataset_tag(cfg.get('data_root', 'dataset'))
    horizon = max(1, int(cfg.get('future_len', 80)) // 10)
    return f'models/mtr/cluster_64_center_dict_{tag}_{horizon}s.pkl'


def _set_intention_file(cfg, path, decoder_node='MOTION_DECODER'):
    """Point both the merged-top-level and method decoder nodes at `path`.

    decoder_node is the cfg key that holds INTENTION_POINTS_FILE: MTR uses
    'MOTION_DECODER', BeTopNet uses 'DECODER'.
    """
    for node_owner in (cfg, cfg.get('method', None)):
        node = getattr(node_owner, decoder_node, None) if node_owner is not None else None
        if node is not None:
            node['INTENTION_POINTS_FILE'] = path


def _write_back_yaml(path, data_root, yaml_path=MTR_YAML):
    """Persist the resolved INTENTION_POINTS_FILE into MTR.yaml (so future runs reuse it)."""
    if not os.path.exists(yaml_path):
        return
    with open(yaml_path) as f:
        text = f.read()
    new_line = (f"  INTENTION_POINTS_FILE: '{path}'"
                f"  # auto-generated from {data_root} (delete file or set to 'auto' to regenerate)")
    new_text, n = re.subn(r'^\s*INTENTION_POINTS_FILE:.*$', new_line, text, count=1, flags=re.M)
    if n:
        with open(yaml_path, 'w') as f:
            f.write(new_text)
        print(f'[intention] updated {yaml_path} -> {path}')


def ensure_intention_points(cfg, build_dataset_fn, log=print, yaml_path=MTR_YAML,
                            decoder_node='MOTION_DECODER'):
    """If the intention-point file is 'auto'/null/missing, cluster the training set,
    save the pkl, persist the path into the method yaml, and update cfg in place. No-op
    otherwise.

    decoder_node selects which cfg node holds INTENTION_POINTS_FILE ('MOTION_DECODER'
    for MTR, 'DECODER' for BeTopNet); yaml_path is the method yaml to write the resolved
    path back into. The clustering itself is model-agnostic (k-means over the training
    set's GT future endpoints), so MTR-family decoders share this path.

    Returns the resolved path. DDP-safe: only LOCAL_RANK 0 clusters; other ranks wait.
    """
    dec = getattr(cfg, decoder_node, None)
    if dec is None:
        return None
    current = dec.get('INTENTION_POINTS_FILE', None)
    needs = (current is None
             or str(current).strip().lower() in AUTO_SENTINELS
             or not os.path.exists(str(current)))
    if not needs:
        return str(current)  # concrete file already present — normal path, do nothing

    out = _resolve_out_path(cfg, current)
    is_rank0 = os.environ.get('LOCAL_RANK', '0') == '0'

    if os.path.exists(out):
        log(f'[intention] reusing existing {out}')
    elif is_rank0:
        log(f'[intention] no anchor file for "{current}"; clustering training set -> {out}')
        ds = build_dataset_fn(cfg)  # MTRDataset over the training split
        generate_to_file(ds, out, log=log)
        _write_back_yaml(out, cfg.get('data_root', 'dataset'), yaml_path=yaml_path)
    else:
        # non-rank0: rank0 produces the file before DDP subprocesses start training,
        # so this normally returns immediately; poll defensively just in case.
        deadline = time.time() + 1800
        while not os.path.exists(out) and time.time() < deadline:
            time.sleep(5)

    _set_intention_file(cfg, out, decoder_node=decoder_node)
    return out


# ─────────────────────────────────── CLI ────────────────────────────────────────

def _main():
    import hydra
    from omegaconf import OmegaConf
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from datasets import build_dataset

    @hydra.main(version_base=None, config_path='../configs', config_name='config')
    def run(cfg):
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, cfg.method)
        out_file = cfg.get('out_file', None) or _resolve_out_path(cfg, None)
        cfg.onthefly = True  # read pkls directly, no h5 cache build
        print(f'Building MTR training dataset from {cfg.data_root} ...')
        ds = build_dataset(cfg, val=False)
        print(f'  {len(ds)} scenes')
        generate_to_file(ds, out_file)
        print(f'\nSet MTR.yaml -> MOTION_DECODER.INTENTION_POINTS_FILE: {out_file}, then retrain.')

    run()


if __name__ == '__main__':
    _main()
