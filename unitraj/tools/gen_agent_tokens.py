"""Generate / auto-bootstrap SMART agent motion-token vocabulary (k-means of local-frame 0.5s segments).

SMART is a GPT-style discrete model: it does NOT regress coordinates, it classifies each
0.5s (shift=5 @10Hz) move into one of `token_size` motion tokens. The achievable minADE is
therefore capped by how well that fixed codebook tiles the *training set's* real motion — so a
Waymo-derived vocab (urban, mean ~28 km/h, hard cap ~129 km/h) underfits highD highway speeds
(100-140 km/h) and floors minADE at ~4 m no matter the learning rate. This tool re-fits the
codebook on the actual training set.

Token format (reverse-engineered from tokens/cluster_frame_5_2048.pkl, must be reproduced exactly):
  dict{'token','traj','token_all'} each = {cat: array}, cat in {'veh','ped','cyc'}
    traj[cat]      (K, 6, 3)     cluster-center 0.5s segment [x, y, heading] in LOCAL frame
                                 (frame 0 forced to origin, heading 0)
    token_all[cat] (K, 6, 4, 2)  per-frame 4-corner box = cal_polygon_contour(traj, cat_box)
    token[cat]     (K, 4, 2)     the segment END contour = token_all[:, -1]  (the match template)
  Verified: token_all[f] == cal_polygon_contour(traj[f], w, L) exactly, veh box (w=2.0, L=4.8).

Two entry points (mirrors tools/gen_intention_points.py for MTR):
  * CLI (regenerate explicitly):
        python tools/gen_agent_tokens.py method=SMART data_root=../data/highd \
            prediction_targets=all "max_data_num=[null,null]" \
            "+out_file=models/smart/tokens/cluster_frame_5_2048_highd.pkl"
  * ensure_agent_tokens(cfg, build_dataset_fn): called from train.py before the model is built;
    if agent_token_path is 'auto'/null/missing, it clusters the training set, writes the pkl,
    persists the path back into SMART.yaml, and returns it. No-op once a concrete file exists.
"""
import os
import pickle
import re
import sys
import time

import numpy as np

SHIFT = 5                          # 5 frames @10Hz = 0.5s per token (== TokenProcessor.shift)
SEG_LEN = SHIFT + 1                # 6 points per token segment
CATS = ['veh', 'ped', 'cyc']
# track_infos object_type code -> SMART category (1=vehicle, 2=pedestrian, 3=cyclist; see types.py)
TYPE_TO_CAT = {1: 'veh', 2: 'ped', 3: 'cyc'}
# per-category box (width, length); matches match_token's shift>2 branch in smart_utils.py
CAT_BOX = {'veh': (2.0, 4.8), 'cyc': (1.0, 2.0), 'ped': (1.0, 1.0)}
# trajs column layout (N, T, 10): [x, y, z, l, w, h, heading, vx, vy, valid]
COL_X, COL_Y, COL_HEAD, COL_VALID = 0, 1, 6, 9
FALLBACK_PKL = 'models/smart/tokens/cluster_frame_5_2048.pkl'  # Waymo vocab, kept for sparse types
SMART_YAML = 'configs/method/SMART.yaml'
AUTO_SENTINELS = {'', 'auto', 'none', 'null'}
DEFAULT_MAX_SEG_PER_CAT = 200000   # reservoir cap per category (k-means input size)


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _circmean(angles, axis=0):
    """Circular mean of angles along `axis`."""
    return np.arctan2(np.sin(angles).mean(axis=axis), np.cos(angles).mean(axis=axis))


def _tangent_heading(xy):
    """Heading per frame from the path tangent (central diff); frame 0 = 0. xy: (T,2) -> (T,)."""
    d = np.zeros_like(xy)
    d[1:-1] = xy[2:] - xy[:-2]
    d[-1] = xy[-1] - xy[-2]
    h = np.arctan2(d[:, 1], d[:, 0])
    h[0] = 0.0
    return h


def cal_polygon_contour_vec(x, y, theta, width, length):
    """Vectorized cal_polygon_contour (corner order [LF, RF, RB, LB], matches smart.py).
    x, y, theta: broadcastable arrays -> contour (..., 4, 2)."""
    x, y, theta = np.asarray(x), np.asarray(y), np.asarray(theta)
    cos, sin = np.cos(theta), np.sin(theta)
    hl, hw = 0.5 * length, 0.5 * width
    lf = np.stack([x + hl * cos - hw * sin, y + hl * sin + hw * cos], axis=-1)
    rf = np.stack([x + hl * cos + hw * sin, y + hl * sin - hw * cos], axis=-1)
    rb = np.stack([x - hl * cos + hw * sin, y - hl * sin - hw * cos], axis=-1)
    lb = np.stack([x - hl * cos - hw * sin, y - hl * sin + hw * cos], axis=-1)
    return np.stack([lf, rf, rb, lb], axis=-2)


def collect_segments(dataset, max_per_cat=DEFAULT_MAX_SEG_PER_CAT, max_scenes=None, log=print):
    """Iterate the (on-the-fly) training dataset; return {cat: (M,6,3)} local-frame 0.5s segments.

    Each segment = a non-overlapping [s, s+SHIFT] window (all 6 frames valid), re-anchored so
    frame 0 is at the origin with heading 0 — exactly the frame in which SMART tokens live.
    Reservoir-samples to max_per_cat per category to bound k-means input size."""
    from scenarionet.common_utils import read_scenario

    buckets = {c: [] for c in CATS}
    counts = {c: 0 for c in CATS}
    rng = np.random.default_rng(0)
    idx_list = list(getattr(dataset, 'onthefly_index', []))
    if not idx_list:
        raise RuntimeError('[agent_tokens] dataset has empty onthefly_index — build it with onthefly=True')
    n = len(idx_list) if max_scenes is None else min(len(idx_list), int(max_scenes))

    for si in range(n):
        data_path, scene_name = idx_list[si]
        try:
            dataset.starting_frame = dataset.onthefly_starting_frames[data_path]
            scenario = read_scenario(data_path, dataset.onthefly_mappings[data_path], scene_name)
            info = dataset.preprocess(scenario)
        except Exception:
            continue
        if not info or 'track_infos' not in info:
            continue
        ti = info['track_infos']
        trajs = np.asarray(ti['trajs'], dtype=np.float32)
        otype = np.asarray(ti['object_type'])
        if trajs.ndim != 3 or trajs.shape[0] == 0:
            continue
        x, y = trajs[..., COL_X], trajs[..., COL_Y]
        hd, val = trajs[..., COL_HEAD], trajs[..., COL_VALID] > 0
        N, T, _ = trajs.shape
        for a in range(N):
            cat = TYPE_TO_CAT.get(int(otype[a]), None)
            if cat is None:
                continue
            va = val[a]
            for s in range(0, T - SHIFT, SHIFT):
                e = s + SHIFT
                if not va[s:e + 1].all():
                    continue
                px, py, ph = x[a, s:e + 1], y[a, s:e + 1], hd[a, s:e + 1]
                th0 = ph[0]
                c, sn = np.cos(-th0), np.sin(-th0)
                dx, dy = px - px[0], py - py[0]
                seg = np.stack([c * dx - sn * dy, sn * dx + c * dy, _wrap(ph - th0)], axis=-1)
                if counts[cat] < max_per_cat:
                    buckets[cat].append(seg.astype(np.float32))
                else:
                    j = rng.integers(0, counts[cat] + 1)
                    if j < max_per_cat:
                        buckets[cat][j] = seg.astype(np.float32)
                counts[cat] += 1
        if (si + 1) % 200 == 0:
            log(f'  scene {si + 1}/{n}  segments: ' + ', '.join(f'{c}={counts[c]}' for c in CATS))

    log('[agent_tokens] total segments seen: ' + ', '.join(f'{c}={counts[c]}' for c in CATS))
    return {c: (np.stack(v) if v else np.zeros((0, SEG_LEN, 3), np.float32)) for c, v in buckets.items()}


def cluster_to_dict(buckets, token_size, fallback_pkl=FALLBACK_PKL, log=print):
    """buckets {cat:(M,6,3)} -> {'token','traj','token_all'} vocab dict.
    Categories with < token_size segments keep the fallback (Waymo) vocab for that category."""
    from sklearn.cluster import MiniBatchKMeans

    fb = None  # Load only if a category needs fallback tokens.
    token, traj, token_all = {}, {}, {}
    for cat in CATS:
        segs = buckets.get(cat, np.zeros((0, SEG_LEN, 3), np.float32))
        if len(segs) >= token_size:
            log(f'[{cat}] {len(segs)} segments -> MiniBatchKMeans k={token_size}')
            feat = segs[:, :, :2].reshape(len(segs), -1)  # (M, 12) local positions
            km = MiniBatchKMeans(n_clusters=token_size, random_state=0, batch_size=4096,
                                 n_init=3, max_iter=300, max_no_improvement=50)
            lab = km.fit_predict(feat)
            xy = km.cluster_centers_.reshape(token_size, SEG_LEN, 2).astype(np.float32)
            head = np.zeros((token_size, SEG_LEN), np.float32)
            for k in range(token_size):
                m = segs[lab == k]
                head[k] = _circmean(m[:, :, 2], axis=0) if len(m) else _tangent_heading(xy[k])
            xy[:, 0, :] = 0.0          # frame 0 anchored at origin
            head[:, 0] = 0.0           # heading 0 at anchor
            tr = np.concatenate([xy, head[..., None]], axis=-1).astype(np.float32)   # (K,6,3)
            w, L = CAT_BOX[cat]
            ta = cal_polygon_contour_vec(tr[..., 0], tr[..., 1], tr[..., 2], w, L).astype(np.float32)
            traj[cat], token_all[cat], token[cat] = tr, ta, ta[:, -1].astype(np.float64)
            disp = np.linalg.norm(xy[:, -1], axis=1)
            log(f'   move(m/0.5s): max={disp.max():.2f} mean={disp.mean():.2f} '
                f'p99={np.percentile(disp, 99):.2f}  -> max speed {disp.max() / 0.5:.1f} m/s')
        else:
            log(f'[{cat}] only {len(segs)} segments (<{token_size}) -> keep fallback Waymo vocab')
            if fb is None:
                if not fallback_pkl or not os.path.isfile(fallback_pkl):
                    raise ValueError(f'{cat} has {len(segs)} segments; {token_size} are required. '
                                     f'Provide normal training data with enough examples per category '
                                     f'or your own compatible fallback tokens at {fallback_pkl}.')
                with open(fallback_pkl, 'rb') as f:
                    fb = pickle.load(f)
            traj[cat] = np.asarray(fb['traj'][cat])
            token_all[cat] = np.asarray(fb['token_all'][cat])
            token[cat] = np.asarray(fb['token'][cat])
    return {'token': token, 'traj': traj, 'token_all': token_all}


def generate_to_file(dataset, out_file, token_size, max_per_cat=DEFAULT_MAX_SEG_PER_CAT,
                     max_scenes=None, log=print):
    """Cluster `dataset`'s motion segments and write the vocab pkl to out_file."""
    buckets = collect_segments(dataset, max_per_cat=max_per_cat, max_scenes=max_scenes, log=log)
    vocab = cluster_to_dict(buckets, token_size, log=log)
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(vocab, f)
    log(f'[agent_tokens] wrote {out_file}')
    return out_file


# ─────────────────────────── train-time auto-bootstrap ───────────────────────────

def _dataset_tag(data_root):
    base = os.path.basename(os.path.realpath(str(data_root)))
    return re.sub(r'[^A-Za-z0-9]+', '_', base).strip('_') or 'dataset'


def _token_size(cfg):
    try:
        return int(cfg.Model.decoder.token_size)
    except Exception:
        try:
            return int(cfg.method.Model.decoder.token_size)
        except Exception:
            return 2048


def _resolve_out_path(cfg, current, token_size):
    """concrete missing .pkl -> generate there; auto/null -> tokens/cluster_frame_5_<size>_<tag>.pkl"""
    if current and str(current).strip().lower() not in AUTO_SENTINELS and str(current).endswith('.pkl'):
        return str(current)
    tag = _dataset_tag(cfg.get('data_root', 'dataset'))
    return f'models/smart/tokens/cluster_frame_5_{token_size}_{tag}.pkl'


def _set_token_path(cfg, path):
    """Point both the merged-top-level and method nodes at `path`."""
    for owner in (cfg, cfg.get('method', None)):
        if owner is not None:
            owner['agent_token_path'] = path


def _write_back_yaml(path, data_root, yaml_path=SMART_YAML):
    """Persist the resolved agent_token_path into SMART.yaml (replace existing key, else append)."""
    if not os.path.exists(yaml_path):
        return
    with open(yaml_path) as f:
        text = f.read()
    new_line = (f"agent_token_path: '{path}'"
                f"  # auto-generated from {data_root} (delete file or set to 'auto'/null to regenerate)")
    new_text, n = re.subn(r'^\s*agent_token_path:.*$', new_line, text, count=1, flags=re.M)
    if n == 0:
        new_text = text.rstrip('\n') + '\n\n' + new_line + '\n'
    with open(yaml_path, 'w') as f:
        f.write(new_text)
    print(f'[agent_tokens] updated {yaml_path} -> {path}')


def ensure_agent_tokens(cfg, build_dataset_fn, log=print, yaml_path=SMART_YAML):
    """If SMART's agent_token_path is 'auto'/null/missing, cluster the training set, save the pkl,
    persist the path into SMART.yaml, and update cfg in place. No-op when a concrete file exists.

    Returns the resolved path. DDP-safe: only LOCAL_RANK 0 clusters; other ranks wait.

    NOTE: switching vocab invalidates any pre-baked SMART token h5 cache (cache_path) — rebuild it
    (run once with onthefly=True, or point cache_path at a fresh dir) so tokens match the new vocab.
    """
    current = cfg.get('agent_token_path', None)
    needs = (current is None
             or str(current).strip().lower() in AUTO_SENTINELS
             or not os.path.exists(str(current)))
    if not needs:
        return str(current)

    token_size = _token_size(cfg)
    out = _resolve_out_path(cfg, current, token_size)
    is_rank0 = os.environ.get('LOCAL_RANK', '0') == '0'

    if os.path.exists(out):
        log(f'[agent_tokens] reusing existing {out}')
    elif is_rank0:
        log(f'[agent_tokens] no vocab for "{current}"; clustering training set -> {out}')
        ds = build_dataset_fn(cfg)
        generate_to_file(ds, out, token_size=token_size, log=log)
        _write_back_yaml(out, cfg.get('data_root', 'dataset'), yaml_path=yaml_path)
    else:
        deadline = time.time() + 3600
        while not os.path.exists(out) and time.time() < deadline:
            time.sleep(5)

    _set_token_path(cfg, out)
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
        token_size = _token_size(cfg)
        out_file = cfg.get('out_file', None) or _resolve_out_path(cfg, None, token_size)
        cfg.onthefly = True  # read pkls directly, no h5 cache build
        print(f'Building SMART training dataset from {cfg.data_root} ...')
        ds = build_dataset(cfg, val=False)
        generate_to_file(ds, out_file, token_size=token_size,
                         max_scenes=cfg.get('max_scenes', None))
        print(f'\nSet SMART.yaml -> agent_token_path: {out_file}, rebuild token cache, then retrain.')

    run()


if __name__ == '__main__':
    _main()
