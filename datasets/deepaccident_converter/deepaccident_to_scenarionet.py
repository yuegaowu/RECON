"""
Convert a single DeepAccident scenario → ScenarioNet / MetaDrive pkl dict.

Coordinate handling
-------------------
- Label positions are in the *lidar* frame; calib provides ``ego_to_world``
  (4×4) to map them into the CARLA world frame.
- Velocities are recomputed from world-frame positions via central
  differencing (DA label vx/vy fields are unreliable for non-ego agents —
  sign-flipped relative to true position motion). This makes velocity
  frame-agnostic and consistent across ego/non-ego tracks.
- Headings are corrected:  world_heading = lidar_heading + ego_yaw
  where ego_yaw = arctan2(R[1,0], R[0,0]).
- Ego position = translation component of ``ego_to_world``.
- CARLA world is right-handed (x→forward, y→right, z→up) — same as MetaDrive.
"""

import os, pickle
from collections import defaultdict

import numpy as np

from .deepaccident_types import (
    CARLA_TO_METADRIVE_OBJECT,
    DATASET_VERSION,
    AGENT_FILTER_MIN_VALID_FRAMES,
    COORDINATE,
    FRAME_RATE_HZ,
)

# Default ego dimensions (compact sedan) when not available from labels
DEFAULT_EGO_DIMS = (4.3, 1.8, 1.5)


def _rotation_yaw(R):
    """Extract yaw angle from a 3×3 or 4×4 rotation matrix."""
    return np.arctan2(R[1, 0], R[0, 0])


def _velocity_from_positions(pos_xy: np.ndarray, valid: np.ndarray,
                             dt: float) -> np.ndarray:
    """Central-difference velocity from world-frame xy positions, gap-aware.

    Returns ``(T, 2)`` float32. Invalid frames stay zero; valid frames at
    track boundaries use forward/backward differencing to neighbors.
    """
    T = pos_xy.shape[0]
    v = np.zeros((T, 2), dtype=np.float32)
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return v
    xy = pos_xy[idx]
    t_grid = idx.astype(np.float32) * dt
    if len(idx) >= 3:
        v[idx[1:-1]] = ((xy[2:] - xy[:-2]) /
                        (t_grid[2:] - t_grid[:-2])[:, None]).astype(np.float32)
    v[idx[0]]  = ((xy[1]  - xy[0])  / (t_grid[1]  - t_grid[0])).astype(np.float32)
    v[idx[-1]] = ((xy[-1] - xy[-2]) / (t_grid[-1] - t_grid[-2])).astype(np.float32)
    return v


def _load_calib_and_label(label_dir, calib_dir):
    """
    Load all label and calib files, sorted by frame number.

    Parameters
    ----------
    label_dir : str   e.g. ``.../ego_vehicle/label/Town03_.../``
    calib_dir : str   e.g. ``.../ego_vehicle/calib/Town03_.../``

    Returns
    -------
    label_paths : list of str
    calib_paths : list of str
    ns : list of int   frame numbers
    """
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f'Missing label dir: {label_dir}')

    label_files = sorted([
        f for f in os.listdir(label_dir) if f.endswith('.txt')
    ])
    label_info = []
    for f in label_files:
        stem = f.replace('.txt', '')
        try:
            frame_num = int(stem.split('_')[-1])
        except ValueError:
            frame_num = len(label_info)
        label_info.append((frame_num, f))
    label_info.sort(key=lambda x: x[0])

    calib_files_map = {}
    if os.path.isdir(calib_dir):
        for f in os.listdir(calib_dir):
            if f.endswith('.pkl'):
                stem = f.replace('.pkl', '')
                try:
                    fn = int(stem.split('_')[-1])
                except ValueError:
                    continue
                calib_files_map[fn] = os.path.join(calib_dir, f)

    label_paths = [os.path.join(label_dir, f) for _, f in label_info]
    calib_paths = [calib_files_map.get(n, None) for n, _ in label_info]
    ns = [n for n, _ in label_info]
    return label_paths, calib_paths, ns


def _load_meta(meta_path):
    """Parse a meta .txt file; return dict with keys below."""
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path) as f:
        lines = f.readlines()

    meta = {}
    # Line 0: <weather> <ego_id> <ego_type> <other_id> <other_type> <dist> <pos> <collision_frame>
    fields0 = lines[0].strip().split()
    meta['weather'] = fields0[0]
    try:
        meta['ego_id'] = int(fields0[1])
    except (ValueError, IndexError):
        meta['ego_id'] = None
    meta['ego_type'] = fields0[2] if len(fields0) > 2 else 'car'
    try:
        meta['other_id'] = int(fields0[3]) if len(fields0) > 3 else None
    except ValueError:
        meta['other_id'] = None
    meta['other_type'] = fields0[4] if len(fields0) > 4 else 'car'
    try:
        meta['collision_frame'] = int(fields0[-1]) if len(fields0) >= 8 else None
    except ValueError:
        meta['collision_frame'] = None

    for line in lines[1:]:
        line = line.strip()
        if line.startswith('colliding agents:'):
            roles = line.split(':')[1].strip().split()
            meta['colliding_roles'] = roles  # list of 2 role names, e.g. ['ego', 'other']
        elif line.startswith('agents id:'):
            ids = line.split(':')[1].strip().split()
            meta['agents_id'] = [int(x) for x in ids]
        elif line.startswith('road_type:'):
            meta['road_type'] = line.split(':')[1].strip()
        elif line.startswith('another_vehicle_spawn_side:'):
            meta['another_vehicle_spawn_side'] = line.split(':')[1].strip()
        elif line.startswith('ego_vehicle_direction:'):
            meta['ego_vehicle_direction'] = line.split(':')[1].strip()
        elif line.startswith('other_vehicle_direction:'):
            meta['other_vehicle_direction'] = line.split(':')[1].strip()
    return meta


def _source_frame_to_local(source_frame, frame_ns):
    """Map a DeepAccident source frame number onto the local track index."""
    if source_frame is None:
        return None
    matches = [index for index, value in enumerate(frame_ns)
               if int(value) == int(source_frame)]
    if len(matches) != 1:
        raise ValueError(
            f'collision source frame {source_frame} has {len(matches)} '
            f'matches in exported frame range '
            f'{frame_ns[0] if frame_ns else None}..'
            f'{frame_ns[-1] if frame_ns else None}'
        )
    return matches[0]


def _parse_label_line(line):
    """Parse one agent line from label txt → dict or None."""
    parts = line.strip().split()
    if len(parts) < 12:
        return None
    try:
        return {
            'type': parts[0],
            'cx': float(parts[1]), 'cy': float(parts[2]), 'cz': float(parts[3]),
            'length': float(parts[4]), 'width': float(parts[5]), 'height': float(parts[6]),
            'yaw': float(parts[7]),
            'vx': float(parts[8]), 'vy': float(parts[9]),
            'track_id': parts[10],
            'class_id': int(parts[11]),
            'is_valid': parts[12].lower() == 'true' if len(parts) > 12 else True,
        }
    except (ValueError, IndexError):
        return None


def convert_scenario(label_dir, calib_dir, meta_path, scenario_id, town, map_features):
    """
    Convert a single DeepAccident scenario.

    Parameters
    ----------
    label_dir : str
        Path to ``ego_vehicle/label/{scenario_name}/`` containing per-frame .txt files.
    calib_dir : str
        Path to ``ego_vehicle/calib/{scenario_name}/`` containing per-frame .pkl files.
    meta_path : str
        Path to ``meta/{scenario_name}.txt``.
    scenario_id : str
    town : str
    map_features : dict
        Pre-built ScenarioNet map_features for this town.

    Returns
    -------
    sd_scenario : dict   ScenarioNet-format scenario dict.
    """
    label_paths, calib_paths, frame_ns = _load_calib_and_label(label_dir, calib_dir)
    T = len(label_paths)
    if T == 0:
        raise RuntimeError(f'No label files in {label_dir}')

    meta = _load_meta(meta_path)
    ego_meta_id = meta.get('ego_id')
    raw_field9 = meta.get('collision_frame')  # meta 第 9 列;只有真碰撞时才是 collision_frame

    # ── Scenario category & collision semantics ──────────────────────────────
    # DeepAccident 在 accident/ 目录下混合三类:
    #   * 真双方碰撞    colliding agents 两侧都是 ego/other/*_behind
    #   * 单方碰撞      一侧是 none(撞静止物或路边)
    #   * near-miss     'none none'(危险但避险成功),meta 第 9 列是 scene_length
    in_accident_dir = 'accident' in scenario_id
    colliding_roles = meta.get('colliding_roles', ['none', 'none'])
    n_real_roles = sum(1 for r in colliding_roles if r != 'none')

    if not in_accident_dir:
        scenario_category = 'normal'
        is_collided = False
        collision_frame = None
    elif n_real_roles == 0:
        scenario_category = 'near_miss'
        is_collided = False
        collision_frame = None  # raw_field9 在此类是 scene_length,不是 collision_frame
    elif n_real_roles == 1:
        scenario_category = 'single_collision'
        is_collided = True
        collision_frame = _source_frame_to_local(raw_field9, frame_ns)
    else:
        scenario_category = 'mutual_collision'
        is_collided = True
        collision_frame = _source_frame_to_local(raw_field9, frame_ns)

    # ── Resolve collision pairs ───────────────────────────────────────────────
    # agents_id order: [ego, ego_behind, other, other_behind]
    ROLE_TO_INDEX = {'ego': 0, 'ego_behind': 1, 'other': 2, 'other_behind': 3}
    collision_info = {
        'scenario_category': scenario_category,    # normal/near_miss/single_collision/mutual_collision
        'is_collided': is_collided,                 # 是否真发生碰撞(取代误用的 is_accident)
        'is_accident': in_accident_dir,             # 保留向后兼容:仅表示来自 accident/ 目录
        'collision_frame': collision_frame,         # near-miss/normal 时为 None
        'collision_source_frame': raw_field9 if is_collided else None,
        'source_frame_start': frame_ns[0] if frame_ns else None,
        'source_frame_end': frame_ns[-1] if frame_ns else None,
        'collision_pairs': [],
        'colliding_roles': colliding_roles,
        'road_type': meta.get('road_type', ''),
        'weather': meta.get('weather', ''),
    }
    agents_id_list = meta.get('agents_id', [])
    if is_collided and len(colliding_roles) == 2 and len(agents_id_list) >= 4:
        for role in colliding_roles:
            if role in ROLE_TO_INDEX and role != 'none':
                idx = ROLE_TO_INDEX[role]
                if idx < len(agents_id_list):
                    # Build pairs: each colliding role paired with the other
                    for role2 in colliding_roles:
                        if role2 in ROLE_TO_INDEX and role2 != 'none' and role2 != role:
                            idx2 = ROLE_TO_INDEX[role2]
                            if idx2 < len(agents_id_list):
                                pair = sorted([str(agents_id_list[idx]), str(agents_id_list[idx2])])
                                if pair not in collision_info['collision_pairs']:
                                    collision_info['collision_pairs'].append(pair)

    # ── 1. Build per-agent track arrays (list of dicts per timestep) ────────
    # Keyed by track_id → list of (frame_idx, state_dict_or_None)
    raw_tracks: dict = defaultdict(lambda: [None] * T)

    ego_positions = np.zeros((T, 3), dtype=np.float64)
    ego_headings = np.zeros(T, dtype=np.float64)
    ego_dimensions = None  # determined from first detection

    for t in range(T):
        # ── Calib ───────────────────────────────────────────────────────────
        calib_path = calib_paths[t]
        if calib_path is None:
            continue
        with open(calib_path, 'rb') as f:
            calib = pickle.load(f)
        E2W = calib['ego_to_world']
        ego_yaw = _rotation_yaw(E2W)
        ego_positions[t] = E2W[:3, 3]
        ego_headings[t] = ego_yaw

        # ── Label ───────────────────────────────────────────────────────────
        with open(label_paths[t]) as f:
            label_lines = f.readlines()
        if len(label_lines) < 1:
            continue
        # Line 0 = ego velocity (label-reported, frame unclear) — IGNORED;
        # ego velocity is recomputed from positions below alongside agents.
        for line in label_lines[1:]:
            agent = _parse_label_line(line)
            if agent is None:
                continue
            tid = agent['track_id']

            # Transform position: lidar → world
            pos_lidar = np.array([agent['cx'], agent['cy'], agent['cz'], 1.0])
            pos_world = E2W @ pos_lidar
            heading_world = agent['yaw'] + ego_yaw

            raw_tracks[tid][t] = {
                'type': agent['type'],
                'position': pos_world[:3].astype(np.float32),
                'length': agent['length'],
                'width': agent['width'],
                'height': agent['height'],
                'heading': heading_world,
                'valid': agent['is_valid'],
            }

    # ── 2. Assemble tracks dict ─────────────────────────────────────────────
    tracks = {}
    object_summary = {}

    # Determine ego dimensions from agents parked near (0,0) or use defaults
    if ego_dimensions is None:
        ego_dimensions = DEFAULT_EGO_DIMS

    sdc_id = str(ego_meta_id) if ego_meta_id is not None and ego_meta_id > 0 else 'ego'

    # Add ego as a track (velocity from world-frame position diff)
    ego_valid = np.ones(T, dtype=bool)
    ego_velocity = _velocity_from_positions(
        ego_positions[:, :2].astype(np.float32), ego_valid, 1.0 / FRAME_RATE_HZ)
    ego_track = {
        'type': 'VEHICLE',
        'state': {
            'position': ego_positions.astype(np.float32),
            'length': np.full((T, 1), ego_dimensions[0], dtype=np.float32),
            'width': np.full((T, 1), ego_dimensions[1], dtype=np.float32),
            'height': np.full((T, 1), ego_dimensions[2], dtype=np.float32),
            'heading': ego_headings.astype(np.float32).reshape(-1, 1),
            'velocity': ego_velocity,
            'valid': ego_valid,
        },
        'metadata': {'object_id': sdc_id},
    }
    tracks[sdc_id] = ego_track

    ego_pos = ego_track['state']['position']
    ego_moving = float(np.sum(np.linalg.norm(np.diff(ego_pos, axis=0)[:, :2], axis=1))) if T > 1 else 0.0
    object_summary[sdc_id] = {
        'type': 'VEHICLE',
        'object_id': sdc_id,
        'track_length': T,
        'moving_distance': ego_moving,
        'valid_length': T,
        'continuous_valid_length': T,
    }

    # Add tracked agents
    for tid, states in raw_tracks.items():
        valid_frames = sum(1 for s in states if s is not None and s.get('valid', False))
        if valid_frames < AGENT_FILTER_MIN_VALID_FRAMES:
            continue

        # Allocate arrays
        pos = np.zeros((T, 3), dtype=np.float32)
        length = np.zeros((T, 1), dtype=np.float32)
        width = np.zeros((T, 1), dtype=np.float32)
        height = np.zeros((T, 1), dtype=np.float32)
        heading = np.zeros((T, 1), dtype=np.float32)
        valid = np.zeros(T, dtype=bool)

        agent_type_str = 'VEHICLE'
        for t, s in enumerate(states):
            if s is None or not s.get('valid', False):
                continue
            pos[t] = s['position']
            length[t] = s['length']
            width[t] = s['width']
            height[t] = s['height']
            heading[t] = s['heading']
            valid[t] = True
            agent_type_str = s['type']

        # Velocity from world-frame position diff (label vx/vy is in an
        # unreliable frame — see module docstring).
        velocity = _velocity_from_positions(pos[:, :2], valid, 1.0 / FRAME_RATE_HZ)

        md_type = CARLA_TO_METADRIVE_OBJECT.get(agent_type_str, 'OTHER')
        agent_id = str(tid)
        tracks[agent_id] = {
            'type': md_type,
            'state': {
                'position': pos,
                'length': length,
                'width': width,
                'height': height,
                'heading': heading,
                'velocity': velocity,
                'valid': valid,
            },
            'metadata': {'object_id': agent_id},
        }

        # Summary
        valid_pos = pos[valid, :2]
        if len(valid_pos) > 1:
            moving_dist = float(np.sum(np.linalg.norm(np.diff(valid_pos, axis=0), axis=1)))
        else:
            moving_dist = 0.0

        continuous_valid = 0
        cur_run = 0
        for v in valid:
            if v:
                cur_run += 1
                continuous_valid = max(continuous_valid, cur_run)
            else:
                cur_run = 0

        object_summary[agent_id] = {
            'type': md_type,
            'object_id': agent_id,
            'track_length': T,
            'moving_distance': moving_dist,
            'valid_length': int(valid.sum()),
            'continuous_valid_length': continuous_valid,
        }

    # ── 3. Pad short scenarios to Waymo length (91 frames) ─────────────────
    # base_dataset.preprocess() pads at the BEGINNING for N < 91, which makes
    # current_time_index=10 land in zero-padding → all agents filtered.
    # Padding at the END fixes this: early frames stay real, padded frames invisible.
    PAD_TARGET = 91
    if T < PAD_TARGET:
        pad_n = PAD_TARGET - T
        for agent_id in list(tracks.keys()):
            t = tracks[agent_id]
            for key in ['position', 'length', 'width', 'height', 'heading', 'velocity']:
                arr = t['state'][key]
                pad_shape = list(arr.shape)
                pad_shape[0] = pad_n
                pad_block = np.zeros(pad_shape, dtype=arr.dtype)
                t['state'][key] = np.concatenate([arr, pad_block], axis=0)
            valid = t['state']['valid']
            t['state']['valid'] = np.concatenate([valid, np.zeros(pad_n, dtype=bool)])
        # Extend ts and length
        ts = [t / FRAME_RATE_HZ for t in range(T)]
        ts_pad = [ts[-1] + (i + 1) / FRAME_RATE_HZ for i in range(pad_n)]
        ts = ts + ts_pad
        T = PAD_TARGET

    if T >= PAD_TARGET:
        ts = [t / FRAME_RATE_HZ for t in range(T)]

    # ── 4. tracks_to_predict ─────────────────────────────────────────────────
    # Align with base_dataset.trajectory_filter() criteria:
    #   1. type ∈ [VEHICLE, PEDESTRIAN, CYCLIST]
    #   2. valid_ratio >= 0.5
    #   3. VEHICLE: moving_distance >= 2.0m
    #   4. valid at current_time_index (frame 10 after padding, or T//4 before)
    check_frame = min(10, T // 4)  # approx current_time_index before padding
    tracks_to_predict = {}
    for idx, (aid, summary) in enumerate(object_summary.items()):
        tp = summary['type']
        if tp not in ('VEHICLE', 'PEDESTRIAN', 'CYCLIST'):
            continue
        track = tracks.get(aid)
        if track is None:
            continue
        valid_arr = track['state']['valid']
        if check_frame >= len(valid_arr) or not valid_arr[check_frame]:
            continue
        valid_ratio = summary['valid_length'] / max(summary['track_length'], 1)
        if valid_ratio < 0.5:
            continue
        if tp == 'VEHICLE' and summary['moving_distance'] < 2.0:
            continue
        tracks_to_predict[aid] = {
            'track_index': idx,
            'track_id': aid,
            'difficulty': 0,
            'object_type': tp,
        }

    # ── 5. number_summary ───────────────────────────────────────────────────
    type_counts = defaultdict(int)
    for s in object_summary.values():
        type_counts[s['type']] += 1
    number_summary = {
        'num_objects': len(tracks),
        'object_types': tuple(type_counts.keys()),
        'num_objects_each_type': dict(type_counts),
        'num_moving_objects': sum(1 for s in object_summary.values() if s['moving_distance'] > 0.1),
        'num_map_features': len(map_features),
        'map_height_diff': 0.0,
    }

    # ── 6. current_time_index ───────────────────────────────────────────────
    if collision_frame is not None and 5 < collision_frame < T - 20:
        current_time_index = collision_frame - 5  # 0.5s before collision
    else:
        current_time_index = min(10, max(2, T // 4))

    # ── 7. Assemble ─────────────────────────────────────────────────────────

    sd_scenario = {
        'id': scenario_id,
        'version': DATASET_VERSION,
        'length': T,
        'tracks': tracks,
        'dynamic_map_states': {},
        'map_features': map_features,
        'metadata': {
            'dataset': 'deepaccident',
            'metadrive_processed': True,
            'id': scenario_id,
            'scenario_id': scenario_id,
            'coordinate': COORDINATE,
            'ts': ts,
            'current_time_index': current_time_index,
            'sdc_id': sdc_id,
            'tracks_to_predict': tracks_to_predict,
            'object_summary': object_summary,
            'number_summary': number_summary,
            'collision': collision_info,
        },
    }
    return sd_scenario
