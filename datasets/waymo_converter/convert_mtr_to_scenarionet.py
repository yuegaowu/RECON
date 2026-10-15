"""
Convert MTR-preprocessed Waymo pkl files to ScenarioNet format for UniTraj.

Usage:
    python convert_mtr_to_scenarionet.py --split training --workers 16
    python convert_mtr_to_scenarionet.py --split validation --workers 8
    python convert_mtr_to_scenarionet.py --split training --max_files 100  # test run
"""

import argparse
import os
import pickle
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ─── Type mappings (Waymo/MTR → MetaDrive/ScenarioNet strings) ────────────────

LANE_TYPE_MAP = {
    'TYPE_FREEWAY':           'LANE_FREEWAY',
    'TYPE_SURFACE_STREET':    'LANE_SURFACE_STREET',
    'TYPE_BIKE_LANE':         'LANE_BIKE_LANE',
}

ROAD_LINE_TYPE_MAP = {
    'TYPE_UNKNOWN':                  'UNKNOWN_LINE',
    'TYPE_BROKEN_SINGLE_WHITE':      'ROAD_LINE_BROKEN_SINGLE_WHITE',
    'TYPE_SOLID_SINGLE_WHITE':       'ROAD_LINE_SOLID_SINGLE_WHITE',
    'TYPE_SOLID_DOUBLE_WHITE':       'ROAD_LINE_SOLID_DOUBLE_WHITE',
    'TYPE_BROKEN_SINGLE_YELLOW':     'ROAD_LINE_BROKEN_SINGLE_YELLOW',
    'TYPE_BROKEN_DOUBLE_YELLOW':     'ROAD_LINE_BROKEN_DOUBLE_YELLOW',
    'TYPE_SOLID_SINGLE_YELLOW':      'ROAD_LINE_SOLID_SINGLE_YELLOW',
    'TYPE_SOLID_DOUBLE_YELLOW':      'ROAD_LINE_SOLID_DOUBLE_YELLOW',
    'TYPE_PASSING_DOUBLE_YELLOW':    'ROAD_LINE_PASSING_DOUBLE_YELLOW',
}

ROAD_EDGE_TYPE_MAP = {
    'TYPE_UNKNOWN':             'ROAD_EDGE_BOUNDARY',
    'TYPE_ROAD_EDGE_BOUNDARY':  'ROAD_EDGE_BOUNDARY',
    'TYPE_ROAD_EDGE_MEDIAN':    'ROAD_EDGE_MEDIAN',
}

OBJECT_TYPE_MAP = {
    'TYPE_VEHICLE':    'VEHICLE',
    'TYPE_PEDESTRIAN': 'PEDESTRIAN',
    'TYPE_CYCLIST':    'CYCLIST',
    'TYPE_OTHER':      'OTHER',
}

TL_STATE_MAP = {
    'LANE_STATE_UNKNOWN':         'LANE_STATE_UNKNOWN',
    'LANE_STATE_ARROW_STOP':      'LANE_STATE_ARROW_STOP',
    'LANE_STATE_ARROW_CAUTION':   'LANE_STATE_ARROW_CAUTION',
    'LANE_STATE_ARROW_GO':        'LANE_STATE_ARROW_GO',
    'LANE_STATE_STOP':            'LANE_STATE_STOP',
    'LANE_STATE_CAUTION':         'LANE_STATE_CAUTION',
    'LANE_STATE_GO':              'LANE_STATE_GO',
    'LANE_STATE_FLASHING_STOP':   'LANE_STATE_FLASHING_STOP',
    'LANE_STATE_FLASHING_CAUTION':'LANE_STATE_FLASHING_CAUTION',
}

DATASET_VERSION = 'v1.0'


def _get_polyline(all_polylines, polyline_index):
    """Extract (x,y,z) polyline from all_polylines using polyline_index."""
    s, e = polyline_index
    pts = all_polylines[s:e, :3]  # (N, 3) x,y,z
    return pts.astype(np.float32)


def _split_polyline_on_gaps(polyline, gap_threshold=2.0):
    """Split a nominally continuous polyline where source points are disconnected."""
    if len(polyline) < 2:
        return [polyline]
    gaps = np.where(
        np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1) > gap_threshold
    )[0] + 1
    return [part for part in np.split(polyline, gaps) if len(part) >= 2]


def convert_one_scenario(mtr_data: dict) -> dict:
    """Convert a single MTR scenario dict to ScenarioNet/MetaDrive format."""
    scenario_id = mtr_data['scenario_id']
    all_polylines = mtr_data['map_infos']['all_polylines']  # (N_pts, 7)
    track_infos = mtr_data['track_infos']
    trajs = track_infos['trajs']          # (N_agents, T, 10)
    obj_ids = track_infos['object_id']    # list length N_agents
    obj_types = track_infos['object_type']  # list length N_agents
    timestamps = mtr_data['timestamps_seconds']  # list length T
    current_time_index = mtr_data['current_time_index']
    sdc_idx = mtr_data['sdc_track_index']
    tracks_to_predict_raw = mtr_data['tracks_to_predict']

    T = trajs.shape[1]  # 91

    # ── 1. Build tracks ────────────────────────────────────────────────────────
    tracks = {}
    object_summary = {}
    for i, (oid, otype) in enumerate(zip(obj_ids, obj_types)):
        agent_id = str(oid)
        traj = trajs[i]  # (T, 10): x,y,z,l,w,h,heading,vx,vy,valid
        valid = traj[:, 9].astype(bool)

        md_type = OBJECT_TYPE_MAP.get(otype, 'OTHER')
        tracks[agent_id] = {
            'type': md_type,
            'state': {
                'position': traj[:, 0:3].astype(np.float32),   # (T, 3)
                'length':   traj[:, 3:4].astype(np.float32),   # (T, 1)
                'width':    traj[:, 4:5].astype(np.float32),   # (T, 1)
                'height':   traj[:, 5:6].astype(np.float32),   # (T, 1)
                'heading':  traj[:, 6:7].astype(np.float32),   # (T, 1)
                'velocity': traj[:, 7:9].astype(np.float32),   # (T, 2)
                'valid':    valid,                              # (T,) bool
            },
            'metadata': {'object_id': agent_id}
        }

        # Compute moving distance (sum of valid displacements)
        valid_pos = traj[valid, :2]
        if len(valid_pos) > 1:
            moving_dist = float(np.sum(np.linalg.norm(np.diff(valid_pos, axis=0), axis=1)))
        else:
            moving_dist = 0.0

        # Continuous valid length (longest valid run)
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

    # ── 2. Build tracks_to_predict ────────────────────────────────────────────
    tracks_to_predict = {}
    for idx, diff, otype in zip(
        tracks_to_predict_raw['track_index'],
        tracks_to_predict_raw['difficulty'],
        tracks_to_predict_raw['object_type'],
    ):
        oid = str(obj_ids[idx])
        tracks_to_predict[oid] = {
            'track_index': int(idx),
            'track_id': oid,
            'difficulty': int(diff),
            'object_type': OBJECT_TYPE_MAP.get(otype, 'VEHICLE'),
        }

    sdc_id = str(obj_ids[sdc_idx])

    # ── 3. Build map_features ─────────────────────────────────────────────────
    map_features = {}

    # Lanes
    for lane in mtr_data['map_infos'].get('lane', []):
        fid = str(lane['id'])
        polyline = _get_polyline(all_polylines, lane['polyline_index'])
        md_type = LANE_TYPE_MAP.get(lane.get('type', ''), 'LANE_SURFACE_STREET')
        map_features[fid] = {
            'type': md_type,
            'polyline': polyline,
            'speed_limit_mph': lane.get('speed_limit_mph', None),
            'interpolating': lane.get('interpolating', None),
            'entry_lanes': lane.get('entry_lanes', []),
            'exit_lanes': lane.get('exit_lanes', []),
            'left_neighbor': [],   # simplified
            'right_neighbor': [],
        }

    # Road lines
    for rl in mtr_data['map_infos'].get('road_line', []):
        fid = str(rl['id'])
        polyline = _get_polyline(all_polylines, rl['polyline_index'])
        md_type = ROAD_LINE_TYPE_MAP.get(rl.get('type', ''), 'UNKNOWN_LINE')
        map_features[fid] = {
            'type': md_type,
            'polyline': polyline,
        }

    # Road edges
    for re in mtr_data['map_infos'].get('road_edge', []):
        fid = str(re['id'])
        polyline = _get_polyline(all_polylines, re['polyline_index'])
        md_type = ROAD_EDGE_TYPE_MAP.get(re.get('type', ''), 'ROAD_EDGE_BOUNDARY')
        parts = _split_polyline_on_gaps(polyline)
        for part_idx, part in enumerate(parts):
            part_fid = fid if part_idx == 0 else f'{fid}__part{part_idx}'
            map_features[part_fid] = {
                'type': md_type,
                'polyline': part,
            }

    # Stop signs
    for ss in mtr_data['map_infos'].get('stop_sign', []):
        fid = str(ss['id'])
        pos = np.array(ss['position'], dtype=np.float32).reshape(1, -1)
        if pos.shape[1] == 2:
            pos = np.concatenate([pos, np.zeros((1, 1), dtype=np.float32)], axis=1)
        map_features[fid] = {
            'type': 'STOP_SIGN',
            'position': pos,
            'lane': ss.get('lane', []),
        }

    # Crosswalks
    for cw in mtr_data['map_infos'].get('crosswalk', []):
        fid = str(cw['id'])
        pts = _get_polyline(all_polylines, cw['polyline_index'])
        map_features[fid] = {
            'type': 'CROSSWALK',
            'polygon': pts,
        }

    # Speed bumps
    for sb in mtr_data['map_infos'].get('speed_bump', []):
        fid = str(sb['id'])
        pts = _get_polyline(all_polylines, sb['polyline_index'])
        map_features[fid] = {
            'type': 'SPEED_BUMP',
            'polygon': pts,
        }

    # Driveways are polygons, not lane centerlines. Keep their semantic type so
    # BaseDataset can ignore them instead of interpolating polygon edges as lanes.
    for dw in mtr_data['map_infos'].get('driveway', []):
        fid = str(dw['id'])
        polyline = _get_polyline(all_polylines, dw['polyline_index'])
        map_features[fid] = {
            'type': 'DRIVEWAY',
            'polygon': polyline,
        }

    # ── 4. Build dynamic_map_states (traffic lights) ──────────────────────────
    # ScenarioNet format: dict keyed by tl_id → {type, state:{object_state:[T states]}, lane, stop_point}
    # MTR format: per-timestep lists: lane_id[t] shape (1, N_tl), state[t] shape (1, N_tl), stop_point[t] shape (1, N_tl, 3)
    dmi = mtr_data['dynamic_map_infos']
    lane_ids_per_step = dmi.get('lane_id', [])
    states_per_step = dmi.get('state', [])
    stop_pts_per_step = dmi.get('stop_point', [])

    dynamic_map_states = {}  # tl_id_str → {type, state, lane, stop_point}
    for t in range(T):
        if t >= len(lane_ids_per_step):
            break
        ids_arr = np.array(lane_ids_per_step[t]).flatten()
        st_arr = np.array(states_per_step[t]).flatten() if t < len(states_per_step) else np.array([])
        sp_arr = np.array(stop_pts_per_step[t]) if t < len(stop_pts_per_step) else np.array([])
        sp_flat = sp_arr.reshape(-1, 3) if sp_arr.size > 0 else np.zeros((0, 3), dtype=np.float32)

        for j, lid in enumerate(ids_arr):
            try:
                lid_int = int(lid)
            except (ValueError, OverflowError):
                continue
            tl_id = str(lid_int)
            if tl_id not in dynamic_map_states:
                dynamic_map_states[tl_id] = {
                    'type': 'TRAFFIC_LIGHT',
                    'state': {'object_state': [None] * T},
                    'lane': lid_int,
                    'stop_point': sp_flat[j].astype(np.float32) if j < len(sp_flat) else np.zeros(3, dtype=np.float32),
                    'metadata': {'track_length': T, 'type': 'TRAFFIC_LIGHT', 'object_id': tl_id, 'dataset': 'waymo'},
                }
            tl_state = st_arr[j] if j < len(st_arr) else 'LANE_STATE_UNKNOWN'
            dynamic_map_states[tl_id]['state']['object_state'][t] = str(tl_state)

    # ── 5. Build number_summary ───────────────────────────────────────────────
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

    # ── 6. Assemble ScenarioNet dict ──────────────────────────────────────────
    ts = list(timestamps[:T])

    sd_scenario = {
        'id': scenario_id,
        'version': DATASET_VERSION,
        'length': T,
        'tracks': tracks,
        'dynamic_map_states': dynamic_map_states,
        'map_features': map_features,
        'metadata': {
            'dataset': 'waymo',
            'metadrive_processed': True,
            'id': scenario_id,
            'scenario_id': scenario_id,
            'coordinate': 'right-handed',
            'ts': ts,
            'current_time_index': current_time_index,
            'sdc_id': sdc_id,
            'tracks_to_predict': tracks_to_predict,
            'object_summary': object_summary,
            'number_summary': number_summary,
        }
    }
    return sd_scenario


def process_chunk(args):
    """Worker function for multiprocessing."""
    input_files, output_dir, chunk_id = args
    results = {}
    errors = 0
    for fpath in input_files:
        try:
            with open(fpath, 'rb') as f:
                mtr_data = pickle.load(f)
            sd_scenario = convert_one_scenario(mtr_data)
            scenario_id = sd_scenario['id']
            out_fname = f'sd_waymo_{DATASET_VERSION}_{scenario_id}.pkl'
            out_path = os.path.join(output_dir, out_fname)
            with open(out_path, 'wb') as f:
                pickle.dump(sd_scenario, f)
            # Summary entry (metadata only, no large arrays)
            results[out_fname] = sd_scenario['metadata']
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f'[chunk {chunk_id}] ERROR on {os.path.basename(fpath)}: {e}')
                traceback.print_exc()
    return results


def convert_split(input_dir: str, output_dir: str, num_workers: int = 16, max_files: int = None):
    """Convert all MTR pkl files in input_dir to ScenarioNet format in output_dir."""
    input_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.startswith('sample_') and f.endswith('.pkl')
    ])
    if max_files:
        input_files = input_files[:max_files]

    print(f'[convert] {len(input_files)} files → {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    # Split into chunks
    chunks = np.array_split(input_files, num_workers)
    chunk_args = [(list(c), output_dir, i) for i, c in enumerate(chunks) if len(c) > 0]

    summary = {}
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            for chunk_result in tqdm(pool.imap_unordered(process_chunk, chunk_args), total=len(chunk_args), desc='chunks'):
                summary.update(chunk_result)
    else:
        for args in tqdm(chunk_args, desc='chunks'):
            summary.update(process_chunk(args))

    # mapping: all files are in the root output_dir (no subdirs)
    mapping = {fname: '' for fname in summary}

    summary_path = os.path.join(output_dir, 'dataset_summary.pkl')
    mapping_path = os.path.join(output_dir, 'dataset_mapping.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)

    print(f'[convert] Done. {len(summary)} scenarios written.')
    print(f'  summary  → {summary_path}')
    print(f'  mapping  → {mapping_path}')
    return len(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['training', 'validation', 'testing'], required=True)
    parser.add_argument('--input_root', default='data/waymo')
    parser.add_argument('--output_root', default='data/waymo/waymo_scenarionet')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--max_files', type=int, default=None, help='Limit for testing')
    args = parser.parse_args()

    split_map = {
        'training':   'processed_scenarios_training',
        'validation': 'processed_scenarios_validation',
        'testing':    'processed_scenarios_testing',
    }
    input_dir = os.path.join(args.input_root, split_map[args.split])
    output_dir = os.path.join(args.output_root, args.split)

    convert_split(input_dir, output_dir, num_workers=args.workers, max_files=args.max_files)
