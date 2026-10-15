"""
Step 1 of UniTraj data preparation:
Convert raw Waymo tfrecord files → MTR intermediate pkl files.

Adapted from MTR (https://arxiv.org/abs/2209.13508) by Shaoshuai Shi.

Requirements:
    pip install waymo-open-dataset-tf-2-6-0

Usage (standalone):
    python datasets/waymo_converter/waymo_preprocess.py \
        --raw_data_path data/waymo/scenario \
        --output_path data/waymo \
        --split all \
        --num_workers 16

Output layout:
    {output_path}/processed_scenarios_training/   sample_<id>.pkl per scenario
    {output_path}/processed_scenarios_validation/ sample_<id>.pkl per scenario
    {output_path}/processed_scenarios_training_infos.pkl
    {output_path}/processed_scenarios_val_infos.pkl
"""

import sys
import os
import numpy as np
import pickle
import multiprocessing
import glob
from tqdm import tqdm

# Allow running both as a script (python datasets/waymo_converter/waymo_preprocess.py) and as a module
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, polyline_type


def decode_tracks_from_proto(tracks):
    track_infos = {
        'object_id': [],
        'object_type': [],
        'trajs': []
    }
    for cur_data in tracks:
        cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading,
                              x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
        cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp, 10)
        track_infos['object_id'].append(cur_data.id)
        track_infos['object_type'].append(object_type[cur_data.object_type])
        track_infos['trajs'].append(cur_traj)
    track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 10)
    return track_infos


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1e9)
    return polyline_dir


def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [], 'road_line': [], 'road_edge': [],
        'stop_sign': [], 'crosswalk': [], 'speed_bump': [], 'driveway': []
    }
    polylines = []
    point_cnt = 0

    for cur_data in map_features:
        cur_info = {'id': cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = lane_type[cur_data.lane.type]
            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)
            cur_info['left_boundary'] = [{
                'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                'feature_id': x.boundary_feature_id, 'boundary_type': x.boundary_type
            } for x in cur_data.lane.left_boundaries]
            cur_info['right_boundary'] = [{
                'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                'feature_id': x.boundary_feature_id, 'boundary_type': road_line_type[x.boundary_type]
            } for x in cur_data.lane.right_boundaries]
            cur_info['left_neighbors'] = [{
                'feature_id': x.feature_id, 'self_start_index': x.self_start_index,
                'self_end_index': x.self_end_index, 'neighbor_start_index': x.neighbor_start_index,
                'neighbor_end_index': x.neighbor_end_index
            } for x in cur_data.lane.left_neighbors]
            cur_info['right_neighbors'] = [{
                'feature_id': x.feature_id, 'self_start_index': x.self_start_index,
                'self_end_index': x.self_end_index, 'neighbor_start_index': x.neighbor_start_index,
                'neighbor_end_index': x.neighbor_end_index
            } for x in cur_data.lane.right_neighbors]
            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([p.x, p.y, p.z, global_type]) for p in cur_data.lane.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, :3])
            cur_polyline = np.concatenate((cur_polyline[:, :3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
            map_infos['lane'].append(cur_info)

        elif cur_data.road_line.ByteSize() > 0:
            cur_info['type'] = road_line_type[cur_data.road_line.type]
            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([p.x, p.y, p.z, global_type]) for p in cur_data.road_line.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, :3])
            cur_polyline = np.concatenate((cur_polyline[:, :3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = road_edge_type[cur_data.road_edge.type]
            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([p.x, p.y, p.z, global_type]) for p in cur_data.road_edge.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, :3])
            cur_polyline = np.concatenate((cur_polyline[:, :3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
            map_infos['road_edge'].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])
            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)
            map_infos['stop_sign'].append(cur_info)

        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack([np.array([p.x, p.y, p.z, global_type]) for p in cur_data.crosswalk.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, :3])
            cur_polyline = np.concatenate((cur_polyline[:, :3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP']
            cur_polyline = np.stack([np.array([p.x, p.y, p.z, global_type]) for p in cur_data.speed_bump.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, :3])
            cur_polyline = np.concatenate((cur_polyline[:, :3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
            map_infos['speed_bump'].append(cur_info)

        elif hasattr(cur_data, 'driveway') and cur_data.driveway.ByteSize() > 0:
            global_type = -1
            cur_polyline = np.stack([np.array([p.x, p.y, p.z, global_type]) for p in cur_data.driveway.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, :3])
            cur_polyline = np.concatenate((cur_polyline[:, :3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
            map_infos['driveway'].append(cur_info)

        else:
            print(f'Unknown map feature type, id: {cur_data.id}')
            continue

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except Exception:
        polylines = np.zeros((0, 7), dtype=np.float32)
        print('Warning: empty polylines')
    map_infos['all_polylines'] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    dynamic_map_infos = {'lane_id': [], 'state': [], 'stop_point': []}
    for cur_data in dynamic_map_states:
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])
        dynamic_map_infos['lane_id'].append(np.array([lane_id]))
        dynamic_map_infos['state'].append(np.array([state]))
        dynamic_map_infos['stop_point'].append(np.array([stop_point]))
    return dynamic_map_infos


def process_waymo_data_with_scenario_proto(data_file, output_path=None):
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    # Resume support: skip tfrecord files that were already fully processed.
    # A .done marker is written after all scenarios in a tfrecord are saved.
    done_marker = os.path.join(
        output_path, '.' + os.path.basename(data_file).replace('/', '_') + '.done'
    )
    if os.path.isfile(done_marker):
        # Re-read the saved info from the marker (list of info dicts)
        with open(done_marker, 'rb') as f:
            return pickle.load(f)

    dataset = tf.data.TFRecordDataset(data_file, compression_type='')
    ret_infos = []
    for data in dataset:
        info = {}
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))

        info['scenario_id'] = scenario.scenario_id
        info['timestamps_seconds'] = list(scenario.timestamps_seconds)
        info['current_time_index'] = scenario.current_time_index
        info['sdc_track_index'] = scenario.sdc_track_index
        info['objects_of_interest'] = list(scenario.objects_of_interest)
        info['tracks_to_predict'] = {
            'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
            'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
        }

        track_infos = decode_tracks_from_proto(scenario.tracks)
        info['tracks_to_predict']['object_type'] = [
            track_infos['object_type'][cur_idx] for cur_idx in info['tracks_to_predict']['track_index']
        ]

        map_infos = decode_map_features_from_proto(scenario.map_features)
        dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

        save_infos = {'track_infos': track_infos, 'dynamic_map_infos': dynamic_map_infos, 'map_infos': map_infos}
        save_infos.update(info)

        output_file = os.path.join(output_path, f'sample_{scenario.scenario_id}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(save_infos, f)

        ret_infos.append(info)

    # Write done marker so this tfrecord is skipped on resume
    with open(done_marker, 'wb') as f:
        pickle.dump(ret_infos, f)

    return ret_infos


def get_infos_from_protos(data_path, output_path, num_workers=16):
    from functools import partial
    os.makedirs(output_path, exist_ok=True)
    func = partial(process_waymo_data_with_scenario_proto, output_path=output_path)
    src_files = sorted(glob.glob(os.path.join(data_path, '*.tfrecord*')))
    with multiprocessing.Pool(num_workers) as p:
        data_infos = list(tqdm(p.imap(func, src_files), total=len(src_files)))
    return [item for infos in data_infos for item in infos]


def preprocess_split(raw_data_path, output_path, split, num_workers=16):
    """
    Convert tfrecord files for one split → MTR intermediate pkl.

    Args:
        raw_data_path: directory containing training/validation/testing subdirs
        output_path:   root output dir (e.g. data/waymo)
        split:         'training', 'validation', or 'testing'
        num_workers:   parallel processes
    """
    split_dir_map = {
        'training':   'processed_scenarios_training',
        'validation': 'processed_scenarios_validation',
        'testing':    'processed_scenarios_testing',
    }
    info_file_map = {
        'training':   'processed_scenarios_training_infos.pkl',
        'validation': 'processed_scenarios_val_infos.pkl',
        'testing':    'processed_scenarios_testing_infos.pkl',
    }

    out_dir = os.path.join(output_path, split_dir_map[split])
    info_file = os.path.join(output_path, info_file_map[split])

    print(f'[waymo_preprocess] Processing {split} split: {raw_data_path}/{split} → {out_dir}')
    infos = get_infos_from_protos(
        data_path=os.path.join(raw_data_path, split),
        output_path=out_dir,
        num_workers=num_workers
    )
    with open(info_file, 'wb') as f:
        pickle.dump(infos, f)
    print(f'[waymo_preprocess] {split}: {len(infos)} scenarios → {info_file}')
    return len(infos)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Step 1: Waymo tfrecord → MTR intermediate pkl')
    parser.add_argument('--raw_data_path', type=str, default='data/waymo/scenario',
                        help='Directory containing training/validation/testing tfrecord subdirs')
    parser.add_argument('--output_path', type=str, default='data/waymo',
                        help='Root output directory (e.g. data/waymo)')
    parser.add_argument('--split', type=str, default='all',
                        choices=['all', 'training', 'validation', 'testing'],
                        help='Which split to process')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    splits = ['training', 'validation'] if args.split == 'all' else [args.split]
    for split in splits:
        preprocess_split(args.raw_data_path, args.output_path, split, args.num_workers)
