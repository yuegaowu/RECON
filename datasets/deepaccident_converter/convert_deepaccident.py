"""
DeepAccident → UniTraj ScenarioNet converter (main orchestrator).

Usage::

    python convert_deepaccident.py \
        --source_root data/deepaccident_raw \
        --output_root  data/deepaccident/scenarionet \
        --map_cache_root data/deepaccident_raw/_maps \
        --xodr_dir data/deepaccident_raw/carla_xodr \
        --workers 16
"""

import argparse
import os, pickle, traceback
from collections import defaultdict
from multiprocessing import Pool

from .deepaccident_types import DATASET_VERSION
from .deepaccident_map_builder import build_town_map
from .deepaccident_to_scenarionet import convert_scenario


def _discover_scenarios(source_root):
    """
    Return list of (label_dir, calib_dir, meta_path, scenario_id, town) tuples.

    Directory layout::

        {source_root}/{type_info}/
            ego_vehicle/label/{scenario_name}/*.txt
            ego_vehicle/calib/{scenario_name}/*.pkl
            meta/{scenario_name}.txt
    """
    scenarios = []
    for type_info in sorted(os.listdir(source_root)):
        type_path = os.path.join(source_root, type_info)
        if not os.path.isdir(type_path):
            continue
        meta_dir = os.path.join(type_path, 'meta')
        if not os.path.isdir(meta_dir):
            continue
        for meta_file in sorted(os.listdir(meta_dir)):
            if not meta_file.endswith('.txt'):
                continue
            scenario_name = meta_file.replace('.txt', '')
            town = scenario_name.split('_')[0]
            label_dir = os.path.join(type_path, 'ego_vehicle', 'label', scenario_name)
            calib_dir = os.path.join(type_path, 'ego_vehicle', 'calib', scenario_name)
            meta_path = os.path.join(meta_dir, meta_file)
            if not os.path.isdir(label_dir):
                continue
            scenario_id = f'{scenario_name}_{type_info}'
            scenarios.append((label_dir, calib_dir, meta_path, scenario_id, town))
    return scenarios


# scenario_category → 输出子目录名
# mutual_collision = 车-车互撞(我们的预警管线能处理)
# single_collision = 车-环境(护栏/路边静态物体,预警管线不处理)
_CATEGORY_TO_SUBDIR = {
    'normal':            'normal',
    'near_miss':         'near_miss',
    'mutual_collision':  'collision_vehicle',
    'single_collision':  'collision_object',
}


def _process_one(args):
    """Worker: convert one scenario to pkl, return (subdir, fname, metadata)."""
    label_dir, calib_dir, meta_path, scenario_id, town, map_features, output_root = args
    try:
        sd = convert_scenario(label_dir, calib_dir, meta_path, scenario_id, town, map_features)
        category = sd['metadata']['collision'].get('scenario_category', 'normal')
        subdir = _CATEGORY_TO_SUBDIR.get(category, 'normal')
        out_dir = os.path.join(output_root, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fname = f'sd_deepaccident_{DATASET_VERSION}_{scenario_id}.pkl'
        with open(os.path.join(out_dir, fname), 'wb') as f:
            pickle.dump(sd, f)
        return subdir, fname, sd['metadata']
    except Exception as e:
        print(f'  ERROR [{scenario_id}]: {e}')
        traceback.print_exc()
        return None, None, None


def convert_all(source_root, output_root, map_cache_root, xodr_dir=None, num_workers=4):
    """Run the full conversion pipeline."""
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(map_cache_root, exist_ok=True)

    # 1. Discover scenarios
    all_scenarios = _discover_scenarios(source_root)
    print(f'[convert] Found {len(all_scenarios)} scenarios')

    # 2. Build per-town maps
    town_label_dirs = defaultdict(list)
    for ldir, cdir, mp, sid, town in all_scenarios:
        town_label_dirs[town].append(ldir)

    town_maps = {}
    for town, label_dirs in town_label_dirs.items():
        print(f'[map] Building map for {town} ({len(label_dirs)} scenarios)')
        town_maps[town] = build_town_map(town, label_dirs, map_cache_root, xodr_dir=xodr_dir)

    # 3. Convert scenarios
    jobs = []
    for ldir, cdir, mp, sid, town in all_scenarios:
        jobs.append((ldir, cdir, mp, sid, town, town_maps[town], output_root))

    # 按 category 子目录分桶 summary
    summaries = defaultdict(dict)   # subdir -> {fname: metadata}
    total = 0
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            for subdir, fname, meta in pool.imap_unordered(_process_one, jobs):
                if fname is not None:
                    summaries[subdir][fname] = meta
                    total += 1
    else:
        for job in jobs:
            subdir, fname, meta = _process_one(job)
            if fname is not None:
                summaries[subdir][fname] = meta
                total += 1

    print(f'[convert] Converted {total} / {len(all_scenarios)} scenarios')

    # 4. 每个子目录独立写 dataset_summary.pkl + dataset_mapping.pkl
    print(f'[convert] Writing per-category summary/mapping...')
    for subdir, summary in summaries.items():
        out_dir = os.path.join(output_root, subdir)
        summary_path = os.path.join(out_dir, 'dataset_summary.pkl')
        mapping_path = os.path.join(out_dir, 'dataset_mapping.pkl')
        mapping = {fname: '' for fname in summary}
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        with open(mapping_path, 'wb') as f:
            pickle.dump(mapping, f)
        print(f'  [{subdir:12s}] {len(summary):4d} pkl  -> {out_dir}')

    print(f'[convert] Done. Total {total} pkl across {len(summaries)} categories.')
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepAccident -> ScenarioNet converter')
    parser.add_argument('--source_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--map_cache_root', required=True)
    parser.add_argument('--xodr_dir', default=None, help='Directory containing Town*.xodr files')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    convert_all(args.source_root, args.output_root, args.map_cache_root,
                xodr_dir=args.xodr_dir, num_workers=args.workers)
