"""
UniTraj Data Preparation Pipeline
==================================
Converts raw Waymo tfrecord files to ScenarioNet pkl format ready for training.

Pipeline (two steps, each skipped if output already exists):

  Step 1  tfrecord → MTR intermediate pkl   (requires waymo-open-dataset)
          Input:  {data_root}/scenario/{split}/*.tfrecord*
          Output: {data_root}/processed_scenarios_{split}/sample_*.pkl
                  {data_root}/processed_scenarios_{split}_infos.pkl

  Step 2  MTR pkl → ScenarioNet pkl          (no extra deps beyond numpy)
          Input:  {data_root}/processed_scenarios_{split}/sample_*.pkl
          Output: {data_root}/waymo_scenarionet/{split}/sd_waymo_v1.0_*.pkl
                  {data_root}/waymo_scenarionet/{split}/dataset_summary.pkl
                  {data_root}/waymo_scenarionet/{split}/dataset_mapping.pkl

After both steps, train with:
  onthefly: True  → reads directly from waymo_scenarionet/{split}/
  onthefly: False → additionally generates h5 cache on first run

Usage:
  # Full pipeline for training + validation (most common)
  python datasets/waymo_converter/prepare_data.py

  # Only a specific split
  python datasets/waymo_converter/prepare_data.py --split training

  # Custom data root
  python datasets/waymo_converter/prepare_data.py --data_root data/waymo

  # Force redo step 2 even if scenarionet pkl already exists
  python datasets/waymo_converter/prepare_data.py --force_step2

  # Limit for testing (process only first N tfrecord files in step 1)
  python datasets/waymo_converter/prepare_data.py --max_files 5

  # Adjust parallelism
  python datasets/waymo_converter/prepare_data.py --num_workers 32
"""

import argparse
import glob
import os
import sys

# Allow importing sibling modules (waymo_preprocess, convert_mtr_to_scenarionet)
_datasets_dir = os.path.dirname(os.path.abspath(__file__))
if _datasets_dir not in sys.path:
    sys.path.insert(0, _datasets_dir)


# ── Step-existence checks ──────────────────────────────────────────────────────

def _mtr_pkl_exists(data_root: str, split: str) -> bool:
    """Return True if MTR intermediate pkl files already exist for this split."""
    split_dir_map = {
        'training':   'processed_scenarios_training',
        'validation': 'processed_scenarios_validation',
        'testing':    'processed_scenarios_testing',
    }
    out_dir = os.path.join(data_root, split_dir_map[split])
    if not os.path.isdir(out_dir):
        return False
    pkls = glob.glob(os.path.join(out_dir, 'sample_*.pkl'))
    return len(pkls) > 0


def _scenarionet_exists(data_root: str, split: str) -> bool:
    """Return True if ScenarioNet pkl files already exist for this split."""
    sn_dir = os.path.join(data_root, 'waymo_scenarionet', split)
    summary = os.path.join(sn_dir, 'dataset_summary.pkl')
    return os.path.isfile(summary)


# ── Step runners ──────────────────────────────────────────────────────────────

def _check_tf_available():
    """Check TensorFlow and waymo-open-dataset are installed; give clear install instructions if not."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        raise ImportError(
            '\n'
            'Step 1 requires TensorFlow and the Waymo Open Dataset API.\n'
            'These are NOT installed in the current environment.\n\n'
            'Recommended: create a dedicated preprocessing environment:\n'
            '  conda create -n waymo_preprocess python=3.9 -y\n'
            '  conda activate waymo_preprocess\n'
            '  pip install waymo-open-dataset-tf-2-11-0 tqdm numpy\n'
            '  python datasets/waymo_converter/prepare_data.py\n\n'
            'Alternatively, if you already have a TensorFlow environment:\n'
            '  conda run -n <your_tf_env> python datasets/waymo_converter/prepare_data.py\n\n'
            'Note: TensorFlow and PyTorch share CUDA libraries and may conflict\n'
            'in the same environment. A separate environment is strongly recommended.'
        )
    try:
        from waymo_open_dataset.protos import scenario_pb2  # noqa: F401
    except ImportError:
        raise ImportError(
            '\n'
            'TensorFlow is installed but waymo-open-dataset is missing.\n'
            'Install it with:\n'
            '  pip install waymo-open-dataset-tf-2-11-0\n'
        )


def run_step1(data_root: str, split: str, num_workers: int, max_files: int = None):
    """Step 1: tfrecord → MTR intermediate pkl."""
    _check_tf_available()
    from waymo_preprocess import preprocess_split

    raw_data_path = os.path.join(data_root, 'scenario')
    if not os.path.isdir(os.path.join(raw_data_path, split)):
        raise FileNotFoundError(
            f'Raw tfrecord directory not found: {raw_data_path}/{split}\n'
            f'Download Waymo Open Motion Dataset (scenario protocol) and place under {raw_data_path}/'
        )

    if max_files is not None:
        print(f'[prepare_data] Step 1 ({split}): --max_files={max_files} (test mode)')
        # Temporarily limit by preprocessing only max_files tfrecords
        # We call the low-level function directly with a subset
        import glob as _glob
        import pickle
        from functools import partial
        import multiprocessing
        from tqdm import tqdm
        from waymo_preprocess import process_waymo_data_with_scenario_proto

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
        out_dir = os.path.join(data_root, split_dir_map[split])
        info_file = os.path.join(data_root, info_file_map[split])
        os.makedirs(out_dir, exist_ok=True)

        src_files = sorted(_glob.glob(os.path.join(raw_data_path, split, '*.tfrecord*')))[:max_files]
        func = partial(process_waymo_data_with_scenario_proto, output_path=out_dir)
        with multiprocessing.Pool(min(num_workers, len(src_files))) as p:
            data_infos = list(tqdm(p.imap(func, src_files), total=len(src_files)))
        infos = [item for sub in data_infos for item in sub]
        with open(info_file, 'wb') as f:
            pickle.dump(infos, f)
        print(f'[prepare_data] Step 1 ({split}): {len(infos)} scenarios written')
    else:
        preprocess_split(raw_data_path, data_root, split, num_workers)


def run_step2(data_root: str, split: str, num_workers: int):
    """Step 2: MTR intermediate pkl → ScenarioNet pkl."""
    from convert_mtr_to_scenarionet import convert_split

    split_dir_map = {
        'training':   'processed_scenarios_training',
        'validation': 'processed_scenarios_validation',
        'testing':    'processed_scenarios_testing',
    }
    input_dir = os.path.join(data_root, split_dir_map[split])
    output_dir = os.path.join(data_root, 'waymo_scenarionet', split)
    convert_split(input_dir, output_dir, num_workers=num_workers)


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare(data_root: str, splits: list, num_workers: int,
            force_step1: bool = False, force_step2: bool = False,
            max_files: int = None):
    for split in splits:
        print(f'\n{"="*60}')
        print(f'[prepare_data] Processing split: {split}')
        print(f'{"="*60}')

        # ── Step 1 ────────────────────────────────────────────────────────────
        if _scenarionet_exists(data_root, split) and not force_step1 and not force_step2:
            print(f'[prepare_data] ScenarioNet pkl already exists for {split}, skipping both steps.')
            print(f'  → {data_root}/waymo_scenarionet/{split}/')
            continue

        if _mtr_pkl_exists(data_root, split) and not force_step1:
            print(f'[prepare_data] Step 1 ({split}): MTR pkl already exists, skipping.')
            print(f'  → {data_root}/processed_scenarios_{split if split != "validation" else "validation"}/')
        else:
            print(f'[prepare_data] Step 1 ({split}): tfrecord → MTR pkl ...')
            run_step1(data_root, split, num_workers, max_files=max_files)

        # ── Step 2 ────────────────────────────────────────────────────────────
        if _scenarionet_exists(data_root, split) and not force_step2:
            print(f'[prepare_data] Step 2 ({split}): ScenarioNet pkl already exists, skipping.')
            print(f'  → {data_root}/waymo_scenarionet/{split}/')
        else:
            print(f'[prepare_data] Step 2 ({split}): MTR pkl → ScenarioNet pkl ...')
            run_step2(data_root, split, num_workers)

    print(f'\n[prepare_data] Done. Data is ready at:')
    for split in splits:
        print(f'  {data_root}/waymo_scenarionet/{split}/')
    print('\nNext step — train with UniTraj:')
    print('  cd unitraj && python train.py method=MTR')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UniTraj data preparation: tfrecord → ScenarioNet pkl',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--data_root', type=str, default='data/waymo',
                        help='Root data directory (default: data/waymo, symlinked to data/waymo)')
    parser.add_argument('--split', type=str, default='all',
                        choices=['all', 'training', 'validation', 'testing'],
                        help='Which split(s) to process (default: all = training + validation)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers (default: 16)')
    parser.add_argument('--force_step1', action='store_true',
                        help='Re-run step 1 even if MTR pkl already exists')
    parser.add_argument('--force_step2', action='store_true',
                        help='Re-run step 2 even if ScenarioNet pkl already exists')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Limit number of tfrecord files in step 1 (for testing)')
    args = parser.parse_args()

    # Default: convert ALL splits so nothing is missing when training starts
    splits = ['training', 'validation'] if args.split == 'all' else [args.split]

    prepare(
        data_root=args.data_root,
        splits=splits,
        num_workers=args.num_workers,
        force_step1=args.force_step1,
        force_step2=args.force_step2,
        max_files=args.max_files,
    )
