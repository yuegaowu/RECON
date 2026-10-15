import glob
import hashlib
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
import h5py
import numpy as np
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from omegaconf import OmegaConf
from scenarionet.common_utils import read_scenario, read_dataset_summary
from torch.utils.data import Dataset
from tqdm import tqdm

from unitraj.datasets import common_utils
from unitraj.datasets.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty, get_trajectory_type
from unitraj.datasets.types import object_type, polyline_type
from unitraj.utils.visualization import check_loaded_data


def _rank0_print(*args, **kwargs):
    """Print only from global rank 0 to avoid 8x duplicate messages in DDP."""
    if int(os.environ.get('GLOBAL_RANK', '0')) == 0:
        print(*args, **kwargs)


def _resolve_data_count(usage, full_count):
    """把 max_data_num / max_val_data_num 的值解析成绝对场景数。

    支持三种语义:
      None              → 不截断,用全量(返回 None)
      int N (≥ 1)        → 取前 min(N, full_count) 个场景
      float f ∈ (0, 1)   → 取 int(full_count × f) 个场景(比例)
      float == 1.0       → 同 None(100% = 全量)

    返回:
      None  → 调用方应跳过截断
      int   → 截断到此数量
    """
    if usage is None:
        return None
    if isinstance(usage, bool):
        raise ValueError(f"max_data_num must be int / float / None, got bool {usage}")
    if isinstance(usage, float) and 0 < usage <= 1.0:
        if usage == 1.0:
            return None
        return max(1, int(full_count * usage))
    if isinstance(usage, (int, float)) and usage >= 1:
        return min(int(usage), full_count)
    raise ValueError(
        f"max_data_num={usage} 不合法。应为 None / int ≥1 / float ∈ (0, 1]"
    )
from functools import lru_cache

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)

# Path to UniTraj/datasets/ for importing conversion scripts
_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets', 'waymo_converter')


class EvaluationSceneSamples(list):
    """Picklable scene result, including legitimate exclusions for the parent loader."""

    def __init__(self, samples=(), audit=None):
        super().__init__(samples)
        self.audit = audit


class BaseDataset(Dataset):

    def __init__(self, config=None, is_validation=False):
        if is_validation:
            self.data_path = list(config['val_data_path'])
        else:
            self.data_path = list(config['train_data_path'])

        if len(self.data_path) == 0:
            raise ValueError(
                "No dataset paths configured.  Set train_data_path and val_data_path "
                "in config.yaml to at least one ScenarioNet directory."
            )

        joint_train = config.get('joint_train', False)
        if joint_train and len(self.data_path) < 2:
            raise ValueError(
                f"joint_train=True requires at least 2 dataset paths, "
                f"got {len(self.data_path)}.  Add more paths or set joint_train=False."
            )

        if not joint_train:
            self.data_path = self.data_path[:1]
            config['max_data_num'] = config['max_data_num'][:1]
            config['starting_frame'] = config['starting_frame'][:1]
            if 'max_val_data_num' in config:
                config['max_val_data_num'] = config['max_val_data_num'][:1]

        self.is_validation = is_validation
        self.config = config
        self.data_loaded_memory = []
        self.file_cache = {}
        # on-the-fly mode state
        self.onthefly_index = []        # list of (data_path, scene_name)
        self.onthefly_mappings = {}     # data_path -> mapping dict
        self.onthefly_starting_frames = {}  # data_path -> starting_frame
        # h5 cache mode state (scenario-level)
        self.h5_scenario_index = []     # list of scene_name (one entry per scenario)
        self.h5_scenario_groups = {}    # scene_name -> list of h5 keys for that scenario
        self._ensure_scenarionet_data()
        self.load_data()

    # ------------------------------------------------------------------
    # Data availability check — runs before load_data()
    # ------------------------------------------------------------------

    def _ensure_scenarionet_data(self):
        """Check that ScenarioNet pkl exists for ALL configured splits (training + validation).
        Runs at startup regardless of which split this instance handles, so both splits
        are ready before any training or validation begins.

        Logic per split:
          - ScenarioNet pkl exists      → skip (already ready)
          - MTR pkl exists, no SN pkl   → auto-run Step 2 (no TensorFlow needed)
          - Neither exists              → raise error with Step 1 setup instructions
        """
        split_dir_map = {
            'training':   'processed_scenarios_training',
            'validation': 'processed_scenarios_validation',
            'testing':    'processed_scenarios_testing',
        }

        # Check ALL configured paths (train + val) regardless of current instance type
        all_paths = list(self.config.get('train_data_path', [])) + \
                    list(self.config.get('val_data_path', []))

        missing_step1 = []  # splits where even MTR pkl is missing

        for data_path in all_paths:
            summary_path = os.path.join(data_path, 'dataset_summary.pkl')
            if os.path.isfile(summary_path):
                continue  # ScenarioNet pkl already exists for this split

            # data_path = <root>/waymo_scenarionet/<split>
            split = os.path.basename(data_path)
            data_root = os.path.dirname(os.path.dirname(data_path))
            mtr_dir = os.path.join(data_root, split_dir_map.get(split, f'processed_scenarios_{split}'))
            mtr_pkls = glob.glob(os.path.join(mtr_dir, 'sample_*.pkl'))

            if len(mtr_pkls) > 0:
                # Step 2: MTR pkl → ScenarioNet pkl (no TensorFlow, runs in unitraj env)
                _rank0_print(f'[UniTraj] ScenarioNet pkl missing for "{split}". '
                      f'Found {len(mtr_pkls)} MTR pkl files. '
                      f'Auto-generating ScenarioNet pkl (Step 2, full split)...')
                if _DATASETS_DIR not in sys.path:
                    sys.path.insert(0, _DATASETS_DIR)
                from convert_mtr_to_scenarionet import convert_split
                num_workers = max(1, min(os.cpu_count() // 2, 16))
                convert_split(mtr_dir, data_path, num_workers=num_workers)
                _rank0_print(f'[UniTraj] ScenarioNet pkl ready: {data_path}')
            else:
                missing_step1.append(split)

        if missing_step1:
            raise FileNotFoundError(
                f'\n\nStep 1 data not found for splits: {missing_step1}\n'
                f'Neither MTR pkl nor ScenarioNet pkl exists.\n\n'
                f'You must run Step 1 first (requires TensorFlow environment):\n\n'
                f'  # One-time environment setup:\n'
                f'  conda create -n waymo_preprocess python=3.9 -y\n'
                f'  conda activate waymo_preprocess\n'
                f'  pip install -r datasets/waymo_converter/requirements_preprocess.txt\n\n'
                f'  # Convert ALL splits (training + validation) — run once:\n'
                f'  cd /path/to/UniTraj\n'
                f'  python datasets/waymo_converter/prepare_data.py   # default: --split all\n\n'
                f'  # Then switch back and train:\n'
                f'  conda activate unitraj\n'
                f'  python train.py method=MTR\n'
            )

    def load_data(self):
        self.data_loaded = {}
        if self.is_validation:
            _rank0_print('Loading validation data...')
        else:
            _rank0_print('Loading training data...')

        # On-the-fly mode: skip h5 entirely, build lightweight index from summary
        if self.config.get('onthefly', False):
            self._load_data_onthefly()
            return

        for cnt, data_path in enumerate(self.data_path):
            phase, dataset_name = data_path.split('/')[-2],data_path.split('/')[-1]
            self.cache_path = os.path.join(self.config['cache_path'], dataset_name, phase)

            data_usage_this_dataset = self.config['max_data_num'][cnt]
            self.starting_frame = self.config['starting_frame'][cnt]
            if self.config['use_cache'] or is_ddp():
                file_list = self.get_data_list(data_usage_this_dataset)
            else:
                cache_exists = os.path.exists(self.cache_path)
                overwrite = self.config.get('overwrite_cache', False)

                # overwrite_cache only applies to training h5.
                # Validation h5 is full and fixed — never delete it.
                should_overwrite = overwrite and not self.is_validation

                if cache_exists and not should_overwrite:
                    if cache_exists and overwrite and self.is_validation:
                        _rank0_print(f'[h5] Validation cache kept (overwrite_cache ignored for validation): '
                              f'{self.cache_path}')
                    else:
                        _rank0_print(f'[h5] Cache exists, skipping generation: {self.cache_path}')
                    file_list = self.get_data_list(data_usage_this_dataset)
                else:

                    _, summary_list, mapping = read_dataset_summary(data_path)

                    # 训练用 max_data_num,验证用 max_val_data_num(两者都支持 int / float 比例 / None)
                    if self.is_validation:
                        usage_list = self.config.get('max_val_data_num', [None] * len(self.data_path))
                    else:
                        usage_list = self.config['max_data_num']
                    usage_raw = usage_list[cnt] if cnt < len(usage_list) else None
                    target_count = _resolve_data_count(usage_raw, len(summary_list))

                    if target_count is not None and target_count < len(summary_list):
                        full_size = len(summary_list)
                        np.random.shuffle(summary_list)
                        summary_list = summary_list[:target_count]
                        split_tag = 'Validation' if self.is_validation else 'Training'
                        _rank0_print(f'[h5] {split_tag}: generating h5 for {len(summary_list)} / {full_size} scenarios '
                              f'(usage={usage_raw})')
                    else:
                        _rank0_print(f'[h5] {"Validation" if self.is_validation else "Training"}: '
                              f'generating h5 for all {len(summary_list)} scenarios')

                    if os.path.exists(self.cache_path):
                        shutil.rmtree(self.cache_path)
                    os.makedirs(self.cache_path, exist_ok=True)
                    process_num = os.cpu_count()//2
                    _rank0_print('Using {} processes to load data...'.format(process_num))

                    data_splits = np.array_split(summary_list, process_num)

                    data_splits = [(data_path, mapping, list(data_splits[i]), dataset_name) for i in range(process_num)]
                    # save the data_splits in a tmp directory
                    os.makedirs('tmp', exist_ok=True)
                    for i in range(process_num):
                        with open(os.path.join('tmp', '{}.pkl'.format(i)), 'wb') as f:
                            pickle.dump(data_splits[i], f)

                    # results = self.process_data_chunk(0)
                    with Pool(processes=process_num) as pool:
                        results = pool.map(self.process_data_chunk, list(range(process_num)))

                    # concatenate the results
                    file_list = {}
                    for result in results:
                        file_list.update(result)

                    with open(os.path.join(self.cache_path, 'file_list.pkl'), 'wb') as f:
                        pickle.dump(file_list, f)
                    with open(os.path.join(self.cache_path, 'cache_meta.pkl'), 'wb') as f:
                        pickle.dump(self._cache_metadata(), f)

                    # All objects from the selected scenarios are written to h5.
                    # max_data_num controls scenario count; all resulting objects are indexed.

            # Build scenario-level groups: scene_name -> [h5_key_0, h5_key_1, ...]
            # h5 key format: "{dataset_name}__{scene_name}__{obj_idx}"
            for key in file_list:
                parts = key.split('__')
                scene_name = parts[1] if len(parts) >= 3 else key
                self.h5_scenario_groups.setdefault(scene_name, []).append(key)

            self.data_loaded.update(file_list)

            if self.config['store_data_in_memory']:
                _rank0_print('Loading data into memory...')
                for data_path in file_list.keys():
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    self.data_loaded_memory.append(data)
                _rank0_print('Loaded {} data into memory'.format(len(self.data_loaded_memory)))

        self.data_loaded_keys = list(self.data_loaded.keys())
        # h5 mode: scenario-level index (one entry per scenario, consistent with on-the-fly)
        self.h5_scenario_index = list(self.h5_scenario_groups.keys())
        n_scenarios = len(self.h5_scenario_index)
        n_objects = len(self.data_loaded_keys)
        if n_scenarios > 0:
            _rank0_print(f'Loaded {n_scenarios} scenarios ({n_objects} total objects) from h5 cache')
        _rank0_print('Data loaded')

    # ------------------------------------------------------------------
    # On-the-fly mode: no h5 cache, load ScenarioNet pkl in each worker
    # ------------------------------------------------------------------

    def _load_data_onthefly(self):
        """Build a scene-level index: one entry per scene.
        __len__ = N_scenes.  Each __getitem__ loads the pkl once and returns
        ALL interesting agents from that scene as a list of dicts.
        collate_fn flattens the list before stacking into batch tensors.

        This avoids the N-fold redundant I/O that would occur if we indexed
        per-agent and re-loaded the same pkl N times per scene per epoch."""
        for cnt, data_path in enumerate(self.data_path):
            starting_frame = self.config['starting_frame'][cnt]

            summary_dict, summary_list, mapping = read_dataset_summary(data_path)

            self.onthefly_mappings[data_path] = mapping
            self.onthefly_starting_frames[data_path] = starting_frame

            # 训练用 max_data_num,验证用 max_val_data_num(两者都支持 int / float 比例 / None)
            if self.is_validation:
                usage_list = self.config.get('max_val_data_num', [None] * len(self.data_path))
            else:
                usage_list = self.config['max_data_num']
            usage_raw = usage_list[cnt] if cnt < len(usage_list) else None
            target_count = _resolve_data_count(usage_raw, len(summary_list))

            if target_count is not None and target_count < len(summary_list):
                full_size = len(summary_list)
                if self.is_validation and self.config.get('strict_eval_support', False):
                    summary_list = sorted(summary_list)
                    order = np.random.default_rng(int(self.config.get('seed', 42))).permutation(len(summary_list))
                    summary_list = [summary_list[i] for i in order[:target_count]]
                else:
                    np.random.shuffle(summary_list)
                    summary_list = summary_list[:target_count]
                split_tag = 'val' if self.is_validation else 'train'
                _rank0_print(f'[BaseDataset] {split_tag} 截断: '
                             f'{data_path} → {target_count} / {full_size} (usage={usage_raw})')

            n_agent_entries = 0
            for scene_name in summary_list:
                scene_meta = summary_dict.get(scene_name, {})
                ttp = scene_meta.get('tracks_to_predict', {})
                self.onthefly_index.append((data_path, scene_name))
                n_agent_entries += len(ttp)

            _rank0_print(f'On-the-fly: indexed {len(summary_list)} scenes '
                         f'(≈{n_agent_entries} agent-samples) in {data_path}')

        _rank0_print('Data loaded')

    def _getitem_onthefly(self, idx, _depth=0):
        """Load ScenarioNet pkl once, process all agents, return the full agent list.

        Index entry: (data_path, scene_name)
        Returns a list of per-agent dicts so collate_fn can flatten across scenes.
        One pkl read + one preprocess/process call covers all agents in the scene,
        eliminating the N-fold redundant I/O of the previous per-agent indexing scheme.

        _depth: recursion counter; raises RuntimeError after 50 retries."""
        if self.is_validation and self.config.get('strict_eval_support', False):
            return self._getitem_eval_strict(idx)
        if _depth >= 50:
            raise RuntimeError(
                f'[BaseDataset] _getitem_onthefly: failed to find a valid sample after 50 retries. '
                f'Check that object_type config matches agents present in the dataset.'
            )

        data_path, scene_name = self.onthefly_index[idx]
        mapping = self.onthefly_mappings[data_path]
        self.starting_frame = self.onthefly_starting_frames[data_path]

        # ── 事故域「碰撞前 TTC 滑窗」评测分支(accident_ttc_eval=True 时启用,默认关 → 训练/常规评测零影响)──
        #   只对肇事车:在碰撞帧 C 之前 T=offsets 秒各锚一次,把场景裁成以该锚点为第 11 帧的子窗,
        #   现有 process/get_agent_data 原样复用,每条记录打 accident_ttc=T 标签供 benchmark_eval 分组。
        if self.config.get('accident_ttc_eval', False):
            try:
                output = self._build_accident_ttc_samples(data_path, mapping, scene_name)
            except RuntimeError:
                raise
            except Exception:
                output = None
            if not output:
                return self._getitem_onthefly(
                    np.random.randint(0, len(self.onthefly_index)), _depth=_depth + 1)
            return output

        try:
            scenario = read_scenario(data_path, mapping, scene_name)
            output = self.preprocess(scenario)
            output = self.process(output)
            if output is None or len(output) == 0:
                return self._getitem_onthefly(
                    np.random.randint(0, len(self.onthefly_index)), _depth=_depth + 1)
            output = self.postprocess(output)
        except RuntimeError:
            raise  # propagate depth-limit error
        except Exception:
            # Some scenarios have no valid prediction targets after filtering —
            # silently fall back to a random scene. Not an error.
            return self._getitem_onthefly(
                np.random.randint(0, len(self.onthefly_index)), _depth=_depth + 1)

        # Return ALL agents from this scene.
        # collate_fn will flatten the list-of-lists into a flat batch.
        return output

    def _getitem_eval_strict(self, idx):
        """Evaluate the requested scene once; never replace it with a random scene."""
        data_path, scene_name = self.onthefly_index[idx]
        mapping = self.onthefly_mappings[data_path]
        self.starting_frame = self.onthefly_starting_frames[data_path]
        is_ttc = bool(self.config.get('accident_ttc_eval', False))
        audit = {'index': int(idx), 'data_path': str(data_path), 'scene_file': str(scene_name),
                 'protocol': 'ttc' if is_ttc else 'regular', 'starting_frame': int(self.starting_frame)}
        self._strict_eval_exclusions = []
        try:
            if is_ttc:
                output = self._build_accident_ttc_samples(data_path, mapping, scene_name)
            else:
                scenario = read_scenario(data_path, mapping, scene_name)
                info = self.preprocess(scenario)
                audit['scene'] = str(info['scenario_id'])
                audit['anchor_index'] = int(info['current_time_index'])
                selected = info['tracks_to_predict']['track_index']
                audit['selected_ids'] = [str(info['track_infos']['object_id'][i]) for i in selected]
                audit['anchor_valid_ids'] = [str(info['track_infos']['object_id'][i]) for i in selected
                                            if info['track_infos']['trajs'][i, info['current_time_index'], -1]]
                output = self.process(info)
                if output:
                    output = self.postprocess(output)
        except Exception as exc:
            raise RuntimeError(f'Strict evaluation failed for {data_path}/{scene_name}: {exc}') from exc
        output = output or []
        audit['n_targets'] = len(output)
        audit['excluded_targets'] = self._strict_eval_exclusions
        audit['status'] = 'ok' if output else 'excluded_no_eligible_target'
        audit['output_keys'] = [[str(r['scenario_id']), str(r['center_objects_id']),
                                 int(r['accident_ttc']) if 'accident_ttc' in r else None]
                                for r in output]
        return EvaluationSceneSamples(output, audit)

    # ─────────────────────────────────────────────────────────────────────────
    # 事故域「碰撞前 TTC 滑窗」评测支持(仅 accident_ttc_eval=True 启用)
    # ─────────────────────────────────────────────────────────────────────────
    def _collision_anchor(self, ret):
        """返回 (碰撞帧 C(局部 0-based), 肇事车 object_id 集合)。
        DA:metadata.collision.collision_frame 直接给(已本地化);
        realaccident:collision_frame 为 None → 碰撞锚定末帧 → 取肇事车最后 valid 帧。
        无肇事标注 → (None, None)。"""
        col = ret.get('collision', {}) or {}
        pairs = col.get('collision_pairs', []) or []
        obj_ids = ret['track_infos']['object_id']
        collider_ids = {str(x) for pr in pairs for x in pr if str(x) in obj_ids}
        if not collider_ids:
            return None, None
        cf = col.get('collision_frame', None)
        if cf is not None:
            return int(cf), collider_ids
        trajs = ret['track_infos']['trajs']
        last = -1
        for cid in collider_ids:
            v = np.flatnonzero(trajs[obj_ids.index(cid)][:, -1] > 0)
            if len(v):
                last = max(last, int(v.max()))
        return (last if last >= 0 else None), collider_ids

    def _reanchor_scene(self, ret, anchor_frame, collider_ids):
        """把整条场景裁成以 anchor_frame 为「当前帧(第 past_len-1 帧)」的 91 帧子窗:
        history=[A-10, A],future=[A+1, A+80];越界帧补零(valid=0)→ 自动被存活门控掉。
        tracks_to_predict 只留在锚点帧 valid 的肇事车(出现晚的车在远 T 自动被排除)。
        返回可直接喂 process() 的 internal_format;无可评肇事车 → None。"""
        past_len = self.config['past_len']
        future_len = self.config['future_len']
        W = past_len + future_len
        lo = anchor_frame - (past_len - 1)
        trajs = ret['track_infos']['trajs']
        N, T, D = trajs.shape
        idxs = lo + np.arange(W)
        in_range = (idxs >= 0) & (idxs < T)
        new = np.zeros((N, W, D), dtype=trajs.dtype)
        if in_range.any():
            new[:, in_range] = trajs[:, idxs[in_range]]

        obj_ids = ret['track_infos']['object_id']
        otypes = ret['track_infos']['object_type']
        sel_idx, sel_types = [], []
        for cid in collider_ids:
            if cid in obj_ids:
                i = obj_ids.index(cid)
                if new[i, past_len - 1, -1] > 0:   # 肇事车在锚点帧必须 valid
                    sel_idx.append(i)
                    sel_types.append(otypes[i])
        if not sel_idx:
            return None

        sub = dict(ret)
        sub_ti = dict(ret['track_infos'])
        sub_ti['trajs'] = new
        sub['track_infos'] = sub_ti
        sub['timestamps_seconds'] = (np.arange(W, dtype=np.float32) * 0.1)
        sub['current_time_index'] = past_len - 1
        sub['track_length'] = W
        sub['tracks_to_predict'] = {'track_index': sel_idx, 'object_type': sel_types}
        return sub

    def _build_accident_ttc_samples(self, data_path, mapping, scene_name):
        """对一个碰撞场景,按 accident_ttc_offsets 的每个 T(秒)生成一组肇事车预测样本。
        锚点 A = C - 10*T(10Hz);每条记录加 accident_ttc=T。无任何可评样本 → None。"""
        scenario = read_scenario(data_path, mapping, scene_name)
        ret = self.preprocess(scenario)
        C, collider_ids = self._collision_anchor(ret)
        if C is None or not collider_ids:
            return None
        offsets = self.config.get('accident_ttc_offsets', [1, 2, 3, 4, 5, 6, 7, 8])
        records = []
        for T in offsets:
            sub = self._reanchor_scene(ret, int(round(C - 10 * float(T))), collider_ids)
            if sub is None:
                continue
            out = self.process(sub)
            if not out:
                continue
            for rec in out:
                rec['accident_ttc'] = np.int64(int(T))
            out = self.postprocess(out)
            records.extend(out)
        return records or None

    def process_data_chunk(self, worker_index):
        with open(os.path.join('tmp', '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f'{worker_index}.h5')

        with h5py.File(hdf5_path, 'w') as f:
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f'{cnt}/{len(data_list)} data processed', flush=True)
                scenario = read_scenario(data_path, mapping, file_name)

                try:
                    output = self.preprocess(scenario)

                    output = self.process(output)

                    output = self.postprocess(output)

                except Exception as e:
                    _rank0_print('Warning: {} in {}'.format(e, file_name))
                    output = None

                if output is None: continue

                for i, record in enumerate(output):
                    # Key format: "{dataset_name}__{scene_filename}__{obj_idx}"
                    # The double-underscore separator allows parsing scene_name at load time.
                    grp_name = f'{dataset_name}__{file_name}__{i}'
                    grp = f.create_group(grp_name)
                    for key, value in record.items():
                        if isinstance(value, str):
                            value = np.bytes_(value)
                        grp.create_dataset(key, data=value)
                    file_info = {}
                    kalman_difficulty = np.stack([x['kalman_difficulty'] for x in output])
                    file_info['kalman_difficulty'] = kalman_difficulty
                    file_info['h5_path'] = hdf5_path
                    file_list[grp_name] = file_info
                del scenario
                del output

        return file_list

    def preprocess(self, scenario):

        traffic_lights = scenario['dynamic_map_states']
        tracks = scenario['tracks']
        map_feat = scenario['map_features']

        past_length = self.config['past_len']
        future_length = self.config['future_len']
        total_steps = past_length + future_length
        starting_fame = self.starting_frame
        strict_ttc = (self.is_validation and self.config.get('strict_eval_support', False)
                      and self.config.get('accident_ttc_eval', False))
        if strict_ttc:
            if starting_fame != 0:
                raise ValueError('Strict TTC requires original scene frame indices (starting_frame=0)')
            # Reanchor against the full original clip. Truncating to 91 first
            # loses later collision frames and can shift short clips' anchors.
            total_steps = max(len(v['state']['valid']) for v in tracks.values())
        ending_fame = starting_fame + total_steps
        trajectory_sample_interval = self.config['trajectory_sample_interval']
        frequency_mask = generate_mask(past_length - 1, total_steps, trajectory_sample_interval)

        track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': []
        }

        for k, v in tracks.items():

            state = v['state']
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [state['position'], state['length'], state['width'], state['height'], state['heading'],
                         state['velocity'], state['valid']]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)
            # all_state = all_state[::sample_inverval]
            if all_state.shape[0] < ending_fame:
                missing = ending_fame - all_state.shape[0]
                # Revision protocol: regular evaluation anchors on the original
                # scene's 11th frame; pad unavailable FUTURE, not observed history.
                # Keep training and the separately reanchored legacy TTC path intact.
                strict_regular = (self.is_validation and self.config.get('strict_eval_support', False)
                                  and not self.config.get('accident_ttc_eval', False))
                padding = (0, missing) if strict_regular or strict_ttc else (missing, 0)
                all_state = np.pad(all_state, (padding, (0, 0)))
            all_state = all_state[starting_fame:ending_fame]

            assert all_state.shape[0] == total_steps, f'Error: {all_state.shape[0]} != {total_steps}'
            
            track_infos['object_id'].append(k)
            track_infos['object_type'].append(object_type[v['type']])
            track_infos['trajs'].append(all_state)

        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)
        # scenario['metadata']['ts'] = scenario['metadata']['ts'][::sample_inverval]
        track_infos['trajs'][..., -1] *= frequency_mask[np.newaxis]
        scenario['metadata']['ts'] = scenario['metadata']['ts'][:total_steps]

        # x,y,z,type
        map_infos = {
            'lane': [],
            'road_line': [],
            'road_edge': [],
            'stop_sign': [],
            'crosswalk': [],
            'speed_bump': [],
        }
        polylines = []
        point_cnt = 0
        for k, v in map_feat.items():
            polyline_type_ = polyline_type[v['type']]
            if polyline_type_ == 0:
                continue

            cur_info = {'id': k}
            cur_info['type'] = v['type']
            if polyline_type_ in [1, 2, 3]:
                cur_info['speed_limit_mph'] = v.get('speed_limit_mph', None)
                cur_info['interpolating'] = v.get('interpolating', None)
                cur_info['entry_lanes'] = v.get('entry_lanes', None)
                try:
                    cur_info['left_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['left_neighbor']
                    ]
                    cur_info['right_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['right_neighbor']
                    ]
                except:
                    cur_info['left_boundary'] = []
                    cur_info['right_boundary'] = []
                polyline = v['polyline']
                map_infos['lane'].append(cur_info)
            elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:
                try:
                    polyline = v['polyline']
                except:
                    polyline = v['polygon']
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [15, 16]:
                polyline = v['polyline']
                cur_info['type'] = 7
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [17]:
                cur_info['lane_ids'] = v['lane']
                cur_info['position'] = v['position']
                map_infos['stop_sign'].append(cur_info)
                polyline = v['position'][np.newaxis]
            elif polyline_type_ in [18]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            elif polyline_type_ in [19]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            if polyline.shape[-1] == 2:
                polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)
            try:
                cur_polyline_dir = get_polyline_dir(polyline)
                type_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = polyline_type_
                cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1)
            except:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
        map_infos['all_polylines'] = polylines

        dynamic_map_infos = {
            'lane_id': [],
            'state': [],
            'stop_point': []
        }
        for k, v in traffic_lights.items():  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in v['state']['object_state']:  # (num_observed_signals)
                lane_id.append(str(v['lane']))
                state.append(cur_signal)
                if type(v['stop_point']) == list:
                    stop_point.append(v['stop_point'])
                else:
                    stop_point.append(v['stop_point'].tolist())
            # lane_id = lane_id[::sample_inverval]
            # state = state[::sample_inverval]
            # stop_point = stop_point[::sample_inverval]
            lane_id = lane_id[:total_steps]
            state = state[:total_steps]
            stop_point = stop_point[:total_steps]
            dynamic_map_infos['lane_id'].append(np.array([lane_id]))
            dynamic_map_infos['state'].append(np.array([state]))
            dynamic_map_infos['stop_point'].append(np.array([stop_point]))

        ret = {
            'track_infos': track_infos,
            'dynamic_map_infos': dynamic_map_infos,
            'map_infos': map_infos
        }
        ret.update(scenario['metadata'])
        ret['timestamps_seconds'] = ret.pop('ts')
        ret['current_time_index'] = self.config['past_len'] - 1
        ret['sdc_track_index'] = track_infos['object_id'].index(ret['sdc_id'])

        ret['tracks_to_predict'] = self._build_tracks_to_predict(ret, track_infos)

        ret['map_center'] = scenario['metadata'].get('map_center', np.zeros(3))[np.newaxis]

        ret['track_length'] = total_steps
        return ret

    def _build_tracks_to_predict(self, ret, track_infos):
        """按全局 prediction_targets(crash/sdc/official/all)选预测目标(center_object / SMART 窗口锚点)。

        规范见 UniTraj/PREDICTION_TARGETS.md。MTR 等 agent-centric 与 SMART 窗口化共用此选择,口径一致。
        铁律:crash ⊆ official ⊆ all 且 sdc ⊆ official ⊆ all;official/all = 基筛选 ∪ crash ∪ sdc
        (crash/sdc 强制并入,永不被筛选踢掉)。依赖 ret 已 update(scenario metadata)(含 sdc_id、
        native tracks_to_predict)。返回 {'track_index': [...], 'object_type': [...]},不修改 ret。
        """
        mode = self.config.get('prediction_targets', 'official')
        obj_ids = track_infos['object_id']

        crash_ids = [i for i in self._collect_crash_ids(ret) if i in obj_ids]
        sdc_as_target = self.config.get('sdc_as_target', True)   # realaccident/highD/exiD 置 False
        sdc_ids = [ret['sdc_id']] if (sdc_as_target and ret.get('sdc_id') in obj_ids) else []

        def _official_base():
            # official base = native tracks_to_predict(可用时)否则严格筛选。
            # native 仅含 crash 对象(realaccident 只标事故车)→ 不可用 → 回退严格筛选。
            native = ret.get('tracks_to_predict', None)
            if native:
                keys = [k for k in set(native.keys()) if k in obj_ids]
                if any(k not in crash_ids for k in keys):
                    return keys
            return list(self.trajectory_filter(ret, min_moving_distance=2.0).keys())

        if mode == 'sdc':
            sample_list = list(sdc_ids)
        elif mode == 'crash':
            sample_list = list(crash_ids)
        elif mode == 'all':
            base = list(self.trajectory_filter(ret, min_moving_distance=0.0).keys())
            sample_list = base + crash_ids + sdc_ids
        else:  # 'official' 及未知模式
            sample_list = _official_base() + crash_ids + sdc_ids

        seen, sel_index, sel_types = set(), [], []
        for _id in sample_list:
            if _id not in obj_ids:
                continue
            idx = obj_ids.index(_id)
            if idx in seen:
                continue
            seen.add(idx)
            sel_index.append(idx)
            sel_types.append(track_infos['object_type'][idx])
        return {'track_index': sel_index, 'object_type': sel_types}

    def process(self, internal_format):

        info = internal_format
        scene_id = info['scenario_id']

        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)
        # 时间戳归一化到窗口首帧=0:模型训练时 time-embedding 通道见到的是 0~1.0(Waymo/DA/highD 的 ts 本就从 0 起);
        #   realaccident 用「事件相对」的负时间戳(ts=[-9.1,…,0.0]),不归一化会把 -9 级 OOD 值喂进 time-embedding
        #   (见 get_agent_data 的 object_time_embedding[...,-1]=timestamps),导致模型严重低估速度。对 0 起点数据是 no-op。
        timestamps = timestamps - timestamps[0]

        track_infos = info['track_infos']

        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )
        if center_objects is None: return None

        sample_num = center_objects.shape[0]

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
         obj_trajs_future_mask, center_gt_trajs,
         center_gt_trajs_mask, center_gt_final_valid_idx,
         track_index_to_predict_new) = self.get_agent_data(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types
        )

        ret_dict = {
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],
            'map_center': info['map_center'],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        if info['map_infos']['all_polylines'].__len__() == 0:
            info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            _rank0_print(f'Warning: empty HDMap {scene_id}')

        if self.config.manually_split_lane:
            map_polylines_data, map_polylines_mask, map_polylines_center = self.get_manually_split_map_data(
                center_objects=center_objects, map_infos=info['map_infos'])
        else:
            map_polylines_data, map_polylines_mask, map_polylines_center = self.get_map_data(
                center_objects=center_objects, map_infos=info['map_infos'])

        ret_dict['map_polylines'] = map_polylines_data
        ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
        ret_dict['map_polylines_center'] = map_polylines_center

        # masking out unused attributes to Zero
        masked_attributes = self.config['masked_attributes']
        if 'z_axis' in masked_attributes:
            ret_dict['obj_trajs'][..., 2] = 0
            ret_dict['map_polylines'][..., 2] = 0
        if 'size' in masked_attributes:
            ret_dict['obj_trajs'][..., 3:6] = 0
        if 'velocity' in masked_attributes:
            ret_dict['obj_trajs'][..., 25:27] = 0
        if 'acceleration' in masked_attributes:
            ret_dict['obj_trajs'][..., 27:29] = 0
        if 'heading' in masked_attributes:
            ret_dict['obj_trajs'][..., 23:25] = 0

        # change every thing to float32
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        if self.config['remove_outliers']: #remove agents that are further away than 5m from the nearest lane at the central time step -> filtering step done in fmae/emp data preprocessing
            lane_samples = torch.from_numpy(map_polylines_data[..., :2]).view(-1, 2)
            # CPU-only: DataLoader workers must not call .cuda() — CUDA contexts are not fork-safe.
            distances_to_nearest_lane = torch.cdist(torch.from_numpy(ret_dict["obj_trajs_pos"][:, 1:, self.config["past_len"]-1, :2]).float(), lane_samples.float()).min(dim=-1).values
            ret_dict["obj_trajs_mask"][:, 1:] = np.array(torch.where(
                (distances_to_nearest_lane >= 5).unsqueeze(-1),
                torch.zeros_like(torch.from_numpy(ret_dict["obj_trajs_mask"][:, 1:])),
                torch.from_numpy(ret_dict["obj_trajs_mask"][:, 1:])
            ))

        ret_dict['map_center'] = ret_dict['map_center'].repeat(sample_num, axis=0)
        ret_dict['dataset_name'] = [info['dataset']] * sample_num

        ret_list = []
        for i in range(sample_num):
            ret_dict_i = {}
            for k, v in ret_dict.items():
                ret_dict_i[k] = v[i]
            ret_list.append(ret_dict_i)

        return ret_list

    def postprocess(self, output):

        if self.is_validation and self.config.get('strict_eval_support', False):
            # An agent with no observed future has no defined displacement error.
            # Record its exclusion; never substitute another scene or agent.
            eligible = []
            for sample in output:
                if np.asarray(sample['center_gt_trajs_mask']).any():
                    eligible.append(sample)
                else:
                    self._strict_eval_exclusions.append({
                        'scene': str(sample['scenario_id']),
                        'oid': str(sample['center_objects_id']),
                        'ttc': int(sample['accident_ttc']) if 'accident_ttc' in sample else None,
                        'reason': 'no_valid_future'})
            get_kalman_difficulty(eligible)
            typed = []
            for sample in eligible:
                if int(sample['center_gt_final_valid_idx']) < 1:
                    # A single future frame is evaluable but has no future heading.
                    sample['trajectory_type'] = -1
                else:
                    typed.append(sample)
            get_trajectory_type(typed)
            return eligible

        # Add the trajectory difficulty
        get_kalman_difficulty(output)

        # Add the trajectory type (stationary, straight, right turn...)
        get_trajectory_type(output)

        return output

    def collate_fn(self, data_list):
        # On-the-fly mode: each item in data_list is a list of per-agent dicts
        # (one scene → multiple agents). Flatten before collation so the rest of
        # the pipeline sees a flat list of agent dicts, exactly as before.
        # H5 mode: each item is already a single agent dict — no change needed.
        strict = self.is_validation and self.config.get('strict_eval_support', False)
        audit = [item.audit for item in data_list if isinstance(item, EvaluationSceneSamples)]
        batch_list = []
        for item in data_list:
            if isinstance(item, list):
                batch_list.extend(item)   # expand scene → agents
            else:
                batch_list.append(item)   # h5 mode: already one agent

        # Hard cap: prevent accidental OOM on batches that happen to contain many
        # agent-dense scenes. The cap is hit only in rare cases (e.g. ten 8-agent
        # scenes in one batch); randomly sub-sample to stay within the limit.
        # Agents dropped here will appear in the same scene in later epochs.
        max_agents = self.config.get('max_agents_per_batch', 128)
        if len(batch_list) > max_agents and not strict:
            keep = np.random.choice(len(batch_list), max_agents, replace=False)
            batch_list = [batch_list[int(i)] for i in sorted(keep)]

        batch_size = len(batch_list)
        if strict and not batch_size:
            return {'batch_size': 0, 'batch_sample_count': 0, 'input_dict': {},
                    'evaluation_support': audit, 'eval_microbatch_size': int(max_agents)}
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            try:
                input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
            except (ValueError, TypeError):
                # Variable-length or non-array fields (e.g. strings, ragged arrays) → keep as list
                input_dict[key] = val_list

        input_dict['center_objects_type'] = input_dict['center_objects_type'].numpy()

        batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_size}
        if strict:
            batch_dict.update(evaluation_support=audit, eval_microbatch_size=int(max_agents))
        return batch_dict

    def __len__(self):
        if self.config.get('onthefly', False):
            # one entry per scene — __getitem__ returns a list of all agents in the scene
            return len(self.onthefly_index)
        # h5 mode: one entry per scenario
        return len(self.h5_scenario_index)

    def _get_file(self, file_path):
        # Do NOT use @lru_cache here: h5py.File handles are not fork-safe.
        # Each DataLoader worker inherits the parent's lru_cache via fork(), and
        # multiple workers sharing the same h5py handle causes data corruption.
        # The per-instance file_cache dict (set in __getitem__) is safe because
        # each forked worker gets its own copy of self.file_cache.
        return h5py.File(file_path, 'r')

    def __getitem__(self, idx):
        if self.config.get('onthefly', False):
            return self._getitem_onthefly(idx)

        # h5 mode: return ALL objects of this scenario as a list (consistent with on-the-fly),
        #   so collate_fn flattens across train_batch_size scenes and caps at max_agents_per_batch.
        #   This makes h5 and on-the-fly identical in batch semantics: train_batch_size = #scenes,
        #   effective batch = min(sum of objects over those scenes, max_agents_per_batch).
        #   (Previously this sampled ONE random object per scene → h5 batch = train_batch_size
        #   objects, which did NOT match on-the-fly; switching modes silently changed the batch.)
        scene_name = self.h5_scenario_index[idx]
        keys = self.h5_scenario_groups[scene_name]
        records = []
        for file_key in keys:
            file_path = self.data_loaded[file_key]['h5_path']
            if file_path not in self.file_cache:
                self.file_cache[file_path] = self._get_file(file_path)
            group = self.file_cache[file_path][file_key]
            records.append({k: group[k][()].decode('utf-8') if group[k].dtype.type == np.bytes_ else group[k][()] for k in group.keys()})
        return records

    def close_files(self):
        for f in self.file_cache.values():
            f.close()
        self.file_cache.clear()

    def get_data_list(self, data_usage):
        file_list_path = os.path.join(self.cache_path, 'file_list.pkl')
        if not os.path.exists(file_list_path):
            raise ValueError(
                f'H5 cache not found: {file_list_path}\n'
                f'Set use_cache: False to generate it, or switch to onthefly: True mode.'
            )
        meta_path = os.path.join(self.cache_path, 'cache_meta.pkl')
        if not os.path.exists(meta_path):
            raise ValueError(
                f'H5 cache predates the unified input protocol: {self.cache_path}\n'
                f'Use a new cache_path, or regenerate explicitly with overwrite_cache: True.'
            )
        with open(meta_path, 'rb') as f:
            cached_meta = pickle.load(f)
        expected_meta = self._cache_metadata()
        if cached_meta != expected_meta:
            raise ValueError(
                f'H5 cache protocol mismatch: {self.cache_path}\n'
                f'cached={cached_meta}\nexpected={expected_meta}\n'
                f'Use a separate cache_path or regenerate explicitly.'
            )
        # Return all objects from the h5 cache without truncation.
        # Scenario count was already limited at h5 generation time via max_data_num.
        data_loaded = pickle.load(open(file_list_path, 'rb'))
        return data_loaded

    def _cache_metadata(self):
        keys = (
            'h5_cache_format',
            'model_name',
            'past_len',
            'future_len',
            'trajectory_sample_interval',
            'starting_frame',
            'max_data_num',
            'max_val_data_num',
            'prediction_targets',
            'sdc_as_target',
            'object_type',
            'line_type',
            'masked_attributes',
            'remove_outliers',
            'max_num_agents',
            'max_agent_distance',
            'agent_center_offset',
            'center_offset_of_map',
            'map_range',
            'map_crop_shape',
            'max_num_roads',
            'max_points_per_lane',
            'manually_split_lane',
            'point_sampled_interval',
            'num_points_each_polyline',
            'vector_break_dist_thresh',
            'smart_window_radius',
            'smart_map_crop_radius',
        )

        def normalize(value):
            if OmegaConf.is_config(value):
                return OmegaConf.to_container(value, resolve=True)
            if hasattr(value, 'items'):
                return {str(k): normalize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value

        from unitraj.utils.model_names import legacy_model_key
        protocol = {key: normalize(self.config.get(key, None)) for key in keys}
        # A naming alias must not invalidate an otherwise identical H5 cache.
        protocol['model_name'] = legacy_model_key(protocol['model_name'])
        serialized = json.dumps(protocol, sort_keys=True, separators=(',', ':'))
        return {
            'version': 1,
            'protocol': protocol,
            'sha256': hashlib.sha256(serialized.encode('utf-8')).hexdigest(),
        }

    def get_agent_data(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types
    ):

        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 1, :, 0] = 1
        object_onehot_mask[:, obj_types == 2, :, 1] = 1
        object_onehot_mask[:, obj_types == 3, :, 2] = 1
        object_onehot_mask[np.arange(num_center_objects), track_index_to_predict, :, 3] = 1
        object_onehot_mask[:, sdc_track_index, :, 4] = 1

        object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps

        object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]

        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, 0:6],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9],
            acce,
        ], axis=-1)

        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)
        valid_past_indices = np.flatnonzero(valid_past_mask)
        filtered_track_indices = np.searchsorted(valid_past_indices, track_index_to_predict)
        if not np.array_equal(valid_past_indices[filtered_track_indices], track_index_to_predict):
            raise RuntimeError("Prediction target was removed by valid-past filtering")

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        max_num_agents = self.config['max_num_agents']
        selection_center = np.asarray(
            self.config.get('agent_center_offset', self.config.get('center_offset_of_map', (30.0, 0.0))),
            dtype=np.float32,
        )
        object_dist_to_center = np.linalg.norm(
            obj_trajs_data[:, :, -1, 0:2] - selection_center[None, None, :],
            axis=-1,
        )

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
        max_agent_distance = self.config.get('max_agent_distance', None)
        if max_agent_distance is not None and float(max_agent_distance) > 0:
            out_of_range_mask = object_dist_to_center > float(max_agent_distance)
            center_obj_idxs = np.arange(num_center_objects)
            out_of_range_mask[center_obj_idxs, filtered_track_indices] = False
            obj_trajs_data[out_of_range_mask] = 0
            obj_trajs_mask[out_of_range_mask] = 0
            obj_trajs_pos[out_of_range_mask] = 0
            obj_trajs_last_pos[out_of_range_mask] = 0
            obj_trajs_future_state[out_of_range_mask] = 0
            obj_trajs_future_mask[out_of_range_mask] = 0
            object_dist_to_center[out_of_range_mask] = 1e10
        center_obj_idxs = np.arange(num_center_objects)
        object_dist_to_center[center_obj_idxs, filtered_track_indices] = -1.0
        topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents]

        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)

        obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask.astype(bool), center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new)

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        track_index_to_predict_selected = []
        selected_type = self.config['object_type']
        selected_type = [object_type[x] for x in selected_type]
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                continue  # agent not valid at current frame, skip silently
            if obj_types[obj_idx] not in selected_type:
                continue  # expected: scenes contain pedestrians/cyclists we don't train on

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)
        if len(center_objects_list) == 0:
            return None, []
        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    def transform_trajs_to_center_coords(self, obj_trajs, center_xyz, center_heading, heading_index,
                                         rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = np.tile(obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
            angle=-center_heading
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
                angle=-center_heading
            ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def get_map_data(self, center_objects, map_infos):

        num_center_objects = center_objects.shape[0]

        def transform_to_center_coordinates(neighboring_polylines):
            neighboring_polylines[:, :, 0:3] -= center_objects[:, None, 0:3]
            neighboring_polylines[:, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 0:2],
                angle=-center_objects[:, 6]
            )
            neighboring_polylines[:, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 3:5],
                angle=-center_objects[:, 6]
            )

            return neighboring_polylines

        polylines = np.expand_dims(map_infos['all_polylines'].copy(), axis=0).repeat(num_center_objects, axis=0)

        map_polylines = transform_to_center_coordinates(neighboring_polylines=polylines)
        num_of_src_polylines = self.config['max_num_roads']
        map_infos['polyline_transformed'] = map_polylines

        all_polylines = map_infos['polyline_transformed']
        max_points_per_lane = self.config.get('max_points_per_lane', 20)
        line_type = self.config.get('line_type', [])
        map_range = self.config.get('map_range', None)
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))
        num_agents = all_polylines.shape[0]
        polyline_list = []
        polyline_mask_list = []

        for k, v in map_infos.items():
            if k == 'all_polylines' or k not in line_type:
                continue
            if len(v) == 0:
                continue
            for polyline_dict in v:
                polyline_index = polyline_dict.get('polyline_index', None)
                polyline_segment = all_polylines[:, polyline_index[0]:polyline_index[1]]
                # Skip zero-length polylines (polyline_index = [N, N], can occur in some
                # scenarionet pkl after point filtering). Prevents downstream empty-mask crash.
                if polyline_segment.shape[1] == 0:
                    continue
                polyline_segment_xy = polyline_segment[:, :, 0:2] - np.asarray(
                    center_offset, dtype=np.float32
                )[None, None, :]
                if self.config.get('map_crop_shape', 'circle') == 'circle':
                    in_range_mask = np.linalg.norm(polyline_segment_xy, axis=-1) <= map_range
                else:
                    in_range_mask = (
                        (np.abs(polyline_segment_xy[..., 0]) <= map_range)
                        & (np.abs(polyline_segment_xy[..., 1]) <= map_range)
                    )

                segment_index_list = []
                for i in range(polyline_segment.shape[0]):
                    segment_index_list.append(find_true_segments(in_range_mask[i]))
                max_segments = max([len(x) for x in segment_index_list])

                segment_list = np.zeros([num_agents, max_segments, max_points_per_lane, 7], dtype=np.float32)
                segment_mask_list = np.zeros([num_agents, max_segments, max_points_per_lane], dtype=np.int32)

                for i in range(polyline_segment.shape[0]):
                    if in_range_mask[i].sum() == 0:
                        continue
                    segment_i = polyline_segment[i]
                    segment_index = segment_index_list[i]
                    for num, seg_index in enumerate(segment_index):
                        segment = segment_i[seg_index]
                        if segment.shape[0] > max_points_per_lane:
                            segment_list[i, num] = segment[
                                np.linspace(0, segment.shape[0] - 1, max_points_per_lane, dtype=int)]
                            segment_mask_list[i, num] = 1
                        else:
                            segment_list[i, num, :segment.shape[0]] = segment
                            segment_mask_list[i, num, :segment.shape[0]] = 1

                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)
        if len(polyline_list) == 0: return np.zeros((num_agents, 0, max_points_per_lane, 7)), np.zeros(
            (num_agents, 0, max_points_per_lane))
        batch_polylines = np.concatenate(polyline_list, axis=1)
        batch_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        polyline_xy_offsetted = batch_polylines[:, :, :, 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
        polyline_center_dist = np.linalg.norm(polyline_xy_offsetted, axis=-1).sum(-1) / np.clip(
            batch_polylines_mask.sum(axis=-1).astype(float), a_min=1.0, a_max=None)
        polyline_center_dist[batch_polylines_mask.sum(-1) == 0] = 1e10
        topk_idxs = np.argsort(polyline_center_dist, axis=-1)[:, :num_of_src_polylines]

        # Ensure topk_idxs has the correct shape for indexing
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        map_polylines = np.take_along_axis(batch_polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(batch_polylines_mask, topk_idxs[..., 0], axis=1)

        # pad map_polylines and map_polylines_mask to num_of_src_polylines
        map_polylines = np.pad(map_polylines,
                               ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask,
                                    ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(float)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0,
                                                  a_max=None)  # (num_center_objects, num_polylines, 3)

        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        map_types = map_polylines[:, :, :, -1]
        map_polylines = map_polylines[:, :, :, :-1]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask, map_polylines_center

    def get_manually_split_map_data(self, center_objects, map_infos):
        """
        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict):
                all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_offset (2):, [offset_x, offset_y]
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
        """
        num_center_objects = center_objects.shape[0]
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))

        # transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].reshape(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)

            # use pre points to map
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:3]
            xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = np.concatenate((neighboring_polylines, xy_pos_pre), axis=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask

        polylines = map_infos['all_polylines'].copy()
        center_objects = center_objects

        point_dim = polylines.shape[-1]

        point_sampled_interval = self.config['point_sampled_interval']
        vector_break_dist_thresh = self.config['vector_break_dist_thresh']
        num_points_each_polyline = self.config['num_points_each_polyline']

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]),
                                       axis=-1)  # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = \
            (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4],
                            axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        batch_polylines = np.stack(ret_polylines, axis=0)
        batch_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        # collect a number of closest polylines for each center objects
        num_of_src_polylines = self.config['max_num_roads']

        if len(batch_polylines) > num_of_src_polylines:
            # Sum along a specific axis and divide by the minimum clamped sum
            polyline_center = np.sum(batch_polylines[:, :, 0:2], axis=1) / np.clip(
                np.sum(batch_polylines_mask, axis=1)[:, None].astype(float), a_min=1.0, a_max=None)
            # Convert the center_offset to a numpy array and repeat it for each object
            center_offset_rot = np.tile(np.array(center_offset, dtype=np.float32)[None, :], (num_center_objects, 1))

            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot[:, None, :],
                angle=center_objects[:, 6]
            )

            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot[:, 0]

            dist = np.linalg.norm(pos_of_map_centers[:, None, :] - polyline_center[None, :, :], axis=-1)

            # Getting the top-k smallest distances and their indices
            topk_idxs = np.argsort(dist, axis=1)[:, :num_of_src_polylines]
            map_polylines = batch_polylines[
                topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[
                topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)

        else:
            map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 0)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 0)

            map_polylines = np.pad(map_polylines,
                                   ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
            map_polylines_mask = np.pad(map_polylines_mask,
                                        ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(np.float32)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1)[:, :, np.newaxis].astype(float),
                                                  a_min=1.0, a_max=None)

        map_types = map_polylines[:, :, :, 6]
        xy_pos_pre = map_polylines[:, :, :, 7:]
        map_polylines = map_polylines[:, :, :, :6]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)

        return map_polylines, map_polylines_mask, map_polylines_center

    def sample_from_distribution(self, original_array, m=100):
        distribution = [
            ("-10,0", 0),
            ("0,10", 23.952629169758517),
            ("10,20", 24.611144221251667),
            ("20,30.0", 21.142773679220554),
            ("30,40.0", 15.996653629820514),
            ("40,50.0", 9.446714336574939),
            ("50,60.0", 3.7812939732733786),
            ("60,70", 0.8821063091988663),
            ("70,80.0", 0.1533644322320915),
            ("80,90.0", 0.027777741552241064),
            ("90,100.0", 0.005542507117231198),
        ]

        # Define bins and calculate sample sizes for each bin
        bins = np.array([float(range_.split(',')[1]) for range_, _ in distribution])
        sample_sizes = np.array([round(perc / 100 * m) for _, perc in distribution])

        # Digitize the original array into bins
        bin_indices = np.digitize(original_array, bins)

        # Sample from each bin
        sampled_indices = []
        for i, size in enumerate(sample_sizes):
            # Find indices of original array that fall into current bin
            indices_in_bin = np.where(bin_indices == i)[0]
            # Sample without replacement to avoid duplicates
            sampled_indices_in_bin = np.random.choice(indices_in_bin, size=min(size, len(indices_in_bin)),
                                                      replace=False)
            sampled_indices.extend(sampled_indices_in_bin)

        # Extract the sampled elements and their original indices
        sampled_array = original_array[sampled_indices]
        _rank0_print('total sample:', len(sampled_indices))
        # Verify distribution (optional, for demonstration)
        for i, (range_, _) in enumerate(distribution):
            _rank0_print(
                f"Bin {range_}: Expected {distribution[i][1]}%, Actual {len(np.where(bin_indices[sampled_indices] == i)[0]) / len(sampled_indices) * 100}%")

        return sampled_array, sampled_indices

    def _collect_crash_ids(self, ret):
        # crash 表来源(无需 CSV,metadata 已带):优先 accident_vids(realaccident),
        # 否则 collision.collision_pairs 展平(DeepAccident);无碰撞数据集返回空。
        # 详见 UniTraj/PREDICTION_TARGETS.md。
        raw = []
        av = ret.get('accident_vids')
        if av:
            raw = [str(x) for x in av]
        else:
            col = ret.get('collision')
            if isinstance(col, dict):
                for pair in (col.get('collision_pairs') or []):
                    raw.extend(str(x) for x in pair)
        seen, out = set(), []
        for i in raw:
            if i not in seen:
                seen.add(i); out.append(i)
        return out

    def trajectory_filter(self, data, min_moving_distance=2.0):
        # min_moving_distance: VEHICLE 的最小总移动距离阈值。
        #   official(严格)=2.0(默认);all(宽松)=0.0(不按移动距离过滤)。
        #   详见 UniTraj/PREDICTION_TARGETS.md 四表规范。
        trajs = data['track_infos']['trajs']
        current_idx = data['current_time_index']
        obj_summary = data['object_summary']

        tracks_to_preidct = {}
        for idx,(k,v) in enumerate(obj_summary.items()):
            type = v['type']
            positions = trajs[idx, :, 0:2]
            validity = trajs[idx, :, -1]
            if type not in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']: continue
            valid_ratio = v['valid_length']/v['track_length']
            if valid_ratio < 0.5: continue
            moving_distance = v['moving_distance']
            if moving_distance < min_moving_distance and type=='VEHICLE': continue
            is_valid_at_m = validity[current_idx]>0
            if not is_valid_at_m: continue

            # past_traj = positions[:current_idx+1, :]  # Time X (x,y)
            # gt_future = positions[current_idx+1:, :]
            # valid_past = count_valid_steps_past(validity[:current_idx+1])


            future_mask =validity[current_idx+1:]
            future_mask[-1]=0
            idx_of_first_zero = np.where(future_mask == 0)[0]
            idx_of_first_zero = len(future_mask) if len(idx_of_first_zero) == 0 else idx_of_first_zero[0]

            #past_trajectory_valid = past_traj[-valid_past:, :]  # Time(valid) X (x,y)

            # try:
            #     kalman_traj = estimate_kalman_filter(past_trajectory_valid, idx_of_first_zero)  # (x,y)
            #     kalman_diff = calculate_epe(kalman_traj, gt_future[idx_of_first_zero-1])
            # except:
            #     continue
            # if kalman_diff < 20: continue

            tracks_to_preidct[k] = {'track_index': idx, 'track_id': k, 'difficulty': 0, 'object_type': type}

        return tracks_to_preidct



import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def draw_figures(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    train_set = build_dataset(cfg)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0,
                                               collate_fn=train_set.collate_fn)
    # for data in train_loader:
    #     inp = data['input_dict']
    #     plt = check_loaded_data(inp, 0)
    #     plt.show()

    concat_list = [4, 4, 4, 4, 4, 4, 4, 4]
    images = []
    for n, data in tqdm(enumerate(train_loader)):
        for i in range(data['batch_size']):
            plt = check_loaded_data(data['input_dict'], i)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf)
            images.append(img)
        if len(images) >= sum(concat_list):
            break
    final_image = concatenate_varying(images, concat_list)
    final_image.show()

    # kalman_dict = {}
    # # create 10 buckets with length 10 as the key
    # for i in range(10):
    #     kalman_dict[i] = {}
    #
    # data_list = []
    # for data in train_loader:
    #     inp = data['input_dict']
    #     kalman_diff = inp['kalman_difficulty']
    #     for idx,k in enumerate(kalman_diff):
    #         k6 = np.floor(k[2]/10)
    #         if k6 in kalman_dict and len(kalman_dict[k6]) == 0:
    #             kalman_dict[k6]['kalman'] = k[2]
    #             kalman_dict[k6]['data'] = inp
    #             check_loaded_data()
    #


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def split_data(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    train_set = build_dataset(cfg)

    copy_dir = ''
    for data in tqdm(train_set.data_loaded_keys):
        shutil.copy(data, copy_dir)


if __name__ == '__main__':
    from unitraj.datasets import build_dataset
    from unitraj.utils.utils import set_seed
    import io
    from PIL import Image
    from unitraj.utils.visualization import concatenate_varying

    split_data()
    # draw_figures()
