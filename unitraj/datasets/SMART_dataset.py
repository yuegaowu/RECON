import os
import pickle
import zlib
import torch
import h5py
from collections import defaultdict, Counter
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
from torch_geometric.data import Dataset
from unitraj.models.smart.smart_utils import Logging
import numpy as np
from torch_geometric.data import HeteroData, Batch
from torch_geometric.data.storage import NodeStorage
from torch_geometric.loader.dataloader import Collater
from torch_geometric.transforms import BaseTransform
from unitraj.models.smart.smart_utils import TokenProcessor
from .base_dataset import BaseDataset
from metadrive.scenario.scenario_description import MetaDriveType
from scenarionet.common_utils import read_scenario
from unitraj.datasets.common_utils import (
    get_polyline_dir,
    find_true_segments,
    generate_mask,
    get_polyline_mag,
    merge_tuple_key,
    save_item,
    load_item,
)
from unitraj.models.smart.smart_utils import wrap_angle
from unitraj.datasets.types import (
    object_type,
    polyline_type,
    smart_polygon_type,
    smart_traffic_light_state_to_int,
)

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)
polygon_type = defaultdict(lambda: default_value, smart_polygon_type)
traffic_light_state_to_int = defaultdict(
    lambda: default_value, smart_traffic_light_state_to_int
)


class SMARTDataset(BaseDataset):
    """SMART 数据加载(窗口化 / 类 MTR agent-centric)。

    每个 prediction_target(由全局 prediction_targets=sdc/crash/official/all 选出)锚一个局部窗口:
    锚点前 center_offset_of_map 为中心,半径 smart_window_radius 选 agent,半径 smart_map_crop_radius
    裁地图;跨场景展平后按 max_agents_per_batch 随机截断再合并。窗口化恒为 on,
    onthefly 与独立 SMART H5 两种模式使用同一窗口构建函数。
    (旧的 scene-centric 整场景实现已废弃删除;锚点不再用 only_train_on_ego,统一走 prediction_targets。)
    """

    def __init__(self, config=None, is_validation=False):
        # token_size + motion-token vocab come from config (SMART.yaml); agent_token_path lets the
        # tokenizer use a dataset-specific vocab (e.g. highD-fitted) instead of the default Waymo one.
        # Read from the config arg directly — self.config is only set by super().__init__ below.
        _token_size, _token_path = 2048, None
        if config is not None:
            try:
                _token_size = int(config.Model.decoder.token_size)
            except Exception:
                _token_size = int(config.get('token_size', 2048)) if hasattr(config, 'get') else 2048
            _token_path = config.get('agent_token_path', None) if hasattr(config, 'get') else None
        self.token_processor = TokenProcessor(_token_size, agent_token_path=_token_path)
        self.target_transform = WaymoTargetBuilder(11, 80)
        self.scene_centric = True
        # self.scene_centric = False
        super().__init__(config, is_validation)
        self.logger = Logging().log(level="DEBUG")

    def process_data_chunk(self, worker_index):
        """窗口化 h5 生成:每个场景按 official 目标展开成 N 个局部窗口,各烘成一个独立 h5 group。
        ▸ 与 scene-centric 的 SMART_dataset.process_data_chunk 关键不同:这里不走
          process()/postprocess()(那是整场景一条),而是 _smart_windows_from_internal(pre)
          建【全部 official 窗口】(eval 同款,不传 only_anchors),逐窗各一条记录。
        ▸ grp_name 改用 base_dataset 的 `__` 分隔格式 `{dataset}__{worker}-{cnt}__{i}`:
          base.load_data 按 key.split('__')[1] 分组 → 一个场景('{worker}-{cnt}')的全部窗口
          自动归到同一个 h5_scenario_group,__len__=场景数,__getitem__ 按场景读回全部窗口。
        ▸ 生成阶段【全烘不丢】:该丢哪些窗口是每个 epoch 随机的事,放到读取时由 collate 按
          max_agents_per_batch 截断(与 onthefly 一致),故这里把每个场景的窗口全部写盘。"""
        with open(os.path.join("tmp", "{}.pkl".format(worker_index)), "rb") as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f"{worker_index}.h5")

        with h5py.File(hdf5_path, "w") as f:
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f"{cnt}/{len(data_list)} data processed", flush=True)
                scenario = read_scenario(data_path, mapping, file_name)

                try:
                    pre = self.preprocess(scenario)
                    output = self._smart_windows_from_internal(pre)  # 全部 official 窗口
                except Exception as e:
                    print("Warning: {} in {}".format(e, file_name))
                    output = None

                if not output:
                    del scenario
                    continue

                for i, record in enumerate(output):
                    grp_name = f"{dataset_name}__{worker_index}-{cnt}__{i}"
                    grp = f.create_group(grp_name)
                    if isinstance(record, HeteroData):
                        record = record.to_dict()
                        global_items = record.pop("_global_store", None)
                        if global_items:
                            record.update(global_items)
                    for key, value in record.items():
                        key = merge_tuple_key(key)
                        save_item(grp, key, value)
                    file_list[grp_name] = {"h5_path": hdf5_path}
                del scenario
                del output

        return file_list

    def preprocess(self, scenario):

        traffic_lights = scenario["dynamic_map_states"]
        tracks = scenario["tracks"]
        map_feat = scenario["map_features"]

        past_length = self.config["past_len"]
        future_length = self.config["future_len"]
        total_steps = past_length + future_length
        starting_fame = self.starting_frame
        ending_fame = starting_fame + total_steps
        trajectory_sample_interval = self.config["trajectory_sample_interval"]
        frequency_mask = generate_mask(
            past_length - 1, total_steps, trajectory_sample_interval
        )

        track_infos = {
            "object_id": [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            "object_type": [],
            "trajs": [],
        }

        for k, v in tracks.items():

            state = v["state"]
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [
                state["position"],
                state["length"],
                state["width"],
                state["height"],
                state["heading"],
                state["velocity"],
                state["valid"],
            ]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)
            # all_state = all_state[::sample_inverval]
            if all_state.shape[0] < ending_fame:
                all_state = np.pad(
                    all_state, ((ending_fame - all_state.shape[0], 0), (0, 0))
                )
            all_state = all_state[starting_fame:ending_fame]

            assert (
                all_state.shape[0] == total_steps
            ), f"Error: {all_state.shape[0]} != {total_steps}"

            track_infos["object_id"].append(k)
            track_infos["object_type"].append(object_type[v["type"]])
            track_infos["trajs"].append(all_state)

        track_infos["trajs"] = np.stack(track_infos["trajs"], axis=0)
        # scenario['metadata']['ts'] = scenario['metadata']['ts'][::sample_inverval]
        track_infos["trajs"][..., -1] *= frequency_mask[np.newaxis]
        scenario["metadata"]["ts"] = scenario["metadata"]["ts"][:total_steps]

        # x,y,z,type
        map_infos = {
            "lane": [],
            "road_line": [],
            "road_edge": [],
            "stop_sign": [],
            "crosswalk": [],
            "speed_bump": [],
        }
        polylines = []
        point_cnt = 0
        point_to_polyline_edge_index = None
        polyline_light_type = None
        polygon_type_ = None
        point_type = None
        point_orientation = None
        point_mag = None
        polyline_cnt = 0
        for k, v in map_feat.items():
            polyline_type_ = polyline_type[v["type"]]
            # if polyline_type_ == 0:
            # continue

            cur_info = {"id": k}
            cur_info["type"] = v["type"]
            if k in traffic_lights.keys():
                cur_info["light_type"] = traffic_light_state_to_int[
                    traffic_lights[k]["state"]["object_state"][past_length - 1]
                ]  # take light states at 11
            else:
                cur_info["light_type"] = traffic_light_state_to_int[None]
            if polyline_type_ in [1, 2, 3]:
                cur_info["speed_limit_mph"] = v.get("speed_limit_mph", None)
                cur_info["interpolating"] = v.get("interpolating", None)
                cur_info["entry_lanes"] = v.get("entry_lanes", None)
                try:
                    cur_info["left_boundary"] = [
                        {
                            "start_index": x["self_start_index"],
                            "end_index": x["self_end_index"],
                            "feature_id": x["feature_id"],
                            "boundary_type": "UNKNOWN",  # roadline type
                        }
                        for x in v["left_neighbor"]
                    ]
                    cur_info["right_boundary"] = [
                        {
                            "start_index": x["self_start_index"],
                            "end_index": x["self_end_index"],
                            "feature_id": x["feature_id"],
                            "boundary_type": "UNKNOWN",  # roadline type
                        }
                        for x in v["right_neighbor"]
                    ]
                except:
                    cur_info["left_boundary"] = []
                    cur_info["right_boundary"] = []
                polyline = v["polyline"]
                # polyline = interpolate_polyline(polyline)
                map_infos["lane"].append(cur_info)
                if polyline_type_ == 3:
                    cur_info["polygon_type"] = polygon_type["BIKE"]
                else:
                    cur_info["polygon_type"] = polygon_type["VEHICLE"]
            elif polyline_type_ in [0]:
                cur_info["speed_limit_mph"] = v.get("speed_limit_mph", None)
                cur_info["interpolating"] = v.get("interpolating", None)
                cur_info["entry_lanes"] = v.get("entry_lanes", None)
                cur_info["left_boundary"] = []
                cur_info["right_boundary"] = []
                try:
                    polyline = v["polyline"]
                except:
                    polyline = v["polygon"]
                # polyline = interpolate_polyline(polyline)
                cur_info["polygon_type"] = polygon_type["BUS"]  # unknown
                continue
            elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:
                try:
                    polyline = v["polyline"]
                except:
                    polyline = v["polygon"]
                # polyline = interpolate_polyline(polyline)
                map_infos["road_line"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["VEHICLE"]
            elif polyline_type_ in [15, 16]:
                polyline = v["polyline"]
                # polyline = interpolate_polyline(polyline)
                cur_info["type"] = 7
                map_infos["road_line"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["VEHICLE"]
            elif polyline_type_ in [17]:  # stopsign
                cur_info["lane_ids"] = v["lane"]
                cur_info["position"] = v["position"]
                map_infos["stop_sign"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["BUS"]
                polyline = v["position"][np.newaxis]
            elif polyline_type_ in [18]:
                map_infos["crosswalk"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["PEDESTRIAN"]
                polyline = v["polygon"]
            elif polyline_type_ in [19]:  # speed bump
                map_infos["crosswalk"].append(cur_info)
                cur_info["polygon_type"] = polygon_type["BUS"]
                polyline = v["polygon"]
                continue
            # 归一化为 2-D (P, C):部分 map feature(如 stop_sign 的 position 已是 (1,3)
            # 又被 [np.newaxis] 成 (1,1,3))否则下游 cur_polyline_dir[:,1] 越界,
            # 导致 ~85% 场景 preprocess 抛 IndexError 被静默丢弃。
            polyline = np.asarray(polyline)
            if polyline.ndim == 1:
                polyline = polyline[None, :]
            elif polyline.ndim > 2:
                polyline = polyline.reshape(-1, polyline.shape[-1])
            if polyline.shape[-1] == 2:
                polyline = np.concatenate(
                    (polyline, np.zeros((polyline.shape[0], 1))), axis=-1
                )
            try:
                cur_polyline_dir = get_polyline_dir(polyline)
                cur_polyline_mag = get_polyline_mag(polyline)
                type_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = polyline_type_
                light_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = cur_info["light_type"]
                cur_polyline = np.concatenate(
                    (polyline, cur_polyline_dir, light_array, type_array), axis=-1
                )
            except:
                cur_polyline = np.zeros((0, 8), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
            cur_edge_index = np.stack(
                [
                    np.arange(point_cnt, point_cnt + len(cur_polyline)),
                    np.repeat(polyline_cnt, len(cur_polyline)),
                ],
                axis=0,
            )
            if point_to_polyline_edge_index is None:
                point_to_polyline_edge_index = cur_edge_index
            else:
                point_to_polyline_edge_index = np.concatenate(
                    [point_to_polyline_edge_index, cur_edge_index], axis=1
                )

            if polyline_light_type is None:
                polyline_light_type = np.array([cur_info["light_type"]])
            else:
                polyline_light_type = np.concatenate(
                    [polyline_light_type, np.array([cur_info["light_type"]])]
                )

            if point_type is None:
                point_type = np.repeat(polyline_type_, len(cur_polyline))
            else:
                point_type = np.concatenate(
                    [point_type, np.repeat(polyline_type_, len(cur_polyline))], axis=0
                )

            if point_orientation is None:
                point_orientation = np.arctan2(
                    cur_polyline_dir[:, 1], cur_polyline_dir[:, 0]
                )
            else:
                point_orientation = np.concatenate(
                    [
                        point_orientation,
                        np.arctan2(cur_polyline_dir[:, 1], cur_polyline_dir[:, 0]),
                    ],
                    axis=0,
                )

            if point_mag is None:
                point_mag = np.array(cur_polyline_mag)
            else:
                point_mag = np.concatenate(
                    [point_mag, np.array(cur_polyline_mag)], axis=0
                )

            if polygon_type_ is None:
                polygon_type_ = np.array([cur_info["polygon_type"]])
            else:
                # if cur_info['polygon_type']
                polygon_type_ = np.concatenate(
                    [polygon_type_, np.array([cur_info["polygon_type"]])]
                )
            point_cnt += len(cur_polyline)
            polyline_cnt += 1

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 8), dtype=np.float32)
        map_infos["all_polylines"] = polylines  # points (n, 8)
        map_infos["point_to_polyline_edge_index"] = point_to_polyline_edge_index
        map_infos["point_type"] = point_type
        map_infos["point_orientation"] = point_orientation
        map_infos["point_magnitude"] = point_mag
        map_infos["polyline_light_type"] = polyline_light_type
        map_infos["polyline_type"] = polygon_type_

        dynamic_map_infos = {"lane_id": [], "state": [], "stop_point": []}
        for k, v in traffic_lights.items():  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in v["state"]["object_state"]:  # (num_observed_signals)
                lane_id.append(str(v["lane"]))
                state.append(cur_signal)
                if type(v["stop_point"]) == list:
                    stop_point.append(v["stop_point"])
                else:
                    stop_point.append(v["stop_point"].tolist())
            # lane_id = lane_id[::sample_inverval]
            # state = state[::sample_inverval]
            # stop_point = stop_point[::sample_inverval]
            lane_id = lane_id[:total_steps]
            state = state[:total_steps]
            stop_point = stop_point[:total_steps]
            dynamic_map_infos["lane_id"].append(np.array([lane_id]))
            dynamic_map_infos["state"].append(np.array([state]))
            dynamic_map_infos["stop_point"].append(np.array([stop_point]))

        ret = {
            "track_infos": track_infos,
            "dynamic_map_infos": dynamic_map_infos,
            "map_infos": map_infos,
        }
        ret.update(scenario["metadata"])
        ret["timestamps_seconds"] = ret.pop("ts")
        ret["current_time_index"] = self.config["past_len"] - 1
        ret["sdc_track_index"] = track_infos["object_id"].index(ret["sdc_id"])
        # 窗口锚点 = 全局 prediction_targets(sdc/crash/official/all)选出的目标,与 MTR 等共用
        #   base_dataset._build_tracks_to_predict,口径一致(不再用 only_train_on_ego)。
        ret["tracks_to_predict"] = self._build_tracks_to_predict(ret, track_infos)

        ret["map_center"] = scenario["metadata"].get("map_center", np.zeros(3))[
            np.newaxis
        ]

        ret["track_length"] = total_steps
        return ret

    def process(self, internal_format):

        info = internal_format
        scene_id = info["scenario_id"]

        sdc_track_index = info["sdc_track_index"]
        current_time_index = info["current_time_index"]
        timestamps = np.array(
            info["timestamps_seconds"][: current_time_index + 1], dtype=np.float32
        )
        # 与 base_dataset 同口径:时间戳归一化到窗口首帧=0,避免 realaccident 的负时间戳
        #   (ts=[-9.1,…,0.0])污染 object_time_embedding[...,-1]。对 0 起点数据是 no-op。
        timestamps = timestamps - timestamps[0]

        track_infos = info["track_infos"]

        track_index_to_predict = np.array(info["tracks_to_predict"]["track_index"])
        obj_types = np.array(track_infos["object_type"])
        obj_trajs_full = track_infos["trajs"]  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, : current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1 :]

        if self.scene_centric:
            center_objects = None
        else:
            center_objects, track_index_to_predict = self.get_interested_agents(
                track_index_to_predict=track_index_to_predict,
                obj_trajs_full=obj_trajs_full,
                current_time_index=current_time_index,
                obj_types=obj_types,
                scene_id=scene_id,
            )
        # if center_objects is None: return None
        obj_ids_global = None  # scene-centric:每个 agent 节点对应的原始 object_id(供肇事分桶)
        if center_objects is None:
            sample_num = 1
            (
                obj_trajs_data,
                obj_trajs_mask,
                obj_trajs_pos,
                obj_trajs_last_pos,
                obj_trajs_future_state,
                obj_trajs_future_mask,
                track_index_to_predict_new,
                agent_valid_mask,
            ) = self.get_agent_data_global(
                obj_trajs_past=obj_trajs_past,
                obj_trajs_future=obj_trajs_future,
                track_index_to_predict=track_index_to_predict,
                sdc_track_index=sdc_track_index,
                timestamps=timestamps,
                obj_types=obj_types,
            )
            obj_ids_global = np.array(track_infos["object_id"])[agent_valid_mask]
        else:
            sample_num = center_objects.shape[0]
            (
                obj_trajs_data,
                obj_trajs_mask,
                obj_trajs_pos,
                obj_trajs_last_pos,
                obj_trajs_future_state,
                obj_trajs_future_mask,
                center_gt_trajs,
                center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new,
            ) = self.get_agent_data(
                center_objects=center_objects,
                obj_trajs_past=obj_trajs_past,
                obj_trajs_future=obj_trajs_future,
                track_index_to_predict=track_index_to_predict,
                sdc_track_index=sdc_track_index,
                timestamps=timestamps,
                obj_types=obj_types,
            )
        dynamic_map_infos = info["dynamic_map_infos"]

        ret_dict = {
            "scenario_id": (
                np.array([scene_id] * len(track_index_to_predict_new))
                if center_objects
                else np.array([scene_id])
            ),
            "obj_trajs": obj_trajs_data,
            "obj_trajs_mask": obj_trajs_mask,
            "track_index_to_predict": track_index_to_predict_new,
            "obj_trajs_pos": obj_trajs_pos,
            "obj_trajs_last_pos": obj_trajs_last_pos,
            "map_center": info["map_center"],
            "obj_trajs_future_state": obj_trajs_future_state,
            "obj_trajs_future_mask": obj_trajs_future_mask,
            # 'light_lane_id': np.array(dynamic_map_infos['lane_id']),
            # 'light_state': np.array(dynamic_map_infos['state']),
            # 'light_stop_point': np.array(dynamic_map_infos['stop_point'])
        }
        if obj_ids_global is not None:
            ret_dict["obj_ids"] = obj_ids_global  # 每 agent 节点 → 原始 object_id(肇事分桶)
        if center_objects is not None:
            ret_dict.update(
                {
                    "center_objects_world": center_objects,
                    "center_objects_id": np.array(track_infos["object_id"])[
                        track_index_to_predict
                    ],
                    "center_objects_type": np.array(track_infos["object_type"])[
                        track_index_to_predict
                    ],
                    "center_gt_trajs": center_gt_trajs,
                    "center_gt_trajs_mask": center_gt_trajs_mask,
                    "center_gt_final_valid_idx": center_gt_final_valid_idx,
                    "center_gt_trajs_src": obj_trajs_full[track_index_to_predict],
                }
            )

        if info["map_infos"]["all_polylines"].__len__() == 0:
            info["map_infos"]["all_polylines"] = np.zeros((2, 8), dtype=np.float32)
            print(f"Warning: empty HDMap {scene_id}")

        if self.config.manually_split_lane:
            map_polylines_data, map_polylines_mask, map_polylines_center = (
                self.get_manually_split_map_data(
                    center_objects=center_objects, map_infos=info["map_infos"]
                )
            )
            ret_dict["map_polylines_center"] = map_polylines_center
        else:
            if self.scene_centric:
                pass
                # map_polylines_data, map_polylines_mask = self.get_map_data_global(
                #     map_infos=info['map_infos'], light_infos=dynamic_map_infos)
            else:
                map_polylines_data, map_polylines_mask, map_polylines_center = (
                    self.get_map_data(
                        center_objects=center_objects, map_infos=info["map_infos"]
                    )
                )
                ret_dict["map_polylines_center"] = map_polylines_center

        # ret_dict['map_polylines'] = map_polylines_data
        # ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
        ret_dict["map_points_pos"] = info["map_infos"]["all_polylines"].copy()[
            None, ...
        ]
        ret_dict["point_to_polyline_edge_index"] = info["map_infos"][
            "point_to_polyline_edge_index"
        ].copy()[None, ...]
        ret_dict["map_points_type"] = info["map_infos"]["point_type"].copy()[None, ...]
        ret_dict["map_points_orientation"] = info["map_infos"][
            "point_orientation"
        ].copy()[None, ...]
        ret_dict["map_points_magnitude"] = info["map_infos"]["point_magnitude"].copy()[
            None, ...
        ]
        ret_dict["polyline_light_type"] = info["map_infos"][
            "polyline_light_type"
        ].copy()[None, ...]
        ret_dict["polyline_type"] = info["map_infos"]["polyline_type"].copy()[None, ...]

        # masking out unused attributes to Zero
        masked_attributes = self.config["masked_attributes"]
        if "z_axis" in masked_attributes:
            ret_dict["obj_trajs"][..., 2] = 0
            # ret_dict['map_polylines'][..., 2] = 0
            ret_dict["map_points_pos"][..., 2] = 0
        if "size" in masked_attributes:
            ret_dict["obj_trajs"][..., 3:6] = 0
        if "velocity" in masked_attributes:
            ret_dict["obj_trajs"][..., 25:27] = 0
        if "acceleration" in masked_attributes:
            ret_dict["obj_trajs"][..., 27:29] = 0
        if "heading" in masked_attributes:
            ret_dict["obj_trajs"][..., 23:25] = 0

        # change every thing to float32
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        ret_dict["map_center"] = ret_dict["map_center"].repeat(sample_num, axis=0)
        ret_dict["dataset_name"] = [info["dataset"]] * sample_num

        ret_list = []
        for i in range(sample_num):
            ret_dict_i = {}
            for k, v in ret_dict.items():
                if k in ("track_index_to_predict", "obj_ids") and self.scene_centric:
                    ret_dict_i[k] = v       # 整条数组不按 sample 切(scene-centric sample_num=1)
                else:
                    ret_dict_i[k] = v[i]
            ret_list.append(ret_dict_i)

        return ret_list

    def get_agent_data_global(
        self,
        obj_trajs_past,
        obj_trajs_future,
        track_index_to_predict,
        sdc_track_index,
        timestamps,
        obj_types,
    ):
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        obj_trajs = obj_trajs_past

        object_onehot_mask = np.zeros((num_objects, num_timestamps, 5))
        object_onehot_mask[obj_types == 1, :, 0] = 1
        object_onehot_mask[obj_types == 2, :, 1] = 1
        object_onehot_mask[obj_types == 3, :, 2] = 1
        object_onehot_mask[track_index_to_predict, :, 3] = 1  # track2pred
        object_onehot_mask[sdc_track_index, :, 4] = 1  # av

        object_time_embedding = np.zeros(
            (num_objects, num_timestamps, num_timestamps + 1)
        )
        for i in range(num_timestamps):
            object_time_embedding[:, i, i] = 1
        object_time_embedding[:, :, -1] = timestamps

        object_heading_embedding = np.zeros((num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, 0] = np.sin(obj_trajs[:, :, 6])
        object_heading_embedding[:, :, 1] = np.cos(obj_trajs[:, :, 6])

        vel = obj_trajs[:, :, 7:9]
        vel_pre = np.roll(vel, shift=1, axis=1)
        acce = (vel - vel_pre) / 0.1
        acce[:, 0, :] = acce[:, 1, :]

        obj_trajs_data = np.concatenate(
            [
                obj_trajs[:, :, 0:6],
                object_onehot_mask,
                object_time_embedding,
                object_heading_embedding,
                obj_trajs[:, :, 7:9],
                acce,
            ],
            axis=-1,
        )

        obj_trajs_mask = obj_trajs[:, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0
        obj_trajs_mask[:, 1:] = obj_trajs_mask[:, 1:] * obj_trajs_mask[:, :-1]
        obj_trajs_mask[:, 0] = False

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future_state = obj_trajs_future[
            :, :, [0, 1, 3, 4, 5, 6, 7, 8]
        ]  # (x, y, vx, vy)
        obj_trajs_future_mask = obj_trajs_future[:, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[0]
        # valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)
        # valid_past_mask = obj_trajs_past[:, num_timestamps-1, -1].astype(bool)
        valid_past_mask = (obj_trajs_past[:, num_timestamps - 1, :2] != 0).all(axis=-1)

        obj_trajs_mask = obj_trajs_mask[valid_past_mask]
        obj_trajs_data = obj_trajs_data[valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, 0:3]
        num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, k, :][cur_valid_mask]

        max_num_agents = self.config["max_num_agents"]
        num_objs = obj_trajs.shape[0]

        obj_trajs_data = obj_trajs_data[None, ...]
        obj_trajs_mask = obj_trajs_mask[None, ...]
        obj_trajs_pos = obj_trajs_pos[None, ...]
        obj_trajs_last_pos = obj_trajs_last_pos[None, ...]
        obj_trajs_future_state = obj_trajs_future_state[None, ...]
        obj_trajs_future_mask = obj_trajs_future_mask[None, ...]
        # track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)[None, ...]
        # 守卫:ego(sdc)当前帧无效(位置=0)时,它不在 valid_past_mask 里,下面 [0][0] 会 IndexError。
        #   SMART scene-centric 以 ego 为参考系,ego 无效则该场景无法处理 → 抛清晰异常,交上层跳过
        #   (与 fast path smart_convert_from_internal 的 `not valid_past_mask[sdc_track_index]: return []` 一致)。
        #   DeepAccident/CARLA 偶发(8/137);Waymo 不会出现。
        if (not valid_past_mask.any()) or (not valid_past_mask[sdc_track_index]):
            raise ValueError("SMART: ego(sdc) 当前帧无效,scene-centric 无参考系,跳过该场景")
        track_index_to_predict = np.where(
            np.arange(num_objs)[valid_past_mask] == sdc_track_index
        )[0][0]

        return (
            obj_trajs_data,
            obj_trajs_mask.astype(bool),
            obj_trajs_pos,
            obj_trajs_last_pos,
            obj_trajs_future_state,
            obj_trajs_future_mask,
            track_index_to_predict,
            valid_past_mask,  # agent 节点顺序 = 原始 object 顺序的 valid 子集(供肇事分桶对齐 object_id)
        )

    def get_map_data_global(self, map_infos, light_infos=None):
        map_polylines = np.expand_dims(
            map_infos["all_polylines"].copy(), axis=0
        ).repeat(1, axis=0)

        num_of_src_polylines = self.config["max_num_roads"]
        map_infos["polyline_transformed"] = map_polylines

        all_polylines = map_infos["polyline_transformed"]
        max_points_per_lane = self.config.get("max_points_per_lane", 20)
        line_type = self.config.get("line_type", [])
        # map_range = self.config.get('map_range', None)
        # center_offset = self.config.get('center_offset_of_map', (30.0, 0))
        num_agents = all_polylines.shape[0]
        polyline_list = []
        polyline_mask_list = []

        for k, v in map_infos.items():
            if k == "all_polylines" or k not in line_type:
                continue
            if len(v) == 0:
                continue
            for polyline_dict in v:
                polyline_index = polyline_dict.get("polyline_index", None)
                polyline_segment = all_polylines[
                    :, polyline_index[0] : polyline_index[1]
                ]
                in_range_mask = np.ones(polyline_segment.shape[:2], dtype=bool)

                segment_index_list = []
                for i in range(polyline_segment.shape[0]):
                    segment_index_list.append(find_true_segments(in_range_mask[i]))
                max_segments = max([len(x) for x in segment_index_list])

                segment_list = np.zeros(
                    [num_agents, max_segments, max_points_per_lane, 8], dtype=np.float32
                )
                segment_mask_list = np.zeros(
                    [num_agents, max_segments, max_points_per_lane], dtype=np.int32
                )

                for i in range(polyline_segment.shape[0]):
                    if in_range_mask[i].sum() == 0:
                        continue
                    segment_i = polyline_segment[i]
                    segment_index = segment_index_list[i]
                    for num, seg_index in enumerate(segment_index):
                        segment = segment_i[seg_index]
                        if segment.shape[0] > max_points_per_lane:
                            segment_list[i, num] = segment[
                                np.linspace(
                                    0,
                                    segment.shape[0] - 1,
                                    max_points_per_lane,
                                    dtype=int,
                                )
                            ]
                            segment_mask_list[i, num] = 1
                        else:
                            segment_list[i, num, : segment.shape[0]] = segment
                            segment_mask_list[i, num, : segment.shape[0]] = 1

                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)
        if len(polyline_list) == 0:
            return np.zeros((num_agents, 0, max_points_per_lane, 8)), np.zeros(
                (num_agents, 0, max_points_per_lane)
            )
        map_polylines = np.concatenate(polyline_list, axis=1)
        map_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        light_types = map_polylines[:, :, :, -2]
        light_types = np.eye(9)[light_types.astype(int)]  # use 9 for light types

        map_types = map_polylines[:, :, :, -1]
        map_polylines = map_polylines[:, :, :, :-2]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        map_polylines = np.concatenate(
            (map_polylines, xy_pos_pre, light_types, map_types), axis=-1
        )
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask

    def get_interested_agents(
        self,
        track_index_to_predict,
        obj_trajs_full,
        current_time_index,
        obj_types,
        scene_id,
    ):
        center_objects_list = []
        track_index_to_predict_selected = []
        selected_type = self.config["object_type"]
        selected_type = [object_type[x] for x in selected_type]
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                print(
                    f"Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}"
                )
                continue
            if obj_types[obj_idx] not in selected_type:
                print(
                    f"Missing obj_types {obj_types[obj_idx]} at time step {current_time_index}, scene_id={scene_id}"
                )
                continue

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)
        if len(center_objects_list) == 0:
            print(
                f"Warning: no center objects at time step {current_time_index}, scene_id={scene_id}"
            )
            return None, []
        center_objects = np.stack(
            center_objects_list, axis=0
        )  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    def postprocess(self, output):

        # get_kalman_difficulty(output)

        # get_trajectory_type(output)

        return self.smart_convert(output)

    def _crop_map_infos_for_smart(self, map_infos, centers_xy):
        radius = self.config.get("smart_map_crop_radius", None)
        if radius is None or radius <= 0 or centers_xy.size == 0:
            return map_infos

        all_points = map_infos["all_polylines"]
        edge_index = map_infos["point_to_polyline_edge_index"]
        if all_points is None or edge_index is None or len(all_points) == 0:
            return map_infos

        centers_xy = np.asarray(centers_xy, dtype=np.float32)
        centers_xy = centers_xy[np.isfinite(centers_xy).all(axis=1)]
        if len(centers_xy) == 0:
            return map_infos

        keep_point = np.zeros(len(all_points), dtype=bool)
        radius_sq = float(radius) ** 2
        chunk = 4096
        point_xy = all_points[:, :2].astype(np.float32, copy=False)
        for start in range(0, len(point_xy), chunk):
            cur = point_xy[start : start + chunk]
            dist_sq = ((cur[:, None, :] - centers_xy[None, :, :]) ** 2).sum(axis=-1)
            keep_point[start : start + chunk] = dist_sq.min(axis=1) <= radius_sq

        if not keep_point.any():
            return map_infos

        point_to_pl = edge_index[1]
        valid_polylines = []
        new_point_indices = []
        for pl_idx in np.unique(point_to_pl[keep_point]):
            old_indices = edge_index[0, point_to_pl == pl_idx]
            old_indices = old_indices[keep_point[old_indices]]
            if len(old_indices) <= 2:
                continue
            valid_polylines.append(int(pl_idx))
            new_point_indices.append(old_indices)

        if not new_point_indices:
            return map_infos

        new_point_indices = np.concatenate(new_point_indices)
        old_to_new = -np.ones(len(all_points), dtype=np.int64)
        old_to_new[new_point_indices] = np.arange(len(new_point_indices), dtype=np.int64)
        valid_polyline_array = np.asarray(valid_polylines, dtype=np.int64)
        pl_remap = {old: new for new, old in enumerate(valid_polylines)}

        kept_old_edges = np.isin(edge_index[0], new_point_indices) & np.isin(edge_index[1], valid_polyline_array)
        old_edges = edge_index[:, kept_old_edges]
        new_edges = np.empty_like(old_edges)
        new_edges[0] = old_to_new[old_edges[0]]
        new_edges[1] = np.asarray([pl_remap[int(x)] for x in old_edges[1]], dtype=old_edges.dtype)

        cropped = dict(map_infos)
        cropped["all_polylines"] = all_points[new_point_indices].copy()
        cropped["point_to_polyline_edge_index"] = new_edges
        cropped["point_type"] = map_infos["point_type"][new_point_indices].copy()
        cropped["point_orientation"] = map_infos["point_orientation"][new_point_indices].copy()
        cropped["point_magnitude"] = map_infos["point_magnitude"][new_point_indices].copy()
        cropped["polyline_light_type"] = map_infos["polyline_light_type"][valid_polyline_array].copy()
        cropped["polyline_type"] = map_infos["polyline_type"][valid_polyline_array].copy()
        return cropped

    def _smart_windows_from_internal(self, info, only_anchors=None):
        """窗口化 scene-centric:每个 prediction_target 锚一个局部窗口,产出 N 条 tokenizer-ready 记录。

        only_anchors:可选的原始 track_index 子集——只为这些锚点建窗口(训练时只建 1 个随机锚点,
          避免白建 N 个;为 None 时建全部目标,供 eval 全覆盖)。

        与 smart_convert_from_internal(单窗口锚 SDC)同款字段口径,差别只在:
          - 锚点 = 每个 tracks_to_predict 目标(而非 SDC);
          - agent 选择 = 锚点半径(smart_window_radius)内 + 最近 max_num_agents 个,强留锚点;
          - 每条记录带 agent_ids(逗号 join 的窗口内 object_id)+ anchor_id(锚点 object_id),
            供分桶与"只报锚点指标"用。
        loss 仍由模型对窗口内全部 agent 计算;只有 eval/log 用 anchor_id 过滤到锚点。
        """
        scene_id = info["scenario_id"]
        current_time_index = info["current_time_index"]
        sdc_track_index = info["sdc_track_index"]

        track_infos = info["track_infos"]
        obj_types = np.array(track_infos["object_type"])
        object_ids_all = np.array(track_infos["object_id"])
        obj_trajs_full = track_infos["trajs"].astype(np.float32)
        obj_trajs_past = obj_trajs_full[:, : current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1 :]

        num_objects, num_historical_steps, _ = obj_trajs_past.shape
        if num_objects == 0:
            return []

        valid_full = (obj_trajs_past[:, num_historical_steps - 1, :2] != 0).all(axis=-1)
        if not valid_full.any():
            return []

        anchors = [
            int(i)
            for i in info.get("tracks_to_predict", {}).get("track_index", [])
            if 0 <= int(i) < num_objects and valid_full[int(i)]
        ]
        if only_anchors is not None:
            keep = {int(a) for a in only_anchors}
            anchors = [a for a in anchors if a in keep]
        if not anchors:
            return []

        max_agents = self.config.get("max_num_agents", None)
        radius = self.config.get("smart_window_radius", None)
        current_xy = obj_trajs_past[:, num_historical_steps - 1, :2]
        current_heading = obj_trajs_past[:, num_historical_steps - 1, 6]
        # 窗口中心 = 锚点【前方 center_offset_of_map】(沿锚点航向旋转,前=+x),与 MTR
        #   center_offset_of_map 同义。车辆未来主要朝前 → 把中心前移,可用更小半径覆盖
        #   前方场景,agent/地图都以此单点为圆心选,显著省 token/显存。
        offset = self.config.get("center_offset_of_map", [30.0, 0.0])
        off_fwd, off_lat = float(offset[0]), float(offset[1])

        records = []
        for anchor_idx in anchors:
            valid_past_mask = valid_full.copy()
            anchor_xy = current_xy[anchor_idx]
            _h = float(current_heading[anchor_idx])
            _cos, _sin = np.cos(_h), np.sin(_h)
            center_xy = (anchor_xy + np.array(
                [off_fwd * _cos - off_lat * _sin, off_fwd * _sin + off_lat * _cos],
                dtype=np.float32,
            )).astype(np.float32)

            if radius is not None and float(radius) > 0:
                within = np.linalg.norm(current_xy - center_xy[None, :], axis=1) <= float(radius)
                within[anchor_idx] = True
                valid_past_mask &= within

            if max_agents is not None and max_agents > 0 and valid_past_mask.sum() > max_agents:
                keep_set = {int(anchor_idx)}
                candidates = np.where(valid_past_mask)[0]
                dist = np.linalg.norm(current_xy[candidates] - center_xy[None, :], axis=1)
                for idx in candidates[np.argsort(dist)]:
                    keep_set.add(int(idx))
                    if len(keep_set) >= int(max_agents):
                        break
                selected_mask = np.zeros_like(valid_past_mask, dtype=bool)
                selected_mask[list(keep_set)] = True
                valid_past_mask &= selected_mask

            if not valid_past_mask[anchor_idx]:
                continue

            old_to_new = -np.ones(num_objects, dtype=np.int64)
            old_to_new[np.where(valid_past_mask)[0]] = np.arange(valid_past_mask.sum())
            av_index = int(old_to_new[anchor_idx])

            past_valid = obj_trajs_past[:, :, -1].astype(bool).copy()
            past_valid[:, 1:] = past_valid[:, 1:] & past_valid[:, :-1]
            past_valid[:, 0] = False
            future_valid = obj_trajs_future[:, :, -1].astype(bool)

            past_position = obj_trajs_past[:, :, 0:3].copy()
            past_shape = obj_trajs_past[:, :, 3:6].copy()
            past_heading = obj_trajs_past[:, :, 6].copy()
            past_velocity_xy = obj_trajs_past[:, :, 7:9].copy()
            invalid_past = ~(obj_trajs_past[:, :, -1].astype(bool))
            past_position[invalid_past] = 0
            past_shape[invalid_past] = 0
            past_heading[invalid_past] = 0
            past_velocity_xy[invalid_past] = 0

            future_position = obj_trajs_future[:, :, [0, 1, 3]].copy()
            future_shape = obj_trajs_future[:, :, 3:6].copy()
            future_heading = obj_trajs_future[:, :, 6].copy()
            future_velocity_xy = obj_trajs_future[:, :, 7:9].copy()
            invalid_future = ~future_valid
            future_position[invalid_future] = 0
            future_shape[invalid_future] = 0
            future_heading[invalid_future] = 0
            future_velocity_xy[invalid_future] = 0

            agent = {}
            agent["num_nodes"] = int(valid_past_mask.sum())
            agent["av_index"] = av_index
            agent["valid_mask"] = np.concatenate([past_valid, future_valid], axis=1)[valid_past_mask]
            predict_mask = np.zeros_like(agent["valid_mask"], dtype=bool)
            predict_mask[:, num_historical_steps:] = True
            predict_mask[~agent["valid_mask"]] = False
            agent["predict_mask"] = predict_mask

            agent_type = np.zeros_like(obj_types, dtype=np.uint8)
            agent_type[obj_types == 2] = 1
            agent_type[obj_types == 3] = 2
            agent["type"] = agent_type[valid_past_mask]
            agent["category"] = np.zeros(agent["num_nodes"], dtype=np.uint8)
            agent["position"] = np.concatenate([past_position, future_position], axis=1)[valid_past_mask].astype(np.float32)
            agent["heading"] = np.concatenate([past_heading, future_heading], axis=1)[valid_past_mask].astype(np.float32)
            agent["shape"] = np.concatenate([past_shape, future_shape], axis=1)[valid_past_mask].astype(np.float32)
            velocity_xy = np.concatenate([past_velocity_xy, future_velocity_xy], axis=1)[valid_past_mask]
            agent["velocity"] = np.pad(
                velocity_xy, pad_width=((0, 0), (0, 0), (0, 1)),
                mode="constant", constant_values=0.0,
            ).astype(np.float32)

            data = {
                "scenario_id": scene_id,
                "agent": NodeStorage(agent),
                # 窗口内每个 agent 节点的 object_id(顺序同 agent),逗号 join 成字符串标量存活;
                #   anchor_id = 锚点 object_id,eval/log 用它把指标过滤到锚点。
                "agent_ids": ",".join(object_ids_all[valid_past_mask].astype(str).ravel().tolist()),
                "anchor_id": str(object_ids_all[anchor_idx]),
            }

            # 地图裁剪以【窗口中心】单点为圆心(而非窗口内全部 agent 的并集),
            #   半径 = smart_map_crop_radius。单点圆 → 地图 token 远少于"每辆车圆盘并集",
            #   这是窗口化省显存的关键(并集会随车辆分散而爆炸,正是早期 OOM 之源)。
            map_infos = self._crop_map_infos_for_smart(info["map_infos"], center_xy[None, :])
            map_all = map_infos["all_polylines"]
            if map_all.__len__() == 0:
                map_all = np.zeros((2, 8), dtype=np.float32)
                map_infos["all_polylines"] = map_all

            data["map_polygon"] = NodeStorage({
                "num_nodes": map_infos["polyline_light_type"].shape[0],
                "light_type": map_infos["polyline_light_type"].copy(),
                "type": map_infos["polyline_type"].copy(),
            })
            data["map_point"] = NodeStorage({
                "num_nodes": map_all.shape[0],
                "position": map_all[:, :2].copy(),
                "orientation": map_infos["point_orientation"].copy(),
                "magnitude": map_infos["point_magnitude"].copy(),
                "height": np.zeros_like(map_infos["point_magnitude"]).astype(np.float32),
                "type": map_infos["point_type"].copy(),
            })
            data["map_point", "to", "map_polygon"] = NodeStorage(
                {"edge_index": map_infos["point_to_polyline_edge_index"].copy()}
            )

            data = self.target_transform(data)
            records.append(self.token_processor.preprocess(data))
        return records

    def smart_convert_from_internal(self, info):
        """Build SMART HeteroData directly from preprocess() output.

        SMART is scene-centric and ultimately needs raw agent trajectories plus raw
        map graph fields. The older on-the-fly path first builds MTR-style
        29-D obj_trajs in process(), then smart_convert() slices that tensor back
        into raw position/heading/velocity/shape. This path keeps the same SMART
        fields while skipping that intermediate representation.

        smart_windowed=True 时改走窗口化路径:每个 prediction_target 锚一个局部窗口
        (锚点 + 半径/近邻内 agent + 局部地图),产出 N 条记录(见 _smart_windows_from_internal
        与 UniTraj/PREDICTION_TARGETS.md)。默认 False,保持下面单窗口(锚 SDC)的原行为不变。
        """
        if bool(self.config.get("smart_windowed", False)):
            return self._smart_windows_from_internal(info)
        scene_id = info["scenario_id"]
        current_time_index = info["current_time_index"]
        sdc_track_index = info["sdc_track_index"]

        track_infos = info["track_infos"]
        obj_types = np.array(track_infos["object_type"])
        obj_trajs_full = track_infos["trajs"].astype(np.float32)
        obj_trajs_past = obj_trajs_full[:, : current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1 :]

        num_objects, num_historical_steps, _ = obj_trajs_past.shape
        if num_objects == 0:
            return []

        valid_past_mask = (obj_trajs_past[:, num_historical_steps - 1, :2] != 0).all(axis=-1)
        if not valid_past_mask.any() or not valid_past_mask[sdc_track_index]:
            return []

        max_agents = self.config.get("max_num_agents", None)
        if max_agents is not None and max_agents > 0 and valid_past_mask.sum() > max_agents:
            current_xy = obj_trajs_past[:, num_historical_steps - 1, :2]
            sdc_xy = current_xy[sdc_track_index]
            keep_indices = [int(sdc_track_index)]

            for idx in info.get("tracks_to_predict", {}).get("track_index", []):
                if 0 <= int(idx) < num_objects and valid_past_mask[int(idx)]:
                    keep_indices.append(int(idx))

            keep_set = set(keep_indices)
            candidates = np.where(valid_past_mask)[0]
            dist = np.linalg.norm(current_xy[candidates] - sdc_xy[None, :], axis=1)
            for idx in candidates[np.argsort(dist)]:
                keep_set.add(int(idx))
                if len(keep_set) >= int(max_agents):
                    break

            selected_mask = np.zeros_like(valid_past_mask, dtype=bool)
            selected_mask[list(keep_set)] = True
            valid_past_mask &= selected_mask

        old_to_new = -np.ones(num_objects, dtype=np.int64)
        old_to_new[np.where(valid_past_mask)[0]] = np.arange(valid_past_mask.sum())
        av_index = int(old_to_new[sdc_track_index])

        past_valid = obj_trajs_past[:, :, -1].astype(bool)
        past_valid[:, 1:] = past_valid[:, 1:] & past_valid[:, :-1]
        past_valid[:, 0] = False
        future_valid = obj_trajs_future[:, :, -1].astype(bool)

        past_position = obj_trajs_past[:, :, 0:3].copy()
        past_shape = obj_trajs_past[:, :, 3:6].copy()
        past_heading = obj_trajs_past[:, :, 6].copy()
        past_velocity_xy = obj_trajs_past[:, :, 7:9].copy()

        # 清零依据用「原始逐帧有效标志」，不要用 cumulative 的 past_valid。
        #   past_valid 做了 [:,1:] &= [:,:-1] / [:,0]=False（要求与前一帧连续有效），会把
        #   「确有观测、但因不连续被判无效」的帧（如孤立帧）也清零。一旦这些帧的真实坐标被抹成 0，
        #   token 处理器（smart_utils.tokenize_agent）就无法借 position!=0 把它们插值/重新激活，
        #   导致该 agent 的 token 锚点丢失 → inference rollout 从原点起步 → 预测落在局部系（误差上万米）。
        #   slow path(process()+smart_convert())正是保留这些原始坐标，故两条路径在此对齐。
        invalid_past = ~(obj_trajs_past[:, :, -1].astype(bool))
        past_position[invalid_past] = 0
        past_shape[invalid_past] = 0
        past_heading[invalid_past] = 0
        past_velocity_xy[invalid_past] = 0

        # Preserve the legacy process()+smart_convert() layout exactly:
        # obj_trajs_future_state = [x, y, length, width, height, heading, vx, vy].
        future_position = obj_trajs_future[:, :, [0, 1, 3]].copy()
        future_shape = obj_trajs_future[:, :, 3:6].copy()
        future_heading = obj_trajs_future[:, :, 6].copy()
        future_velocity_xy = obj_trajs_future[:, :, 7:9].copy()

        invalid_future = ~future_valid
        future_position[invalid_future] = 0
        future_shape[invalid_future] = 0
        future_heading[invalid_future] = 0
        future_velocity_xy[invalid_future] = 0

        agent = {}
        agent["num_nodes"] = int(valid_past_mask.sum())
        agent["av_index"] = av_index
        agent["valid_mask"] = np.concatenate([past_valid, future_valid], axis=1)[valid_past_mask]

        predict_mask = np.zeros_like(agent["valid_mask"], dtype=bool)
        predict_mask[:, num_historical_steps:] = True
        predict_mask[~agent["valid_mask"]] = False
        agent["predict_mask"] = predict_mask

        # SMART token processor uses 0=vehicle, 1=pedestrian, 2=cyclist.
        # Match the legacy one-hot+argmax behavior: unknown/other falls back to 0.
        agent_type = np.zeros_like(obj_types, dtype=np.uint8)
        agent_type[obj_types == 2] = 1
        agent_type[obj_types == 3] = 2
        agent["type"] = agent_type[valid_past_mask]
        agent["category"] = np.zeros(agent["num_nodes"], dtype=np.uint8)
        agent["position"] = np.concatenate([past_position, future_position], axis=1)[valid_past_mask].astype(np.float32)
        agent["heading"] = np.concatenate([past_heading, future_heading], axis=1)[valid_past_mask].astype(np.float32)
        agent["shape"] = np.concatenate([past_shape, future_shape], axis=1)[valid_past_mask].astype(np.float32)

        velocity_xy = np.concatenate([past_velocity_xy, future_velocity_xy], axis=1)[valid_past_mask]
        agent["velocity"] = np.pad(
            velocity_xy,
            pad_width=((0, 0), (0, 0), (0, 1)),
            mode="constant",
            constant_values=0.0,
        ).astype(np.float32)

        data = {
            "scenario_id": scene_id,
            "agent": NodeStorage(agent),
        }

        current_centers = agent["position"][:, num_historical_steps - 1, :2]
        map_infos = self._crop_map_infos_for_smart(info["map_infos"], current_centers)
        map_all = map_infos["all_polylines"]
        if map_all.__len__() == 0:
            map_all = np.zeros((2, 8), dtype=np.float32)
            map_infos["all_polylines"] = map_all
            print(f"Warning: empty HDMap {scene_id}")

        data["map_polygon"] = NodeStorage(
            {
                "num_nodes": map_infos["polyline_light_type"].shape[0],
                "light_type": map_infos["polyline_light_type"].copy(),
                "type": map_infos["polyline_type"].copy(),
            }
        )
        data["map_point"] = NodeStorage(
            {
                "num_nodes": map_all.shape[0],
                "position": map_all[:, :2].copy(),
                "orientation": map_infos["point_orientation"].copy(),
                "magnitude": map_infos["point_magnitude"].copy(),
                "height": np.zeros_like(map_infos["point_magnitude"]).astype(np.float32),
                "type": map_infos["point_type"].copy(),
            }
        )
        data["map_point", "to", "map_polygon"] = NodeStorage(
            {"edge_index": map_infos["point_to_polyline_edge_index"].copy()}
        )

        data = self.target_transform(data)
        return [self.token_processor.preprocess(data)]

    def smart_convert(self, output):
        data = []
        for i in output:
            d = {
                "scenario_id": i["scenario_id"],
                # 'kalman_difficulty': i['kalman_difficulty'],
                # 'trajectory_type': i['trajectory_type'],
            }
            if i.get("obj_ids") is not None:
                # 每 agent 节点对应的原始 object_id(顺序与 agent NodeStorage 一致),供肇事分桶。
                #   存成【逗号 join 的字符串标量】:数组形式会被 HeteroData/to_dict 压成 0-d 标量
                #   (只剩首元素),而字符串标量能像 scenario_id 一样原样存活。object_id 是无逗号的
                #   数字串,join/split 无歧义。
                d["agent_ids"] = ",".join(
                    np.asarray(i["obj_ids"]).astype(str).ravel().tolist()
                )
            obj_trajs = i["obj_trajs"]
            num_historical_steps = obj_trajs.shape[1]

            agent = {}
            agent["num_nodes"] = i["obj_trajs"].shape[0]
            type_one_hot_mask = obj_trajs[:, :, 6:11]
            agent["av_index"] = i["track_index_to_predict"]
            # agent['av_index'] = i['sdc_track_index']

            valid_mask = np.concatenate(
                (i["obj_trajs_mask"], i["obj_trajs_future_mask"].astype(bool)), axis=-1
            )
            agent["valid_mask"] = valid_mask

            predict_mask = np.zeros_like(valid_mask, dtype=bool)
            predict_mask[:, num_historical_steps:] = True
            predict_mask[~valid_mask] = False
            agent["predict_mask"] = predict_mask

            agent["type"] = np.argmax(i["obj_trajs"][:, 0, 6:11], axis=1)
            # agent['type'] = np.array([self.custom_argmax(row) for row in i['obj_trajs'][:, 0, 6:11]])
            agent["category"] = np.zeros(agent["num_nodes"], dtype=np.uint8)

            agent["position"] = np.concatenate(
                [i["obj_trajs_pos"], i["obj_trajs_future_state"][..., :3]], axis=1
            ).astype(np.float32)

            obj_his_heading_encoding = i["obj_trajs"][..., 23:25]
            obj_his_heading = np.arctan2(
                obj_his_heading_encoding[..., 0], obj_his_heading_encoding[..., 1]
            )
            obj_future_heading = i["obj_trajs_future_state"][..., 5]
            agent["heading"] = np.concatenate(
                [obj_his_heading, obj_future_heading], axis=1
            ).astype(np.float32)

            velo_all = np.concatenate(
                [i["obj_trajs"][..., 25:27], i["obj_trajs_future_state"][..., 3:5]],
                axis=1,
            )
            agent["velocity"] = np.pad(
                velo_all,
                pad_width=((0, 0), (0, 0), (0, 1)),
                mode="constant",
                constant_values=0.0,
            ).astype(np.float32)

            if i.get("center_objects_world"):
                agent["center_objects_world"] = (
                    i["center_objects_world"].reshape(1, -1).astype(np.float32)
                )

            agent["shape"] = np.concatenate(
                [i["obj_trajs"][..., 3:6], i["obj_trajs_future_state"][..., 2:5]],
                axis=1,
            ).astype(np.float32)
            d["agent"] = NodeStorage(agent)

            d["map_polygon"] = {}
            d["map_polygon"]["num_nodes"] = i["polyline_light_type"].shape[0]
            d["map_polygon"]["light_type"] = i["polyline_light_type"].copy()
            d["map_polygon"]["type"] = i["polyline_type"].copy()
            d["map_polygon"] = NodeStorage(d["map_polygon"])

            d["map_point"] = {}
            d["map_point"]["num_nodes"] = i["map_points_pos"].shape[0]
            d["map_point"]["position"] = i["map_points_pos"][:, :2]
            d["map_point"]["orientation"] = i["map_points_orientation"].copy()

            d["map_point"]["magnitude"] = i["map_points_magnitude"].copy()
            d["map_point"]["height"] = np.zeros_like(i["map_points_magnitude"]).astype(
                np.float32
            )
            d["map_point"]["type"] = i["map_points_type"].copy()
            d["map_point"] = NodeStorage(d["map_point"])

            edge_index = i["point_to_polyline_edge_index"].copy()
            d["map_point", "to", "map_polygon"] = NodeStorage(
                {"edge_index": edge_index}
            )
            if i.get("center_objects_type"):
                d["center_objects_type"] = i["center_objects_type"]

            d = self.target_transform(d)

            data.append(self.token_processor.preprocess(d))
        return data

    def collate_fn(self, data_list):
        # 评测时失败场景为 None,丢弃;整批都失败则返回 None(由评测循环跳过)。
        data_list = [d for d in data_list if d is not None]
        # 窗口化:__getitem__ 每个场景可能返回 list[dict](N 窗口)→ 展平为扁平样本列表。
        records = []
        for d in data_list:
            if isinstance(d, list):
                records.extend([x for x in d if x is not None])
            else:
                records.append(d)
        if len(records) == 0:
            return None
        # eval:每个窗口独立成一个 batch(collect_smart 逐窗前向,全锚点都要报,不可丢,
        #   且避免 N 窗口并一次前向 OOM)。
        if self.config.get("eval", False):
            return [self._merge_records([r]) for r in records]
        # 训练:MTR 同款——跨场景展平后的窗口(center_object)总数若超 max_agents_per_batch,
        #   随机截断到上限(每窗是独立 sub-scene,随机丢安全;丢掉的下个 epoch 再现),再合并成
        #   一个 batch 前向。max_agents_per_batch 由此对窗口化生效(与 agent-centric 同义)。
        cap = self.config.get("max_agents_per_batch", None)
        if cap is not None and int(cap) > 0 and len(records) > int(cap):
            keep = np.random.choice(len(records), int(cap), replace=False)
            records = [records[int(i)] for i in sorted(keep)]
        return self._merge_records(records)

    def _merge_records(self, data_list):
        batch_size = len(data_list)
        merged_data = HeteroData()

        if self.scene_centric:
            single_keys = ["scenario_id"]
        else:
            single_keys = [
                "scenario_id",
                "kalman_difficulty",
                "trajectory_type",
                "center_objects_type",
            ]
        all_keys = [
            "map_save",
            "pt_token",
            "agent",
            "map_point",
            "map_polygon",
            "map_point_to_map_polygon",
        ]
        sample = data_list[0]
        pt_offset = 0
        pl_offset = 0
        for k in single_keys:
            if k == "scenario_id":
                merged_data[k] = [data[k] for data in data_list]
            elif k == "kalman_difficulty":
                merged_data[k] = torch.from_numpy(
                    np.concatenate([data[k] for data in data_list], axis=0)
                )
            else:
                merged_data[k] = torch.tensor([data[k] for data in data_list])

        for key in all_keys:
            merged_data[key] = {}
            if key != "map_point_to_map_polygon":
                for sub_key in sample[key].keys():
                    if sub_key == "num_nodes":
                        continue
                    elif sub_key == "drivable":
                        merged_data[key][sub_key] = [
                            data[key][sub_key] for data in data_list
                        ]
                    try:
                        merged_data[key][sub_key] = torch.from_numpy(
                            np.concatenate(
                                [data[key][sub_key] for data in data_list], axis=0
                            )
                        )
                    except Exception as e:
                        merged_data[key][sub_key] = torch.tensor(
                            [data[key][sub_key] for data in data_list]
                        )
            else:
                merged_edges = []
                for data in data_list:
                    edge_index = data[key]["edge_index"]
                    edge_index[0] += pt_offset
                    edge_index[1] += pl_offset
                    merged_edges.append(edge_index)
                    pt_offset += edge_index[0].max().item() + 1
                    pl_offset += edge_index[1].max().item() + 1
                merged_data[key]["edge_index"] = torch.from_numpy(
                    np.concatenate(merged_edges, axis=1)
                )
            if key not in ["map_save", "map_point_to_map_polygon"]:
                ptr = torch.tensor([0])
                num_nodes_list = [data[key]["num_nodes"] for data in data_list]
                ptr = torch.cat([ptr, torch.tensor(num_nodes_list, dtype=torch.int64)])
                merged_data[key]["ptr"] = ptr
                merged_data[key]["num_nodes"] = torch.tensor(sum(num_nodes_list))
                merged_data[key]["batch"] = torch.arange(len(ptr)).repeat_interleave(
                    ptr
                )
        # 肇事分桶用:把每个 agent 节点的 (原始 object_id, scenario_id) 拼成两个【扁平 python
        #   字符串列表】(长度 = 全场景 agent 总数,顺序与上面 agent 子键 concat 完全一致)。
        #   不用 numpy 数组——Batch.from_data_list 会对变长 ndarray 属性做整理而打乱;纯 list
        #   会被原样裹成 [list],评测端取 [0] 即可。
        if self.scene_centric and ("agent_ids" in data_list[0]):
            flat_ids, flat_scene, flat_anchor = [], [], []
            for data in data_list:
                s = str(np.asarray(data["agent_ids"]).ravel()[0]) if hasattr(
                    data["agent_ids"], "ravel") else str(data["agent_ids"])
                oids = s.split(",") if s else []
                sid = str(np.asarray(data["scenario_id"]).ravel()[0])
                # 窗口化:anchor_id = 该窗口锚点的 object_id;窗口内 oid 唯一,故 oid==anchor
                #   即标出锚点节点(eval/log 只报锚点)。非窗口化无 anchor_id → 全 False。
                anchor = None
                if "anchor_id" in data:
                    anchor = str(np.asarray(data["anchor_id"]).ravel()[0]) if hasattr(
                        data["anchor_id"], "ravel") else str(data["anchor_id"])
                flat_ids.extend(oids)
                flat_scene.extend([sid] * len(oids))
                flat_anchor.extend([(o == anchor) for o in oids])
            merged_data["agent_ids"] = flat_ids
            merged_data["agent_scene"] = flat_scene
            merged_data["agent_is_anchor"] = flat_anchor
            # TTC 评测:每 agent 节点 → 该窗口的 accident_ttc(与 flat_ids 顺序一致),供评测端按锚点分组。
            if "accident_ttc" in data_list[0]:
                flat_ttc = []
                for data in data_list:
                    s = str(np.asarray(data["agent_ids"]).ravel()[0]) if hasattr(
                        data["agent_ids"], "ravel") else str(data["agent_ids"])
                    oids = s.split(",") if s else []
                    ttc = int(np.asarray(data["accident_ttc"]).ravel()[0]) if hasattr(
                        data["accident_ttc"], "ravel") else int(data["accident_ttc"])
                    flat_ttc.extend([ttc] * len(oids))
                merged_data["agent_ttc"] = flat_ttc
        return Batch.from_data_list([merged_data])

    def __getitem__(self, idx):
        if self.config.get("onthefly", False):
            return self._getitem_onthefly_smart(idx)
        # h5 cache 模式(窗口化):idx 索引到一个【场景】,读回它【全部窗口】成 list[dict],
        #   与 onthefly 完全一致——collate 跨场景展平后 cap 到 max_agents_per_batch(训练)或逐窗
        #   成独立 batch(eval)。base.load_data 已按 `__` 把同场景的窗口归到 h5_scenario_groups。
        scene_name = self.h5_scenario_index[idx]
        keys = self.h5_scenario_groups[scene_name]
        records = []
        for file_key in keys:
            file_path = self.data_loaded[file_key]["h5_path"]
            if file_path not in self.file_cache:
                self.file_cache[file_path] = self._get_file(file_path)
            group = self.file_cache[file_path][file_key]
            records.append({k: load_item(group[k]) for k in group.keys()})
        return records

    @staticmethod
    def _flatten_smart_record(record):
        """HeteroData 窗口 → 扁平 dict(与 _getitem_onthefly_smart 内 _flatten 同款)。"""
        if isinstance(record, HeteroData):
            record = record.to_dict()
            record.update(record.pop("_global_store", None) or {})
        return {merge_tuple_key(k): v for k, v in record.items()}

    def _build_accident_ttc_windows_smart(self, data_path, scene_name):
        """对一个碰撞场景,按 accident_ttc_offsets 的每个 T(秒)为肇事车建一组 SMART 窗口。
        锚点 A=C-10*T(10Hz);每个窗口 dict 加 accident_ttc=T。无可评样本 → None。
        说明:SMART preprocess 把轨迹裁到前 91 帧 → realaccident(92 帧)的末帧碰撞锚点会偏 0.1s,
        其余与 agent-centric 路径一致(碰撞帧定位/存活门控复用 BaseDataset)。"""
        scenario = read_scenario(data_path, self.onthefly_mappings[data_path], scene_name)
        ret = self.preprocess(scenario)
        C, collider_ids = self._collision_anchor(ret)
        if C is None or not collider_ids:
            return None
        offsets = self.config.get("accident_ttc_offsets", [1, 2, 3, 4, 5, 6, 7, 8])
        records = []
        for T in offsets:
            sub = self._reanchor_scene(ret, int(round(C - 10 * float(T))), collider_ids)
            if sub is None:
                continue
            windows = self._smart_windows_from_internal(
                sub, only_anchors=sub["tracks_to_predict"]["track_index"])
            if not windows:
                continue
            for r in windows:
                fr = self._flatten_smart_record(r)
                fr["accident_ttc"] = np.int64(int(T))
                records.append(fr)
        return records or None

    def _getitem_onthefly_smart(self, idx, _depth=0):
        """直接读 ScenarioNet pkl 现处理,转成与 h5 路径一致的扁平 dict
        (HeteroData.to_dict() + merge_tuple_key),collate_fn 两种模式通用。
        训练时失败场景随机回退(保证 batch 满);评测时(eval=True)失败场景返回 None 由 collate
        丢弃——避免"随机重复填充"污染基准口径(每个场景至多评一次,不可处理者直接不计)。"""
        if _depth >= 50:
            raise RuntimeError("[SMARTDataset] on-the-fly: 连续 50 次取样失败,检查数据/object_type。")
        is_eval = bool(self.config.get("eval", False))
        data_path, scene_name = self.onthefly_index[idx]
        self.starting_frame = self.onthefly_starting_frames[data_path]
        # 评测:失败→None(丢弃);训练:失败→随机换样
        on_fail = (lambda: None) if is_eval else (
            lambda: self._getitem_onthefly_smart(np.random.randint(len(self.onthefly_index)), _depth + 1))
        # 事故域「碰撞前 TTC 滑窗」评测(只评肇事车,每锚点一组窗口,标 accident_ttc)。
        #   复用 BaseDataset 的 _collision_anchor / _reanchor_scene(SMART preprocess 同样 update(metadata)、
        #   置 current_time_index,_smart_windows_from_internal 也尊重 current_time_index)。
        if self.config.get("accident_ttc_eval", False):
            try:
                out = self._build_accident_ttc_windows_smart(data_path, scene_name)
            except Exception:
                out = None
            return out if out else on_fail()
        try:
            scenario = read_scenario(data_path, self.onthefly_mappings[data_path], scene_name)
            preprocessed = self.preprocess(scenario)
            # 窗口化:每个 prediction_target 一个局部窗口(独立 sub-scene,等价 MTR center_object)。
            #   eval=建全部锚点(都要报指标,不可丢);训练=与 MTR 同款,本场景所有 official 目标
            #   各开一窗收成 list,跨场景展平后由 collate 按 max_agents_per_batch 随机截断再合并前向。
            #   单场景最多建 max_agents_per_batch 个(再多在本步也会被截掉,省 tokenize 开销),超出随机取子集。
            if is_eval:
                # 评测提速(可选):smart_eval_window_cap=N → 每场景至多随机评 N 个 official 窗口
                #   (只给大集开:waymo/highd/deepaccident_normal;事故集不设此键 → 仍建全部锚点,
                #    保证 collider 不被抽掉、肇事分桶/对照有效)。随机种子按 scene_name 固定 → 可复现。
                cap = self.config.get("smart_eval_window_cap", None)
                if cap is not None and int(cap) > 0:
                    ti = list(preprocessed.get("tracks_to_predict", {}).get("track_index", []))
                    if len(ti) == 0:
                        return on_fail()
                    if len(ti) > int(cap):
                        rng = np.random.RandomState(zlib.crc32(str(scene_name).encode()) & 0xffffffff)
                        sel = rng.choice(len(ti), int(cap), replace=False)
                        anchors = [int(ti[int(i)]) for i in sel]
                    else:
                        anchors = [int(x) for x in ti]
                    output = self._smart_windows_from_internal(preprocessed, only_anchors=anchors)
                else:
                    output = self._smart_windows_from_internal(preprocessed)
            else:
                ti = list(preprocessed.get("tracks_to_predict", {}).get("track_index", []))
                if len(ti) == 0:
                    return on_fail()
                cap = self.config.get("max_agents_per_batch", None)
                if cap is not None and int(cap) > 0 and len(ti) > int(cap):
                    sel = np.random.choice(len(ti), int(cap), replace=False)
                    anchors = [int(ti[i]) for i in sel]
                else:
                    anchors = [int(x) for x in ti]
                output = self._smart_windows_from_internal(preprocessed, only_anchors=anchors)
        except Exception:
            return on_fail()
        if not output:
            return on_fail()

        def _flatten(record):
            if isinstance(record, HeteroData):
                record = record.to_dict()
                record.update(record.pop("_global_store", None) or {})
            return {merge_tuple_key(k): v for k, v in record.items()}

        # 窗口化:返回该场景全部窗口 list[dict]。eval→collate 逐窗成独立 batch;
        #   训练→collate 跨场景展平 + 随机截断 + 合并成一个 batch(MTR 同款)。
        return [_flatten(r) for r in output]

    def custom_argmax(self, row):
        reversed_row = row[::-1]
        max_val = np.max(row)
        if max_val == 0:
            return 0
        else:
            last_idx_reversed = reversed_row.argmax()
            original_idx = len(row) - last_idx_reversed - 1
            return original_idx + 1


class WaymoTargetBuilder(BaseTransform):

    def __init__(
        self, num_historical_steps: int, num_future_steps: int, mode="train"
    ) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.mode = mode
        self.num_features = 3
        self.augment = False
        self.logger = Logging().log(level="DEBUG")

    def score_ego_agent(self, agent):
        av_index = agent["av_index"]
        agent["category"][av_index] = 5
        return agent

    def clip(self, agent, max_num=32):
        av_index = agent["av_index"]
        valid = agent["valid_mask"]
        ego_pos = agent["position"][av_index]
        obstacle_mask = agent["type"] == 3
        distance = torch.norm(
            agent["position"][:, self.num_historical_steps - 1, :2]
            - ego_pos[self.num_historical_steps - 1, :2],
            dim=-1,
        )  # keep the closest 100 vehicles near the ego car
        distance[obstacle_mask] = 10e5
        sort_idx = distance.sort()[1]
        mask = torch.zeros(valid.shape[0])
        mask[sort_idx[:max_num]] = 1
        mask = mask.to(torch.bool)
        mask[av_index] = True
        new_av_index = mask[:av_index].sum()
        agent["num_nodes"] = int(mask.sum())
        agent["av_index"] = int(new_av_index)
        excluded = ["num_nodes", "av_index", "ego"]
        for key, val in agent.items():
            if key in excluded:
                continue
            if key == "id":
                val = list(np.array(val)[mask])
                agent[key] = val
                continue
            if len(val.size()) > 1:
                agent[key] = val[mask, ...]
            else:
                agent[key] = val[mask]
        return agent

    def score_nearby_vehicle(self, agent, max_num=10):
        av_index = agent["av_index"]
        agent["category"] = torch.zeros_like(agent["category"])
        obstacle_mask = agent["type"] == 3
        pos = agent["position"][av_index, self.num_historical_steps, :2]
        distance = torch.norm(
            agent["position"][:, self.num_historical_steps, :2] - pos, dim=-1
        )
        distance[obstacle_mask] = 10e5
        sort_idx = distance.sort()[1]
        nearby_mask = torch.zeros(distance.shape[0])
        nearby_mask[sort_idx[1:max_num]] = 1
        nearby_mask = nearby_mask.bool()
        agent["category"][nearby_mask] = 3
        agent["category"][obstacle_mask] = 0

    def score_trained_vehicle(self, agent, max_num=10, min_distance=0):
        av_index = agent["av_index"]
        agent["category"] = np.zeros_like(agent["category"])
        pos = agent["position"][av_index, self.num_historical_steps, :2]
        distance = np.linalg.norm(
            agent["position"][:, self.num_historical_steps, :2] - pos, axis=-1
        )
        distance_all_time = np.linalg.norm(
            agent["position"][:, :, :2] - agent["position"][av_index, :, :2], axis=-1
        )
        invalid_mask = (
            distance_all_time < 150
        )  # we do not believe the perception out of range of 150 meters
        agent["valid_mask"] = agent["valid_mask"] * invalid_mask
        # we do not predict vehicle  too far away from ego car
        closet_vehicle = distance < 100
        valid = agent["valid_mask"]
        valid_current = valid[:, self.num_historical_steps :]
        valid_counts = valid_current.sum(axis=1)
        counts_vehicle = valid_counts >= 1
        no_backgroud = agent["type"] != 3
        vehicle2pred = closet_vehicle & counts_vehicle & no_backgroud
        if vehicle2pred.sum() > max_num:
            # too many still vehicle so that train the model using the moving vehicle as much as possible
            true_indices = np.nonzero(vehicle2pred)[0]
            selected_indices = np.random.choice(true_indices, max_num, replace=False)
            vehicle2pred[:] = False
            vehicle2pred[selected_indices] = True
        agent["category"][vehicle2pred] = 3

    def rotate_agents(
        self, position, heading, num_nodes, num_historical_steps, num_future_steps
    ):
        origin = position[:, num_historical_steps - 1]
        theta = heading[:, num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(num_nodes, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        target = origin.new_zeros(num_nodes, num_future_steps, 4)
        target[..., :2] = torch.bmm(
            position[:, num_historical_steps:, :2] - origin[:, :2].unsqueeze(1), rot_mat
        )
        his = origin.new_zeros(num_nodes, num_historical_steps, 4)
        his[..., :2] = torch.bmm(
            position[:, :num_historical_steps, :2] - origin[:, :2].unsqueeze(1), rot_mat
        )
        if position.size(2) == 3:
            target[..., 2] = position[:, num_historical_steps:, 2] - origin[
                :, 2
            ].unsqueeze(-1)
            his[..., 2] = position[:, :num_historical_steps, 2] - origin[
                :, 2
            ].unsqueeze(-1)
            target[..., 3] = wrap_angle(
                heading[:, num_historical_steps:] - theta.unsqueeze(-1)
            )
            his[..., 3] = wrap_angle(
                heading[:, :num_historical_steps] - theta.unsqueeze(-1)
            )
        else:
            target[..., 2] = wrap_angle(
                heading[:, num_historical_steps:] - theta.unsqueeze(-1)
            )
            his[..., 2] = wrap_angle(
                heading[:, :num_historical_steps] - theta.unsqueeze(-1)
            )
        return his, target

    def __call__(self, data) -> HeteroData:
        agent = data["agent"]
        self.score_ego_agent(agent)
        self.score_trained_vehicle(agent, max_num=32)
        return HeteroData(data)
