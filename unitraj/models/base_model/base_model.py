import json

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

import unitraj.datasets.common_utils as common_utils
import unitraj.utils.visualization as visualization


class BaseModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pred_dicts = []

        if config.get('eval_nuscenes', False):
            self.init_nuscenes()

    def init_nuscenes(self):
        if self.config.get('eval_nuscenes', False):
            from nuscenes import NuScenes

            from nuscenes.eval.prediction.config import PredictionConfig

            from nuscenes.prediction import PredictHelper
            nusc = NuScenes(version='v1.0-trainval', dataroot=self.config['nuscenes_dataroot'])

            # Prediction helper and configs:
            self.helper = PredictHelper(nusc)

            with open('models/base_model/nuscenes_config.json', 'r') as f:
                pred_config = json.load(f)
            self.pred_config5 = PredictionConfig.deserialize(pred_config, self.helper)

    def forward(self, batch):
        """
        Forward pass for the model
        :param batch: input batch
        :return: prediction: {
                'predicted_probability': (batch_size,modes)),
                'predicted_trajectory': (batch_size,modes, future_len, 2)
                }
                loss (with gradient)
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        batch_size = len(batch['input_dict']['center_objects_id'])
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=batch_size)
        self.log_info(batch, batch_idx, prediction, status='train')
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status='val')
        return loss

    def predict(self, batch):
        prediction, loss = self.forward(batch)
        return prediction


    def on_validation_epoch_end(self):
        if self.config.get('eval_waymo', False):
            import pickle
            exp_name = self.config.get('exp_name', 'eval')
            from pathlib import Path
            output = Path(self.config.get('prediction_output_dir', 'outputs'))
            output.mkdir(parents=True, exist_ok=True)
            save_path = output / f'{exp_name}_pred_dicts.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(self.pred_dicts, f)
            print(f'[Waymo Eval] Saved {len(self.pred_dicts)} pred_dicts -> {save_path}')
            metric_results, result_format_str = self.compute_metrics_waymo(self.pred_dicts)
            print(metric_results)
            print(result_format_str)

        elif self.config.get('eval_nuscenes', False):
            import os
            os.makedirs('submission', exist_ok=True)
            json.dump(self.pred_dicts, open(os.path.join('submission', "evalai_submission.json"), "w"))
            metric_results = self.compute_metrics_nuscenes(self.pred_dicts)
            print('\n', metric_results)
            
        elif self.config.get('eval_argoverse2', False):
            metric_results = self.compute_metrics_av2(self.pred_dicts)
            
        self.pred_dicts = []

    def configure_optimizers(self):
        raise NotImplementedError

    def compute_metrics_nuscenes(self, pred_dicts):
        from nuscenes.eval.prediction.compute_metrics import compute_metrics
        metric_results = compute_metrics(pred_dicts, self.helper, self.pred_config5)
        return metric_results

    def compute_metrics_waymo(self, pred_dicts):
        from unitraj.models.base_model.waymo_eval import waymo_evaluation
        try:
            num_modes_for_eval = pred_dicts[0]['pred_trajs'].shape[0]
        except:
            num_modes_for_eval = 6
        metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts,
                                                             num_modes_for_eval=num_modes_for_eval)

        metric_result_str = '\n'
        for key in metric_results:
            metric_results[key] = metric_results[key]
            metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
        metric_result_str += '\n'
        metric_result_str += result_format_str

        return metric_result_str, metric_results

    def compute_metrics_av2(self, pred_dicts):
        from unitraj.models.base_model.av2_eval import argoverse2_evaluation
        try:
            num_modes_for_eval = pred_dicts[0]['pred_trajs'].shape[0]
        except:
            num_modes_for_eval = 6
        metric_results = argoverse2_evaluation(pred_dicts=pred_dicts,
                                               num_modes_for_eval=num_modes_for_eval)
        self.log('val/av2_official_minADE6', metric_results['min_ADE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_minFDE6', metric_results['min_FDE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_brier_minADE', metric_results['brier_min_ADE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_brier_minFDE', metric_results['brier_min_FDE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_miss_rate', metric_results['miss_rate'], prog_bar=True, on_step=False, on_epoch=True)
        
        # metric_result_str = '\n'
        # for key, value in metric_results.items():
        #     metric_result_str += '%s: %.4f\n' % (key, value)
        # metric_result_str += '\n'
        # print(metric_result_str)
        return metric_results
        
    def compute_official_evaluation(self, batch_dict, prediction):
        if self.config.get('eval_waymo', False):

            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']
            pred_trajs = prediction['predicted_trajectory']
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape

            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]

            pred_dict_list = []

            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][bs_idx],
                    'pred_trajs': pred_trajs_world[bs_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[bs_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][bs_idx],
                    'object_type': input_dict['center_objects_type'][bs_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][bs_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][bs_idx].cpu().numpy()
                }
                pred_dict_list.append(single_pred_dict)

            assert len(pred_dict_list) == batch_dict['batch_size']

            self.pred_dicts += pred_dict_list

        elif self.config.get('eval_nuscenes', False):
            from nuscenes.eval.prediction.data_classes import Prediction
            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']
            pred_trajs = prediction['predicted_trajectory']
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
            # assert num_feat == 7

            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]
            pred_dict_list = []

            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'instance': input_dict['scenario_id'][bs_idx].split('_')[1],
                    'sample': input_dict['scenario_id'][bs_idx].split('_')[2],
                    'prediction': pred_trajs_world[bs_idx, :, 4::5, 0:2].cpu().numpy(),
                    'probabilities': pred_scores[bs_idx, :].cpu().numpy(),
                }

                pred_dict_list.append(
                    Prediction(instance=single_pred_dict["instance"], sample=single_pred_dict["sample"],
                               prediction=single_pred_dict["prediction"],
                               probabilities=single_pred_dict["probabilities"]).serialize())

            self.pred_dicts += pred_dict_list
        
        elif self.config.get('eval_argoverse2', False):

            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']
            pred_trajs = prediction['predicted_trajectory']
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape

            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]

            pred_dict_list = []

            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][bs_idx],
                    'pred_trajs': pred_trajs_world[bs_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[bs_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][bs_idx],
                    'object_type': input_dict['center_objects_type'][bs_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][bs_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][bs_idx].cpu().numpy()
                }
                pred_dict_list.append(single_pred_dict)

            assert len(pred_dict_list) == batch_dict['batch_size']

            self.pred_dicts += pred_dict_list

    def log_info(self, batch, batch_idx, prediction, status='train'):
        ## logging
        # Split based on dataset
        inputs = batch['input_dict']
        gt_traj = inputs['center_gt_trajs'].unsqueeze(1)  # .transpose(0, 1).unsqueeze(0)
        gt_traj_mask = inputs['center_gt_trajs_mask'].unsqueeze(1)
        center_gt_final_valid_idx = inputs['center_gt_final_valid_idx']

        predicted_traj = prediction['predicted_trajectory']
        predicted_prob = prediction['predicted_probability'].detach().cpu().numpy()

        # Calculate ADE losses
        ade_diff = torch.norm(predicted_traj[:, :, :, :2] - gt_traj[:, :, :, :2], 2, dim=-1)
        ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / torch.sum(gt_traj_mask, dim=-1)
        ade_losses = ade_losses.cpu().detach().numpy()
        minade = np.min(ade_losses, axis=1)
        # Calculate FDE losses
        bs, modes, future_len = ade_diff.shape
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, modes, 1).to(torch.int64)

        fde = torch.gather(ade_diff, -1, center_gt_final_valid_idx).cpu().detach().numpy().squeeze(-1)
        minfde = np.min(fde, axis=-1)

        best_fde_idx = np.argmin(fde, axis=-1)
        # ── top-1(可部署)指标:argmax(score) 那条的误差 —— V3 排序蒸馏的直接考核量 ──
        # 用原始 [B,modes] 概率取 argmax,须在下一行把 predicted_prob 覆盖成标量之前算。
        top1_idx = np.argmax(predicted_prob, axis=-1)
        top1_FDE = fde[np.arange(bs), top1_idx]
        top1_ADE = ade_losses[np.arange(bs), top1_idx]
        MissRate1 = (top1_FDE > 2.0)
        top1_is_best = (top1_idx == best_fde_idx).astype(np.float32)
        predicted_prob = predicted_prob[np.arange(bs), best_fde_idx]
        miss_rate = (minfde > 2.0)
        brier_fde = minfde + np.square(1 - predicted_prob)

        loss_dict = {
            'minADE6': minade,
            'minFDE6': minfde,
            'miss_rate': miss_rate.astype(np.float32),
            'brier_fde': brier_fde,
            'top1_FDE': top1_FDE,
            'top1_ADE': top1_ADE,
            'MissRate1': MissRate1.astype(np.float32),
            'top1_is_best': top1_is_best}

        important_metrics = list(loss_dict.keys())

        # ── Waymo-style 2Hz metrics ──────────────────────────────────────────────
        # Mirrors the official Waymo evaluation protocol (no TF dependency).
        # Evaluated at 3 checkpoints (2Hz subsampling of 10Hz predictions):
        #   step 5  → t=2.5s (10Hz idx 24), MissRate L2 threshold 2.0m
        #   step 9  → t=4.5s (10Hz idx 44), MissRate L2 threshold 3.6m
        #   step 15 → t=7.5s (10Hz idx 74), MissRate L2 threshold 6.0m
        # Note: MissRate thresholds use the Waymo longitudinal threshold as L2 proxy.
        # Key logged metrics (matching official Waymo table columns):
        #   val/waymo/{type}_minADE   — avg of steps 5/9/15  (primary comparison metric)
        #   val/waymo/{type}_minFDE   — avg of steps 5/9/15
        #   val/waymo/{type}_MissRate — avg of steps 5/9/15
        #   val/waymo/minADE          — avg of 3 types        (overall headline metric)
        #   val/waymo/minFDE          — avg of 3 types
        #   val/waymo/MissRate        — avg of 3 types
        _WAYMO_STEPS = {
            5:  {'10hz_idx': 24, 'miss_thresh': 2.0},
            9:  {'10hz_idx': 44, 'miss_thresh': 3.6},
            15: {'10hz_idx': 74, 'miss_thresh': 6.0},
        }
        ade_diff_np = ade_diff.cpu().detach().numpy()   # (bs, modes, future_len)
        mask_np = gt_traj_mask.squeeze(1).cpu().numpy() # (bs, future_len)

        # waymo_per_agent: '{metric}{step}' -> (bs,) array
        # metric names: minADE / minFDE / MissRate  (no redundant "waymo_" prefix)
        waymo_per_agent = {}
        for step, cfg in _WAYMO_STEPS.items():
            idx = cfg['10hz_idx']
            if idx >= future_len:
                continue
            hz2_indices = list(range(4, idx + 1, 5))          # 2Hz frames 1..step in 10Hz indices
            step_l2   = ade_diff_np[:, :, hz2_indices]         # (bs, modes, S)
            step_mask = mask_np[:, hz2_indices]                 # (bs, S)
            valid_sum = np.maximum(step_mask.sum(axis=-1, keepdims=True), 1)
            ade_per_mode = (step_l2 * step_mask[:, np.newaxis, :]).sum(axis=-1) / valid_sum  # (bs, modes)
            waymo_per_agent[f'minADE{step}']   = ade_per_mode.min(axis=-1)
            fde_per_mode = ade_diff_np[:, :, idx]
            fde_valid    = mask_np[:, idx]
            waymo_per_agent[f'minFDE{step}']   = fde_per_mode.min(axis=-1) * fde_valid
            waymo_per_agent[f'MissRate{step}'] = ((fde_per_mode.min(axis=-1) > cfg['miss_thresh'])
                                                   & (fde_valid > 0)).astype(np.float32)
        # ────────────────────────────────────────────────────────────────────────

        new_dict = {}
        dataset_names = inputs['dataset_name']
        unique_dataset_names = np.unique(dataset_names)
        for dataset_name in unique_dataset_names:
            batch_idx_for_this_dataset = np.argwhere([n == str(dataset_name) for n in dataset_names])[:, 0]
            for key in loss_dict.keys():
                new_dict[dataset_name + '/' + key] = loss_dict[key][batch_idx_for_this_dataset]

        # merge new_dict with log_dict
        loss_dict.update(new_dict)
        # loss_dict.update(avg_dict)

        if status == 'val':
            # ── Per-agent-type logging ────────────────────────────────────────────
            # Logged with sync_dist=False to avoid DDP deadlock on conditional keys.
            # (sync_dist=True requires every rank to call self.log() for the SAME key;
            #  conditional per-type logging breaks this symmetry → all_reduce hangs.)
            agent_type_dict = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}

            # Accumulates per-type Waymo averages for the final overall summary.
            # type_waymo_avgs[metric_name] = list of per-type mean values
            type_waymo_avgs = {'minADE': [], 'minFDE': [], 'MissRate': []}

            for atype, aname in agent_type_dict.items():
                batch_idx_for_type = np.where(inputs['center_objects_type'] == atype)[0]
                n = len(batch_idx_for_type)
                if n == 0:
                    continue

                # ── UniTraj simple per-type metrics (secondary) ──────────────────
                for key in important_metrics:
                    self.log(
                        f"val/agent_type/{aname}_{key}",
                        float(np.mean(loss_dict[key][batch_idx_for_type])),
                        on_step=False, on_epoch=True, prog_bar=False,
                        sync_dist=False, batch_size=n,
                    )

                # ── Waymo-style 2Hz per-step metrics (secondary) ─────────────────
                if not waymo_per_agent:
                    continue
                step_vals = {'minADE': [], 'minFDE': [], 'MissRate': []}
                for step in [5, 9, 15]:
                    for metric in ['minADE', 'minFDE', 'MissRate']:
                        key = f'{metric}{step}'
                        if key not in waymo_per_agent:
                            continue
                        v = float(np.mean(waymo_per_agent[key][batch_idx_for_type]))
                        self.log(
                            f"val/waymo/{aname}_{key}",
                            v,
                            on_step=False, on_epoch=True, prog_bar=False,
                            sync_dist=False, batch_size=n,
                        )
                        step_vals[metric].append(v)

                # ── Waymo-style per-type summary: average over steps 5/9/15 ──────
                # These match the "VEHICLE / PEDESTRIAN / CYCLIST" rows in the
                # official Waymo evaluation table.
                for metric, vals in step_vals.items():
                    if vals:
                        avg_val = float(np.mean(vals))
                        self.log(
                            f"val/waymo/{aname}_{metric}",
                            avg_val,
                            on_step=False, on_epoch=True, prog_bar=False,
                            sync_dist=False, batch_size=n,
                        )
                        type_waymo_avgs[metric].append(avg_val)

            # ── Waymo-style overall summary: average over 3 types ────────────────
            # These match the "Avg" row in the official Waymo evaluation table.
            # Logged with sync_dist=True (called unconditionally on every rank).
            if waymo_per_agent:
                for metric, type_vals in type_waymo_avgs.items():
                    if type_vals:
                        self.log(
                            f"val/waymo/{metric}",
                            float(np.mean(type_vals)),
                            on_step=False, on_epoch=True, prog_bar=False,
                            sync_dist=True, batch_size=bs,
                        )
                # Also log per-step overall averages (secondary, for detailed analysis)
                for step in [5, 9, 15]:
                    for metric in ['minADE', 'minFDE', 'MissRate']:
                        key = f'{metric}{step}'
                        if key in waymo_per_agent:
                            self.log(
                                f"val/waymo/{key}",
                                float(np.mean(waymo_per_agent[key])),
                                on_step=False, on_epoch=True, prog_bar=False,
                                sync_dist=True, batch_size=bs,
                            )

        if status == 'val' and self.config.get('eval', False):
            # Extra detailed breakdowns (trajectory type, kalman difficulty) — only in eval mode.
            new_dict = {}
            trajectory_types = inputs["trajectory_type"].cpu().numpy()
            trajectory_correspondance = {0: "stationary", 1: "straight", 2: "straight_right",
                                         3: "straight_left", 4: "right_u_turn", 5: "right_turn",
                                         6: "left_u_turn", 7: "left_turn"}
            for traj_type in range(8):
                batch_idx_for_traj_type = np.where(trajectory_types == traj_type)[0]
                if len(batch_idx_for_traj_type) > 0:
                    for key in important_metrics:
                        new_dict["traj_type/" + trajectory_correspondance[traj_type] + "_" + key] = loss_dict[key][
                            batch_idx_for_traj_type]
            loss_dict.update(new_dict)

            new_dict = {}
            kalman_difficulties = inputs["kalman_difficulty"][:, -1].cpu().numpy()
            for kalman_bucket, (low, high) in {"easy": [0, 30], "medium": [30, 60], "hard": [60, 9999999]}.items():
                batch_idx_for_kalman_diff = \
                    np.where(np.logical_and(low <= kalman_difficulties, kalman_difficulties < high))[0]
                if len(batch_idx_for_kalman_diff) > 0:
                    for key in important_metrics:
                        new_dict["kalman/" + kalman_bucket + "_" + key] = loss_dict[key][batch_idx_for_kalman_diff]
            loss_dict.update(new_dict)

        # Take mean for each key; store original length for batch_size weighting.
        # agent_type/* metrics are already logged above with sync_dist=False — they
        # are NOT in loss_dict, so the loop below never sees them.
        size_dict = {key: len(value) for key, value in loss_dict.items()}
        loss_dict = {key: np.mean(value) for key, value in loss_dict.items()}

        # Log all remaining metrics with sync_dist=True (same key called on every rank).
        for k, v in loss_dict.items():
            self.log(
                status + "/" + k, v,
                on_step=(status == 'train'),
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=size_dict[k],
            )

        # if self.local_rank == 0 and status == 'val' and batch_idx == 0:
        #     img = visualization.visualize_prediction(batch, prediction)
        #     wandb.log({"prediction": [wandb.Image(img)]})

        return

    def convert_to_fmae_format(self, input):
        # GPU-native 改写(原版强制 .to("cpu") + numpy → PCIe 来回搬几百 MB / step)。
        # 现在全部在 input 所在 device 上跑;forecast / EMP / MAE 共用,语义完全对齐旧版。
        B = len(input['center_objects_id'])
        N_agent = self.config["max_num_agents"]
        N_lane = self.config["max_num_roads"]
        h_steps = self.config["past_len"]
        f_steps = self.config["future_len"]
        lane_sampling_pts = self.config["max_points_per_lane"]

        obj_trajs              = input["obj_trajs"]
        obj_trajs_pos          = input["obj_trajs_pos"]
        obj_trajs_mask         = input["obj_trajs_mask"]
        obj_trajs_future_state = input["obj_trajs_future_state"]
        obj_trajs_future_mask  = input["obj_trajs_future_mask"]
        map_polylines          = input["map_polylines"]
        map_polylines_mask     = input["map_polylines_mask"]
        center_objects_world   = input["center_objects_world"]

        # 用第一个 tensor 的 device 作为目标设备(Lightning 已把 batch 搬上 GPU)
        device = obj_trajs.device

        data = {}
        data["origin"] = center_objects_world[..., 0:2].clone()
        data["theta"] = center_objects_world[..., 6].clone()

        # ── x: 历史相邻帧 Δxy(相邻两帧任一无效 → 0)
        data["x"] = torch.zeros(B, N_agent, h_steps, 2, device=device, dtype=torch.float32)
        valid_pair_hist = obj_trajs_mask[..., :(h_steps - 1)] & obj_trajs_mask[..., 1:h_steps]  # (B,N,h-1)
        delta_hist = obj_trajs_pos[..., 1:h_steps, 0:2] - obj_trajs_pos[..., :(h_steps - 1), 0:2]
        data["x"][..., 1:h_steps, 0:2] = torch.where(
            valid_pair_hist.unsqueeze(-1), delta_hist, torch.zeros_like(delta_hist)
        )
        # slot 0 已为 0(zeros 初始化)

        # ── x_attr: [primary type | category | combined type](uint8,模型只用 col 2)
        # 向量化原 B×N×elif 嵌套 if;语义对齐:vehicle > pedestrian > bicycle > unknown。
        last  = obj_trajs[:, :, h_steps - 1]   # (B, N, F)
        first = obj_trajs[:, :, 0]             # (B, N, F)
        is_veh  = last[..., 6]  == 1
        is_ped  = last[..., 7]  == 1
        is_bike = last[..., 8]  == 1

        x_attr = torch.zeros((B, N_agent, 3), dtype=torch.uint8, device=device)
        # col 0
        t0 = torch.full((B, N_agent), 9, dtype=torch.uint8, device=device)
        t0[is_bike] = 3
        t0[is_ped]  = 1
        t0[is_veh]  = 0
        x_attr[..., 0] = t0
        # col 2
        t2 = torch.full((B, N_agent), 3, dtype=torch.uint8, device=device)
        t2[is_bike] = 2
        t2[is_ped]  = 1
        t2[is_veh]  = 0
        x_attr[..., 2] = t2
        # col 1
        focal_h = last[..., 9]  == 1
        sdc_h   = last[..., 10] == 1
        sdc_0   = first[..., 10] == 1
        cat = torch.zeros((B, N_agent), dtype=torch.uint8, device=device)
        cat[focal_h | sdc_h]         = 3
        cat[(~focal_h) & (~sdc_0)]   = 2
        x_attr[..., 1] = cat
        data["x_attr"] = x_attr

        data["x_positions"] = obj_trajs[..., :h_steps, :2].clone()
        data["x_centers"]   = obj_trajs[..., h_steps - 1, :2].clone()

        # ── angles & velocity(原版用 numpy → 换成 torch 原生,免 GPU↔CPU)
        data["x_angles"] = torch.asin(obj_trajs[..., h_steps + 12].float())

        v_past = torch.linalg.norm(obj_trajs[..., h_steps + 14:h_steps + 16].float(), dim=-1)
        v_fut  = torch.linalg.norm(obj_trajs_future_state[..., 2:].float(),            dim=-1)
        data["x_velocity"] = torch.cat((v_past, v_fut), dim=-1)

        # ── lanes
        data["lane_positions"] = map_polylines[..., :2].clone()
        mid_l = (lane_sampling_pts // 2) - 1
        mid_r = lane_sampling_pts // 2
        data["lane_centers"] = data["lane_positions"][:, :, mid_l:mid_r + 1].mean(dim=-2)
        data["lane_angles"]  = torch.atan2(
            data["lane_positions"][..., mid_r, 1] - data["lane_positions"][..., mid_l, 1],
            data["lane_positions"][..., mid_r, 0] - data["lane_positions"][..., mid_l, 0],
        )
        # lane_attr / is_intersections / track_id 已确认 model 不读,删除以省搬运。

        # ── y: 未来相对最后一帧的 Δxy(prev_invalid OR future_invalid → 0)
        valid_y = obj_trajs_mask[..., (h_steps - 1)].unsqueeze(-1) & obj_trajs_future_mask  # (B,N,f)
        y_delta = obj_trajs_future_state[..., :, 0:2] - obj_trajs[..., (h_steps - 1), 0:2].unsqueeze(-2)
        data["y"] = torch.where(valid_y.unsqueeze(-1), y_delta, torch.zeros_like(y_delta))

        # ── padding masks
        data["x_padding_mask"]      = torch.cat((~obj_trajs_mask, ~obj_trajs_future_mask), dim=-1)
        data["lane_padding_mask"]   = ~map_polylines_mask.clone()
        data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)

        # 离群:历史最后一帧距原点 > 150m → 全步 mask
        last_pos_norm = torch.linalg.norm(data["x_positions"][..., h_steps - 1, :].float(), dim=-1)  # (B,N)
        far_mask = (last_pos_norm > 150).unsqueeze(-1)                                                 # (B,N,1)
        data["x_padding_mask"] = torch.where(
            far_mask, torch.ones_like(data["x_padding_mask"]), data["x_padding_mask"]
        )
        # 历史最后一帧本身无效 → 全步 mask
        last_invalid = data["x_padding_mask"][..., h_steps - 1].unsqueeze(-1)
        data["x_padding_mask"] = torch.where(
            last_invalid, torch.ones_like(data["x_padding_mask"]), data["x_padding_mask"]
        )
        data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)

        data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)
        data["num_lanes"]  = (~data["lane_key_padding_mask"]).sum(-1)
        data["scenario_id"] = input["scenario_id"]

        # ── velocity_diff(相邻帧任一无效 → 0)
        data["x_velocity_diff"] = torch.zeros(B, N_agent, h_steps, device=device, dtype=torch.float32)
        vel_diff = data["x_velocity"][..., 1:h_steps] - data["x_velocity"][..., :(h_steps - 1)]
        data["x_velocity_diff"][..., 1:h_steps] = torch.where(
            valid_pair_hist, vel_diff, torch.zeros_like(vel_diff)
        )
        # slot 0 已为 0

        return data
    
    def convert_to_qcnet_format(self, input):
        return