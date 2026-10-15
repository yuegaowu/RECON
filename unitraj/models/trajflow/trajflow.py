# TrajFlow: Multimodal Motion Prediction via Flow Matching
# https://arxiv.org/abs/2506.08541  (IROS 2025, Qi Yan et al., UBC)
#
# UniTraj wrapper. TrajFlow is built on the MTR / BeTopNet codebase, so the context
# encoder is the (identical) MTR encoder and the input is the standard UniTraj unified
# MTR-family representation. Only the decoder is new: a conditional flow-matching
# denoiser. This wrapper adapts the vendored TrajFlow model (models/, denoising/) to
# the UniTraj BaseModel contract:
#
#   training: build a `denoiser_dict` (normalized GT) from center_gt_trajs, run the
#             FlowMatcher loss (1-2 inner forwards for cross-time consistency), and
#             return (prediction, loss). `prediction` here is a monitoring readout
#             (NMS over the last decoder layer's noised prediction) used by log_info.
#   eval:     run the FlowMatcher sampler (noise -> trajectory), which internally runs
#             the decoder in eval mode and stores the 6-mode NMS prediction in
#             batch['pred_trajs'/'pred_scores'] — that is what TrajFlow reports.

import os
import copy
import logging
from pathlib import Path

import torch
from easydict import EasyDict

from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.trajflow.models.dmt_model import DenoisingMotionTransformer
from unitraj.models.trajflow.denoising.flow_matching import FlowMatcher
from unitraj.models.trajflow.utils.denoising_data_rescale import shift_data_to_normalize
from unitraj.models.trajflow.models.denoising_decoder.decoder_utils import generate_final_prediction


class TrajFlow(BaseModel):

    def __init__(self, config):
        super(TrajFlow, self).__init__(config)
        self.config = config

        cfg = self._build_dmt_config(config)
        self.data_rescale = cfg.DATA_CONFIG.DATA_RESCALE
        self.num_motion_modes = cfg.MODEL_DMT.DMT.NUM_MOTION_MODES
        self.ctc_loss = cfg.MODEL_DMT.DENOISING.CTC_LOSS

        logger = logging.getLogger('TrajFlow')

        model = DenoisingMotionTransformer(config=cfg, logger=logger)

        # The FlowMatcher *owns* the DMT model — register ONLY the flow_matcher so the
        # DMT parameters appear once in the module tree (no duplicate ckpt keys).
        denoising_cfg = cfg.MODEL_DMT.DENOISING
        self.flow_matcher = FlowMatcher(
            model=model,
            sampling_timesteps=denoising_cfg.FM.SAMPLING_STEPS,
            objective=denoising_cfg.FM.OBJECTIVE,
            t_schedule=denoising_cfg.FM.T_SCHEDULE,
            logger=logger,
            data_rescale=cfg.DATA_CONFIG.DATA_RESCALE,
            ckpt_dir=None,
            model_cfg=copy.deepcopy(cfg.MODEL_DMT),
        )

    # ------------------------------------------------------------------ config
    def _build_dmt_config(self, config):
        """Translate the UniTraj (flat OmegaConf) config into the nested EasyDict tree
        the vendored TrajFlow modules expect, injecting UniTraj global fields."""
        model_dmt = EasyDict(dict(config['MODEL_DMT']))
        # Inject UniTraj globals into the decoder cfg (mirrors betopnet.py / MTR).
        model_dmt.DMT['OBJECT_TYPE'] = list(config['object_type'])
        model_dmt.DMT['NUM_FUTURE_FRAMES'] = int(config['future_len'])
        model_dmt.DMT['CENTER_OFFSET_OF_MAP'] = list(config['center_offset_of_map'])

        # Resolve the intention-points pkl to an absolute path so build_motion_query's
        # `init_cfg().ROOT_DIR / INTENTION_POINTS_FILE` join yields the right file
        # (pathlib: ROOT_DIR / <abs path> == <abs path>).
        ip = model_dmt.DMT['INTENTION_POINTS_FILE']
        if not os.path.isabs(ip):
            # this file lives at unitraj/models/trajflow/trajflow.py -> parents[2] == unitraj/
            unitraj_root = Path(__file__).resolve().parents[2]
            ip = str((unitraj_root / ip).resolve())
        model_dmt.DMT['INTENTION_POINTS_FILE'] = ip

        # FlowMatcher reads model_cfg.DEVICE off a deepcopy of MODEL_DMT, so set it here too.
        model_dmt.DEVICE = 'cuda'

        cfg = EasyDict()
        cfg.MODEL_DMT = model_dmt
        cfg.DEVICE = 'cuda'
        cfg.SAVE_DIR = EasyDict({'CKPT_DIR': None})
        cfg.DATA_CONFIG = EasyDict({'DATA_RESCALE': model_dmt.get('DATA_RESCALE', 'linear')})
        return cfg

    # ------------------------------------------------------------------ forward
    def forward(self, batch):
        input_dict = batch['input_dict']
        num_center_objects = input_dict['center_gt_trajs'].shape[0]

        batch['batch_size'] = num_center_objects
        batch['batch_sample_count'] = [num_center_objects]   # FlowMatcher uses sum(...) == B
        batch['cur_epoch'] = int(self.current_epoch)

        if self.training:
            # Build denoiser_dict (normalized GT future) for the flow-matching target.
            gt_metric = input_dict['center_gt_trajs'][..., 0:2].contiguous()   # [B, T, 2]
            gt_mask = input_dict['center_gt_trajs_mask']                        # [B, T]
            gt_norm = shift_data_to_normalize(gt_metric, gt_mask, self.data_rescale)
            batch['denoiser_dict'] = {
                'gt_traj_metric': gt_metric,
                'gt_traj_normalized': gt_norm,
                'gt_traj_mask': gt_mask,
            }

            disp_dict, wb_dict = {}, {}
            loss_reg, loss_cls, batch = self.flow_matcher(batch, disp_dict=disp_dict, wb_dict=wb_dict)
            loss = loss_reg + loss_cls

            # Monitoring readout for log_info: NMS the last decoder layer's prediction.
            pred_list = self.flow_matcher.model.denoising_decoder.forward_ret_dict['pred_list']
            scores, trajs, _ = generate_final_prediction(pred_list, self.num_motion_modes)
            output = {
                'predicted_probability': scores,          # [B, modes] (sigmoid)
                'predicted_trajectory': trajs,            # [B, modes, T, 7]
            }

            # 兜底(复刻 autobot.py 的防爆做法):flow-matching 解码器对 64 条噪声 query 积分,
            # 单条 loss 又是对 80 帧 .sum() 不平均,量级达几百,grad_clip 偶发兜不住 → 某个
            # batch 产生 NaN/Inf。一旦发生且照常 backward,AdamW 一/二阶动量被写成 NaN,此后
            # 每步永久 NaN 不恢复。这里检测到非有限 loss 时丢弃原计算图,用清洗过的解码器输出
            # 重建一个有梯度的「中性 loss」跳过该 batch,保持参数梯度连通。
            # ⚠️ IEEE754 规定 NaN*0 = NaN,必须先 nan_to_num 再乘 0,否则 backward 仍出 NaN 梯度。
            if not torch.isfinite(loss):
                print(f'[TrajFlow] WARNING: non-finite loss detected (value={loss.item()}); '
                      f'skipping this batch via safe fallback.', flush=True)
                raw_scores, raw_trajs = pred_list[-1][:2]
                safe_scores = torch.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
                safe_trajs = torch.nan_to_num(raw_trajs, nan=0.0, posinf=0.0, neginf=0.0)
                loss = (safe_scores.sum() + safe_trajs.sum()) * 0.0 + 1.0

            return output, loss

        # ---- eval: run the flow sampler; decoder stores the 6-mode NMS prediction ----
        batch['denoiser_dict'] = {}
        _trajs, _cls, batch = self.flow_matcher(batch, flag_sample=True)
        output = {
            'predicted_probability': batch['pred_scores'],   # [B, modes] (sigmoid)
            'predicted_trajectory': batch['pred_trajs'],     # [B, modes, T, 7]
        }
        loss = torch.zeros(1, device=output['predicted_trajectory'].device)
        return output, loss

    # ------------------------------------------------------------------ grad guard
    def on_after_backward(self):
        """梯度层护栏(forward 里的 loss 守卫不够)。

        现象:loss 仍有限(几百量级),但 flow-matching 解码器 / GMM-NLL 损失路径里
        `.norm(dim=-1)`(decoder_utils 多处用它算 anchor 距离)在两点重合时 norm=0,
        其梯度 x/‖x‖ = NaN;`sqrt`/`log` 在边界也会产生 Inf 梯度。
        ⚠️ `grad_clip_norm` 对 NaN 梯度无效:total_norm=NaN → 裁剪系数 max_norm/(NaN+eps)=NaN
        → 把所有梯度乘成 NaN → optimizer.step() 把权重写成 NaN,此后永久不恢复
        (表现为每步 loss=1.0 兜底、minADE6=nan)。
        本钩子在 backward 之后、grad_clip 之前执行:把任何含非有限值的梯度张量 nan_to_num
        清零,使该参数本步变为 no-op,保护权重与 AdamW 动量;其余正常梯度仍由 grad_clip 处理。
        """
        bad = False
        for p in self.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad = True
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if bad:
            print('[TrajFlow] WARNING: non-finite gradients detected; '
                  'sanitized to 0 before grad-clip/optimizer step.', flush=True)

    # ------------------------------------------------------------------ optim
    def configure_optimizers(self):
        decay_steps = [x for x in self.config['learning_rate_sched']]

        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in decay_steps:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * self.config['lr_decay']
            return max(cur_decay, self.config['lr_clip'] / self.config['learning_rate'])

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['learning_rate'],
                                      weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=-1)
        return [optimizer], [scheduler]
