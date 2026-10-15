# Behavioral Topology (BeTopNet): https://arxiv.org/abs/2409.18031
# Published at NeurIPS 2024 (OpenDriveLab)
# Pipeline developed upon Motion Transformer (MTR), so the context encoder is the
# (identical) MTR encoder already shipped in UniTraj; only the decoder adds the
# additive behavioral-topology reasoning head.

import torch
from easydict import EasyDict

from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.mtr.MTR import MTREncoder
from unitraj.models.betopnet.betop_decoder import BeTopDecoder


class BeTopNet(BaseModel):

    def __init__(self, config):
        super(BeTopNet, self).__init__(config)
        self.config = config
        self.model_cfg = EasyDict(config)
        self.pred_dicts = []

        # propagate global UniTraj fields into the decoder cfg (mirrors unitraj MTR)
        self.model_cfg.DECODER['CENTER_OFFSET_OF_MAP'] = self.model_cfg['center_offset_of_map']
        self.model_cfg.DECODER['NUM_FUTURE_FRAMES'] = self.model_cfg['future_len']
        self.model_cfg.DECODER['OBJECT_TYPE'] = self.model_cfg['object_type']

        self.context_encoder = MTREncoder(self.model_cfg.CONTEXT_ENCODER)
        self.motion_decoder = BeTopDecoder(
            in_channels=self.context_encoder.num_out_channels,
            config=self.model_cfg.DECODER
        )

    def forward(self, batch):
        enc_dict = self.context_encoder(batch)
        out_dict = self.motion_decoder(enc_dict)

        output = {}
        if self.training:
            pred_list = self.motion_decoder.forward_ret_dict['pred_list']
            mode_scores, out_dists = pred_list[-1][0], pred_list[-1][1]
            output['predicted_probability'] = mode_scores       # [B, num_query] (logits)
            output['predicted_trajectory'] = out_dists           # [B, num_query, T, 7]
            loss, _, _ = self.motion_decoder.get_loss()
        else:
            output['predicted_probability'] = out_dict['pred_scores']   # [B, modes]
            output['predicted_trajectory'] = out_dict['pred_trajs']     # [B, modes, T, 7]
            # BeTop's training-time loss paths (distinct-anchor selection) don't run at
            # eval; val metrics come from `prediction` via base_model.log_info.
            loss = torch.zeros(1, device=output['predicted_trajectory'].device)

        return output, loss

    def get_loss(self):
        loss, _, _ = self.motion_decoder.get_loss()
        return loss

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

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=-1, verbose=True)
        return [optimizer], [scheduler]
