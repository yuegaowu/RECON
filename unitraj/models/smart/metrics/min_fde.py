from typing import Optional

import torch
from torchmetrics import Metric

from unitraj.models.smart.metrics.utils import topk
from unitraj.models.smart.metrics.utils import valid_filter


class minMultiFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minMultiFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        self.sum += torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                               target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                               p=2, dim=-1).min(dim=-1)[0].sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class minFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 eval_timestep: int = 60,
                 **kwargs) -> None:
        super(minFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        self.eval_timestep = eval_timestep

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        # 单模态 FDE:取「每个 agent 自己最后一个有效步」的位移误差
        # (不是写死在固定步评估,否则该步无效的 agent 会被整体漏掉)。
        eval_timestep = min(self.eval_timestep, pred.shape[1])
        vm = valid_mask[:, :eval_timestep]
        steps = torch.arange(1, eval_timestep + 1, device=pred.device)
        inds_last = (vm * steps).argmax(dim=-1)          # 每个 agent 的最后有效步
        ar = torch.arange(pred.size(0), device=pred.device)
        fde = torch.norm(pred[ar, inds_last] - target[ar, inds_last], p=2, dim=-1)
        has_valid = vm.any(dim=-1)
        self.sum += (fde * has_valid).sum()
        self.count += has_valid.sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
