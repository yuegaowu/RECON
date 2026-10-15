'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
Ported from BeTop/womd/betopnet/utils/loss_utils.py.

Only the BeTop-specific losses live here (focal_loss + topo_loss for the braid
reasoning heads). The GMM regression loss (nll_loss_gmm_direct) is reused from
unitraj.models.mtr.loss_utils.
'''

import torch
import torch.nn.functional as F


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Original implementation from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def topo_loss(
    prediction, targets, valid_mask,
    top_k=False, top_k_ratio=1.):
    """
    build the top-k CE loss for BeTop reasoning
    preds, targets, valid_mask: [b, src, tgt]
    return: loss [b]
    """
    b, s, t, step = prediction.shape
    targets = targets.float()

    loss = focal_loss(
        prediction,
        targets,
        reduction='none',
    )

    loss = loss * valid_mask[..., None]
    loss = loss.view(b, s * t, step)
    valid_mask = valid_mask.view(b, s * t)
    if top_k:
        # Penalises the top-k hardest pixels
        k = int(top_k_ratio * loss.shape[1])
        loss, _ = torch.sort(loss, dim=1, descending=True)
        loss = loss[:, :k]

    mask = torch.sum(valid_mask, dim=-1)
    mask = mask + (mask == 0).float()

    return torch.sum(loss.mean(-1), dim=1) / mask
