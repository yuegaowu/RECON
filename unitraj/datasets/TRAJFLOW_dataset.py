from .base_dataset import BaseDataset


class TrajFlowDataset(BaseDataset):
    """TrajFlow consumes the exact MTR-family unified input (29-dim agents, map
    polylines, future states) — identical to MTR / BeTopNet. The flow-matching
    target (normalized GT future) is built on-the-fly inside the model wrapper from
    center_gt_trajs, so no extra preprocessing is needed: this is BaseDataset
    unchanged, mirroring MTRDataset / BeTopDataset."""

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
