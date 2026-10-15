from .base_dataset import BaseDataset


class BeTopDataset(BaseDataset):
    """BeTopNet consumes the exact MTR-family unified input (29-dim agents, map
    polylines, future states). The behavioral-topology GT braids are computed
    on-the-fly inside the decoder from obj_trajs_future_state + map_polylines, so
    no extra preprocessing is needed — this is BaseDataset unchanged, mirroring
    MTRDataset."""

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
