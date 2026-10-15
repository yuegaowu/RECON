from .base_dataset import BaseDataset


class GameFormerDataset(BaseDataset):
    """UniTraj agent-centric samples adapted by the GameFormer model wrapper."""

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
