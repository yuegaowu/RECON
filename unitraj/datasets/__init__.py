"""Per-model input adapters sharing the project's data processing rules."""
from importlib import import_module

DATASET_SPECS = {
    'autobot': ('autobot_dataset', 'AutoBotDataset'),
    'MTR': ('MTR_dataset', 'MTRDataset'),
    'wayformer': ('wayformer_dataset', 'WayformerDataset'),
    'recon': ('wayformer_dataset', 'WayformerDataset'),
    'MAE': ('fmae_dataset', 'FMAEDataset'),
    'forecast': ('fmae_dataset', 'FMAEDataset'),
    'GAMEFORMER': ('GAMEFORMER_dataset', 'GameFormerDataset'),
    'SENEVA': ('SENEVA_dataset', 'SenevaDataset'),
    'EMP': ('EMP_dataset', 'EMPDataset'),
    'BeTopNet': ('BETOPNET_dataset', 'BeTopDataset'),
    'SMART': ('SMART_dataset', 'SMARTDataset'),
    'TrajFlow': ('TRAJFLOW_dataset', 'TrajFlowDataset'),
}


def build_dataset(config, val=False):
    name = config.method.model_name
    if name not in DATASET_SPECS:
        raise ValueError(f'Unknown model {name!r}. Available: {list(DATASET_SPECS)}')
    module, symbol = DATASET_SPECS[name]
    cls = getattr(import_module('unitraj.datasets.' + module), symbol)
    return cls(config=config, is_validation=val)
