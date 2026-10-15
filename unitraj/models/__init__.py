"""Load the requested model without importing unrelated optional dependencies."""
from importlib import import_module

MODEL_SPECS = {
    'autobot': ('autobot.autobot', 'AutoBotEgo'),
    'MTR': ('mtr.MTR', 'MotionTransformer'),
    'wayformer': ('wayformer.wayformer', 'Wayformer'),
    'MAE': ('fmae.trainer_mae', 'TrainerMAE'),
    'forecast': ('fmae.trainer_forecast', 'TrainerForecast'),
    'GAMEFORMER': ('gameformer.gameformer', 'GameFormerModel'),
    'SENEVA': ('seneva.seneva', 'SenevaModel'),
    'EMP': ('emp.trainer_forecast', 'TrainerEMP'),
    'BeTopNet': ('betopnet.betopnet', 'BeTopNet'),
    'SMART': ('smart.smart', 'SMART'),
    'TrajFlow': ('trajflow.trajflow', 'TrajFlow'),
    'recon': ('recon.recon', 'RECON'),
}


def build_model(config):
    name = config.method.model_name
    if name not in MODEL_SPECS:
        raise ValueError(f'Unknown model {name!r}. Available: {list(MODEL_SPECS)}')
    module, symbol = MODEL_SPECS[name]
    cls = getattr(import_module('unitraj.models.' + module), symbol)
    return cls(config=config)
