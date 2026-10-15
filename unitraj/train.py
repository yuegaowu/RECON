import warnings
# Suppress noisy third-party warnings that are not actionable
warnings.filterwarnings('ignore', category=UserWarning)   # pygame, hydra, knn_utils, etc.
warnings.filterwarnings('ignore', category=FutureWarning) # timm, torch.load, etc.

import os
import sys
import ctypes
import atexit
from pathlib import Path

# ── All processes: die when parent dies ───────────────────────────────────────
# prctl(PR_SET_PDEATHSIG, SIGTERM) makes the kernel send SIGTERM to this
# process the moment its parent dies. Applied to ALL ranks (including rank 0)
# so that every DDP process and DataLoader worker cleans up when its parent exits.
_local_rank = int(os.environ.get('LOCAL_RANK', '0'))
def _set_pdeathsig():
    """Ask the kernel to send SIGTERM to this process when its parent dies.
    Prevents orphan GPU processes after Ctrl+C or parent crash.
    Safe: only affects the calling process, never other users' processes.
    """
    try:
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        libc.prctl.argtypes = [
            ctypes.c_int,   # option
            ctypes.c_ulong, # arg2
            ctypes.c_ulong, # arg3
            ctypes.c_ulong, # arg4
            ctypes.c_ulong, # arg5
        ]
        libc.prctl.restype = ctypes.c_int
        libc.prctl(1, 15, 0, 0, 0)  # PR_SET_PDEATHSIG=1, SIGTERM=15
    except Exception:
        pass


_set_pdeathsig()  # apply to every DDP rank process


def _dataloader_worker_init(worker_id):
    """Each DataLoader worker also dies when its parent DDP rank dies."""
    _set_pdeathsig()

# In DDP, all subprocesses share the same terminal. Silence non-rank-0 processes
# so that TQDM progress bar and print() output don't interleave.
_DEBUG_ALL_RANKS = os.environ.get('UNITRAJ_DEBUG_ALL_RANKS', '0') == '1'
if _local_rank != 0 and not _DEBUG_ALL_RANKS:
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
elif _local_rank != 0 and _DEBUG_ALL_RANKS:
    sys.stdout = open(f'/tmp/unitraj_rank{_local_rank}_stdout.log', 'w', buffering=1)
    sys.stderr = open(f'/tmp/unitraj_rank{_local_rank}_stderr.log', 'w', buffering=1)

import pytorch_lightning as pl
import torch

# Optional allocation budget for explicitly shared-GPU runs.
if os.environ.get("RECON_CUDA_MEMORY_FRACTION"):
    _fraction = float(os.environ["RECON_CUDA_MEMORY_FRACTION"])
    if not 0 < _fraction <= .5:
        raise ValueError("Shared-GPU allocator fraction must be in (0, .5]")
    torch.cuda.set_per_process_memory_fraction(_fraction, _local_rank)


def _cleanup_gpu():
    """Release CUDA context on normal exit (covers: training complete, sys.exit, atexit).
    Does NOT handle SIGKILL — that requires pkill after Ctrl+C if processes get stuck.
    """
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


atexit.register(_cleanup_gpu)

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    # Named RECON comparisons use this SAME Lightning path, with explicit YAML
    # settings and audit callbacks; there is no alternate training engine.
    recon_directory = None
    checkpoint_directory = Path(cfg.get('checkpoint_root', 'unitraj_ckpt')).expanduser().resolve() / cfg.exp_name
    resume_policy = cfg.get('resume_policy', 'auto')
    if resume_policy not in ('auto', 'fresh', 'paired_resume'):
        raise ValueError('Unknown resume policy')
    if resume_policy == 'fresh':
        if cfg.ckpt_path is not None:
            raise ValueError('fresh training cannot restore ckpt_path')
        if list(checkpoint_directory.glob('last*.ckpt')):
            raise FileExistsError('fresh training requires a new exp_name; found a resume checkpoint')
    if cfg.get('recon_profile', False):
        from tools.recon_training import validate_profile, prepare_model, audit_callback, reserve_directory
        recon_directory = validate_profile(cfg)
        if resume_policy == 'paired_resume':
            from tools.recon_recovery import validate_resume_directory
            validate_resume_directory(cfg, recon_directory)
        else:
            reserve_directory(recon_directory)

    # MTR is anchor-based: it needs per-type intention points (clustered GT endpoints)
    # before the decoder is built. If MOTION_DECODER.INTENTION_POINTS_FILE is 'auto'/null
    # or points at a missing file, cluster the training set now, save the pkl, persist the
    # path back into MTR.yaml, then continue. No-op once a concrete file exists.
    if cfg.method.model_name == 'MTR':
        from tools.gen_intention_points import ensure_intention_points
        # Cluster over an on-the-fly view (one-time pass, no h5 cache build) without
        # mutating the training cfg's storage mode. OmegaConf.merge returns a new cfg.
        ensure_intention_points(
            cfg, lambda c: build_dataset(OmegaConf.merge(c, {'onthefly': True}), val=False))

    # BeTopNet is MTR-family and anchor-based too: same intention points, but its decoder
    # node is DECODER (not MOTION_DECODER) and its yaml is BeTopNet.yaml. If
    # DECODER.INTENTION_POINTS_FILE is 'auto'/null/missing (e.g. after switching datasets),
    # cluster this training set's GT endpoints, save the pkl, persist the path into
    # BeTopNet.yaml, then continue. No-op once a concrete file exists.
    if cfg.method.model_name == 'BeTopNet':
        from tools.gen_intention_points import ensure_intention_points
        ensure_intention_points(
            cfg, lambda c: build_dataset(OmegaConf.merge(c, {'onthefly': True}), val=False),
            yaml_path='configs/method/BeTopNet.yaml', decoder_node='DECODER')

    # SMART is a discrete next-token model: its minADE is capped by the motion-token vocabulary.
    # If agent_token_path is 'auto'/null/missing, cluster the training set's 0.5s motion segments
    # now, save the vocab pkl, persist the path back into SMART.yaml, then continue. No-op once a
    # concrete file exists. (Waymo vocab underfits highD highway speeds — see tools/gen_agent_tokens.)
    if cfg.method.model_name == 'SMART':
        from tools.gen_agent_tokens import ensure_agent_tokens
        ensure_agent_tokens(
            cfg, lambda c: build_dataset(OmegaConf.merge(c, {'onthefly': True}), val=False))

    model = build_model(cfg)
    if resume_policy == 'paired_resume':
        from tools.recon_training import validate_structure
        if cfg.recon_arm == 'P':
            for parameter in model.confounder_gate.parameters():
                parameter.requires_grad_(False)
        validate_structure(model, cfg)

    # Weight-init only: load pretrained state_dict, do NOT restore optimizer/epoch.
    # Only applies when no last.ckpt exists for auto-resume (fresh training run).
    _is_rank0 = int(os.environ.get('RANK', os.environ.get('GLOBAL_RANK', '0'))) == 0

    pretrained_path = cfg.get('pretrained_path', None)
    if pretrained_path is not None:
        import glob as _glob
        ckpt_dir = str(checkpoint_directory)
        existing_last = _glob.glob(os.path.join(ckpt_dir, 'last*.ckpt'))
        if existing_last:
            if _is_rank0:
                print(f'[train] Resume checkpoint found ({existing_last[0]}) — ignoring pretrained_path, will auto-resume instead')
        else:
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)  # support both Lightning ckpt and raw state_dict
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if recon_directory is not None:
                if unexpected or (cfg.model_name == 'wayformer' and missing):
                    raise ValueError(f'Unexpected shared initialization keys: {missing}, {unexpected}')
                prepare_model(model, state_dict, cfg, recon_directory)
            if _is_rank0:
                print(f'[train] Loaded pretrained weights from {pretrained_path}')
                if missing:
                    print(f'[train]   Missing keys ({len(missing)}): {missing[:5]}{"..." if len(missing)>5 else ""}')
                if unexpected:
                    print(f'[train]   Unexpected keys ({len(unexpected)}): {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')

    if recon_directory is not None:
        # Model-specific parameter initialization must not change dataset sampling.
        set_seed(cfg.seed)
    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)
    if recon_directory is not None:
        train_set.onthefly_index.sort()
        val_set.onthefly_index.sort()
        if not cfg.get('recon_no_padding_validation', False) and any(len(ds) % len(cfg.devices) for ds in (train_set, val_set)):
            raise ValueError('RECON DDP requires split sizes divisible by world size to avoid sampler padding')

    # batch sizes are per-GPU; effective total = batch_size × num_gpus (handled by DDP automatically)
    train_batch_size = cfg.method['train_batch_size']
    eval_batch_size = cfg.method['eval_batch_size']

    call_backs = []
    if recon_directory is not None:
        call_backs.append(audit_callback(cfg, recon_directory, train_set, val_set))

    # 监控指标可由 method 覆盖:多模态模型用 val/brier_fde;MAE 单模态无 brier,用 val/minADE。
    monitor_metric = cfg.method.get('monitor_metric', 'val/brier_fde')
    if cfg.get('recon_early_stopping'):
        from pytorch_lightning.callbacks import EarlyStopping
        rule = cfg.recon_early_stopping
        call_backs.append(EarlyStopping(
            monitor=rule.monitor, mode='min', patience=int(rule.patience),
            min_delta=float(rule.min_delta), strict=True, check_finite=True,
            check_on_train_epoch_end=False, verbose=True, log_rank_zero_only=True))
    checkpoint_type = ModelCheckpoint
    if cfg.get('checkpoint_last_each_epoch', False):
        from tools.epoch_last_checkpoint import EpochLastCheckpoint
        checkpoint_type = EpochLastCheckpoint
    checkpoint_callback = checkpoint_type(
        monitor=monitor_metric,
        filename=('epoch={epoch:03d}' if cfg.get('checkpoint_keep_each_epoch', False)
                  else '{epoch}-{' + monitor_metric + ':.2f}'),
        auto_insert_metric_name=not cfg.get('checkpoint_keep_each_epoch', False),
        save_top_k=-1 if cfg.get('checkpoint_keep_each_epoch', False) else 1,
        save_last=True,   # RECON's EpochLastCheckpoint also saves on non-improving epochs.
        mode='min',
        dirpath=str(checkpoint_directory),
        enable_version_counter=False,  # always overwrite last.ckpt directly, never create last-v1.ckpt
    )
    call_backs.append(checkpoint_callback)

    progress_bar = TQDMProgressBar(refresh_rate=1)  # update every step so we can see progress immediately
    call_backs.append(progress_bar)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        shuffle=True,  # 每个 epoch 前打乱训练集:单卡下让 limit_train_batches<1 取「随机一半」而非固定前一半;
                       # 多卡 DDP 下 Lightning 会据此把 sampler 包成 DistributedSampler(shuffle=True),行为一致
        collate_fn=train_set.collate_fn, worker_init_fn=_dataloader_worker_init,
        **({'generator': torch.Generator().manual_seed(cfg.seed)} if recon_directory is not None else {}),
        persistent_workers=(cfg.load_num_workers > 0))  # keep workers alive between epochs; avoids re-fork hang

    val_sampler = None
    if recon_directory is not None and cfg.get('recon_no_padding_validation', False):
        from tools.recon_sampling import NoPaddingDistributedSampler, batch_count
        val_sampler = NoPaddingDistributedSampler(val_set, len(cfg.devices), int(os.environ.get('RANK', 0)), eval_batch_size)
        # The consumed prefix must end before DistributedSampler's padded tail.
        consumed = batch_count(len(train_set), len(cfg.devices), train_batch_size, cfg.limit_train_batches) * train_batch_size * len(cfg.devices)
        if consumed > len(train_set):
            raise ValueError('Training prefix would consume padded duplicate scenes')

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        sampler=val_sampler,
        collate_fn=val_set.collate_fn if recon_directory is not None else train_set.collate_fn,
        worker_init_fn=_dataloader_worker_init,
        **({'generator': torch.Generator().manual_seed(cfg.seed + 1)} if recon_directory is not None else {}),
        persistent_workers=(cfg.load_num_workers > 0))

    csv_logger = CSVLogger(
        save_dir=str(checkpoint_directory),
        name='metrics',
        version=''
    )
    loggers = [csv_logger]
    if cfg.get('use_wandb', True):
        loggers.append(WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name))

    _limit_train = cfg.get('limit_train_batches', 1.0)
    _limit_val = cfg.get('limit_val_batches', 1.0)
    _precision = cfg.get('precision', '32-true')
    # DDP strategy 可由 method 覆盖:多数模型用默认 'ddp';SMART 在 train_map=False 时
    # map 预测头不进 loss(unused params),即便单卡也走 DDP,需 'ddp_find_unused_parameters_true'。
    _strategy = cfg.method.get('ddp_strategy', 'ddp')

    trainer = pl.Trainer(
        max_epochs=(int(cfg.recon_recovery.boundary_epochs)
                    if cfg.get('recon_recovery') and cfg.recon_recovery.segment == 'prefix'
                    else cfg.method.max_epochs),
        logger=loggers if not cfg.debug else None,
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        # accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
        accelerator="cpu" if cfg.debug else "gpu",
        strategy="auto" if cfg.debug else _strategy,
        precision=_precision,  # set precision='16-mixed' in hydra to use fp16 AMP
        callbacks=call_backs,
        log_every_n_steps=50,  # log metrics every 50 steps to reduce noise
        num_sanity_val_steps=0,  # skip sanity check; avoids DDP barrier issue in transition to training
        limit_train_batches=_limit_train,
        limit_val_batches=_limit_val,
    )

    # rank-0 echo of the key Trainer settings (some values are silently consumed
    # by Lightning and don't show up in its own startup logs — print them here)
    if _is_rank0:
        print('━' * 72)
        print(f'[train] exp_name              = {cfg.exp_name}')
        print(f'[train] checkpoint_directory  = {checkpoint_directory}')
        print(f'[train] method                = {cfg.method.model_name}')
        print(f'[train] devices               = {cfg.devices}   ({len(cfg.devices)} GPU DDP)')
        print(f'[train] max_epochs            = {cfg.method.max_epochs}')
        print(f'[train] train_batch_size      = {cfg.method.train_batch_size}  (per-GPU, effective = ×{len(cfg.devices)})')
        print(f'[train] max_agents_per_batch  = {cfg.get("max_agents_per_batch", "null")}')
        print(f'[train] max_data_num          = {cfg.max_data_num}')
        print(f'[train] max_val_data_num      = {cfg.get("max_val_data_num", "(not set)")}')
        print(f'[train] limit_train_batches   = {_limit_train}')
        print(f'[train] limit_val_batches     = {_limit_val}')
        print(f'[train] precision             = {_precision}')
        print(f'[train] joint_train           = {cfg.joint_train}')
        print(f'[train] prediction_targets    = {cfg.prediction_targets}')
        print(f'[train] object_type           = {list(cfg.object_type)}')
        if cfg.method.model_name == 'SMART':
            print(f'[train] smart_fast_onthefly  = {cfg.method.get("smart_fast_onthefly", cfg.get("smart_fast_onthefly", True))}')
        print(f'[train] ckpt_path             = {cfg.ckpt_path}')
        print(f'[train] pretrained_path       = {cfg.pretrained_path}')
        print('━' * 72)

    # automatically resume training from the latest last*.ckpt if ckpt_path is not specified.
    # When Lightning crashes mid-save, the newer checkpoint is left as last-v{n}.ckpt while
    # last.ckpt still points to the previous epoch.  Pick the file with the highest epoch.
    if resume_policy == 'auto' and cfg.ckpt_path is None and not cfg.debug:
        import glob
        ckpt_dir = str(checkpoint_directory)
        candidates = glob.glob(os.path.join(ckpt_dir, 'last*.ckpt'))
        best_ckpt, best_epoch = None, -1
        for c in candidates:
            try:
                meta = torch.load(c, map_location='cpu')
                ep = meta.get('epoch', -1)
                if ep > best_epoch:
                    best_epoch, best_ckpt = ep, c
            except Exception:
                pass
        if best_ckpt is not None:
            cfg.ckpt_path = best_ckpt
            if _is_rank0:
                print(f'[train] Auto-resuming from {best_ckpt} (epoch={best_epoch})')

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()
