"""Configuration guards and audit callbacks for the existing Lightning trainer.

No optimizer, training loop, scheduler, or dataset implementation lives here.
"""
import hashlib
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf
from utils.model_names import canonical_model_key

U = Path(__file__).resolve().parents[1]
ABLATION_ARMS = ('NoScoreBranch', 'NoAuxGeo', 'NoProto')


def rank():
    return int(os.environ.get('RANK', '0'))


def rank_directory(directory):
    return directory if rank() == 0 else directory / 'ranks' / f'rank{rank()}'


def reserve_directory(directory):
    if rank() == 0:
        directory.mkdir(parents=True, exist_ok=False)
    else:
        deadline = time.monotonic() + 45
        while not directory.is_dir():
            if time.monotonic() > deadline:
                raise TimeoutError('Rank zero did not reserve output directory')
            time.sleep(.1)
        rank_directory(directory).mkdir(parents=True, exist_ok=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def validate_profile(cfg):
    """Fail before training if the named profile is incomplete or ambiguous."""
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not cfg.get('recon_run') or not cfg.get('recon_profile'):
        raise ValueError('Paired fine-tuning requires an explicit training/ profile and RECON arm; scratch training uses config.yaml')
    if cfg.recon_run.phase not in ('engineering', 'formal'):
        raise ValueError('Unknown RECON phase')
    world = len(cfg.devices)
    if cfg.debug or list(cfg.devices) != list(range(world)) or world not in (1, 2, 6):
        raise ValueError('Use one, two or six contiguous logical GPUs')
    if (world == 1 and cfg.ddp_strategy != 'auto') or (world > 1 and cfg.ddp_strategy != 'ddp_find_unused_parameters_true'):
        raise ValueError('Use auto for single GPU, ddp_find_unused_parameters_true for multiple GPUs')
    if world > 1 and (int(os.environ.get('WORLD_SIZE', '0')) != world or 'RANK' not in os.environ):
        raise ValueError('Launch DDP via torchrun --standalone --nproc_per_node=WORLD_SIZE train.py')
    recovering = cfg.get('recon_recovery') and cfg.recon_recovery.segment == 'continuation'
    if recovering:
        assert cfg.resume_policy == 'paired_resume' and Path(cfg.ckpt_path).is_file()
    elif cfg.resume_policy != 'fresh' or cfg.ckpt_path is not None:
        raise ValueError('RECON paired runs require fresh weight-only initialization, no resume')
    if not cfg.pretrained_path or not Path(cfg.pretrained_path).is_file():
        raise ValueError('Missing shared pretrained checkpoint')
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Set PYTHONHASHSEED=0 before starting Python for stable target ordering')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Set CUDA_VISIBLE_DEVICES before Python imports; devices=[0] is logical')
    cosine = cfg.scheduler == 'cosine_step'
    if cfg.max_epochs < 1 or (not cosine and cfg.onecycle_epochs < cfg.max_epochs):
        raise ValueError('Epoch budget must be positive and fit within the shared OneCycle schedule')
    if cfg.get('recon_early_stopping'):
        assert cfg.recon_early_stopping.patience >= 1 and cfg.recon_early_stopping.min_delta >= 0
        assert cfg.recon_early_stopping.monitor == cfg.monitor_metric
        assert cfg.checkpoint_keep_each_epoch
    if cfg.recon_run.get('full_pool_protocol') and cfg.recon_run.phase == 'formal':
        if cosine:
            assert cfg.max_epochs == 30 and not cfg.get('recon_early_stopping')
            assert cfg.learning_rate == 2e-5 and cfg.cosine_min_lr == 1e-6
            assert cfg.checkpoint_keep_each_epoch and not cfg.get('recon_recovery')
        else:
            assert cfg.max_epochs == cfg.onecycle_epochs == (30 if cfg.get('recon_early_stopping') else 100)
        if cfg.get('recon_early_stopping'):
            assert cfg.recon_early_stopping.patience == 5 and cfg.recon_early_stopping.min_delta == 0.
            assert cfg.recon_early_stopping.monitor == cfg.monitor_metric == 'val/brier_fde'
            assert cfg.checkpoint_keep_each_epoch
        assert list(cfg.max_data_num) == list(cfg.max_val_data_num) == [None]
        assert cfg.limit_train_batches == .3 and cfg.limit_val_batches == 1.0
        assert cfg.recon_no_padding_validation and cfg.recon_arm in ('P', 'F') + ABLATION_ARMS
    if cfg.optimizer != 'AdamW' or cfg.scheduler not in ('onecycle', 'cosine_step'):
        raise ValueError('Expected AdamW with declared OneCycle or step-based cosine schedule')
    if cosine:
        assert 0 < cfg.cosine_min_lr < cfg.learning_rate
        assert not cfg.get('recon_early_stopping') and cfg.recon_arm in ('P', 'F') + ABLATION_ARMS
    if cfg.score_head_only or cfg.clean_norm_v2 or cfg.rank_loss_weight != 0:
        raise ValueError('Probe/ranking-distillation settings are outside this model comparison')
    if cfg.prediction_targets != 'official' or cfg.joint_train or cfg.accident_ttc_eval:
        raise ValueError('The training comparison uses normal Waymo official targets only')
    if len(cfg.train_data_path) != 1 or len(cfg.val_data_path) != 1:
        raise ValueError('Exactly one normal training and validation split is required')
    train, val = Path(cfg.train_data_path[0]).resolve(), Path(cfg.val_data_path[0]).resolve()
    if train.name != 'training' or val.name != 'validation' or train.parent != val.parent:
        raise ValueError('Expected distinct training/validation directories of the same Waymo dataset')
    if not cfg.onthefly or cfg.use_cache or not cfg.strict_eval_support:
        raise ValueError('Use the shared on-the-fly loader and strict validation support')
    arm = cfg.recon_arm
    expected = {'B': ('wayformer', 'dictionary', 0., 0.),
                'C': ('recon', 'mlp', 0., 0.),
                'P': ('recon', 'dictionary', 0., .0005),
                'F': ('recon', 'dictionary', .5, .0005),
                'NoScoreBranch': ('recon', 'none', 0., 0.),
                'NoAuxGeo': ('recon', 'dictionary', .5, .0005),
                'NoProto': ('recon', 'none', 0., 0.)}
    actual = (canonical_model_key(cfg.model_name), cfg.prototype_conditioning,
              cfg.confounder_subtract_scale, cfg.prototype_entropy_weight)
    if arm not in expected or actual != expected[arm]:
        raise ValueError(f'Variant definition mismatch for {arm}: {actual}')
    if arm in ABLATION_ARMS:
        assert cfg.recon_run.get('ablation_protocol') and cosine
        assert cfg.train_batch_size == cfg.eval_batch_size == 4
        assert cfg.load_num_workers == 8 and cfg.max_agents_per_batch == 64
        assert cfg.optimizer_eps == 1e-4 and cfg.weight_decay == .01 and cfg.grad_clip_norm == 5
    return Path(cfg.get('checkpoint_root', 'unitraj_ckpt')).expanduser().resolve() / cfg.exp_name


def validate_structure(model, cfg):
    """Fail before training if the registered network differs from the locked arm."""
    import inspect
    import torch
    arm = cfg.recon_arm
    if arm in ABLATION_ARMS:
        from tools.recon_ablation import validate_ablation_structure
        return validate_ablation_structure(model, cfg)
    if arm == 'B':
        assert type(model).__name__ == 'Wayformer', 'B must use the original Wayformer'
        return {'arm': arm, 'class': type(model).__name__}
    assert arm in ('C', 'P', 'F')
    assert type(model).__name__ == 'RECON'
    assert Path(inspect.getfile(type(model))).resolve() == (U / 'models/recon/recon.py').resolve()
    expected = dict(causal_v1_enabled=False, causal_v2_enabled=True, clean_norm_v2=False,
                    score_head_only=False, rank_loss_weight=0., cls_loss_weight=1.,
                    best6_ade_weight=.2, best6_fde_weight=.1, graph_sparsity_weight=.001,
                    causal_logit_residual_weight=1., kinematic_filter=False,
                    num_modes=6, hidden_size=256, past_len=11, future_len=80,
                    num_queries_enc=192, num_queries_dec=64,
                    num_encoder_layers=2, num_decoder_layers=8)
    assert all(cfg[k] == v for k, v in expected.items()), 'Unexpected backbone/head/loss configuration'
    assert model.causal_v2 and not model.causal_v1 and not model.clean_norm_v2
    assert model.c == 6 and model.d_k == 256 and model.T == 80
    assert model.mode_intervention[0].in_features == 1030
    assert model.deconfounded_prob_predictor[0].in_features == 1286
    assert not hasattr(model, 'deconf_prob_v2')
    conditioning, scale, entropy = {'C': ('mlp', 0., 0.), 'P': ('dictionary', 0., .0005),
                                    'F': ('dictionary', .5, .0005)}[arm]
    assert model.prototype_conditioning == cfg.prototype_conditioning == conditioning
    assert cfg.confounder_subtract_scale == scale and cfg.prototype_entropy_weight == entropy
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    for name, parameter in parameters.items():
        assert parameter.requires_grad == (arm == 'F' or not name.startswith('confounder_gate.')), name
    count = sum(p.numel() for p in parameters.values() if p.requires_grad)
    assert count == (17448338 if arm == 'F' else 17118610), 'Unexpected effective parameter count'
    dictionary = None
    if arm == 'C':
        assert 'confounder_prototypes' not in parameters and 'confounder_prototypes' not in buffers
    else:
        assert 'confounder_prototypes' not in parameters
        prototypes = buffers['confounder_prototypes']
        assert prototypes.shape == (64, 256) and not prototypes.requires_grad
        assert 'confounder_prototypes' in model.state_dict(), 'Dictionary must persist in checkpoints'
        assert torch.allclose(prototypes.norm(dim=-1), torch.ones(64, device=prototypes.device), atol=1e-5)
        path = model._resolve_unitraj_path(cfg.confounder_init_path).resolve()
        assert path == (U / 'models/wayformer/prototypes_large_gpu/normal_mode_prototypes_k64.pt').resolve()
        dictionary = {'path': str(path), 'file_sha256': sha(path), 'shape': [64, 256], 'frozen_buffer': True}
    return {'arm': arm, 'class': type(model).__name__, 'conditioning': conditioning,
            'explicit_subtraction_scale': scale, 'prototype_entropy_weight': entropy,
            'trainable_parameters': count, 'gate_trainable': arm == 'F', 'dictionary': dictionary,
            'backbone_and_loss_checks': expected}


def prepare_model(model, state, cfg, directory):
    import torch
    loaded = model.state_dict()
    if any(k not in loaded or not torch.equal(v, loaded[k]) for k, v in state.items()):
        raise ValueError('Shared pretrained tensors were not copied exactly')
    if cfg.recon_arm in ('C', 'P'):
        for parameter in model.confounder_gate.parameters():
            parameter.requires_grad_(False)
    audit = {'arm': cfg.recon_arm, 'initial_checkpoint': cfg.pretrained_path,
             'initial_sha256': sha(cfg.pretrained_path),
             'tensor_sha256': {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                              for k, v in loaded.items()},
             'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
             'structure': validate_structure(model, cfg),
             'frozen_parameters': [k for k, p in model.named_parameters() if not p.requires_grad]}
    write_json(rank_directory(directory) / 'initialization.json', audit)


def audit_callback(cfg, directory, train_set, val_set):
    import pytorch_lightning as pl
    import torch
    directory = rank_directory(directory)

    sources = [U / 'train.py', Path(__file__), U / 'tools/epoch_last_checkpoint.py', U / 'configs/config.yaml',
               U / 'utils/model_names.py', U / 'models/recon/recon.py', U / 'models/recon_lp/recon_lp.py',
               U / 'models/__init__.py',
               U / 'models/base_model/base_model.py', U / 'models/wayformer/wayformer.py',
               U / 'models/wayformer/wayformer_utils.py',
               U / 'datasets/__init__.py', U / 'datasets/base_dataset.py', U / 'datasets/wayformer_dataset.py',
               U / 'datasets/evaluation_support.py', U / 'configs/method/wayformer.yaml',
               U / 'configs/method/RECON.yaml']
    sources += sorted((U / 'configs/method').glob('RECON*.yaml'))
    # Current recipes only; historical manifests retain their original source hashes.
    sources += sorted((U / 'configs/training').glob('*.yaml'))
    if cfg.recon_run.get('ablation_protocol'):
        sources += [U / 'tools/recon_ablation.py']
    sources += [U / 'tools/recon_sampling.py', U / 'tools/recon_stopping.py']
    if cfg.scheduler == 'cosine_step':
        sources += [U / 'tools/verify_recon_cosine.py']
    if cfg.get('recon_recovery'):
        sources += [U / 'tools/recon_recovery.py']
    index = {'training': train_set.onthefly_index, 'validation': val_set.onthefly_index}
    write_json(directory / 'data_index.json', index)
    OmegaConf.save(cfg, directory / 'resolved_config.yaml', resolve=True)
    manifest = {'status': 'prepared', 'engine': 'train.py / pytorch_lightning.Trainer.fit',
                'rank': rank(), 'world_size': len(cfg.devices),
                'phase': cfg.recon_run.phase, 'arm': cfg.recon_arm, 'seed': cfg.seed,
                'source_sha256': {str(p): sha(p) for p in sources},
                'data_index_sha256': sha(directory / 'data_index.json'),
                'config_sha256': sha(directory / 'resolved_config.yaml'),
                'initial_sha256': sha(cfg.pretrained_path),
                'selection': cfg.monitor_metric, 'resume_policy': cfg.resume_policy,
                'batch_unit': 'scene; target count recorded for each actual batch',
                'cuda_memory_fraction': os.environ.get('RECON_CUDA_MEMORY_FRACTION'),
                'cuda_allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                'scheduler_interval': 'step' if cfg.scheduler == 'cosine_step' else 'epoch (existing model.configure_optimizers)'}
    if cfg.get('recon_recovery'):
        manifest['recovery_segment'] = cfg.recon_recovery.segment
        manifest['restart_boundary_epochs'] = int(cfg.recon_recovery.boundary_epochs)
        prefix = directory / 'recovery_prefix.json'
        if prefix.exists():
            manifest['prefix_provenance'] = json.loads(prefix.read_text())
            manifest['timing_start_epoch'] = int(cfg.recon_recovery.boundary_epochs)
            manifest['fit_seconds_scope'] = 'current continuation only; prefix retained separately'
    write_json(directory / 'training_manifest.json', manifest)

    class Audit(pl.Callback):
        def on_fit_start(self, trainer, pl_module):
            self.started = time.perf_counter()
            self.step_times = []
            manifest.update(status='running', optimizer=type(trainer.optimizers[0]).__name__,
                            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                            gpu_name=torch.cuda.get_device_name(pl_module.device),
                            device=str(pl_module.device),
                            schedulers=[{'name': type(s.scheduler).__name__, 'interval': s.interval}
                                        for s in trainer.lr_scheduler_configs])
            write_json(directory / 'training_manifest.json', manifest)

        def on_train_start(self, trainer, pl_module):
            if cfg.resume_policy == 'paired_resume':
                from tools.recon_recovery import verify_restored_state
                verify_restored_state(trainer, directory, cfg)
            manifest['train_sampler'] = type(trainer.train_dataloader.sampler).__name__
            write_json(directory / 'training_manifest.json', manifest)

        def on_train_epoch_start(self, trainer, pl_module):
            if not cfg.get('recon_no_padding_validation', False):
                return
            import itertools
            import numpy as np
            sampler = trainer.train_dataloader.sampler
            sampler.set_epoch(trainer.current_epoch)
            count = int(trainer.num_training_batches) * cfg.train_batch_size
            selected = list(itertools.islice(iter(sampler), count))
            assert len(selected) == len(set(selected)) == count
            row = dict(epoch=trainer.current_epoch, pool_scenes=len(train_set),
                       rank=trainer.global_rank, scenes=count, steps=int(trainer.num_training_batches),
                       sampler_seed=sampler.seed, sampler_epoch=sampler.epoch,
                       selected_sha256=hashlib.sha256(np.asarray(selected, dtype='<i8').tobytes()).hexdigest())
            with (directory / 'sampling_epochs.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')

        def record(self, trainer, batch, batch_idx, split):
            fields = batch['input_dict']
            keys = [[str(s), str(o)] for s, o in zip(fields['scenario_id'], fields['center_objects_id'])]
            digest = hashlib.sha256()
            audit_tensor = cfg.recon_run.audit_inputs and batch_idx % int(cfg.recon_run.get('audit_tensor_every', 1)) == 0
            if audit_tensor:
                for name, value in sorted(fields.items()):
                    if torch.is_tensor(value):
                        digest.update(name.encode())
                        digest.update(str((value.dtype, tuple(value.shape))).encode())
                        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            row = {'epoch': trainer.current_epoch, 'step': trainer.global_step, 'batch_idx': batch_idx,
                   'targets': len(keys), 'keys': keys,
                   'input_tensor_sha256': digest.hexdigest() if audit_tensor else None,
                   'lr': trainer.optimizers[0].param_groups[0]['lr']}
            with (directory / f'{split}_batches.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.step_started = time.perf_counter()
            self.record(trainer, batch, batch_idx, 'train')

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            torch.cuda.synchronize(pl_module.device)
            self.step_times.append(time.perf_counter() - self.step_started)

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            finite = [torch.isfinite(p.grad).all() for p in pl_module.parameters() if p.grad is not None]
            if finite and not torch.stack(finite).all():
                raise FloatingPointError('Nonfinite training gradient')

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                       if torch.is_tensor(v) and v.numel() == 1}
            with (directory / 'epoch_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps({'epoch': trainer.current_epoch, 'step': trainer.global_step,
                                         'metrics': metrics}) + '\n')

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.record(trainer, batch, batch_idx, 'validation')

        def on_fit_end(self, trainer, pl_module):
            import re
            from tools.recon_stopping import validate_completion
            epochs = [json.loads(line) for line in (directory / 'epoch_metrics.jsonl').read_text().splitlines()]
            pausing = cfg.get('recon_recovery') and cfg.recon_recovery.segment == 'prefix'
            if pausing:
                assert len(epochs) == cfg.recon_recovery.boundary_epochs
                manifest.update(completed_epochs=len(epochs), stop_reason='matched_restart_boundary')
            else:
                manifest.update(validate_completion(cfg, epochs))
            if cfg.scheduler == 'cosine_step':
                from tools.verify_recon_cosine import verify_cosine
                manifest['cosine_verification'] = verify_cosine(directory, trainer)
            if cfg.get('checkpoint_keep_each_epoch', False):
                manifest['epoch_checkpoints'] = {
                    str(int(re.search(r'epoch=(\d+)', Path(path).name).group(1))): path
                    for path in trainer.checkpoint_callback.best_k_models}
                assert len(manifest['epoch_checkpoints']) == len(epochs)
            manifest.update(status='paused' if pausing else 'complete', global_step=trainer.global_step,
                            fit_seconds=time.perf_counter() - self.started,
                            train_step_seconds=self.step_times,
                            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                            best_checkpoint=trainer.checkpoint_callback.best_model_path,
                            last_checkpoint=trainer.checkpoint_callback.last_model_path)
            write_json(directory / 'training_manifest.json', manifest)
            trainer.strategy.barrier('recon_audit_finished')

        def on_exception(self, trainer, pl_module, exception):
            manifest.update(status='failed', reason=str(exception), global_step=trainer.global_step)
            write_json(directory / 'training_manifest.json', manifest)

    return Audit()
