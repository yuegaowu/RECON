"""Retain a completed prefix and resume Lightning state at a matched boundary.

The old run is immutable. No optimizer or training loop is implemented here.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

U = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(U), str(U.parent)]
from omegaconf import OmegaConf
from tools.recon_training import sha, write_json, rank_directory


def read(path):
    return json.loads(Path(path).read_text())


def state_summary(checkpoint):
    early = [v for k, v in checkpoint['callbacks'].items() if k.startswith('EarlyStopping')]
    assert len(early) == 1
    return {'epoch_human': checkpoint['epoch'] + 1, 'global_step': checkpoint['global_step'],
            'optimizer_lr': checkpoint['optimizer_states'][0]['param_groups'][0]['lr'],
            'optimizer_state_entries': len(checkpoint['optimizer_states'][0]['state']),
            'scheduler_last_epoch': checkpoint['lr_schedulers'][0]['last_epoch'],
            'early_wait_count': early[0]['wait_count'], 'early_best_score': float(early[0]['best_score'])}


def remap_callbacks(checkpoint, old, new):
    def remap(value):
        if isinstance(value, str) and (value == str(old) or value.startswith(str(old) + '/')):
            return str(new) + value[len(str(old)):]
        if isinstance(value, dict):
            return {remap(k): remap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [remap(v) for v in value]
        return value
    checkpoint['callbacks'] = remap(checkpoint['callbacks'])
    return checkpoint


def retain_prefix(source, destination, boundary):
    """Copy only completed epochs; leave the interrupted tail in its original run."""
    import torch
    source, destination = Path(source).resolve(), Path(destination).resolve()
    assert source != destination and not destination.exists()
    cfg = OmegaConf.load(source / 'resolved_config.yaml')
    checkpoint_path = source / f'epoch={boundary-1:03d}.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == boundary - 1
    expected_steps = checkpoint['global_step']
    destination.mkdir(parents=True)
    for epoch in range(boundary):
        original = source / f'epoch={epoch:03d}.ckpt'
        assert original.is_file()
        shutil.copy2(original, destination / original.name)
    shutil.copy2(checkpoint_path, destination / 'last.ckpt')
    for rank in range(len(cfg.devices)):
        suffix = Path('.') if rank == 0 else Path('ranks') / f'rank{rank}'
        old, new = source / suffix, destination / suffix
        new.mkdir(parents=True, exist_ok=True)
        for name in ('initialization.json', 'data_index.json'):
            shutil.copy2(old / name, new / name)
        for name in ('training_manifest.json', 'resolved_config.yaml'):
            shutil.copy2(old / name, new / ('prefix_' + name))
        for name in ('train_batches.jsonl', 'validation_batches.jsonl', 'epoch_metrics.jsonl', 'sampling_epochs.jsonl'):
            count = 0
            with (old / name).open() as inp, (new / name).open('x') as out:
                for line in inp:
                    row = json.loads(line)
                    if row['epoch'] < boundary:
                        out.write(line); count += 1
            if name == 'train_batches.jsonl':
                assert count == expected_steps
            if name in ('epoch_metrics.jsonl', 'sampling_epochs.jsonl'):
                assert count == boundary
        write_json(new / 'recovery_prefix.json', {
            'source_directory': str(old), 'boundary_epochs': boundary,
            'original_checkpoint': str(checkpoint_path), 'original_checkpoint_sha256': sha(checkpoint_path),
            'source_manifest_sha256': sha(old / 'training_manifest.json'),
            'source_code_sha256': read(old / 'training_manifest.json')['source_sha256'],
            'retained_files_sha256': {p.name: sha(p) for p in new.glob('*.jsonl')}})
    prepare_continuation(destination, boundary, checkpoint_path, source)


def prepare_continuation(directory, boundary, source_checkpoint=None, source_directory=None):
    import torch
    directory = Path(directory).resolve()
    checkpoint_path = Path(source_checkpoint) if source_checkpoint else directory / 'last.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == boundary - 1
    old = Path(source_directory) if source_directory else directory
    if old == directory:
        cfg = OmegaConf.load(directory / 'resolved_config.yaml')
        for rank in range(len(cfg.devices)):
            d = directory if rank == 0 else directory / 'ranks' / f'rank{rank}'
            manifest = read(d / 'training_manifest.json')
            assert manifest['status'] == 'paused' and manifest['completed_epochs'] == boundary
            assert manifest['global_step'] == checkpoint['global_step']
            for name in ('training_manifest.json', 'resolved_config.yaml'):
                shutil.copy2(d / name, d / ('prefix_' + name))
            write_json(d / 'recovery_prefix.json', {
                'source_directory': str(d), 'boundary_epochs': boundary,
                'source_manifest_sha256': sha(d / 'prefix_training_manifest.json'),
                'source_code_sha256': manifest['source_sha256'],
                'retained_files_sha256': {p.name: sha(p) for p in d.glob('*.jsonl')}})
    before = state_summary(checkpoint)
    remap_callbacks(checkpoint, old, directory)
    assert state_summary(checkpoint) == before
    output = directory / 'resume_boundary.ckpt'
    assert not output.exists()
    torch.save(checkpoint, output)
    write_json(directory / 'resume_expected.json', {
        'checkpoint': str(output), 'sha256': sha(output), **before,
        'note': 'Only callback output paths remapped; model/optimizer/scheduler/early-stop/loop states retained. RNG is restarted identically for both arms at this boundary.'})


def validate_resume_directory(cfg, directory):
    expected = read(directory / 'resume_expected.json')
    assert str(Path(cfg.ckpt_path).resolve()) == expected['checkpoint']
    assert sha(cfg.ckpt_path) == expected['sha256']
    assert expected['epoch_human'] == cfg.recon_recovery.boundary_epochs
    local = rank_directory(directory)
    prefix = read(local / 'recovery_prefix.json')
    assert prefix['boundary_epochs'] == cfg.recon_recovery.boundary_epochs
    for name, digest in prefix['retained_files_sha256'].items():
        assert sha(local / name) == digest, ('Changed retained prefix', name)


def verify_restored_state(trainer, directory, cfg):
    from pytorch_lightning.callbacks import EarlyStopping
    expected = read(Path(cfg.ckpt_path).parent / 'resume_expected.json')
    early = next(c for c in trainer.callbacks if isinstance(c, EarlyStopping))
    optimizer = trainer.optimizers[0]
    actual = {'epoch_human': trainer.current_epoch,
              'global_step': trainer.global_step,
              'optimizer_lr': optimizer.param_groups[0]['lr'],
              'optimizer_state_entries': len(optimizer.state),
              'scheduler_last_epoch': trainer.lr_scheduler_configs[0].scheduler.last_epoch,
              'early_wait_count': early.wait_count, 'early_best_score': float(early.best_score)}
    assert all(actual[k] == expected[k] for k in actual), (actual, expected)
    write_json(directory / 'resume_verification.json', {'status': 'passed', **actual})


def continuation_command(command, directory):
    return command + ['resume_policy=paired_resume', 'recon_recovery.segment=continuation',
                      f'ckpt_path={directory}/resume_boundary.ckpt']


def prepare_queue(root, parent):
    """Extend a newly prepared queue; never edit the parent protocol or evidence."""
    from hydra import compose, initialize_config_dir
    root, parent = root.resolve(), parent.resolve()
    q, p, old = read(root/'queue.json'), read(root/'protocol.json'), read(parent/'queue.json')
    parent_protocol = read(parent/'protocol.json')
    assert q['status'] == 'prepared' and not p['enforce_gpu_free_memory_floor']
    old_p = next(j for j in old['jobs'] if j['arm'] == 'P')
    # Actual model, loss, dataset and method YAML are the original experiment.
    for path, digest in parent_protocol['source_sha256'].items():
        f = Path(path)
        if any(folder in f.relative_to(U).parts for folder in ('models', 'datasets', 'utils')) or f.parent == U/'configs/method':
            assert sha(f) == digest, ('Changed training algorithm', f)
    with initialize_config_dir(version_base=None, config_dir=str(U/'configs')):
        cfg = compose(config_name=p['config_name'], overrides=q['jobs'][0]['overrides'])
    boundary = int(cfg.recon_recovery.boundary_epochs)
    before = read(Path(old_p['directory'])/'initialization.json')
    after = read(root/'initialization_preflight.json')['P']
    assert all(before[k] == after[k] for k in after)
    plans = {}
    for job in q['jobs']:
        directory = Path(job['directory'])
        command = job['command']
        plan = {'arm': job['arm'], 'boundary_epochs': boundary, 'directory': str(directory),
                'prefix_command': command if job['arm'] == 'F' else None,
                'continuation_command': continuation_command(command, directory)}
        if job['arm'] == 'P':
            retain_prefix(old_p['directory'], directory, boundary)
        plan_path = root/f"{job['arm']}.recovery_plan.json"
        write_json(plan_path, plan)
        plans[job['arm']] = {'path': str(plan_path), 'sha256': sha(plan_path)}
        job['command'] = [sys.executable, '-u', 'tools/recon_recovery.py', '--run-arm', str(plan_path), '--root', str(root)]
        with initialize_config_dir(version_base=None,config_dir=str(U/'configs')):
            overrides=job['overrides'] + ['resume_policy=paired_resume','recon_recovery.segment=continuation',f'ckpt_path={directory}/resume_boundary.ckpt']
            resolved=compose(config_name=p['config_name'],overrides=overrides)
        OmegaConf.set_struct(resolved,False)
        resolved=OmegaConf.merge(resolved,resolved.method)
        OmegaConf.save(resolved,root/f"{job['arm']}.continuation.planned.yaml",resolve=True)
    p['recovery'] = {'parent_root': str(parent), 'parent_protocol_sha256': sha(parent/'protocol.json'),
                     'boundary_epochs': boundary, 'plans': plans,
                     'parent_P_directory': old_p['directory'],
                     'policy': 'P retains completed prefix; F pauses/restarts at the same epoch. Same seeds and fresh loaders after the boundary; not bitwise equivalence to an uninterrupted run.'}
    p['stop_policy'] = 'Validation patience/max epochs or real training error/nonfinite/disk exhaustion; free GPU memory telemetry never stops or blocks this queue'
    write_json(root/'protocol.json',p);q['protocol_sha256']=sha(root/'protocol.json')
    write_json(root/'queue.json',q)
    print('RECOVERY PREPARED',root,flush=True)


def run_arm(plan_path, root):
    plan, protocol = read(plan_path), read(root/'protocol.json')
    assert sha(plan_path) == protocol['recovery']['plans'][plan['arm']]['sha256']
    directory = Path(plan['directory'])
    if plan['prefix_command']:
        subprocess.run(plan['prefix_command'], cwd=U, check=True)
        prepare_continuation(directory, plan['boundary_epochs'])
    subprocess.run(plan['continuation_command'], cwd=U, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--run-arm', type=Path)
    args = parser.parse_args()
    if args.run_arm:
        run_arm(args.run_arm, args.root)
    else:
        assert args.parent
        prepare_queue(args.root, args.parent)
