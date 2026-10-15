"""Support-preserving CPU helpers for the opt-in RECON revision evaluator."""
import hashlib
import json
from pathlib import Path


def iter_agent_batches(loader, support_log=None):
    """Retain every target, split dense batches for inference, skip empty scenes."""
    for batch in loader:
        if support_log is not None:
            support_log.extend(batch.get('evaluation_support', []))
        n = int(batch['batch_size'])
        if not n:
            continue
        cap = int(batch.get('eval_microbatch_size', n))
        if cap <= 0:
            raise ValueError('eval_microbatch_size must be positive')
        for start in range(0, n, cap):
            stop = min(start + cap, n)
            if start == 0 and stop == n:
                yield batch
            else:
                fields = batch['input_dict']
                if any(len(value) != n for value in fields.values()):
                    raise ValueError('Non-target-leading field in evaluation batch')
                out = dict(batch)
                out.update(batch_size=stop-start, batch_sample_count=stop-start,
                           input_dict={k: v[start:stop] for k, v in fields.items()})
                yield out


def write_support_manifest(path, audits, records, expected_scenes, config, sources):
    if len(audits) != expected_scenes:
        raise ValueError('Scene support audit is incomplete')
    requested = [(x['data_path'], x['scene_file']) for x in audits]
    if len(set(requested)) != len(requested):
        raise ValueError('Repeated requested scene in evaluation')
    keys = [(str(r['scene']), str(r['oid']), r.get('ttc')) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Repeated output target key in evaluation')
    expected = [tuple(k) for row in audits for k in row['output_keys']]
    if len(expected) != len(set(expected)) or set(expected) != set(keys):
        raise ValueError('Collected records differ from preprocessing support')
    result = {'config': config, 'n_requested_scenes': expected_scenes,
              'n_excluded_scenes': sum(r['status'] != 'ok' for r in audits),
              'n_unique_targets': len(keys), 'scenes': audits,
              'source_sha256': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
