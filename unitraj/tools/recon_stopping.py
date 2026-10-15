"""Replay the predeclared validation stopping rule; no training implementation."""
import math


def validate_completion(cfg, epochs):
    maximum = int(cfg.max_epochs)
    assert 0 < len(epochs) <= maximum, 'Invalid completed epoch count'
    assert [e['epoch'] for e in epochs] == list(range(len(epochs)))
    rule = cfg.get('recon_early_stopping')
    if not rule:
        assert len(epochs) == maximum, 'Fixed-budget training ended early'
        return {'completed_epochs': len(epochs), 'stop_reason': 'max_epochs'}
    best, wait, first_stop = float('inf'), 0, None
    for i, row in enumerate(epochs):
        value = row['metrics'][rule.monitor]
        assert math.isfinite(value), 'Nonfinite validation metric is a failure, not successful early stopping'
        if value < best - float(rule.min_delta):
            best, wait = value, 0
        else:
            wait += 1
        if wait >= int(rule.patience):
            first_stop = i + 1
            break
    expected = first_stop if first_stop is not None else maximum
    assert len(epochs) == expected, ('Premature stop or continued after patience', len(epochs), expected)
    return {'completed_epochs': len(epochs),
            'stop_reason': 'validation_patience' if first_stop is not None else 'max_epochs',
            'patience': int(rule.patience), 'min_delta': float(rule.min_delta),
            'wait_count': wait, 'monitor': rule.monitor}
