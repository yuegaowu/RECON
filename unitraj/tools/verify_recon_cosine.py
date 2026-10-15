"""Check every recorded optimizer learning rate against the declared cosine curve."""
import json
import math


def verify_cosine(directory, trainer):
    scheduler = trainer.lr_scheduler_configs[0].scheduler
    assert type(scheduler).__name__ == 'CosineAnnealingLR'
    total = trainer.global_step
    assert scheduler.T_max == scheduler.last_epoch == total
    high, low = scheduler.base_lrs[0], scheduler.eta_min
    count = 0; previous = high
    with (directory/'train_batches.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row['step'] == count
            expected = low + (high-low)*(1+math.cos(math.pi*count/total))/2
            assert math.isclose(row['lr'], expected, rel_tol=1e-8, abs_tol=1e-14)
            assert low <= row['lr'] <= previous + 1e-14
            previous = row['lr']; count += 1
    assert count == total
    assert math.isclose(trainer.optimizers[0].param_groups[0]['lr'], low, abs_tol=1e-14)
    return {'status': 'passed', 'updates': total, 'start_lr': high, 'end_lr_after_final_update': low,
            'last_used_lr': previous, 'interval': 'step', 'checked_updates': count}
