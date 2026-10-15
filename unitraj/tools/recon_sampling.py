"""Sampling arithmetic for the original Lightning loaders; no training loop."""
import math
from torch.utils.data import DistributedSampler


def batch_count(scenes, world, batch, limit):
    available = math.ceil(math.ceil(scenes / world) / batch)
    return min(available, limit) if isinstance(limit, int) else int(available * limit)


class NoPaddingDistributedSampler(DistributedSampler):
    """Disjoint validation shards. Require equal batch counts for sync_dist logs."""
    def __init__(self, dataset, num_replicas, rank, batch_size):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        counts = [len(range(r, len(dataset), num_replicas)) for r in range(num_replicas)]
        if len({math.ceil(n / batch_size) for n in counts}) != 1:
            raise ValueError('No-padding validation requires equal per-rank batch counts')
        self.num_samples = counts[rank]
        self.total_size = len(dataset)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return self.num_samples
