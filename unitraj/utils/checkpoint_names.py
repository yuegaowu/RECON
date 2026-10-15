"""Keep historical evaluation IDs stable across checkpoint directory aliases.

Display names and canonical paths live in configs/checkpoint_catalog.json.
Saved training configs, checkpoint contents and result IDs are immutable.
This module deliberately has no torch/model imports.
"""
import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _experiment_ids():
    catalog = Path(__file__).resolve().parents[1] / "configs/checkpoint_catalog.json"
    entries = json.loads(catalog.read_text(encoding="utf-8"))["entries"]
    mapping = {}
    for entry in entries:
        for name in (entry["canonical_name"], entry["legacy_name"]):
            previous = mapping.setdefault(name, entry["legacy_name"])
            if previous != entry["legacy_name"]:
                raise ValueError("Conflicting checkpoint alias: " + name)
    return mapping


def checkpoint_experiment_id(checkpoint_path, default="model"):
    """Return the original run ID; unknown/new runs retain their directory name."""
    parts = os.path.normpath(str(checkpoint_path)).split(os.sep)
    if "unitraj_ckpt" not in parts:
        return default
    index = parts.index("unitraj_ckpt") + 1
    if index >= len(parts):
        return default
    name = parts[index]
    return _experiment_ids().get(name, name)
