"""Repair existing DeepAccident ScenarioNet collision frames atomically.

The current local dataset was converted after discarding ``frame_ns``.  A
full-dataset audit established that every labelled clip uses source frame 11
as local index 0, so the recoverable mapping is ``local = source - 11``.
Future conversions must use the exact frame-number lookup implemented in
``deepaccident_to_scenarionet.py`` instead of this migration utility.
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path


COLLISION_BUCKETS = ("collision_vehicle", "collision_object")


def repair_file(path: Path, source_start: int, apply: bool) -> str:
    with path.open("rb") as handle:
        scene = pickle.load(handle)
    metadata = scene.setdefault("metadata", {})
    collision = metadata.setdefault("collision", {})
    if not collision.get("is_collided", False):
        return "not_collision"

    if collision.get("collision_source_frame") is not None:
        source_frame = int(collision["collision_source_frame"])
        local_frame = int(collision["collision_frame"])
        if local_frame != source_frame - source_start:
            raise ValueError(f"Inconsistent already-migrated frame: {path}")
        return "already_migrated"

    source_frame = int(collision["collision_frame"])
    local_frame = source_frame - source_start
    length = int(scene.get("length", 0))
    if not 0 <= local_frame < length:
        raise ValueError(
            f"Recovered local frame out of bounds: {path}, "
            f"source={source_frame}, local={local_frame}, length={length}"
        )

    collision.update(
        {
            "collision_source_frame": source_frame,
            "collision_frame": local_frame,
            "source_frame_start": source_start,
            "source_frame_end": source_start + length - 1,
            "collision_frame_mapping": "source_frame_minus_11_recovered",
        }
    )
    # Keep current_time_index consistent with a clean conversion.
    metadata["current_time_index"] = (
        local_frame - 5
        if 5 < local_frame < length - 20
        else min(10, max(2, length // 4))
    )
    if not apply:
        return "would_repair"

    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(scene, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return "repaired"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-start", type=int, default=11)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    counts = {}
    for bucket in COLLISION_BUCKETS:
        files = sorted((args.root / bucket).glob("sd_*.pkl"))
        for path in files:
            status = repair_file(path, args.source_start, args.apply)
            counts[status] = counts.get(status, 0) + 1
    print({"root": str(args.root), "apply": args.apply, "counts": counts})


if __name__ == "__main__":
    main()
