"""Split converted levelX ScenarioNet data by source recording.

The split is stratified by location and keeps every clip from one recording in
the same subset. Validation/testing files are hard-linked before their original
training directory entries are removed, so no scenario payload is copied.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


SPLITS = ("training", "validation", "testing")
EVAL_FRACTION = 0.1
SEARCH_TRIALS = 50_000


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _write_pickle_atomic(path: Path, value) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(value, f)
    os.replace(tmp_path, path)


def _recording_counts(summary: Dict[str, dict]) -> Tuple[Counter, Dict[str, str]]:
    counts = Counter()
    locations = {}
    for metadata in summary.values():
        recording_id = str(metadata["source_recording_id"])
        location_id = str(metadata.get("locationId", "unknown"))
        counts[recording_id] += 1
        previous = locations.setdefault(recording_id, location_id)
        if previous != location_id:
            raise ValueError(
                f"Recording {recording_id} has multiple locations: "
                f"{previous}, {location_id}"
            )
    return counts, locations


def _allocate_eval_counts(
    recordings_by_location: Dict[str, list],
) -> Dict[str, int]:
    total_recordings = sum(len(values) for values in recordings_by_location.values())
    target_total = int(round(total_recordings * EVAL_FRACTION))
    eligible = {
        location_id: len(values)
        for location_id, values in recordings_by_location.items()
        if len(values) >= 3
    }
    if target_total < len(eligible):
        raise ValueError(
            f"Cannot represent all {len(eligible)} locations with only "
            f"{target_total} evaluation recordings"
        )

    allocated = {location_id: 1 for location_id in eligible}
    while sum(allocated.values()) < target_total:
        candidates = [
            location_id
            for location_id, num_recordings in eligible.items()
            if allocated[location_id] < (num_recordings - 1) // 2
        ]
        if not candidates:
            raise ValueError("Not enough recordings to allocate evaluation splits")
        location_id = max(
            candidates,
            key=lambda value: (
                eligible[value] * EVAL_FRACTION - allocated[value],
                eligible[value],
                -int(value),
            ),
        )
        allocated[location_id] += 1
    return allocated


def _choose_location_assignment(
    recordings: Iterable[str],
    counts: Counter,
    seed: int,
    num_eval: int,
) -> Dict[str, str]:
    recordings = sorted(recordings, key=lambda value: int(value))
    if num_eval == 0:
        return {recording_id: "training" for recording_id in recordings}

    target = sum(counts[recording_id] for recording_id in recordings) * EVAL_FRACTION
    rng = random.Random(seed)
    best = None
    best_score = None
    for _ in range(SEARCH_TRIALS):
        candidate = recordings.copy()
        rng.shuffle(candidate)
        validation = tuple(sorted(candidate[:num_eval], key=lambda value: int(value)))
        testing = tuple(
            sorted(candidate[num_eval : 2 * num_eval], key=lambda value: int(value))
        )
        validation_scenes = sum(counts[recording_id] for recording_id in validation)
        testing_scenes = sum(counts[recording_id] for recording_id in testing)
        score = (
            abs(validation_scenes - target) + abs(testing_scenes - target),
            abs(validation_scenes - testing_scenes),
            validation,
            testing,
        )
        if best_score is None or score < best_score:
            best_score = score
            best = (validation, testing)

    validation, testing = best
    assignment = {recording_id: "training" for recording_id in recordings}
    assignment.update({recording_id: "validation" for recording_id in validation})
    assignment.update({recording_id: "testing" for recording_id in testing})
    return assignment


def build_assignment(summary: Dict[str, dict], seed: int) -> Dict[str, str]:
    counts, locations = _recording_counts(summary)
    recordings_by_location = defaultdict(list)
    for recording_id, location_id in locations.items():
        recordings_by_location[location_id].append(recording_id)

    eval_counts = _allocate_eval_counts(recordings_by_location)
    assignment = {}
    for location_offset, location_id in enumerate(
        sorted(recordings_by_location, key=lambda value: int(value))
    ):
        location_assignment = _choose_location_assignment(
            recordings_by_location[location_id],
            counts,
            seed + location_offset,
            eval_counts.get(location_id, 0),
        )
        assignment.update(location_assignment)
    return assignment


def _partition_indexes(
    summary: Dict[str, dict],
    mapping: Dict[str, str],
    assignment: Dict[str, str],
):
    summaries = {split: {} for split in SPLITS}
    mappings = {split: {} for split in SPLITS}
    for filename, metadata in summary.items():
        recording_id = str(metadata["source_recording_id"])
        split = assignment[recording_id]
        summaries[split][filename] = metadata
        mappings[split][filename] = mapping.get(filename, "")
    return summaries, mappings


def _manifest(
    dataset_root: Path,
    seed: int,
    assignment: Dict[str, str],
    summaries: Dict[str, dict],
) -> dict:
    recording_locations = {}
    for metadata in summaries["training"].values():
        recording_locations[str(metadata["source_recording_id"])] = str(
            metadata.get("locationId", "unknown")
        )
    for split in ("validation", "testing"):
        for metadata in summaries[split].values():
            recording_locations[str(metadata["source_recording_id"])] = str(
                metadata.get("locationId", "unknown")
            )

    split_recordings = {
        split: sorted(
            (recording_id for recording_id, value in assignment.items() if value == split),
            key=lambda value: int(value),
        )
        for split in SPLITS
    }
    return {
        "dataset_root": str(dataset_root.resolve()),
        "strategy": "location-stratified source-recording split",
        "seed": seed,
        "target_scene_fractions": {
            "training": 0.8,
            "validation": 0.1,
            "testing": 0.1,
        },
        "scene_counts": {split: len(summaries[split]) for split in SPLITS},
        "recording_counts": {
            split: len(split_recordings[split]) for split in SPLITS
        },
        "recordings": split_recordings,
        "recording_locations": recording_locations,
    }


def _validate_source(training_dir: Path, summary: Dict[str, dict]) -> None:
    payload_files = {path.name for path in training_dir.glob("sd_*.pkl")}
    summary_files = set(summary)
    if payload_files != summary_files:
        missing = len(summary_files - payload_files)
        extra = len(payload_files - summary_files)
        raise ValueError(
            f"Training payload/index mismatch: missing={missing}, extra={extra}"
        )


def apply_split(dataset_root: Path, seed: int, dry_run: bool) -> dict:
    manifest_path = dataset_root / "split_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} already exists; this dataset has already been split"
        )

    training_dir = dataset_root / "training"
    summary = _load_pickle(training_dir / "dataset_summary.pkl")
    mapping = _load_pickle(training_dir / "dataset_mapping.pkl")
    _validate_source(training_dir, summary)
    if set(summary) != set(mapping):
        raise ValueError("dataset_summary.pkl and dataset_mapping.pkl keys differ")

    assignment = build_assignment(summary, seed)
    summaries, mappings = _partition_indexes(summary, mapping, assignment)
    manifest = _manifest(dataset_root, seed, assignment, summaries)
    if dry_run:
        return manifest

    for split in ("validation", "testing"):
        split_dir = dataset_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        existing_payloads = list(split_dir.glob("sd_*.pkl"))
        if existing_payloads:
            raise FileExistsError(
                f"{split_dir} already contains {len(existing_payloads)} scenario files"
            )

    # Create second directory entries first. os.link does not duplicate payload data.
    for split in ("validation", "testing"):
        split_dir = dataset_root / split
        for filename in summaries[split]:
            os.link(training_dir / filename, split_dir / filename)
        _write_pickle_atomic(split_dir / "dataset_summary.pkl", summaries[split])
        _write_pickle_atomic(split_dir / "dataset_mapping.pkl", mappings[split])

    # Commit the training index, then remove only files now reachable via eval dirs.
    _write_pickle_atomic(training_dir / "dataset_summary.pkl", summaries["training"])
    _write_pickle_atomic(training_dir / "dataset_mapping.pkl", mappings["training"])
    for split in ("validation", "testing"):
        for filename in summaries[split]:
            (training_dir / filename).unlink()

    tmp_manifest_path = manifest_path.with_suffix(".json.tmp")
    tmp_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_manifest_path, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split levelX ScenarioNet clips by source recording"
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    manifest = apply_split(
        dataset_root=args.dataset_root,
        seed=args.seed,
        dry_run=not args.apply,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
