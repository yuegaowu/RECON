"""
Convert levelXdata exiD/highD CSV recordings to UniTraj ScenarioNet-like pkl.

The converter creates fixed 91-frame, 10 Hz clips matching Waymo motion
scenario length. Traffic lights are unavailable in exiD/highD, so
``dynamic_map_states`` is intentionally empty. Map features are built from
available lane information:

* highD: lane markings in ``recordingMeta.csv`` become freeway lane centerlines
  and lane divider polylines.
* exiD: the official OpenDRIVE map matching each recording's ``locationId``
  provides lane centerlines, road markings, road edges, and lane topology.

Example smoke:

    conda activate unitraj
    python datasets/levelx_converter/convert_levelx_to_scenarionet.py \\
      --dataset exid --source_root exid \\
      --output_dir data/exid/waymo_scenarionet/training_smoke \\
      --max_recordings 1 --max_clips 10
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.deepaccident_converter.deepaccident_map_builder import (
    _build_features_from_xodr,
)


CLIP_LEN = 91
TARGET_HZ = 10.0
ORIGINAL_HZ = 25.0
DT = 1.0 / TARGET_HZ
WINDOW_SECONDS = CLIP_LEN / TARGET_HZ

EXID_VERSION = "v2.1"
HIGHD_VERSION = "v1.0"


def _read_one_csv(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def _iter_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as f:
        yield from csv.DictReader(f)


def _f32(shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _object_summary(
    object_id: str,
    object_type: str,
    pos: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, object]:
    valid_pos = pos[valid, :2]
    if len(valid_pos) > 1:
        moving_distance = float(np.linalg.norm(np.diff(valid_pos, axis=0), axis=1).sum())
    else:
        moving_distance = 0.0
    continuous = 0
    cur = 0
    for v in valid:
        if bool(v):
            cur += 1
            continuous = max(continuous, cur)
        else:
            cur = 0
    return {
        "type": object_type,
        "object_id": object_id,
        "track_length": CLIP_LEN,
        "moving_distance": moving_distance,
        "valid_length": int(valid.sum()),
        "continuous_valid_length": continuous,
    }


def _number_summary(object_summary: Dict[str, Dict[str, object]], map_features: Dict[str, dict]) -> Dict[str, object]:
    counts = Counter(s["type"] for s in object_summary.values())
    return {
        "num_objects": len(object_summary),
        "object_types": tuple(counts.keys()),
        "num_objects_each_type": dict(counts),
        "num_moving_objects": sum(1 for s in object_summary.values() if float(s["moving_distance"]) > 0.1),
        "num_map_features": len(map_features),
        "map_height_diff": 0.0,
    }


def _select_sdc(tracks: Dict[str, dict], object_summary: Dict[str, dict], current_idx: int) -> Optional[str]:
    centers = []
    for tid, tr in tracks.items():
        if tr["type"] != "VEHICLE":
            continue
        if not bool(tr["state"]["valid"][current_idx]):
            continue
        centers.append(tr["state"]["position"][current_idx, :2])
    if not centers:
        return None
    scene_center = np.stack(centers, axis=0).mean(axis=0)
    best_id = None
    best_score = None
    for tid, tr in tracks.items():
        if tr["type"] != "VEHICLE":
            continue
        if not bool(tr["state"]["valid"][current_idx]):
            continue
        summary = object_summary.get(tid, {})
        # Prefer a moving vehicle near the scene center.
        dist = float(np.linalg.norm(tr["state"]["position"][current_idx, :2] - scene_center))
        moving_bonus = -1000.0 if float(summary.get("moving_distance", 0.0)) >= 2.0 else 0.0
        score = moving_bonus + dist
        if best_score is None or score < best_score:
            best_score = score
            best_id = tid
    return best_id


def _tracks_to_predict(tracks: Dict[str, dict], object_summary: Dict[str, dict], current_idx: int) -> Dict[str, dict]:
    result = {}
    for track_index, tid in enumerate(tracks.keys()):
        tr = tracks[tid]
        obj_type = tr["type"]
        if obj_type not in ("VEHICLE", "CYCLIST", "PEDESTRIAN"):
            continue
        if not bool(tr["state"]["valid"][current_idx]):
            continue
        summary = object_summary[tid]
        if summary["valid_length"] / CLIP_LEN < 0.5:
            continue
        if obj_type == "VEHICLE" and float(summary["moving_distance"]) < 2.0:
            continue
        result[tid] = {
            "track_index": track_index,
            "track_id": tid,
            "difficulty": 0,
            "object_type": obj_type,
        }
    return result


def _polyline(points: List[Tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    z = np.zeros((arr.shape[0], 1), dtype=np.float32)
    return np.concatenate([arr[:, :2], z], axis=1)


def _resample_polyline_by_arclength(
    points,
    spacing_m: float = 0.5,
) -> np.ndarray:
    """Resample a continuous polyline at approximately uniform arc-length spacing."""
    array = np.asarray(points)
    if array.ndim != 2 or len(array) == 0:
        return array
    if array.shape[1] == 2:
        array = np.concatenate(
            [array, np.zeros((len(array), 1), dtype=array.dtype)], axis=1
        )
    output_dtype = array.dtype
    polyline = array.astype(np.float64, copy=False)
    if len(polyline) < 2:
        return array.copy()

    segment_lengths = np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-6))
    polyline = polyline[keep]
    if len(polyline) < 2:
        return polyline.astype(output_dtype, copy=False)

    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1)))
    )
    total_length = float(distances[-1])
    if total_length <= 1e-6:
        return polyline[:1].astype(output_dtype, copy=False)

    num_segments = max(1, int(math.ceil(total_length / spacing_m)))
    sample_distances = np.linspace(
        0.0, total_length, num_segments + 1, dtype=np.float64
    )
    result = np.column_stack(
        [
            np.interp(sample_distances, distances, polyline[:, dim])
            for dim in range(polyline.shape[1])
        ]
    )
    return result.astype(output_dtype, copy=False)


def _lane_feature(fid: int, points: List[Tuple[float, float]], lane_type: str, speed_limit_mph: Optional[float] = None) -> Tuple[str, dict]:
    feat = {
        "type": lane_type,
        "polyline": _polyline(points),
        "speed_limit_mph": speed_limit_mph,
        "interpolating": False,
        "entry_lanes": [],
        "exit_lanes": [],
        "left_neighbor": [],
        "right_neighbor": [],
    }
    return str(fid), feat


def _line_feature(fid: int, points: List[Tuple[float, float]], line_type: str) -> Tuple[str, dict]:
    return str(fid), {"type": line_type, "polyline": _polyline(points)}


def _safe_speed_limit_mph(meta: Dict[str, str]) -> Optional[float]:
    try:
        speed = float(meta.get("speedLimit", "-1"))
    except ValueError:
        return None
    if speed <= 0:
        return None
    # exiD speedLimit is m/s in the extracted metadata.
    return float(speed * 2.236936)


def _find_exid_xodr(source_root: Path, location_id: str) -> Path:
    matches = sorted((source_root / "maps" / "opendrive").glob(f"{location_id}_*/*.xodr"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one OpenDRIVE map for locationId={location_id}, found {len(matches)}"
        )
    return matches[0]


def _load_exid_official_map(source_root: Path, location_id: str) -> Dict[object, dict]:
    xodr_path = _find_exid_xodr(source_root, location_id)
    root = ET.parse(xodr_path).getroot()
    features = _build_features_from_xodr(
        root,
        flip_y=False,
        orient_lanes_by_travel_direction=True,
        use_lane_link_ids=True,
    )
    for feature in features.values():
        if "polyline" in feature:
            feature["polyline"] = _resample_polyline_by_arclength(
                feature["polyline"], spacing_m=0.5
            )
        feature.setdefault("metadata", {})
        feature["metadata"].update(
            {
                "source": "official_opendrive",
                "location_id": str(location_id),
                "xodr_file": xodr_path.name,
            }
        )
    if not features:
        raise ValueError(f"No map features parsed from {xodr_path}")
    return features


def _load_exid_recording(
    data_dir: Path,
    rec_id: str,
    official_map_features: Optional[Dict[object, dict]] = None,
):
    meta = _read_one_csv(data_dir / f"{rec_id}_recordingMeta.csv")
    track_meta = {}
    for row in _iter_csv(data_dir / f"{rec_id}_tracksMeta.csv"):
        tid = row["trackId"]
        cls = row.get("class", "car").lower()
        object_type = "CYCLIST" if cls in {"motorcycle", "bicycle"} else "VEHICLE"
        length = float(row.get("length") or 0.0)
        width = float(row.get("width") or 0.0)
        if cls == "motorcycle":
            length = length if length > 0 else 2.2
            width = width if width > 0 else 0.8
        track_meta[tid] = {
            "type": object_type,
            "length": length,
            "width": width,
            "height": 1.5 if object_type == "VEHICLE" else 1.7,
            "class": cls,
        }

    rows_by_track = defaultdict(list)
    for row in _iter_csv(data_dir / f"{rec_id}_tracks.csv"):
        tid = row["trackId"]
        frame = float(row["frame"])
        x = float(row["xCenter"])
        y = float(row["yCenter"])
        heading = math.radians(float(row["heading"]))
        rows_by_track[tid].append(
            (
                frame,
                x,
                y,
                heading,
                float(row["xVelocity"]),
                float(row["yVelocity"]),
                float(row.get("lonLaneletPos") or frame),
            )
        )
    if official_map_features is None:
        raise ValueError("exiD conversion requires official OpenDRIVE map features")
    map_features = official_map_features
    return meta, track_meta, rows_by_track, map_features


def _load_highd_recording(data_dir: Path, rec_id: str):
    meta = _read_one_csv(data_dir / f"{rec_id}_recordingMeta.csv")
    track_meta = {}
    for row in _iter_csv(data_dir / f"{rec_id}_tracksMeta.csv"):
        tid = row["id"]
        cls = row.get("class", "Car")
        length = float(row.get("width") or 0.0)
        width = float(row.get("height") or 0.0)
        track_meta[tid] = {
            "type": "VEHICLE",
            "length": length,
            "width": width,
            "height": 3.0 if cls.lower() == "truck" else 1.5,
            "class": cls,
        }

    rows_by_track = defaultdict(list)
    min_x = float("inf")
    max_x = float("-inf")
    for row in _iter_csv(data_dir / f"{rec_id}_tracks.csv"):
        tid = row["id"]
        frame = float(row["frame"])
        length = float(row.get("width") or track_meta[tid]["length"])
        width = float(row.get("height") or track_meta[tid]["width"])
        # highD x/y are the upper-left corner of the vehicle bounding box.
        x = float(row["x"]) + length / 2.0
        y = float(row["y"]) + width / 2.0
        vx = float(row["xVelocity"])
        vy = float(row["yVelocity"])
        heading = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-3 else (0.0 if vx >= 0 else math.pi)
        rows_by_track[tid].append((frame, x, y, heading, vx, vy, frame))
        min_x = min(min_x, x)
        max_x = max(max_x, x)

    map_features = _build_highd_map_features(meta, min_x, max_x, rows_by_track)
    return meta, track_meta, rows_by_track, map_features


def _parse_markings(value: str) -> List[float]:
    return [float(x) for x in value.split(";") if x.strip()]


def _highd_lane_group_direction(
    rows_by_track: Dict[str, List[tuple]],
    lower_y: float,
    upper_y: float,
) -> float:
    velocities = []
    for rows in rows_by_track.values():
        for row in rows:
            y = row[2]
            if lower_y <= y <= upper_y and abs(row[4]) > 1.0:
                velocities.append(row[4])
    if not velocities:
        raise ValueError(
            f"Cannot infer highD lane direction for y=[{lower_y}, {upper_y}]"
        )
    return 1.0 if float(np.median(velocities)) >= 0.0 else -1.0


def _build_highd_map_features(
    meta: Dict[str, str],
    min_x: float,
    max_x: float,
    rows_by_track: Dict[str, List[tuple]],
) -> Dict[str, dict]:
    x0 = min_x - 50.0
    x1 = max_x + 50.0
    speed_limit_mph = _safe_speed_limit_mph(meta)
    features = {}
    fid = 1
    for markings_key in ["upperLaneMarkings", "lowerLaneMarkings"]:
        markings = _parse_markings(meta.get(markings_key, ""))
        if len(markings) < 2:
            continue
        markings = sorted(markings)
        direction = _highd_lane_group_direction(
            rows_by_track, markings[0], markings[-1]
        )
        points_x = (x0, x1) if direction > 0 else (x1, x0)
        for mark_idx, y in enumerate(markings):
            line_type = (
                "ROAD_LINE_SOLID_SINGLE_WHITE"
                if mark_idx in (0, len(markings) - 1)
                else "ROAD_LINE_BROKEN_SINGLE_WHITE"
            )
            k, v = _line_feature(
                fid, [(points_x[0], y), (points_x[1], y)], line_type
            )
            v["polyline"] = _resample_polyline_by_arclength(
                [(points_x[0], y), (points_x[1], y)]
            )
            v["metadata"] = {"source": markings_key}
            features[k] = v
            fid += 1
        for a, b in zip(markings[:-1], markings[1:]):
            y = 0.5 * (a + b)
            k, v = _lane_feature(
                fid,
                [(points_x[0], y), (points_x[1], y)],
                "LANE_FREEWAY",
                speed_limit_mph,
            )
            v["polyline"] = _resample_polyline_by_arclength(
                [(points_x[0], y), (points_x[1], y)]
            )
            v["metadata"] = {"source": markings_key}
            features[k] = v
            fid += 1
    return features


def _resample_track(rows: List[tuple], sample_frames: np.ndarray, dims: Dict[str, object]) -> Optional[dict]:
    rows = sorted(rows, key=lambda x: x[0])
    frames = np.asarray([r[0] for r in rows], dtype=np.float32)
    if len(frames) < 2:
        return None
    valid = (sample_frames >= frames[0]) & (sample_frames <= frames[-1])
    if not valid.any():
        return None

    x = np.interp(sample_frames, frames, [r[1] for r in rows]).astype(np.float32)
    y = np.interp(sample_frames, frames, [r[2] for r in rows]).astype(np.float32)
    heading_raw = np.unwrap(np.asarray([r[3] for r in rows], dtype=np.float32))
    heading = np.interp(sample_frames, frames, heading_raw).astype(np.float32)
    vx = np.interp(sample_frames, frames, [r[4] for r in rows]).astype(np.float32)
    vy = np.interp(sample_frames, frames, [r[5] for r in rows]).astype(np.float32)

    pos = _f32((CLIP_LEN, 3))
    pos[:, 0] = x
    pos[:, 1] = y
    velocity = _f32((CLIP_LEN, 2))
    velocity[:, 0] = vx
    velocity[:, 1] = vy
    length = np.full((CLIP_LEN, 1), float(dims["length"]), dtype=np.float32)
    width = np.full((CLIP_LEN, 1), float(dims["width"]), dtype=np.float32)
    height = np.full((CLIP_LEN, 1), float(dims["height"]), dtype=np.float32)

    pos[~valid] = 0.0
    velocity[~valid] = 0.0
    heading[~valid] = 0.0
    length[~valid] = 0.0
    width[~valid] = 0.0
    height[~valid] = 0.0

    return {
        "type": str(dims["type"]),
        "state": {
            "position": pos,
            "length": length,
            "width": width,
            "height": height,
            "heading": heading.reshape(CLIP_LEN, 1),
            "velocity": velocity,
            "valid": valid.astype(bool),
        },
    }


def _convert_clip(
    dataset: str,
    version: str,
    recording_id: str,
    start_frame: float,
    meta: Dict[str, str],
    track_meta: Dict[str, dict],
    rows_by_track: Dict[str, List[tuple]],
    map_features: Dict[str, dict],
) -> Optional[dict]:
    sample_frames = start_frame + np.arange(CLIP_LEN, dtype=np.float32) * (ORIGINAL_HZ / TARGET_HZ)
    tracks = {}
    object_summary = {}
    for tid in sorted(rows_by_track, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        dims = track_meta.get(tid)
        if dims is None:
            continue
        tr = _resample_track(rows_by_track[tid], sample_frames, dims)
        if tr is None:
            continue
        if int(tr["state"]["valid"].sum()) == 0:
            continue
        object_id = str(tid)
        tr["metadata"] = {"object_id": object_id, "class": dims.get("class")}
        tracks[object_id] = tr
        object_summary[object_id] = _object_summary(object_id, tr["type"], tr["state"]["position"], tr["state"]["valid"])

    current_time_index = 10
    sdc_id = _select_sdc(tracks, object_summary, current_time_index)
    if sdc_id is None:
        return None
    tracks_to_predict = _tracks_to_predict(tracks, object_summary, current_time_index)
    if not tracks_to_predict:
        return None

    scenario_id = f"{dataset}_{recording_id}_{int(round(start_frame)):06d}"
    ts = [round(i * DT, 4) for i in range(CLIP_LEN)]
    metadata = {
        "dataset": dataset,
        "metadrive_processed": True,
        "id": scenario_id,
        "scenario_id": scenario_id,
        "coordinate": "right-handed",
        "ts": ts,
        "current_time_index": current_time_index,
        "sdc_id": sdc_id,
        "tracks_to_predict": tracks_to_predict,
        "object_summary": object_summary,
        "number_summary": _number_summary(object_summary, map_features),
        "source_recording_id": recording_id,
        "source_start_frame": float(start_frame),
        "source_frame_rate": ORIGINAL_HZ,
        "map_center": tracks[sdc_id]["state"]["position"][current_time_index].astype(np.float32),
    }
    for k in ["locationId", "speedLimit", "duration", "numVehicles", "numTracks", "numVRUs", "numCars", "numTrucks"]:
        if k in meta:
            metadata[k] = meta[k]

    return {
        "id": scenario_id,
        "version": version,
        "length": CLIP_LEN,
        "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": map_features,
        "metadata": metadata,
    }


def _recording_ids(data_dir: Path, dataset: str) -> List[str]:
    ids = []
    for p in sorted(data_dir.glob("*_recordingMeta.csv")):
        ids.append(p.name.split("_")[0])
    if dataset == "highd":
        ids = [f"{int(x):02d}" for x in ids]
    return ids


def convert_dataset(
    dataset: str,
    source_root: Path,
    output_dir: Path,
    stride_seconds: float,
    max_recordings: Optional[int] = None,
    max_clips: Optional[int] = None,
) -> int:
    data_dir = source_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    version = EXID_VERSION if dataset == "exid" else HIGHD_VERSION
    loader = _load_exid_recording if dataset == "exid" else _load_highd_recording
    prefix = "sd_exid" if dataset == "exid" else "sd_highd"

    summary = {}
    converted = 0
    skipped = 0
    exid_map_cache = {}
    rec_ids = _recording_ids(data_dir, dataset)
    if max_recordings is not None:
        rec_ids = rec_ids[:max_recordings]

    stride_frames = stride_seconds * ORIGINAL_HZ
    window_frames = WINDOW_SECONDS * ORIGINAL_HZ

    for rec_idx, rec_id in enumerate(rec_ids, start=1):
        print(f"[{dataset}] recording {rec_idx}/{len(rec_ids)} id={rec_id}")
        if dataset == "exid":
            recording_meta = _read_one_csv(data_dir / f"{rec_id}_recordingMeta.csv")
            location_id = recording_meta["locationId"]
            if location_id not in exid_map_cache:
                print(f"  loading official OpenDRIVE map for locationId={location_id}")
                exid_map_cache[location_id] = _load_exid_official_map(source_root, location_id)
                print(f"  map_features={len(exid_map_cache[location_id])}")
            meta, track_meta, rows_by_track, map_features = loader(
                data_dir, rec_id, exid_map_cache[location_id]
            )
        else:
            meta, track_meta, rows_by_track, map_features = loader(data_dir, rec_id)
        all_frames = [row[0] for rows in rows_by_track.values() for row in rows]
        recording_start = min(all_frames)
        recording_end = max(all_frames)
        max_start = recording_end - (CLIP_LEN - 1) * (ORIGINAL_HZ / TARGET_HZ)
        starts = []
        cur = recording_start
        while cur <= max_start + 1e-6:
            starts.append(cur)
            cur += stride_frames
        for start_frame in starts:
            sd = _convert_clip(dataset, version, rec_id, start_frame, meta, track_meta, rows_by_track, map_features)
            if sd is None:
                skipped += 1
                continue
            fname = f"{prefix}_{version}_{sd['id']}.pkl"
            with (output_dir / fname).open("wb") as f:
                pickle.dump(sd, f)
            summary[fname] = sd["metadata"]
            converted += 1
            if converted % 100 == 0:
                print(f"  converted={converted} skipped={skipped}")
            if max_clips is not None and converted >= max_clips:
                break
        if max_clips is not None and converted >= max_clips:
            break

    mapping = {fname: "" for fname in summary}
    with (output_dir / "dataset_summary.pkl").open("wb") as f:
        pickle.dump(summary, f)
    with (output_dir / "dataset_mapping.pkl").open("wb") as f:
        pickle.dump(mapping, f)
    print(f"[{dataset}] done converted={converted} skipped={skipped} output={output_dir}")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert exiD/highD CSV to ScenarioNet-like pkl")
    parser.add_argument("--dataset", choices=["exid", "highd"], required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--stride_seconds",
        type=float,
        default=WINDOW_SECONDS,
        help="Sliding-window stride. Default 9.1 gives non-overlapping clips.",
    )
    parser.add_argument("--max_recordings", type=int, default=None)
    parser.add_argument("--max_clips", type=int, default=None)
    args = parser.parse_args()

    convert_dataset(
        dataset=args.dataset,
        source_root=args.source_root,
        output_dir=args.output_dir,
        stride_seconds=args.stride_seconds,
        max_recordings=args.max_recordings,
        max_clips=args.max_clips,
    )


if __name__ == "__main__":
    main()
