"""按 agent 轨迹半径裁剪 DeepAccident ScenarioNet 地图,根治 SMART 等模型在 CARLA 稠密整镇
地图上的显存爆炸(scene-centric 全图注意力 × 整镇地图)。

做法:对每个 scenario pkl,收集所有 agent 全时刻【有效】位置,只保留"任一折线点距某个有效
agent 位置 <= margin"的 map_feature,其余删除。margin=50m > 模型用到的 pl2a_radius=30m,
故对预测无损。tracks/metadata/dynamic_map_states/文件名一律不动。

安全:只读原目录、写【新目录】(<folder>_mapcrop),原始数据绝不覆盖。

用法:
  python crop_deepaccident_map.py \
      --src data/deepaccident/scenarionet/collision_vehicle \
      --margin 50
  # 输出 -> .../collision_vehicle_mapcrop
"""
import argparse
import glob
import os
import pickle
import shutil

import numpy as np
from scipy.spatial import cKDTree


def collect_agent_xy(scenario):
    xy = []
    for t in scenario["tracks"].values():
        st = t["state"]
        p = np.asarray(st["position"])[:, :2]
        v = np.asarray(st["valid"]).astype(bool)
        if v.any():
            xy.append(p[v])
    return np.concatenate(xy, 0) if xy else None


def crop_one(scenario, margin):
    """按【有效 agent 包围盒外扩 margin 米】的矩形裁剪地图:
    agent bbox 若为 200x200,margin=100 → 保留 400x400 的地图(每边外扩 margin)。
    margin >= pl2a_radius(30m) 故对预测无损;只要 polyline 任一点落在外扩盒内就保留整条。"""
    axy = collect_agent_xy(scenario)
    mf = scenario["map_features"]
    n_before = len(mf)
    if axy is None:                      # 没有有效 agent → 保守保留全部
        return scenario, n_before, n_before
    xmin, ymin = axy.min(0) - margin
    xmax, ymax = axy.max(0) + margin
    keep = {}
    for fid, m in mf.items():
        pl = m.get("polyline")
        if pl is None or len(pl) == 0:   # 无几何的要素(极少)→ 保留
            keep[fid] = m
            continue
        p = np.asarray(pl)[:, :2]
        inside = (p[:, 0] >= xmin) & (p[:, 0] <= xmax) & (p[:, 1] >= ymin) & (p[:, 1] <= ymax)
        if inside.any():
            keep[fid] = m
    scenario["map_features"] = keep
    return scenario, n_before, len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原 scenarionet 文件夹(如 .../collision_vehicle)")
    ap.add_argument("--margin", type=float, default=100.0, help="agent 包围盒外扩多少米(默认 100:200x200→400x400)")
    ap.add_argument("--dst", default=None, help="输出文件夹(默认 <src>_mapcrop)")
    args = ap.parse_args()

    src = os.path.normpath(args.src)
    dst = args.dst or (src + "_mapcrop")
    os.makedirs(dst, exist_ok=True)

    sd_files = sorted(glob.glob(os.path.join(src, "sd_*.pkl")))
    print(f"[crop] src={src}\n[crop] dst={dst}\n[crop] margin={args.margin}m  n_scenario={len(sd_files)}")

    tot_before = tot_after = 0
    for i, f in enumerate(sd_files):
        with open(f, "rb") as fh:
            scenario = pickle.load(fh)
        scenario, nb, na = crop_one(scenario, args.margin)
        tot_before += nb
        tot_after += na
        with open(os.path.join(dst, os.path.basename(f)), "wb") as fh:
            pickle.dump(scenario, fh)
        if i % max(len(sd_files) // 10, 1) == 0:
            print(f"  {i}/{len(sd_files)}  map_features {nb}->{na}")

    # 元数据 / 附属文件原样复制(summary 不含地图点、按文件名索引,直接可用)
    for extra in ("dataset_summary.pkl", "dataset_mapping.pkl", "collider_objects.csv"):
        p = os.path.join(src, extra)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, extra))

    print(f"[crop] 完成。map_features 总量 {tot_before} -> {tot_after} "
          f"(保留 {100*tot_after/max(tot_before,1):.0f}%)")


if __name__ == "__main__":
    main()
