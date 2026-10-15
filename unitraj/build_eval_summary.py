"""
build_eval_summary.py —— 把 benchmark_eval.py 逐数据集产出的 metrics.csv / accident_ttc.csv
归并成「按评测语义分组」的汇总表(行=数据集/子集,列=指标)。纯 pandas,无 torch/hydra 依赖。

由 eval_allsets.sh / eval_accident_ttc.sh 在跑完所有数据集后自动调用,不单独手跑。

两种模式(--kind):
  regular(默认):读各数据集 metrics.csv,按「预测协议是否相同」拆成 2 张表 ——
    ① regular_metrics.csv    : 单次预测口径(场景起点,全车)。行包含:
         waymo / highd / deepaccident_normal / realaccident_single_vehicle
         + 两个碰撞集的 non_collider 行(周围未撞车,口径与上面相同 → 并进来)
    ② accident_collider_metrics.csv : 事故车(collider)单次预测。行包含:
         deepaccident_collision / realaccident_collision(列与 ① 完全相同,放一起便于 DA vs 真实对照)
  ttc:读各碰撞集 accident_ttc.csv,纵向拼成一张 accident_ttc.csv,首列加 dataset 区分。

路由完全靠「csv 里有没有 subset 列」判断,不硬编码数据集名:
  - 无 subset 列  → 单行,进 ① regular,行名=tag
  - 有 subset 列  → collider 行进 ②,non_collider 行进 ① 并加 "(non_collider)" 后缀

用法:
  python build_eval_summary.py --kind regular --out <summary_dir> tag1|csv1 tag2|csv2 ...
  python build_eval_summary.py --kind ttc     --out <summary_dir> tag1|ttc1 tag2|ttc2 ...
  --out 不给时,自动从第一个 csv 的文件夹名推出 <exp>_epoch=<E>_summary。
"""
import os
import sys
import argparse

import pandas as pd


def _friendly(tag):
    """碰撞集标签去掉末尾 _vehicle 作行名:deepaccident_collision_vehicle → deepaccident_collision。"""
    return tag[:-len("_vehicle")] if tag.endswith("_vehicle") else tag


def _derive_summary_dir(first_csv, first_tag):
    """从逐数据集文件夹名 <prefix>_<tag>[_count] 反推 <prefix>_summary。"""
    folder = os.path.basename(os.path.dirname(os.path.abspath(first_csv)))
    idx = folder.find(f"_{first_tag}")
    prefix = folder[:idx] if idx > 0 else folder
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(first_csv))),
                        f"{prefix}_summary")


def _parse_pairs(items):
    pairs = []
    for a in items:
        if "|" not in a:
            continue
        tag, csv = a.split("|", 1)
        pairs.append((tag, csv))
    return pairs


def build_regular(pairs, out_dir):
    """从磁盘 csv 路径归并(CLI 入口用):pairs = [(tag, csv_path)]。读成 DataFrame 后转交 frames 版。"""
    frames = []
    for tag, csvp in pairs:
        if not os.path.isfile(csvp):
            print(f"  [summary] 跳过缺失: {tag} -> {csvp}")
            continue
        frames.append((tag, pd.read_csv(csvp)))
    return build_regular_frames(frames, out_dir)


def build_regular_frames(pairs, out_dir):
    """从内存 DataFrame 归并(benchmark_eval 统一评测用,不落逐集 csv):pairs = [(tag, DataFrame)]。
    路由同 build_regular:无 subset 列→单行进 regular;有 subset 列→collider 行进 collider 表、
    non_collider 行进 regular 表并加 "(non_collider)" 后缀。"""
    regular_rows, collider_rows, col_order = [], [], []
    for tag, df in pairs:
        for c in df.columns:                       # 收集列顺序(排除 subset),保持首次出现序
            if c != "subset" and c not in col_order:
                col_order.append(c)
        if "subset" in df.columns:                 # 碰撞集:拆 collider / non_collider
            for d in df.to_dict("records"):         # records 保留各列原 dtype(n 列保持整数)
                sub = str(d.pop("subset"))
                if sub == "collider":
                    collider_rows.append((_friendly(tag), d))
                else:
                    regular_rows.append((f"{_friendly(tag)} (non_collider)", d))
        else:                                      # 常规集:单行
            regular_rows.append((tag, df.to_dict("records")[0]))

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for fname, rows in [("regular_metrics.csv", regular_rows),
                        ("accident_collider_metrics.csv", collider_rows)]:
        if not rows:
            continue
        cols = ["dataset"] + col_order
        data = [[name] + [d.get(c) for c in col_order] for name, d in rows]
        path = os.path.join(out_dir, fname)
        pd.DataFrame(data, columns=cols).to_csv(path, index=False, float_format="%.4f")
        written.append(path)
        print(f"  [summary] {fname}  ({len(rows)} 行) -> {path}")
    return written


def build_ttc_wide_frames(pairs, out_dir):
    """宽表 TTC 归并(benchmark_eval 统一评测用,内存 DataFrame):pairs=[(tag, 单行 DataFrame)]。
    每碰撞集一行,首列加 dataset;列=ADE1..8 / FDE1..8 / MissRate1..8 / n1..8。
    与逐文件磁盘版 build_ttc(长表矩阵,eval_accident_ttc.sh 用)互不影响。"""
    rows, col_order = [], []
    for tag, df in pairs:
        for c in df.columns:
            if c not in col_order:
                col_order.append(c)
        rows.append((_friendly(tag), df.to_dict("records")[0]))   # records 保留 n 列整数 dtype
    if not rows:
        print("  [summary] 无 TTC 输入")
        return []
    os.makedirs(out_dir, exist_ok=True)
    cols = ["dataset"] + col_order
    data = [[name] + [d.get(c) for c in col_order] for name, d in rows]
    path = os.path.join(out_dir, "accident_ttc.csv")
    pd.DataFrame(data, columns=cols).to_csv(path, index=False, float_format="%.4f")
    print(f"  [summary] accident_ttc.csv  ({len(rows)} 行,事故前 1..8s 对角线) -> {path}")
    return [path]


def build_ttc(pairs, out_dir):
    frames, col_order = [], []
    for tag, csvp in pairs:
        if not os.path.isfile(csvp):
            print(f"  [summary] 跳过缺失: {tag} -> {csvp}")
            continue
        df = pd.read_csv(csvp)
        for c in df.columns:
            if c not in col_order:
                col_order.append(c)
        df.insert(0, "dataset", _friendly(tag))
        frames.append(df)
    if not frames:
        print("  [summary] 无 TTC 输入")
        return []
    os.makedirs(out_dir, exist_ok=True)
    out = pd.concat(frames, ignore_index=True)[["dataset"] + col_order]
    path = os.path.join(out_dir, "accident_ttc.csv")
    out.to_csv(path, index=False, float_format="%.4f")
    print(f"  [summary] accident_ttc.csv  ({len(out)} 行) -> {path}")
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["regular", "ttc"], default="regular")
    ap.add_argument("--out", default=None)
    ap.add_argument("pairs", nargs="+", help="tag|csv_path 列表")
    args = ap.parse_args()

    pairs = _parse_pairs(args.pairs)
    if not pairs:
        print("  [summary] 没有有效的 tag|csv 输入")
        return
    out_dir = args.out or _derive_summary_dir(pairs[0][1], pairs[0][0])
    if args.kind == "ttc":
        build_ttc(pairs, out_dir)
    else:
        build_regular(pairs, out_dir)


if __name__ == "__main__":
    main()
