"""
benchmark_eval.py — 已训练模型的最终指标评测(yaml/CLI 驱动,一次评一个模型)。

与 evaluation.py 的关系:
  evaluation.py 是框架原有入口,trainer.validate 把指标丢给 wandb,横轴固定且依赖
  TF 官方 eval(本环境无 tensorflow/waymo_open_dataset,用不了)。本脚本接管"产出
  最终论文指标"这件事:手动收集每目标 (pred[K,80,2], score[K], gt[80,2], mask[80]),
  numpy 自算 3/5/8s 指标,单/多模态同一套逻辑,缺失指标留空。

统一评测:一条命令评一个模型在【全部数据集】上的指标。
  configs/eval.yaml 里:
    - override method / ckpt_path / devices  → 选模型 / 权重 / GPU
    - datasets 块                            → 数据集清单(每集 val_data_path / data_root /
                                               sample_fraction:null=全量 / 整数N=前N个 / 小数0~1=比例)
  benchmark_eval.py 内部循环 datasets:模型只加载一次,逐集出 metrics.csv,跑完自动归并汇总表。
  eval_batch_size 沿用各模型 method yaml;load_num_workers 沿用 config.yaml;读取固定 onthefly。
  ▸ 带肇事标注的碰撞集(deepaccident/realaccident collision_vehicle)会【在同一次评测里】额外跑
    「事故前 TTC 对角线」评测:撞前 T=1..8s 各锚一次,只用从预测点到撞击瞬间这 [0,T] 段算 ADE_T/FDE_T,
    归并成宽表 accident_ttc.csv(每碰撞集一行)。无需再单独跑 eval_accident_ttc.sh。

输出布局(一个数据集 = 一个文件夹 + 一个单行 csv,跑完再归并出 summary):
  final_metrics/<exp_name>_epoch=<E>_<dataset>[_<count>]/metrics.csv   ← 逐集明细
  final_metrics/<exp_name>_epoch=<E>/regular_metrics.csv …             ← 自动归并汇总
  final_metrics/<exp_name>_epoch=<E>/accident_ttc.csv                  ← 事故前 1..8s 对角线(宽表)
    - exp_name / epoch 从 ckpt_path 解析;dataset = datasets 块的标签(key)
    - count: 全量(null/1.0)省略;整数子集附 _N;比例附 _pctNN
    - csv 无行名(文件夹名已代表该次评测),只有列名 + 一行数值;缺失指标列留空

跑法:
  cd UniTraj/unitraj
  python benchmark_eval.py                       # 读 eval.yaml,评全部数据集
  python benchmark_eval.py method=MTR ckpt_path=unitraj_ckpt/.../brier_fde=3.00.ckpt   # 临时换模型/权重

单集旧路径(事故 TTC 评测 eval_accident_ttc.sh 用):
  传 +run_dataset_label=<标签> val_data_path=[..] data_root=.. → 只评该单集(不走 datasets 循环)。
"""
import os
import re
import sys
import csv
import glob
import pickle
import logging
import warnings
from utils.checkpoint_names import checkpoint_experiment_id
from utils.model_names import canonical_model_key

# 评测脚本:屏蔽第三方库的弃用/Future 警告与 Lightning 的 INFO 日志,保持终端输出干净。
# 必须在导入 torch / timm / pygame(经 datasets、models 间接引入)之前设置。
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
for _name in ("lightning_fabric", "pytorch_lightning", "lightning"):
    logging.getLogger(_name).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# 在任何会初始化 CUDA 的 import 之前锁定可见 GPU。
#   实测:`from datasets import ...` / `from models import ...` 在 import 时就会创建 CUDA
#   上下文(torch.cuda.is_initialized() 变 True)。一旦初始化,再设 CUDA_VISIBLE_DEVICES 无效
#   → 进程默认落到物理 GPU 0。所以必须在这些 import 之前、按 eval.yaml/命令行的 devices 设好。
#   设成单卡可见后,进程内一律用 cuda:0(= 选中的物理卡),同时规避部分模型(BeTopNet)硬编码
#   .cuda() 落 cuda:0 与模型不在同卡的 device mismatch。
# ─────────────────────────────────────────────────────────────────────────────
def _bootstrap_visible_gpu():
    import re as _re
    for _a in sys.argv[1:]:                 # ① 命令行 devices=[N] 优先
        if _a.startswith("devices="):
            _m = _re.search(r"\d+", _a)
            if _m:
                return _m.group(0)
    try:                                    # ② 否则读 configs/eval.yaml 的 devices
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "eval.yaml")
        with open(_p) as _f:
            for _line in _f:
                _m = _re.match(r"\s*devices:\s*\[?\s*(\d+)", _line)
                if _m:
                    return _m.group(1)
    except Exception:
        pass
    return None


_BOOTSTRAP_GPU = _bootstrap_visible_gpu()
if _BOOTSTRAP_GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _BOOTSTRAP_GPU

# 自动多卡 orchestrator 转发给子进程的原始命令行(在 __main__ 摘掉 ckpt_path 后赋值)。
_PARENT_ARGV = []

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset
from datasets.evaluation_support import iter_agent_batches, write_support_manifest
from datasets.common_utils import rotate_points_along_z   # agent-centric → world 反变换(成对碰撞几何用)
from models import build_model
from utils.utils import set_seed
import build_eval_summary

# 时间窗常量(10Hz,future_len=80)。idx_t = round(t*10) - 1。
HORIZONS = {"3s": 29, "5s": 49, "8s": 79}
# 事故 TTC 评测:每个锚点报 horizon=1..8s 的逐帧指标(frame 9/19/.../79)。
#   对锚点 T,碰撞瞬间恰好落在 horizon h=T(=矩阵对角线)。
HORIZONS_TTC = {f"{h}s": 10 * h - 1 for h in range(1, 9)}
# 事故前 TTC「对角线」评测(统一评测内置,宽表每碰撞集一行):
#   对锚点 T(撞前 T 秒预测),只评从预测点到撞击瞬间这 [0,T] 段——撞击恰落在 gt 帧 10T-1。
#   ADE{T}=该段平均位移误差;FDE{T}=撞击瞬间终点误差;MissRate{T}=FDE>2m 比例;n{T}=活到撞击的肇事车数。
TTC_DIAG_OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8]
TTC_DIAG_COLUMN_ORDER = (
    [f"ADE{T}" for T in TTC_DIAG_OFFSETS]
    + [f"FDE{T}" for T in TTC_DIAG_OFFSETS]
    + [f"MissRate{T}" for T in TTC_DIAG_OFFSETS]
    + [f"n{T}" for T in TTC_DIAG_OFFSETS]
)
TYPE_ID2NAME = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
MISS_THRESH = 2.0  # minFDE > 2m 记为 miss(UniTraj 内置约定)

# 论文表列顺序:
#   ① 公平 + 抗变长主指标:top1(置信度最高那条)在固定 3/5/8s 时刻的 ADE/FDE,
#      只统计「活到该时刻」的目标(survive 到 frame 29/49/79 才计入)。单/多模态同一口径,
#      事故场景目标存活时长不一也能横向比 → 这是跨域/事故评测看的主指标。
#   ② 多模态能力上限 + 校准(best-of-6,单模态留空)。
#   ③ 分对象类型(best-of-6 全程;DeepAccident 仅 vehicle 有值,人/自行车留空)。
COLUMN_ORDER = [
    "ADE1@3s", "ADE1@5s", "ADE1@8s", "FDE1@3s", "FDE1@5s", "FDE1@8s",
    "MissRate1@3s", "MissRate1@5s", "MissRate1@8s",  # top1 漏报率(公平口径,单/多模态可比)
    "n@3s", "n@5s", "n@8s",
    "minADE6", "minFDE6", "MissRate", "brier_minFDE6", "top1_is_best",
    # 校准 / 多模态多样性(仅多模态有值;K==1 留空)——分布偏移下的"置信度崩塌"诊断指标。
    #   p_best        : oracle 最优模式上的概率质量(越高=越敢断,越不坍缩);brier 校准项=(1-p_best)²
    #   mode_entropy  : K 条模式概率分布的香农熵(nats);mode_entropy_norm = 熵 / ln(K) ∈[0,1]
    #   endpoint_spread: K 个预测终点两两平均距离(m;越小=模式越挤=越坍缩)
    "p_best", "mode_entropy", "mode_entropy_norm", "endpoint_spread",
    "vehicle_minADE", "vehicle_minFDE", "pedestrian_minADE", "pedestrian_minFDE",
    "cyclist_minADE", "cyclist_minFDE",
    "n_targets",
]


# ─────────────────────────────────────────────────────────────────────────────
# 肇事对象查找表:{scenario_id → set(肇事 track_id)}
#   只有 DeepAccident 的 collision_vehicle(mutual_collision)在 pkl 的
#   metadata.collision.collision_pairs 里存了车-车互撞双方的 track_id。据此即可把
#   评测目标分成「肇事对象 / 非肇事对象」两桶。
#   Waymo 等无碰撞数据集 → collision_pairs 不存在 → 返回空表 → 评测退化为单行输出。
# ─────────────────────────────────────────────────────────────────────────────
def build_collider_lookup(val_data_paths):
    """建 {scenario_id: set(肇事 track_id)}。路径相对 unitraj/ 解析(与 ckpt_path 同款)。

    优先读目录里预算好的 `collider_objects.csv`(列 collider_ids 形如 "7154|7155"),它与 pkl 的
    collision_pairs 完全等价(已逐场景核对 137/137 一致),但避免为读 metadata 而 pickle.load 整个
    带 CARLA 稠密大图的 pkl——快很多。没有该 CSV 时回退到扫 sd_*.pkl 的 collision_pairs。
    normal 等无碰撞数据集:CSV 的 collider_ids 全空 → 得到空表 → 评测自然退化为单行(不分桶)。"""
    base = os.path.dirname(os.path.abspath(__file__))
    lookup = {}
    for p in (val_data_paths or []):
        p = str(p)
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(base, p))
        csv_path = os.path.join(p, "collider_objects.csv")
        if os.path.isfile(csv_path):
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    ids = {x for x in (row.get("collider_ids") or "").split("|") if x}
                    if ids:
                        lookup[str(row["scenario_id"])] = ids
            continue
        for pkl in glob.glob(os.path.join(p, "sd_*.pkl")):   # 无 CSV 时回退
            try:
                meta = pickle.load(open(pkl, "rb"))["metadata"]
            except Exception:
                continue
            pairs = meta.get("collision", {}).get("collision_pairs", []) or []
            ids = {str(x) for pair in pairs for x in pair}
            if ids:
                lookup[str(meta["scenario_id"])] = ids
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# 每目标记录收集:统一成 (err[K,80], mask[80], score[K], type, final_idx, scene, oid)
# ─────────────────────────────────────────────────────────────────────────────
def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj


def collect_agent(model, loader, device, dump_world=False, support_log=None):
    """agent-centric:model.predict(batch) → predicted_trajectory[B,K,80,2]+probability[B,K]。
    误差在 agent-centric frame 直接算(与 base_model.log_info 一致,无需 world 变换)。

    统一用 predict() 而非 forward():MTR/wayformer/autobot 的 forward 返回 (pred, loss),
    而 EMP/forecast(Forecast-MAE 系)的 forward 返回原始 net 字典(y_hat/pi)。两类模型都
    实现了 predict() 并统一吐出 {predicted_trajectory, predicted_probability},故用它最稳。

    dump_world=True 时额外把 pred/gt 反变换回 world 坐标(pred_world[K,80,2]、gt_world[80,2]),
    供离线算【成对碰撞几何 / 碰撞检测】——同一逆变换保证同场景各 agent 落同一世界系。"""
    records = []
    for batch in iter_agent_batches(tqdm(loader, desc="  评测(agent)", ncols=90, unit="batch"), support_log):
        batch = move_to_device(batch, device)
        pred = model.predict(batch)
        traj = pred["predicted_trajectory"][..., :2]        # [B,K,80,2]
        prob = pred["predicted_probability"]                # [B,K]
        inp = batch["input_dict"]
        gt = inp["center_gt_trajs"][..., :2]                # [B,80,2]
        gt_mask = inp["center_gt_trajs_mask"].bool()        # [B,80]
        final_idx = inp["center_gt_final_valid_idx"].long() # [B]
        types = inp["center_objects_type"]                  # [B]
        scenes = inp.get("scenario_id")                     # [B] str(每目标所属场景)
        oids = inp.get("center_objects_id")                 # [B] str(该目标 track_id)
        ttc = inp.get("accident_ttc")                       # [B] int(碰撞前 T 秒;仅 TTC 评测有)
        err = torch.norm(traj - gt.unsqueeze(1), dim=-1).cpu().numpy()   # [B,K,80]
        ep = traj.detach().cpu().numpy()                                 # [B,K,80,2] 预测轨迹(算终点 spread 用)
        prob = prob.detach().cpu().numpy()
        gt_mask = gt_mask.cpu().numpy()
        final_idx = final_idx.cpu().numpy()
        types = types.cpu().numpy() if torch.is_tensor(types) else np.asarray(types)
        pred_world = gt_world = None
        if dump_world:
            # 反变换:world = rotate(agent_centric, +heading) + center_xy(正变换的逆;见 base_dataset
            #   transform_trajs_to_center_coords:用 center_xyz[:,0:3] 平移 + heading=center[:,6] 旋 -heading)。
            cw = inp["center_objects_world"]
            cw = cw.detach().cpu().numpy() if torch.is_tensor(cw) else np.asarray(cw)
            cxy = cw[:, 0:2].astype(np.float64); chead = cw[:, 6].astype(np.float64)
            gt_ac = gt.detach().cpu().numpy().astype(np.float64)                     # [B,80,2]
            gt_world = rotate_points_along_z(gt_ac, chead) + cxy[:, None, :]         # [B,80,2]
            Bn, Kn, Tn, _ = ep.shape
            prw = rotate_points_along_z(ep.reshape(Bn, Kn * Tn, 2).astype(np.float64), chead)
            pred_world = prw.reshape(Bn, Kn, Tn, 2) + cxy[:, None, None, :]          # [B,K,80,2]
        for b in range(err.shape[0]):
            fi_b = int(final_idx[b])
            rec = dict(err=err[b], mask=gt_mask[b], score=prob[b],
                       endpoints=ep[b][:, fi_b, :2],   # [K,2] 各模式在最后有效帧的预测终点
                       type=int(types[b]), final_idx=fi_b,
                       scene=str(scenes[b]) if scenes is not None else None,
                       oid=str(oids[b]) if oids is not None else None,
                       ttc=(int(ttc[b]) if ttc is not None else None))
            if dump_world:
                rec["pred_world"] = pred_world[b].astype(np.float32)   # [K,80,2]
                rec["gt_world"] = gt_world[b].astype(np.float32)       # [80,2]
            records.append(rec)
    return records


def collect_smart(model, loader, device, dump_world=False):
    """scene-centric:model.inference(data) → pred_traj/gt[N,80,2]、valid_mask[N,80]。K=1。
    dump_world=True 时把 pred/gt 直接当 world 存(SMART 本就 scene-centric,同窗各 agent 同帧;
    K=1 故 pred_world 形状 [1,80,2])。"""
    from torch_geometric.data import Batch
    records = []

    def _process_one(data):
        data = data.to(device)
        # SMART 的 inference() 本身不做地图 token 化;validation_step 在 inference 前会先跑
        #   match_token_map + sample_pt_pred(造出 pt_token['pt_valid_mask'] 等),再加 av_index 偏移。
        #   这里照搬,否则 smart_decoder 取 pt_valid_mask 会 KeyError。
        data = model.match_token_map(data)
        data = model.sample_pt_pred(data)
        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
        pred = model.inference(data)
        pred_traj = pred["pred_traj"]                       # [N,80,2]
        gt = pred["gt"]                                     # [N,80,2]
        H = model.num_historical_steps
        valid = data["agent"]["valid_mask"][:, H:].cpu().numpy().astype(bool)   # [N,80]
        eval_mask = data["agent"]["valid_mask"][:, H - 1].cpu().numpy().astype(bool)
        atype = data["agent"]["type"].cpu().numpy()
        err = torch.norm(pred_traj - gt, dim=-1).cpu().numpy()  # [N,80]
        # scene-centric 肇事分桶:collate 已把每个 agent 节点的 (object_id, scenario_id) 拼成
        #   两个扁平 list(顺序与 agent 子键 concat 一致),Batch 包装器把它们裹成 [list],取 [0]。
        flat_oid = data["agent_ids"][0] if "agent_ids" in data else None
        flat_scene = data["agent_scene"][0] if "agent_scene" in data else None
        # 窗口化:只对锚点节点计指标(loss 仍由模型对全窗 agent 计算,这里只过滤"报表")。
        flat_anchor = data["agent_is_anchor"][0] if "agent_is_anchor" in data else None
        flat_ttc = data["agent_ttc"][0] if "agent_ttc" in data else None   # 事故 TTC 评测:每节点的碰撞前 T 秒
        pt_np = pred_traj.detach().cpu().numpy() if dump_world else None
        gt_np = gt.detach().cpu().numpy() if dump_world else None
        for i in np.where(eval_mask)[0]:
            if flat_anchor is not None and not bool(flat_anchor[i]):
                continue
            m = valid[i]
            if m.sum() == 0:
                continue
            final_idx = int(np.where(m)[0].max())
            rec = dict(err=err[i][None, :], mask=m, score=np.ones(1, np.float32),
                       type=int(atype[i]) + 1, final_idx=final_idx,  # SMART type 0/1/2 → 1/2/3
                       scene=(str(flat_scene[i]) if flat_scene is not None else None),
                       oid=(str(flat_oid[i]) if flat_oid is not None else None),
                       ttc=(int(flat_ttc[i]) if flat_ttc is not None else None))
            if dump_world:
                rec["pred_world"] = pt_np[i][None].astype(np.float32)   # [1,80,2]
                rec["gt_world"] = gt_np[i].astype(np.float32)           # [80,2]
            records.append(rec)

    for data in tqdm(loader, desc="  评测(smart)", ncols=90, unit="batch"):
        if data is None:            # 整批场景都不可处理(eval 丢弃模式)→ 跳过
            continue
        # 窗口化 eval:collate 返回 list[Batch](逐窗一个 batch)→ 逐窗前向,避免 N 窗口并一次前向 OOM。
        if isinstance(data, list):
            for sub in data:
                _process_one(sub)
        else:
            _process_one(data)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 指标聚合(multimodal 自动从 K 推断:K==1 时 MissRate/brier/top1 留空)
# ─────────────────────────────────────────────────────────────────────────────
def _min_ade(err, mask, upto=None):
    if upto is not None:
        mask = mask.copy()
        mask[upto + 1:] = False
    if mask.sum() == 0:
        return None
    return float(((err * mask).sum(axis=1) / mask.sum()).min())


def aggregate(records):
    if not records:
        raise RuntimeError("没有收集到任何评测目标(records 为空)。")
    acc = {k: [] for k in [
        "ADE1@3s", "ADE1@5s", "ADE1@8s", "FDE1@3s", "FDE1@5s", "FDE1@8s",
        # top1 漏报率(公平口径:只用置信度最高那条 + 存活门控,单/多模态都能比):
        #   top1 在 h 时刻 FDE1@h > MISS_THRESH 记为 miss。区别于下面 best-of-6 的 MissRate。
        "MissRate1@3s", "MissRate1@5s", "MissRate1@8s",
        "minADE6", "minFDE6", "MissRate", "brier_minFDE6", "top1_is_best",
        "p_best", "mode_entropy", "mode_entropy_norm", "endpoint_spread"]}
    n_horizon = {h: 0 for h in HORIZONS}             # 活到 3/5/8s 的目标数(变长下必报)
    per_type = {n: {"ade": [], "fde": []} for n in TYPE_ID2NAME.values()}

    for r in records:
        err, mask, score, fi = r["err"], r["mask"], r["score"], r["final_idx"]
        tname = TYPE_ID2NAME.get(r["type"])
        multimodal = err.shape[0] > 1
        top1 = int(score.argmax())               # 置信度最高那条(K=1 时 =0)

        # ── ② 多模态能力上限(best-of-K,全程到最后有效帧)──────────────────────
        ade_full = _min_ade(err, mask)
        fde_full = float(err[:, fi].min())
        if ade_full is not None:
            acc["minADE6"].append(ade_full)
            if tname:
                per_type[tname]["ade"].append(ade_full)
        acc["minFDE6"].append(fde_full)
        if tname:
            per_type[tname]["fde"].append(fde_full)
        if multimodal:
            acc["MissRate"].append(1.0 if fde_full > MISS_THRESH else 0.0)
            best = int(err[:, fi].argmin())          # 终点最接近 GT 的那条(oracle)
            acc["brier_minFDE6"].append(fde_full + (1.0 - float(score[best])) ** 2)
            acc["top1_is_best"].append(1.0 if top1 == best else 0.0)
            # ── 校准 / 坍缩诊断 ───────────────────────────────────────────────
            acc["p_best"].append(float(score[best]))            # oracle 最优模式的概率质量
            p = np.clip(score.astype(np.float64), 1e-9, None)
            p = p / p.sum()                                     # 防御性归一(score 已近似概率)
            H = float(-(p * np.log(p)).sum())                   # 香农熵(nats)
            acc["mode_entropy"].append(H)
            acc["mode_entropy_norm"].append(H / np.log(len(p)))  # /ln(K) → [0,1]
            ep = r.get("endpoints")                             # [K,2] 各模式预测终点
            if ep is not None and len(ep) > 1:
                d = np.linalg.norm(ep[:, None, :] - ep[None, :, :], axis=-1)
                iu = np.triu_indices(len(ep), k=1)
                acc["endpoint_spread"].append(float(d[iu].mean()))

        # ── ① 公平主指标:只用 top1 那条,在固定 3/5/8s 时刻;只统计活到该时刻的目标 ──
        e1 = err[top1]                               # [80] 置信度最高那条的逐帧误差
        for hname, hidx in HORIZONS.items():
            if not mask[hidx]:                       # 没活到该时刻 → 不计入该时刻
                continue
            n_horizon[hname] += 1
            acc[f"FDE1@{hname}"].append(float(e1[hidx]))
            acc[f"MissRate1@{hname}"].append(1.0 if e1[hidx] > MISS_THRESH else 0.0)  # top1 漏报
            m_up = mask.copy(); m_up[hidx + 1:] = False
            acc[f"ADE1@{hname}"].append(float((e1 * m_up).sum() / m_up.sum()))

    out = {k: (float(np.mean(v)) if len(v) else np.nan) for k, v in acc.items()}
    for hname in HORIZONS:
        out[f"n@{hname}"] = n_horizon[hname]
    for n, d in per_type.items():
        out[f"{n}_minADE"] = float(np.mean(d["ade"])) if d["ade"] else np.nan
        out[f"{n}_minFDE"] = float(np.mean(d["fde"])) if d["fde"] else np.nan
    out["n_targets"] = len(records)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 事故 TTC 聚合:按「碰撞前 T 秒」锚点分组,每组报 horizon=1..8s 的 top1 指标。
#   对锚点 T,碰撞瞬间落在 horizon h=T(对角线);存活门控:只对该 horizon 仍有效的肇事车计入。
#   FDE1@h / ADE1@h / MissRate1@h 口径与主指标一致(top1 那条;只用置信度最高轨迹)。
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_ttc(records):
    by_ttc = {}
    for r in records:
        if r.get("ttc") is not None:
            by_ttc.setdefault(int(r["ttc"]), []).append(r)
    if not by_ttc:
        raise RuntimeError("TTC 评测没收集到任何带 accident_ttc 的记录。")
    rows = []
    for T in sorted(by_ttc):
        recs = by_ttc[T]
        acc = {f"{m}@{h}": [] for m in ("ADE1", "FDE1", "MissRate1") for h in HORIZONS_TTC}
        nh = {h: 0 for h in HORIZONS_TTC}
        for r in recs:
            err, mask, score = r["err"], r["mask"], r["score"]
            e1 = err[int(score.argmax())]              # 置信度最高那条逐帧误差 [80]
            for hname, hidx in HORIZONS_TTC.items():
                if hidx >= len(mask) or not mask[hidx]:  # 没活到该 horizon → 不计入
                    continue
                nh[hname] += 1
                acc[f"FDE1@{hname}"].append(float(e1[hidx]))
                acc[f"MissRate1@{hname}"].append(1.0 if e1[hidx] > MISS_THRESH else 0.0)
                m_up = mask.copy(); m_up[hidx + 1:] = False
                acc[f"ADE1@{hname}"].append(float((e1 * m_up).sum() / m_up.sum()))
        row = {"ttc_before_crash_s": T, "n_collider": len(recs)}
        for k, v in acc.items():
            row[k] = float(np.mean(v)) if v else np.nan
        for h in HORIZONS_TTC:
            row[f"n@{h}"] = nh[h]
        rows.append(row)
    return rows


def ttc_column_order():
    """accident_ttc.csv 列顺序:行标识 + 逐 horizon 的 ADE1 / FDE1 / MissRate1 / n。"""
    cols = ["ttc_before_crash_s", "n_collider"]
    cols += [f"ADE1@{h}" for h in HORIZONS_TTC]
    cols += [f"FDE1@{h}" for h in HORIZONS_TTC]
    cols += [f"MissRate1@{h}" for h in HORIZONS_TTC]
    cols += [f"n@{h}" for h in HORIZONS_TTC]
    return cols


def aggregate_ttc_diagonal(records):
    """事故前 TTC「对角线」聚合 → 宽表一行(dict)。
    对每个锚点 T(撞前 T 秒预测,模型仍吐 8s 轨迹),只用从预测点到撞击瞬间这 [0,T] 段计指标:
      撞击恰落在 gt 帧 hidx=10T-1;ADE{T}=top1 在 [0,hidx] 的平均位移误差,FDE{T}=帧 hidx 终点误差。
    存活门控:肇事车没活到撞击帧(mask[hidx]=0)→ 不计入该 T(故不足 8s 的车只在其可锚的 T 上有值,
      n{T} 随 T 增大递减);超过 8s 的车也只评到 T=8(offsets 固定 1..8)。"""
    by_ttc = {}
    for r in records:
        if r.get("ttc") is not None:
            by_ttc.setdefault(int(r["ttc"]), []).append(r)
    row = {}
    for T in TTC_DIAG_OFFSETS:
        recs = by_ttc.get(T, [])
        hidx = 10 * T - 1
        ade, fde, miss = [], [], []
        for r in recs:
            err, mask, score = r["err"], r["mask"], r["score"]
            if hidx >= len(mask) or not mask[hidx]:   # 没活到撞击瞬间 → 该锚点不计入
                continue
            e1 = err[int(score.argmax())]             # 置信度最高那条逐帧误差 [80]
            m_up = mask.copy(); m_up[hidx + 1:] = False
            ade.append(float((e1 * m_up).sum() / m_up.sum()))
            fde.append(float(e1[hidx]))
            miss.append(1.0 if e1[hidx] > MISS_THRESH else 0.0)
        row[f"ADE{T}"] = float(np.mean(ade)) if ade else np.nan
        row[f"FDE{T}"] = float(np.mean(fde)) if fde else np.nan
        row[f"MissRate{T}"] = float(np.mean(miss)) if miss else np.nan
        row[f"n{T}"] = len(ade)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# 输出文件夹名
# ─────────────────────────────────────────────────────────────────────────────
def derive_eval_tag(cfg):
    """评测「变体标记」,插进 exp 名(在 _epoch= 之前),让不同评测设置产出不同文件名、互不覆盖。
       规则:① 显式 cfg.eval_tag 优先(任意字符串);② 否则 method.kinematic_filter=True → '_kf'(运动学过滤评测)。
       ⚠ 标记必须放在 _epoch= 之前:离线 _accident_metrics.py 的正则是 `(.+_epoch=\\d+)__`,放 epoch 后会漏读。
       对不含 kinematic_filter 的模型(MTR 等)返回空串 → 行为与改动前完全一致。"""
    explicit = cfg.get("eval_tag", None)
    if explicit:
        return "_" + str(explicit).lstrip("_")
    try:
        kf = bool(cfg.get("method", {}).get("kinematic_filter", False))
    except Exception:
        kf = False
    return "_kf" if kf else ""


def derive_run_prefix(cfg, ckpt_path, ckpt_dict):
    """统一评测的输出文件夹名:<exp_name>[<eval_tag>]_epoch=<E>(一次评测 = 一个文件夹,内含汇总表)。"""
    # exp_name:ckpt 路径里 unitraj_ckpt/<EXP>/... 那一段(epoch=NN-val 子目录或扁平都兼容)
    exp_name = checkpoint_experiment_id(ckpt_path, cfg.get("exp_name", "model"))
    # epoch:先从路径里 epoch=NN 抓,抓不到再读 ckpt 字典
    m = re.search(r"epoch=(\d+)", ckpt_path)
    epoch = m.group(1) if m else str(ckpt_dict.get("epoch", "NA"))
    return f"{exp_name}{derive_eval_tag(cfg)}_epoch={epoch}"


def derive_run_name(cfg, ckpt_path, ckpt_dict):
    """单集旧路径(TTC / 手动单集)的文件夹名:<exp_name>_epoch=<E>_<dataset>[_<count>]。"""
    prefix = derive_run_prefix(cfg, ckpt_path, ckpt_dict)
    # dataset 标签:① 若显式给了 run_dataset_label(命令行覆盖),直接用它——
    #   用于消歧不同来源但同末段名的子集(如 deepaccident 与 realaccident 都有 collision_vehicle/normal,
    #   末段相同会导致输出文件夹同名互相覆盖)。不传则保持原行为,向后兼容。
    #   ② 否则用 val_data_path 末段(如 deepaccident 的 collision_vehicle/normal);
    #   ③ 末段是 validation/testing 这类通用名时回退到 data_root 末段(如 waymo)。
    label_override = cfg.get("run_dataset_label", None)
    val_paths = cfg.get("val_data_path") or []
    leaf = os.path.basename(os.path.normpath(str(val_paths[0]))) if len(val_paths) else ""
    if label_override:
        dataset = str(label_override)
    elif leaf.lower() in ("", "validation", "testing", "training", "val", "test", "train"):
        dataset = os.path.basename(os.path.normpath(str(cfg.data_root)))
    else:
        dataset = leaf
    # count:全量(max_val_data_num 为 null)省略;否则附数量
    mvn = cfg.get("max_val_data_num", None)
    try:                       # ListConfig / list → 取第 1 个;标量直接用
        count = mvn[0]
    except (TypeError, KeyError, IndexError):
        count = mvn
    #   - None / 0 / float 1.0(=100% 全量) → 不加后缀
    #   - float ∈ (0,1)  → 比例模式,加 _pctNN(int(0.5) 会变 0,故单独处理)
    #   - int ≥ 1        → 绝对场景数,加 _N
    if count in (None, 0) or (isinstance(count, float) and count == 1.0):
        suffix = ""
    elif isinstance(count, float) and 0 < count < 1.0:
        suffix = f"_pct{int(round(count * 100))}"
    else:
        suffix = f"_{int(count)}"
    return f"{prefix}_{dataset}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# 单个数据集评测:设好该集数据口径 → 建 loader → 收集 → 算指标,返回 ("metrics"|"ttc", DataFrame)。
#   只在内存出 DataFrame,不落逐集 csv(统一评测最终只写一份汇总,见 main);
#   单集旧路径(TTC / 手动单集)由 main 负责把返回的 df 写成独立文件夹。
#   模型/权重由 main() 只加载一次后传入复用。
# ─────────────────────────────────────────────────────────────────────────────
def _save_revision_support(cfg, label, tag, val_set, audit, records):
    if not cfg.get('strict_eval_support', False):
        return
    from pathlib import Path
    base = Path(__file__).resolve().parent
    ckpt = os.environ.get('BENCH_CKPT_PATH') or cfg.get('ckpt_path', '')
    prefix = derive_run_prefix(cfg, str(ckpt), {})
    path = base / 'final_metrics' / '_support_manifests' / f'{prefix}__{label}.{tag}.json'
    write_support_manifest(path, audit, records, len(val_set),
                           OmegaConf.to_container(cfg, resolve=True),
                           [base/'datasets/base_dataset.py', base/'datasets/evaluation_support.py', Path(__file__)])


def _build_eval_loader(cfg, val_set):
    """统一评测 loader。shard_count>1 时只取「场景下标 % shard_count == shard_index」的分片
    —— 多进程场景分片 = 评测版 DDP(推理无需 rank 同步,末尾把各分片的原始 records 合并后只聚合一次,
    数值与单卡完全一致)。shard_count=1(默认)即整集顺序评测。"""
    sc = int(cfg.get("shard_count", 1) or 1)
    si = int(cfg.get("shard_index", 0) or 0)
    sampler = None
    if sc > 1:
        sampler = [i for i in range(len(val_set)) if i % sc == si]
        print(f"  [shard] {si}/{sc}: 本进程取 {len(sampler)}/{len(val_set)} 个场景")
    return DataLoader(val_set, batch_size=cfg["eval_batch_size"], num_workers=cfg.load_num_workers,
                      shuffle=False, drop_last=False, collate_fn=val_set.collate_fn, sampler=sampler)


def collect_dataset_records(cfg, model, device, label, val_data_path, data_root, sample_fraction,
                            window_cap=None, pred_targets=None):
    """建该集 loader(尊重 shard_*)并收集【原始 per-target records】,不做聚合。"""
    cfg["val_data_path"] = [val_data_path]                  # base_dataset 要求列表形式
    cfg["data_root"] = data_root                            # 决定 train_data_path 插值(启动校验 training split)
    cfg["max_val_data_num"] = [sample_fraction]             # null=全量 / 整数N / 小数比例
    cfg["run_dataset_label"] = label
    # SMART 评测每场景随机窗口上限(只大集设、事故集为 None → 建全部 official 窗口)。每次都显式写入,
    #   避免上一集的值在 cfg 复用中残留;非 SMART 模型忽略此键。
    cfg["smart_eval_window_cap"] = window_cap
    # 该集的预测目标口径(逐集可覆盖全局:事故集设 crash → 只评事故对象)。每次显式写,防 cfg 复用残留。
    if pred_targets is not None:
        cfg["prediction_targets"] = pred_targets
    print(f"\n######## [{label}] val={val_data_path}  data_root={data_root}  "
          f"sample_fraction={sample_fraction} ########")
    val_set = build_dataset(cfg, val=True)
    loader = _build_eval_loader(cfg, val_set)
    dw = bool(cfg.get("dump_accident_records", False))
    with torch.no_grad():
        if cfg.method.model_name == "SMART":
            return collect_smart(model, loader, device, dump_world=dw)
        audit = []
        records = collect_agent(model, loader, device, dump_world=dw, support_log=audit)
        _save_revision_support(cfg, label, 'regular', val_set, audit, records)
        return records


def collect_ttc_diagonal_records(cfg, model, device, label, val_data_path, data_root, sample_fraction,
                                 pred_targets=None):
    """「事故前 TTC 对角线」原始 records 收集(尊重 shard_*):临时置 accident_ttc_eval=True
    单独建 loader(撞前 T=1..8 各锚一次,只评肇事车),跑完恢复标志。TTC 本就只锚肇事车,
    prediction_targets 不影响其目标,但仍同步设置以与该集口径一致。"""
    cfg["val_data_path"] = [val_data_path]
    cfg["data_root"] = data_root
    cfg["max_val_data_num"] = [sample_fraction]
    cfg["run_dataset_label"] = label
    if pred_targets is not None:
        cfg["prediction_targets"] = pred_targets
    cfg["accident_ttc_eval"] = True
    dw = bool(cfg.get("dump_accident_records", False))
    print(f"\n######## [TTC·事故前1..8s][{label}] val={val_data_path} ########")
    try:
        val_set = build_dataset(cfg, val=True)
        loader = _build_eval_loader(cfg, val_set)
        with torch.no_grad():
            if cfg.method.model_name == "SMART":
                records = collect_smart(model, loader, device, dump_world=dw)
            else:
                audit = []
                records = collect_agent(model, loader, device, dump_world=dw, support_log=audit)
                _save_revision_support(cfg, label, 'ttc', val_set, audit, records)
    finally:
        cfg["accident_ttc_eval"] = False
    return records


def records_to_metrics_df(records, val_data_path, ttc_eval=False):
    """把【原始 records】聚合成 DataFrame —— 单卡 / 多卡合并共用此函数,保证数值口径一致。
    ttc_eval=True 走旧的 8×8 矩阵聚合(legacy CLI);否则常规分桶 + 单/多模态聚合。"""
    # ── 事故 TTC 滑窗(legacy 矩阵版):只评肇事车,锚点 × horizon 1..8s ──
    if ttc_eval:
        rows = aggregate_ttc(records)
        cols = ttc_column_order()
        df = pd.DataFrame([[r.get(c) for c in cols] for r in rows], columns=cols)
        print(f"  [TTC] 肇事车记录 {len(records)} 条,{len(rows)} 个锚点(碰撞前 T 秒)")
        return "ttc", df, 0
    # 肇事对象分桶:命中肇事 track_id → collider 桶;同场景其余 → non_collider 桶。
    #   查找表为空(如 Waymo)或无命中 → 退化为单行输出。
    lookup = build_collider_lookup([val_data_path] if val_data_path else None)
    n_collider = 0
    for r in records:
        r["is_collider"] = bool(lookup) and str(r.get("oid")) in lookup.get(str(r.get("scene")), ())
        n_collider += int(r["is_collider"])
    if n_collider > 0:
        subsets = [("collider", [r for r in records if r["is_collider"]]),
                   ("non_collider", [r for r in records if not r["is_collider"]])]
        rows = [(name, aggregate(recs)) for name, recs in subsets if recs]
        cols = [c for c in COLUMN_ORDER if c in rows[0][1]]
        df = pd.DataFrame([[name] + [rr[c] for c in cols] for name, rr in rows],
                          columns=["subset"] + cols)
        print(f"  [肇事分桶] collider={n_collider} / "
              f"non_collider={len(records) - n_collider}(肇事标注场景数={len(lookup)})")
    else:
        row = aggregate(records)
        cols = [c for c in COLUMN_ORDER if c in row]
        df = pd.DataFrame([[row[c] for c in cols]], columns=cols)
        print(f"  n_targets={len(records)}（指标列含义见 final_metrics/README.md）")
    return "metrics", df, n_collider


def _dump_accident_records(cfg, records, label, tag):
    """把原始 records(含 pred_world[K,80,2]/gt_world[80,2]/score/mask/ttc/is_collider/scene/oid)
    落盘到 final_metrics/_accident_records/<exp>_epoch=<E>__<label>.<tag>.pkl(tag=regular|ttc)。
    供离线复算轨迹指标与机制分析；`_`前缀目录不参与排行榜目录扫描。"""
    if not cfg.get("dump_accident_records"):
        return
    ckpt = os.environ.get("BENCH_CKPT_PATH") or cfg.get("ckpt_path", "")
    exp = checkpoint_experiment_id(ckpt)
    exp = exp + derive_eval_tag(cfg)                       # filter 开启等变体 → exp 名带标记,records 不覆盖原文件
    mep = re.search(r"epoch=(\d+)", str(ckpt)); ep = mep.group(1) if mep else "NA"
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_metrics", "_accident_records")
    os.makedirs(ddir, exist_ok=True)
    fp = os.path.join(ddir, f"{exp}_epoch={ep}__{label}.{tag}.pkl")
    with open(fp, "wb") as f:
        pickle.dump(records, f)
    print(f"  [dump:{tag}] {len(records)} 条 records -> _accident_records/{os.path.basename(fp)}")


def compute_dataset_df(cfg, model, device, label, val_data_path, data_root, sample_fraction, ttc_eval,
                       window_cap=None, pred_targets=None):
    """单集评测(单卡 / legacy 单集路径):收集 records → 聚合成 DataFrame。"""
    records = collect_dataset_records(cfg, model, device, label, val_data_path, data_root,
                                      sample_fraction, window_cap=window_cap, pred_targets=pred_targets)
    out = records_to_metrics_df(records, val_data_path, ttc_eval=ttc_eval)
    _dump_accident_records(cfg, records, label, "regular")   # records 此时已带 is_collider(分桶后)
    return out


def compute_ttc_diagonal_df(cfg, model, device, label, val_data_path, data_root, sample_fraction,
                            pred_targets=None):
    """单集 TTC 对角线评测:收集 records → 宽表一行 DataFrame(列见 TTC_DIAG_COLUMN_ORDER)。"""
    records = collect_ttc_diagonal_records(cfg, model, device, label, val_data_path, data_root,
                                           sample_fraction, pred_targets=pred_targets)
    _dump_accident_records(cfg, records, label, "ttc")   # 撞前 1..8s 锚点的肇事车 records(含 world 坐标)
    row = aggregate_ttc_diagonal(records)
    df = pd.DataFrame([[row.get(c) for c in TTC_DIAG_COLUMN_ORDER]], columns=TTC_DIAG_COLUMN_ORDER)
    print(f"  [TTC 对角线] 肇事车记录 {len(records)} 条 → 撞前 1..8s 各锚 ADE_T/FDE_T(宽表一行)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def _merge_shard_summaries(base):
    """多卡场景分片评测的合并:把各 final_metrics/<prefix>__shardR/ 里 dump 的【原始 records】
    按数据集跨分片拼起来,只聚合一次(数值与单卡完全一致),写出正式 final_metrics/<prefix>/ 的
    3 张汇总表,然后删除 __shardR 临时目录。无 GPU/ckpt 依赖。"""
    import json, shutil
    fm = os.path.join(base, "final_metrics")
    parts = sorted(glob.glob(os.path.join(fm, "*__shard[0-9]*")))
    if not parts:
        print("[merge] 未找到 __shard* 分片目录,跳过"); return
    prefixes = sorted({re.sub(r"__shard\d+$", "", os.path.basename(p)) for p in parts})
    for prefix in prefixes:
        pdirs = [p for p in parts if re.sub(r"__shard\d+$", "", os.path.basename(p)) == prefix]
        # 数据集顺序:合并各分片 _meta.json 的 label(保持首次出现序)。
        meta, seen = [], set()
        for p in pdirs:
            mp = os.path.join(p, "_meta.json")
            if os.path.isfile(mp):
                for m in json.load(open(mp)):
                    if m["label"] not in seen:
                        seen.add(m["label"]); meta.append(m)
        frames, ttc_frames = [], []
        for m in meta:
            label, vdp = m["label"], m["val_data_path"]
            recs = []
            for p in pdirs:
                fp = os.path.join(p, f"{label}.regular.pkl")
                if os.path.isfile(fp):
                    recs.extend(pickle.load(open(fp, "rb")))
            if recs:
                print(f"  [merge:{label}] regular records {len(recs)} 条(跨 {len(pdirs)} 分片)")
                _, df, _ = records_to_metrics_df(recs, vdp, ttc_eval=False)
                frames.append((label, df))
            if m.get("want_ttc"):
                trecs = []
                for p in pdirs:
                    fp = os.path.join(p, f"{label}.ttc.pkl")
                    if os.path.isfile(fp):
                        trecs.extend(pickle.load(open(fp, "rb")))
                if trecs:
                    row = aggregate_ttc_diagonal(trecs)
                    tdf = pd.DataFrame([[row.get(c) for c in TTC_DIAG_COLUMN_ORDER]],
                                       columns=TTC_DIAG_COLUMN_ORDER)
                    ttc_frames.append((label, tdf))
        out_dir = os.path.join(fm, prefix)
        print(f"\n############ [merge] {prefix} <- {len(pdirs)} 个分片 → {out_dir}/ ############")
        if frames:
            build_eval_summary.build_regular_frames(frames, out_dir)
        if ttc_frames:
            build_eval_summary.build_ttc_wide_frames(ttc_frames, out_dir)
        for p in pdirs:
            shutil.rmtree(p, ignore_errors=True)
        print(f"[merge] 完成(已删 {len(pdirs)} 个 __shard* 临时目录)")


def _orchestrate_multigpu(dev_list, base):
    """自动多卡:本进程当 orchestrator,每张卡起一个【场景分片】子进程
    (`python benchmark_eval.py … devices=[g] +shard_index=r +shard_count=K`),
    跑完在本进程合并(无 GPU 依赖)。子进程从 eval.yaml/继承的环境读 method/ckpt,
    并继承本进程透传的其余命令行覆盖。本进程不加载模型、不占 GPU。"""
    import subprocess
    K = len(dev_list)
    script = os.path.abspath(__file__)
    passthrough = [a for a in _PARENT_ARGV if not a.startswith("devices=")]
    print(f"\n############ 自动多卡:{K} 张卡场景分片 devices={dev_list} ############")
    procs = []
    for r, g in enumerate(dev_list):
        cmd = ([sys.executable, script] + passthrough
               + [f"devices=[{int(g)}]", f"+shard_index={r}", f"+shard_count={K}"])
        print(f"  [launch] shard {r}/{K} -> GPU {g}")
        procs.append(subprocess.Popen(cmd))
    fail = sum(1 for p in procs if p.wait() != 0)
    print(f"\n############ 分片结束(失败 {fail}/{K})→ 合并 ############")
    _merge_shard_summaries(base)


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    # 多卡分片合并模式(merge_shards=true):只读各 __shardR/ 的 records 合并出汇总,不碰 GPU/模型。
    if cfg.get("merge_shards", False):
        _merge_shard_summaries(os.path.dirname(os.path.abspath(__file__)))
        return
    # 单集旧路径(eval_accident_ttc.sh)的命令行数据口径必须在 merge 前捕获:否则 method yaml
    #   自带的 val_data_path/max_val_data_num(如 SMART.yaml=[5000])会在 merge 时覆盖掉命令行值。
    #   统一评测(datasets 循环)模式不依赖它——那里每集会在 merge 后显式设置数据口径。
    _cli_label = cfg.get("run_dataset_label", None)
    _cli_data = {k: cfg.get(k) for k in ("val_data_path", "data_root", "max_val_data_num")}
    # ── 自动多卡:devices 给了多张卡 且 非子进程(无 shard_index)且 非单集旧路径 →
    #     本进程当 orchestrator,逐卡起场景分片子进程后合并、返回。devices 一张 → 走原单卡路径。
    #     与模型无关(任何 method 都适用,数值与单卡一致);单卡评测行为完全不变。
    _dev = list(cfg.get("devices", [0]) or [0])
    if cfg.get('strict_eval_support', False) and (len(_dev) != 1 or int(cfg.get('shard_count', 1) or 1) != 1):
        raise ValueError('Revision support protocol requires single-device, unsharded evaluation')
    if (_cli_label is None and cfg.get("shard_index", None) is None
            and not cfg.get("merge_shards", False) and len(_dev) > 1):
        _orchestrate_multigpu(_dev, os.path.dirname(os.path.abspath(__file__)))
        return
    # 把 method yaml 合并进来(取其 eval_batch_size / 模型结构 / 锚点 + 特征预处理键,保证与训练同口径)。
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg["eval"] = True
    # 读取方式强制 onthefly:CPU 直读原始 ScenarioNet pkl,不生成/不依赖 h5 缓存。
    cfg["onthefly"] = True
    cfg["use_cache"] = False
    # SMART 评测强制关闭 fast on-the-fly(对历史稀疏 agent 会算错 token 锚点 → 误差爆炸)。
    cfg["smart_fast_onthefly"] = False
    if cfg.get('strict_eval_support', False):
        if canonical_model_key(cfg.method.model_name) not in ('wayformer', 'recon_lp', 'recon'):
            raise ValueError('Revision support protocol currently validated for Wayformer-family datasets only')
        if not str(cfg.get('eval_tag', '')).lstrip('_').startswith('revision_'):
            raise ValueError('Strict evaluation needs a fresh revision_* eval_tag to preserve historical outputs')
        if int(cfg.get('shard_count', 1) or 1) != 1 or len(_dev) != 1:
            raise ValueError('Revision support protocol currently requires single-device evaluation')
        if not cfg.get('dump_accident_records', False):
            raise ValueError('Strict evaluation requires prediction record dumps')

    # ── 设备:可见 GPU 已在 import 前由 _bootstrap_visible_gpu() 锁定(见文件顶部)──────────
    #   此处不能再设 CUDA_VISIBLE_DEVICES——CUDA 早在 import datasets/models 时已初始化,改了无效。
    #   进程内一律 cuda:0 = 选中的物理卡;顺带规避 BeTopNet 等硬编码 .cuda() 落 cuda:0 的 device mismatch。
    dev = cfg.get("devices", [0])
    try:                       # ListConfig / list / tuple → 取第 1 张卡(仅用于显示/校验)
        gpu_id = int(dev[0])
    except (TypeError, KeyError, IndexError):
        gpu_id = int(dev)
    if _BOOTSTRAP_GPU is not None and int(_BOOTSTRAP_GPU) != gpu_id:
        print(f"  [设备] ⚠ eval.yaml/命令行 devices={gpu_id},但启动时锁定的是物理 GPU {_BOOTSTRAP_GPU}"
              f"(以锁定值为准;请确认 devices 写法)")

    set_seed(cfg.seed)
    device = torch.device("cuda:0")   # 物理卡已映射为唯一可见的 cuda:0
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("medium")
    print(f"  [数据] 读取方式=onthefly(原始 pkl 直读,不生成/不使用 h5 缓存)")
    print(f"  [设备] 物理 GPU {_BOOTSTRAP_GPU if _BOOTSTRAP_GPU is not None else gpu_id} "
          f"→ 进程内 cuda:0(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','未设')})")

    # ── 权重 ──────────────────────────────────────────────────────────────────
    # ckpt_path 优先取 __main__ 里从 argv 摘出来的值(绕开 hydra 对 '=' 的语法限制)。
    ckpt_path = os.environ.get("BENCH_CKPT_PATH") or cfg.get("ckpt_path", None)
    if not ckpt_path:
        raise SystemExit("必须指定 ckpt_path(在 eval.yaml 或命令行 ckpt_path=...)。")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path)

    # ── 模型只加载一次,后续逐集复用 ───────────────────────────────────────────
    print(f"\n========== eval: method={cfg.method.model_name} | ckpt={ckpt_path} ==========")
    model = build_model(cfg)
    ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt_dict.get("state_dict", ckpt_dict)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)} (showing 3): {list(missing)[:3]}")
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)} (showing 3): {list(unexpected)[:3]}")
    model.to(device).eval()
    # SMART 的 self.noise 评测强制关掉(确定性 + 用最佳匹配 token,结果可复现)。
    if cfg.method.model_name == "SMART" and hasattr(model, "noise"):
        model.noise = False

    ttc_eval = bool(cfg.get("accident_ttc_eval", False))

    # ── 解析要评测的数据集清单 ─────────────────────────────────────────────────
    #   ① 单集旧路径:命令行给了 +run_dataset_label(事故 TTC 脚本 eval_accident_ttc.sh 用)
    #      → 只评 cfg 顶层 val_data_path/data_root/max_val_data_num 指定的那一个集。
    #   ② 统一评测(默认):循环 eval.yaml 的 datasets 块,逐集评测。
    legacy_label = _cli_label
    if legacy_label is not None:
        mvn = _cli_data.get("max_val_data_num") or [None]
        try:
            sf = list(mvn)[0]
        except (TypeError, IndexError):
            sf = mvn
        ds_items = [(str(legacy_label),
                     str(list(_cli_data["val_data_path"])[0]),
                     str(_cli_data["data_root"]), sf, False, None, None)]
    else:
        datasets = cfg.get("datasets", None)
        if not datasets:
            raise SystemExit("eval.yaml 缺少 datasets 块(统一评测的数据集清单)。")
        # 全局预测口径(逐集可用 prediction_targets 覆盖;事故集设 crash = 只评事故对象)。
        g_pred = cfg.get("prediction_targets", "official")
        ds_items = [(str(label), str(ds["val_data_path"]), str(ds["data_root"]),
                     ds.get("sample_fraction", None), bool(ds.get("ttc_eval", False)),
                     ds.get("smart_eval_window_cap", None),
                     str(ds.get("prediction_targets", g_pred)))
                    for label, ds in datasets.items()]

    base = os.path.dirname(os.path.abspath(__file__))
    # ── 多卡场景分片:shard_count>1 时本进程只评 1/shard_count 的场景,且【只收集原始 records 落盘】,
    #     聚合交给 merge_shards 统一做(避免各分片先算均值再平均把 missrate/best-of-6 算错)。──────
    is_shard = int(cfg.get("shard_count", 1) or 1) > 1 and legacy_label is None
    shard_dir, shard_meta = None, []
    if is_shard:
        si = int(cfg.get("shard_index", 0) or 0)
        shard_dir = os.path.join(base, "final_metrics",
                                 derive_run_prefix(cfg, ckpt_path, ckpt_dict) + f"__shard{si}")
        os.makedirs(shard_dir, exist_ok=True)
    # ── 逐集评测 ───────────────────────────────────────────────────────────────
    frames = []   # [(标签, DataFrame)] —— 内存收集,统一评测最后只写一份汇总
    ttc_frames = []  # [(标签, 宽表一行 DataFrame)] —— 碰撞集的「事故前 1..8s」对角线指标
    n_ok = 0
    for label, val_data_path, data_root, sample_fraction, want_ttc, window_cap, pred_targets in ds_items:
        try:
            if is_shard:
                # 分片:收原始 records 落盘(常规 + 可选 TTC 对角线),不聚合。
                recs = collect_dataset_records(cfg, model, device, label, val_data_path,
                                               data_root, sample_fraction, window_cap=window_cap,
                                               pred_targets=pred_targets)
                with open(os.path.join(shard_dir, f"{label}.regular.pkl"), "wb") as f:
                    pickle.dump(recs, f)
                m = {"label": label, "val_data_path": val_data_path, "want_ttc": False}
                if not ttc_eval and want_ttc:
                    try:
                        trecs = collect_ttc_diagonal_records(cfg, model, device, label,
                                                             val_data_path, data_root, sample_fraction,
                                                             pred_targets=pred_targets)
                        with open(os.path.join(shard_dir, f"{label}.ttc.pkl"), "wb") as f:
                            pickle.dump(trecs, f)
                        m["want_ttc"] = True
                    except Exception as e:
                        print(f"  [{label}] !! TTC 分片收集失败,跳过:{type(e).__name__}: {e}")
                shard_meta.append(m)
                n_ok += 1
                continue

            kind, df, _n_collision = compute_dataset_df(cfg, model, device, label,
                                          val_data_path, data_root, sample_fraction, ttc_eval,
                                          window_cap=window_cap, pred_targets=pred_targets)
            n_ok += 1
            if legacy_label is not None:
                # 单集旧路径(eval_accident_ttc.sh / 手动单集):写独立文件夹(向后兼容,脚本靠 "写入 ->" 抓路径)。
                run_name = derive_run_name(cfg, ckpt_path, ckpt_dict) + ("_ttc" if kind == "ttc" else "")
                out_dir = os.path.join(base, "final_metrics", run_name)
                os.makedirs(out_dir, exist_ok=True)
                fname = "accident_ttc.csv" if kind == "ttc" else "metrics.csv"
                csv_path = os.path.join(out_dir, fname)
                df.to_csv(csv_path, index=False, float_format="%.4f")
                print(f"  写入 -> {csv_path}")
                with pd.option_context("display.max_columns", None, "display.width", 260):
                    print(df.to_string(index=False))
            else:
                frames.append((label, df))
                # 碰撞集(eval.yaml 该集 ttc_eval: true)额外跑「事故前 TTC 对角线」,并入同一次评测。
                #   只对有 collision_pairs 锚点的车-车碰撞集开;单车/normal/validation 无锚点,不开。
                if not ttc_eval and want_ttc:
                    try:
                        ttc_df = compute_ttc_diagonal_df(cfg, model, device, label,
                                                         val_data_path, data_root, sample_fraction,
                                                         pred_targets=pred_targets)
                        ttc_frames.append((label, ttc_df))
                    except Exception as e:
                        if cfg.get('strict_eval_support', False):
                            raise
                        print(f"  [{label}] !! TTC 事故前评测失败,跳过:{type(e).__name__}: {e}")
        except Exception as e:
            if cfg.get('strict_eval_support', False):
                raise
            print(f"  [{label}] !! 评测失败,跳过:{type(e).__name__}: {e}")

    # ── 分片模式:写 _meta.json(数据集顺序 + 路径 + 是否有 TTC)后直接返回,聚合交给 merge_shards ──
    if is_shard:
        import json
        with open(os.path.join(shard_dir, "_meta.json"), "w") as f:
            json.dump(shard_meta, f, ensure_ascii=False)
        print(f"\n############ [shard {cfg.get('shard_index')}] records 收集完成 "
              f"-> {shard_dir}/  ({n_ok}/{len(ds_items)} 集)############")
        return

    # ── 统一评测:全部跑完 → 只写一份汇总到单个文件夹 final_metrics/<exp>_epoch=<E>/ ──────
    #   out_dir_suffix(可选):给汇总目录名加后缀,多卡并行时每个进程写各自子集到独立目录
    #   (final_metrics/<exp>_epoch=<E><suffix>/),跑完由外部脚本把各 suffix 目录的行合并。
    if legacy_label is None and frames:
        suffix = str(cfg.get("out_dir_suffix", "") or "")
        run_dir = os.path.join(base, "final_metrics",
                               derive_run_prefix(cfg, ckpt_path, ckpt_dict) + suffix)
        print(f"\n############ 汇总 -> {run_dir}/ ############")
        build_eval_summary.build_regular_frames(frames, run_dir)
        if ttc_frames:
            build_eval_summary.build_ttc_wide_frames(ttc_frames, run_dir)
    print(f"\n############ DONE:成功 {n_ok}/{len(ds_items)} 个数据集 ############")


if __name__ == "__main__":
    # 在 hydra 解析前,把 ckpt_path=... 从 argv 摘出来塞进环境变量。
    # Waymo 的 ckpt 名里带 '='(epoch=58 / brier_fde=3.11),hydra 的 override 语法会报错;
    # 摘出来后就能不加任何引号直接写 ckpt_path=unitraj_ckpt/.../brier_fde=3.11.ckpt。
    _kept = []
    for _a in sys.argv[1:]:
        if _a.startswith("ckpt_path="):
            os.environ["BENCH_CKPT_PATH"] = _a[len("ckpt_path="):].strip("'\"")
        else:
            _kept.append(_a)
    sys.argv = [sys.argv[0]] + _kept
    _PARENT_ARGV = list(_kept)   # 自动多卡时转发给子进程(devices 会被替换为单卡)
    main()
