"""
Build ScenarioNet map_features from CARLA OpenDRIVE (.xodr) files.

Output 与 Waymo Open Dataset 的 ScenarioNet 表示对齐:
- `LANE_SURFACE_STREET`            : 驾驶车道中心线 + 拓扑(entry/exit/neighbor)
- `ROAD_EDGE_BOUNDARY`             : 路缘(最外侧 driving lane 的外边)
- `ROAD_LINE_{BROKEN,SOLID}_*_{WHITE,YELLOW}` : 车道分隔线(来自 roadMark)

坐标系约定:OpenDRIVE 右手系 → CARLA 世界系(y 取负)。
"""

from __future__ import annotations

import math
import os
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── 常量 ──────────────────────────────────────────────────────────────────────
_DRIVING_TYPES = {'driving', 'bidirectional', 'entry', 'exit',
                  'onRamp', 'offRamp', 'connectingRamp'}
_SAMPLE_DENSITY = 2.0      # 2 个采样点 / 1 m,对齐 Waymo 约 0.5 m

# roadMark (type, color) → Waymo ROAD_LINE 子类型
_ROADMARK_TO_WAYMO = {
    ('broken',         'yellow'): 'ROAD_LINE_BROKEN_SINGLE_YELLOW',
    ('broken',         'white'):  'ROAD_LINE_BROKEN_SINGLE_WHITE',
    ('solid',          'yellow'): 'ROAD_LINE_SOLID_SINGLE_YELLOW',
    ('solid',          'white'):  'ROAD_LINE_SOLID_SINGLE_WHITE',
    ('solid solid',    'yellow'): 'ROAD_LINE_SOLID_DOUBLE_YELLOW',
    ('solid solid',    'white'):  'ROAD_LINE_SOLID_DOUBLE_WHITE',
    ('broken broken',  'yellow'): 'ROAD_LINE_PASSING_DOUBLE_YELLOW',
    ('solid broken',   'yellow'): 'ROAD_LINE_PASSING_DOUBLE_YELLOW',
    ('broken solid',   'yellow'): 'ROAD_LINE_PASSING_DOUBLE_YELLOW',
}


# ── 几何采样(OpenDRIVE planView 段) ───────────────────────────────────────
def _eval_poly(coeffs, s):
    a, b, c, d = coeffs
    return a + s * (b + s * (c + s * d))


def _sample_geometry(geom_el, n_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """采样一个 <geometry> 元素,返回 (xy[N,2] in OpenDRIVE, headings[N] in radians)。"""
    x0  = float(geom_el.get('x'))
    y0  = float(geom_el.get('y'))
    hdg = float(geom_el.get('hdg'))
    length = float(geom_el.get('length'))
    ss = np.linspace(0.0, length, n_pts)
    cos_h, sin_h = math.cos(hdg), math.sin(hdg)

    arc_el   = geom_el.find('arc')
    pp3_el   = geom_el.find('paramPoly3')
    poly3_el = geom_el.find('poly3')

    if arc_el is not None:
        k = float(arc_el.get('curvature'))
        if abs(k) < 1e-9:
            xs = x0 + ss * cos_h
            ys = y0 + ss * sin_h
            heads = np.full(n_pts, hdg)
        else:
            R  = 1.0 / k
            cx = x0 - R * sin_h
            cy = y0 + R * cos_h
            ang = hdg + ss * k
            xs = cx + R * np.sin(ang)
            ys = cy - R * np.cos(ang)
            heads = ang
    elif pp3_el is not None:
        coU = tuple(float(pp3_el.get(k, 0)) for k in ('aU', 'bU', 'cU', 'dU'))
        coV = tuple(float(pp3_el.get(k, 0)) for k in ('aV', 'bV', 'cV', 'dV'))
        u  = _eval_poly(coU, ss)
        v  = _eval_poly(coV, ss)
        xs = x0 + u * cos_h - v * sin_h
        ys = y0 + u * sin_h + v * cos_h
        du = coU[1] + 2 * coU[2] * ss + 3 * coU[3] * ss * ss
        dv = coV[1] + 2 * coV[2] * ss + 3 * coV[3] * ss * ss
        heads = hdg + np.arctan2(dv, du)
    elif poly3_el is not None:
        co = tuple(float(poly3_el.get(k, 0)) for k in ('a', 'b', 'c', 'd'))
        v  = _eval_poly(co, ss)
        xs = x0 + ss * cos_h - v * sin_h
        ys = y0 + ss * sin_h + v * cos_h
        dv = co[1] + 2 * co[2] * ss + 3 * co[3] * ss * ss
        heads = hdg + np.arctan2(dv, np.ones_like(ss))
    else:
        xs = x0 + ss * cos_h
        ys = y0 + ss * sin_h
        heads = np.full(n_pts, hdg)
    return np.stack([xs, ys], axis=1), heads


def _eval_lane_width(lane_el, s_relative: float) -> float:
    """评估 lane <width> 多项式在 s_relative(相对 laneSection 起点)。"""
    widths = sorted(lane_el.findall('width'),
                    key=lambda w: float(w.get('sOffset', 0)))
    if not widths:
        return 3.5
    chosen = widths[0]
    for w in widths:
        if float(w.get('sOffset', 0)) <= s_relative:
            chosen = w
    a = float(chosen.get('a', 0))
    b = float(chosen.get('b', 0))
    c = float(chosen.get('c', 0))
    d = float(chosen.get('d', 0))
    s_w = max(0.0, s_relative - float(chosen.get('sOffset', 0)))
    return float(max(0.0, _eval_poly((a, b, c, d), s_w)))


def _transform_odr_xy(pts_odr: np.ndarray, flip_y: bool) -> np.ndarray:
    """Transform OpenDRIVE XY, optionally applying CARLA's y-axis flip."""
    out = pts_odr.copy()
    if flip_y:
        out[:, 1] = -out[:, 1]
    return out


def _to_3d(xy: np.ndarray) -> np.ndarray:
    """(N, 2) → (N, 3) float32(z=0)。"""
    z = np.zeros((len(xy), 1), dtype=np.float32)
    return np.concatenate([xy.astype(np.float32), z], axis=1)


def _remove_consecutive_duplicate_points(points: np.ndarray) -> np.ndarray:
    """Remove zero-length consecutive segments without changing feature semantics."""
    points = np.asarray(points)
    if len(points) <= 1:
        return points
    keep = np.concatenate(
        ([True], np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) > 1e-6)
    )
    return points[keep]


# ── road 解析 ────────────────────────────────────────────────────────────────
@dataclass
class LaneRecord:
    """OpenDRIVE 中一条 lane 的完整中间表示。"""
    road_id: str
    section_idx: int
    lane_id: int               # signed:>0 左,<0 右,0 中心(不可驾驶)
    lane_type: str
    is_junction: bool
    center_xy: np.ndarray      # (N, 2) CARLA 世界
    outer_xy: np.ndarray       # (N, 2) lane 外侧边(远离 id=0 的一侧)
    section_s: np.ndarray      # (N,) laneSection-local OpenDRIVE s
    road_marks: List[dict] = field(default_factory=list)
    predecessor_lane_id: Optional[int] = None
    successor_lane_id: Optional[int] = None
    feature_id: Optional[int] = None    # 仅 driving lane 才赋值

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.road_id, self.section_idx, self.lane_id)


@dataclass
class RoadRecord:
    road_id: str
    junction_id: str           # '-1' = regular road
    is_junction: bool
    lanes: List[LaneRecord]    # 全部 lane(driving + sidewalk + shoulder + ...)
    predecessor: Optional[tuple]  # (elementType, elementId, contactPoint)
    successor:   Optional[tuple]
    total_length: float

    @property
    def driving_lanes(self) -> List[LaneRecord]:
        return [lr for lr in self.lanes if lr.lane_type in _DRIVING_TYPES]


def _sample_road_refline(road_el) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """采样 road 的 ref line,返回 (xy(N,2) odr, headings(N,), s(N,), total_length)。"""
    plan_view = road_el.find('planView')
    if plan_view is None:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0), 0.0

    xy_segs, head_segs, s_segs = [], [], []
    s_acc = 0.0
    for geom in plan_view.findall('geometry'):
        glen = float(geom.get('length', 0))
        if glen < 0.05:
            continue
        n = max(3, int(math.ceil(glen * _SAMPLE_DENSITY)) + 1)
        xy, heads = _sample_geometry(geom, n)
        s_seg = np.linspace(s_acc, s_acc + glen, n)
        xy_segs.append(xy)
        head_segs.append(heads)
        s_segs.append(s_seg)
        s_acc += glen

    if not xy_segs:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0), 0.0
    return (np.vstack(xy_segs), np.concatenate(head_segs),
            np.concatenate(s_segs), s_acc)


def _slice_refline_section(
    ref_xy: np.ndarray,
    ref_heads: np.ndarray,
    ref_s: np.ndarray,
    s_start: float,
    s_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a road reference line to exact laneSection boundaries."""
    unique_s, unique_indices = np.unique(ref_s, return_index=True)
    unique_xy = ref_xy[unique_indices]
    unique_heads = np.unwrap(ref_heads[unique_indices])
    if len(unique_s) < 2 or s_end - s_start <= 1e-8:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    inner = (unique_s > s_start + 1e-8) & (unique_s < s_end - 1e-8)
    sample_s = np.concatenate(([s_start], unique_s[inner], [s_end]))
    sample_xy = np.column_stack([
        np.interp(sample_s, unique_s, unique_xy[:, axis])
        for axis in range(2)
    ])
    sample_heads = np.interp(sample_s, unique_s, unique_heads)
    return sample_xy, sample_heads, sample_s - s_start


def _parse_road(road_el, flip_y: bool = True) -> Optional[RoadRecord]:
    """解析一个 <road>,返回 RoadRecord 或 None。"""
    road_id = road_el.get('id')
    if road_id is None:
        return None
    junction_id = road_el.get('junction', '-1')
    is_junction = (junction_id != '-1')

    ref_xy, ref_heads, ref_s, total_len = _sample_road_refline(road_el)
    if len(ref_xy) < 2:
        return None

    lanes_el = road_el.find('lanes')
    if lanes_el is None:
        return None

    lane_records: List[LaneRecord] = []
    sections = lanes_el.findall('laneSection')
    # 注意:section 按 s 升序;最后一个 section 的 s_end = total_len
    section_starts = [float(s.get('s', 0)) for s in sections]
    section_ends   = section_starts[1:] + [total_len]

    for sec_idx, (ls_el, s_start, s_end) in enumerate(
            zip(sections, section_starts, section_ends)):

        sec_xy, sec_heads, sec_s_loc = _slice_refline_section(
            ref_xy, ref_heads, ref_s, s_start, s_end
        )
        if len(sec_xy) < 2:
            continue

        # ref line 各点的左侧法向(odr 系)
        perp_x = -np.sin(sec_heads)
        perp_y =  np.cos(sec_heads)

        # 先处理 <center>(id=0,无宽度,承载中心分隔 roadMark)
        center_el = ls_el.find('center')
        if center_el is not None:
            for lane_el in center_el.findall('lane'):
                rms = []
                for rm in lane_el.findall('roadMark'):
                    rms.append({
                        'type':   rm.get('type',   'none'),
                        'color':  rm.get('color',  'white'),
                        'weight': rm.get('weight', 'standard'),
                        'sOffset': float(rm.get('sOffset', 0)),
                    })
                center_xy_carla = _transform_odr_xy(sec_xy, flip_y)
                lane_records.append(LaneRecord(
                    road_id=road_id, section_idx=sec_idx,
                    lane_id=int(lane_el.get('id', 0)),
                    lane_type=lane_el.get('type', 'none'),
                    is_junction=is_junction,
                    center_xy=center_xy_carla,
                    outer_xy=center_xy_carla,    # 中心 lane:outer = ref line 自身
                    section_s=sec_s_loc.copy(),
                    road_marks=rms,
                ))

        for side_tag, sign in (('left', +1), ('right', -1)):
            side_el = ls_el.find(side_tag)
            if side_el is None:
                continue

            # 从靠近中心到外侧排序(left: id 1,2,3...;right: id -1,-2,-3...)
            lane_elems = sorted(side_el.findall('lane'),
                                key=lambda l: sign * int(l.get('id', 0)))
            # 沿 ref 累积的"距 ref 线偏移量"(逐点),sign 之外的部分
            cumulative = np.zeros(len(sec_xy), dtype=np.float64)
            for lane_el in lane_elems:
                lane_id = int(lane_el.get('id', 0))
                ltype   = lane_el.get('type', 'none')
                lane_link_el = lane_el.find('link')
                predecessor_lane_id = successor_lane_id = None
                if lane_link_el is not None:
                    predecessor_el = lane_link_el.find('predecessor')
                    successor_el = lane_link_el.find('successor')
                    if predecessor_el is not None and predecessor_el.get('id') is not None:
                        predecessor_lane_id = int(predecessor_el.get('id'))
                    if successor_el is not None and successor_el.get('id') is not None:
                        successor_lane_id = int(successor_el.get('id'))
                widths  = np.array([_eval_lane_width(lane_el, s)
                                    for s in sec_s_loc])

                center_off = cumulative + widths / 2.0
                outer_off  = cumulative + widths

                cx = sec_xy[:, 0] + sign * center_off * perp_x
                cy = sec_xy[:, 1] + sign * center_off * perp_y
                ox = sec_xy[:, 0] + sign * outer_off  * perp_x
                oy = sec_xy[:, 1] + sign * outer_off  * perp_y

                center_xy_odr = np.stack([cx, cy], axis=1)
                outer_xy_odr  = np.stack([ox, oy], axis=1)

                rms = []
                for rm in lane_el.findall('roadMark'):
                    rms.append({
                        'type':   rm.get('type',   'none'),
                        'color':  rm.get('color',  'white'),
                        'weight': rm.get('weight', 'standard'),
                        'sOffset': float(rm.get('sOffset', 0)),
                    })

                lane_records.append(LaneRecord(
                    road_id=road_id, section_idx=sec_idx,
                    lane_id=lane_id, lane_type=ltype,
                    is_junction=is_junction,
                    center_xy=_transform_odr_xy(center_xy_odr, flip_y),
                    outer_xy=_transform_odr_xy(outer_xy_odr, flip_y),
                    section_s=sec_s_loc.copy(),
                    road_marks=rms,
                    predecessor_lane_id=predecessor_lane_id,
                    successor_lane_id=successor_lane_id,
                ))
                cumulative = outer_off

    link_el = road_el.find('link')
    pred = succ = None
    if link_el is not None:
        pe = link_el.find('predecessor')
        if pe is not None:
            pred = (pe.get('elementType'), pe.get('elementId'),
                    pe.get('contactPoint', 'start'))
        se = link_el.find('successor')
        if se is not None:
            succ = (se.get('elementType'), se.get('elementId'),
                    se.get('contactPoint', 'start'))

    return RoadRecord(
        road_id=road_id, junction_id=junction_id, is_junction=is_junction,
        lanes=lane_records, predecessor=pred, successor=succ,
        total_length=total_len,
    )


# ── junction / 拓扑解析 ─────────────────────────────────────────────────────
def _parse_junctions(root) -> Dict[str, List[dict]]:
    """{junction_id: [{incoming, connecting, contact_point, lane_links:[(from,to)]}]}"""
    out: Dict[str, List[dict]] = {}
    for j in root.findall('.//junction'):
        jid = j.get('id')
        if jid is None:
            continue
        conns = []
        for c in j.findall('connection'):
            ll = []
            for x in c.findall('laneLink'):
                try:
                    ll.append((int(x.get('from')), int(x.get('to'))))
                except (TypeError, ValueError):
                    continue
            conns.append({
                'incoming_road':   c.get('incomingRoad'),
                'connecting_road': c.get('connectingRoad'),
                'contact_point':   c.get('contactPoint', 'start'),
                'lane_links':      ll,
            })
        out[jid] = conns
    return out


def _build_lane_links(
    roads: Dict[str, RoadRecord],
    junctions: Dict[str, List[dict]],
    use_lane_link_ids: bool = False,
) -> Dict[Tuple[str, int, int], Dict[str, List]]:
    """构建 lane-level 拓扑。

    返回:{lane_key: {'predecessors': [lane_key], 'successors': [lane_key]}}

    拓扑来源:
    1. 同一 road 内部 section_idx 之间(section i 末尾接 section i+1 起点,同 lane_id)
    2. road.predecessor / road.successor 链 → 接对方 road 同 lane_id 的对应 section
    3. junction.connection.lane_link → incoming_road 末/首端 lane → connecting_road 起/末端 lane
    """
    links: Dict = defaultdict(lambda: {'predecessors': [], 'successors': []})

    def _add(a, b):
        links[a]['successors'].append(b)
        links[b]['predecessors'].append(a)

    def _section_bounds(road: RoadRecord):
        """返回 (first_section_idx, last_section_idx)。"""
        idxs = sorted({lr.section_idx for lr in road.lanes})
        return (idxs[0] if idxs else 0, idxs[-1] if idxs else 0)

    def _find_lane(road: RoadRecord, sec_idx: int, lane_id: int):
        for lr in road.lanes:
            if lr.section_idx == sec_idx and lr.lane_id == lane_id:
                return lr
        return None

    # 1. road 内 section 间(同 lane_id)
    for road in roads.values():
        if len(road.lanes) == 0:
            continue
        sec_ids = sorted({lr.section_idx for lr in road.lanes})
        for a, b in zip(sec_ids[:-1], sec_ids[1:]):
            for lr_a in road.lanes:
                if lr_a.section_idx != a:
                    continue
                next_lane_id = (
                    lr_a.successor_lane_id
                    if use_lane_link_ids and lr_a.successor_lane_id is not None
                    else lr_a.lane_id
                )
                lr_b = _find_lane(road, b, next_lane_id)
                if lr_b and lr_a.lane_type == lr_b.lane_type:
                    _add(lr_a.key, lr_b.key)

    # 2. road 间 link(elementType='road')
    for road in roads.values():
        first_sec, last_sec = _section_bounds(road)
        for link, my_sec, is_successor in (
                (road.successor,   last_sec,  True),
                (road.predecessor, first_sec, False)):
            if link is None:
                continue
            etype, eid, cp = link
            if etype != 'road' or eid not in roads:
                continue
            other = roads[eid]
            other_first, other_last = _section_bounds(other)
            other_sec = other_first if cp == 'start' else other_last
            for lr_a in road.lanes:
                if lr_a.section_idx != my_sec:
                    continue
                linked_lane_id = lr_a.lane_id
                if use_lane_link_ids:
                    candidate = (
                        lr_a.successor_lane_id if is_successor
                        else lr_a.predecessor_lane_id
                    )
                    if candidate is not None:
                        linked_lane_id = candidate
                lr_b = _find_lane(other, other_sec, linked_lane_id)
                if lr_b is None or lr_a.lane_type != lr_b.lane_type:
                    continue
                if is_successor:
                    _add(lr_a.key, lr_b.key)
                else:
                    _add(lr_b.key, lr_a.key)

    # 3. junction.connection
    for jid, conns in junctions.items():
        for c in conns:
            in_id = c['incoming_road']
            conn_id = c['connecting_road']
            if in_id not in roads or conn_id not in roads:
                continue
            in_road, conn_road = roads[in_id], roads[conn_id]
            in_first, in_last = _section_bounds(in_road)
            conn_first, conn_last = _section_bounds(conn_road)

            # incoming_road 与 connection 接触面:
            # contact_point 描述的是 *connecting_road* 在 junction 中的接触端,
            # 但 OpenDRIVE 中默认 incoming_road 在 junction 的 "出去" 一侧(succ)
            # CARLA 实践:incoming 端取 last(进入 junction 那头),connecting 端按 cp。
            in_sec = in_last
            conn_sec = conn_first if c['contact_point'] == 'start' else conn_last

            for from_lid, to_lid in c['lane_links']:
                lr_a = _find_lane(in_road,   in_sec,   from_lid)
                lr_b = _find_lane(conn_road, conn_sec, to_lid)
                if lr_a is None or lr_b is None:
                    continue
                _add(lr_a.key, lr_b.key)

    # 去重
    for k in links:
        links[k]['predecessors'] = list(dict.fromkeys(links[k]['predecessors']))
        links[k]['successors']   = list(dict.fromkeys(links[k]['successors']))
    return dict(links)


# ── Waymo feature 输出 ───────────────────────────────────────────────────────
def _neighbor_lane(
    road: RoadRecord,
    lr: LaneRecord,
    side: str,
    orient_by_travel_direction: bool = False,
) -> Optional[LaneRecord]:
    """同向相邻 driving lane(必须 lane_id 同号,跨过 id=0 视为对向)。

    side='left'  → driver 视角的左侧;side='right' → 右侧。
    """
    sign = int(np.sign(lr.lane_id))
    if sign == 0:
        return None
    if orient_by_travel_direction:
        # OpenDRIVE:负 lane 沿 +s 行驶,正 lane 沿 -s 行驶。
        # 驾驶员视角左侧:负 lane 靠中心(id+1),正 lane 靠中心(id-1)。
        delta = -sign if side == 'left' else sign
    else:
        delta = +1 if side == 'left' else -1
    target_id = lr.lane_id + delta
    if int(np.sign(target_id)) != sign:
        return None  # 跨过中心 → 是对向车道
    while abs(target_id) <= 10:
        for other in road.lanes:
            if other.section_idx == lr.section_idx and other.lane_id == target_id:
                if other.lane_type in _DRIVING_TYPES:
                    return other
                # 间隔了 shoulder 等,继续向外一格
                break
        target_id += delta
        if int(np.sign(target_id)) != sign:
            return None
    return None


def _make_lane_feature(
    lr: LaneRecord,
    entry_ids: List[int],
    exit_ids: List[int],
    left_id: Optional[int],
    right_id: Optional[int],
    orient_by_travel_direction: bool,
) -> dict:
    center_xy = lr.center_xy
    if orient_by_travel_direction and lr.lane_id > 0:
        center_xy = center_xy[::-1]
    return {
        'type': 'LANE_SURFACE_STREET',
        'polyline': _to_3d(center_xy),
        'speed_limit_mph': None,
        'interpolating': None,
        'entry_lanes':   entry_ids,
        'exit_lanes':    exit_ids,
        'left_neighbor':  [left_id]  if left_id  is not None else [],
        'right_neighbor': [right_id] if right_id is not None else [],
    }


def _travel_oriented_centerline(
    lane: LaneRecord,
    orient_by_travel_direction: bool,
) -> np.ndarray:
    if orient_by_travel_direction and lane.lane_id > 0:
        return lane.center_xy[::-1]
    return lane.center_xy


def _orient_lane_links_by_geometry(
    lane_links: Dict[Tuple[str, int, int], Dict[str, List]],
    lane_records: Dict[Tuple[str, int, int], LaneRecord],
) -> Dict[Tuple[str, int, int], Dict[str, List]]:
    """Orient candidate lane adjacency from travel-end to travel-start."""
    oriented = defaultdict(lambda: {'predecessors': [], 'successors': []})
    candidate_pairs = set()
    for source, links in lane_links.items():
        for target in links.get('successors', []):
            if source != target and source in lane_records and target in lane_records:
                candidate_pairs.add(frozenset((source, target)))

    for pair in candidate_pairs:
        a_key, b_key = tuple(pair)
        a = _travel_oriented_centerline(lane_records[a_key], True)
        b = _travel_oriented_centerline(lane_records[b_key], True)
        if len(a) < 2 or len(b) < 2:
            continue
        distance_a_to_b = float(np.linalg.norm(a[-1] - b[0]))
        distance_b_to_a = float(np.linalg.norm(b[-1] - a[0]))
        source, target = (
            (a_key, b_key)
            if distance_a_to_b <= distance_b_to_a
            else (b_key, a_key)
        )
        oriented[source]['successors'].append(target)
        oriented[target]['predecessors'].append(source)

    for key in oriented:
        oriented[key]['predecessors'] = list(dict.fromkeys(oriented[key]['predecessors']))
        oriented[key]['successors'] = list(dict.fromkeys(oriented[key]['successors']))
    return dict(oriented)


def _outermost_driving(road: RoadRecord, side: int):
    """side=+1 取最左 driving lane,-1 取最右。返回 LaneRecord 或 None。"""
    cand = [lr for lr in road.lanes
            if lr.lane_type in _DRIVING_TYPES and np.sign(lr.lane_id) == side]
    if not cand:
        return None
    return max(cand, key=lambda lr: side * lr.lane_id)


def _slice_polyline_by_s(
    polyline: np.ndarray,
    section_s: np.ndarray,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """Slice a sampled lane boundary at exact laneSection-local s limits."""
    if len(polyline) < 2 or len(section_s) != len(polyline):
        return np.zeros((0, 2), dtype=np.float64)

    # Geometry boundaries can contribute duplicate s samples. np.interp expects
    # a strictly increasing coordinate, so keep the first point at each s.
    unique_s, unique_indices = np.unique(section_s, return_index=True)
    unique_polyline = polyline[unique_indices]
    if len(unique_s) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    start_s = float(np.clip(start_s, unique_s[0], unique_s[-1]))
    end_s = float(np.clip(end_s, unique_s[0], unique_s[-1]))
    if end_s - start_s <= 1e-6:
        return np.zeros((0, 2), dtype=np.float64)

    inner = (unique_s > start_s + 1e-8) & (unique_s < end_s - 1e-8)
    sample_s = np.concatenate(([start_s], unique_s[inner], [end_s]))
    sample_xy = np.column_stack([
        np.interp(sample_s, unique_s, unique_polyline[:, axis])
        for axis in range(2)
    ])
    return sample_xy


def _road_mark_segments(lane: LaneRecord):
    """Yield non-overlapping visible road-mark segments for one lane boundary."""
    marks = sorted(lane.road_marks, key=lambda mark: mark['sOffset'])
    if not marks or len(lane.section_s) < 2:
        return

    section_end = float(np.max(lane.section_s))
    intervals = []
    for index, mark in enumerate(marks):
        start_s = max(0.0, float(mark['sOffset']))
        end_s = (
            float(marks[index + 1]['sOffset'])
            if index + 1 < len(marks)
            else section_end
        )
        waymo_type = _ROADMARK_TO_WAYMO.get((mark['type'], mark['color']))
        if waymo_type is None or end_s - start_s <= 1e-6:
            continue
        if intervals and intervals[-1][2] == waymo_type:
            previous_start, previous_end, _ = intervals[-1]
            if abs(previous_end - start_s) <= 1e-6:
                intervals[-1] = (previous_start, end_s, waymo_type)
                continue
        intervals.append((start_s, end_s, waymo_type))

    for start_s, end_s, waymo_type in intervals:
        segment = _slice_polyline_by_s(
            lane.outer_xy, lane.section_s, start_s, end_s
        )
        if len(segment) >= 2:
            yield start_s, end_s, waymo_type, segment


def _build_features_from_xodr(
    root,
    flip_y: bool = True,
    orient_lanes_by_travel_direction: bool = False,
    use_lane_link_ids: bool = False,
) -> dict:
    """主入口:从 xodr root 构建完整 map_features。"""
    # Pass 1:解析所有 road
    roads: Dict[str, RoadRecord] = {}
    for road_el in root.findall('.//road'):
        rec = _parse_road(road_el, flip_y=flip_y)
        if rec is not None and len(rec.lanes) > 0:
            roads[rec.road_id] = rec

    # Pass 2:解析 junction
    junctions = _parse_junctions(root)

    # Pass 3:构建 lane 拓扑
    lane_links = _build_lane_links(
        roads, junctions, use_lane_link_ids=use_lane_link_ids
    )
    if orient_lanes_by_travel_direction:
        lane_records = {
            lane.key: lane
            for road in roads.values()
            for lane in road.driving_lanes
        }
        lane_links = _orient_lane_links_by_geometry(lane_links, lane_records)

    # Pass 4:给每条 driving lane 分配 feature_id
    features: dict = {}
    fid_counter = [0]
    lane_id_map: Dict[Tuple[str, int, int], int] = {}
    for road in roads.values():
        for lr in road.driving_lanes:
            lr.feature_id = fid_counter[0]
            lane_id_map[lr.key] = fid_counter[0]
            fid_counter[0] += 1

    # Pass 5:emit LANE_SURFACE_STREET
    for road in roads.values():
        for lr in road.driving_lanes:
            entries = [lane_id_map[k]
                       for k in lane_links.get(lr.key, {}).get('predecessors', [])
                       if k in lane_id_map]
            exits   = [lane_id_map[k]
                       for k in lane_links.get(lr.key, {}).get('successors', [])
                       if k in lane_id_map]
            left = _neighbor_lane(
                road, lr, 'left', orient_lanes_by_travel_direction
            )
            right = _neighbor_lane(
                road, lr, 'right', orient_lanes_by_travel_direction
            )
            left_id  = lane_id_map.get(left.key)  if left  else None
            right_id = lane_id_map.get(right.key) if right else None
            features[lr.feature_id] = _make_lane_feature(
                lr, entries, exits, left_id, right_id,
                orient_by_travel_direction=orient_lanes_by_travel_direction)

    # Pass 6:emit ROAD_EDGE_BOUNDARY(每个 section 左右各一条)
    for road in roads.values():
        sec_ids = sorted({lr.section_idx for lr in road.lanes})
        for sec in sec_ids:
            for side in (+1, -1):
                cand = [lr for lr in road.lanes
                        if lr.section_idx == sec
                        and lr.lane_type in _DRIVING_TYPES
                        and np.sign(lr.lane_id) == side]
                if not cand:
                    continue
                outermost = max(cand, key=lambda lr: side * lr.lane_id)
                features[fid_counter[0]] = {
                    'type': 'ROAD_EDGE_BOUNDARY',
                    'polyline': _to_3d(outermost.outer_xy),
                }
                fid_counter[0] += 1

    # Pass 7:emit ROAD_LINE_*(包含 center lane 的中心分隔标线)
    # OpenDRIVE roadMark 在 lane 的"外缘"(远离 id=0 一侧),正是该 lane 与下一外侧
    # lane 之间的分隔线。type='curb'/'none' 不输出(curb 由 ROAD_EDGE 代理)。
    # 中心 lane(id=0)的 roadMark 即对向车道分隔线,使用 ref line。
    for road in roads.values():
        for lr in road.lanes:
            for start_s, end_s, waymo_type, segment in _road_mark_segments(lr):
                features[fid_counter[0]] = {
                    'type': waymo_type,
                    'polyline': _to_3d(segment),
                    'metadata': {
                        'road_id': lr.road_id,
                        'section_idx': lr.section_idx,
                        'lane_id': lr.lane_id,
                        'road_mark_s_start': start_s,
                        'road_mark_s_end': end_s,
                    },
                }
                fid_counter[0] += 1

    for feature in features.values():
        for geometry_key in ('polyline', 'polygon'):
            if geometry_key in feature:
                feature[geometry_key] = _remove_consecutive_duplicate_points(
                    feature[geometry_key]
                )
    return features


# ── 公共入口 ──────────────────────────────────────────────────────────────────
def build_town_map(town_name, scenario_dirs, map_cache_dir, xodr_dir=None):
    """构建 town_name 的 map_features。优先使用 xodr,否则 fallback 到轨迹重建。"""
    cache_path = os.path.join(map_cache_dir, f'{town_name}_map.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    xodr_path = None
    if xodr_dir:
        candidate = os.path.join(xodr_dir, f'{town_name}.xodr')
        if os.path.isfile(candidate):
            xodr_path = candidate

    if xodr_path:
        print(f'  [{town_name}] Parsing xodr: {xodr_path}')
        tree = ET.parse(xodr_path)
        map_features = _build_features_from_xodr(tree.getroot())
        # 简要分布统计
        from collections import Counter
        type_counts = Counter(v['type'] for v in map_features.values())
        print(f'  [{town_name}] features={len(map_features)}  '
              + '  '.join(f'{t}={n}' for t, n in type_counts.most_common()))
    else:
        print(f'  [{town_name}] WARNING: no xodr found, using trajectory reconstruction')
        map_features = _build_from_trajectories(town_name, scenario_dirs)

    os.makedirs(map_cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(map_features, f)
    print(f'  [{town_name}] Map saved → {cache_path}')
    return map_features


# ── Legacy fallback(无 xodr 时用轨迹重建) ─────────────────────────────────
def _build_from_trajectories(town_name, scenario_dirs):
    """轨迹聚类重建(留作没有 xodr 时的兜底)。"""
    from collections import defaultdict
    from scipy.interpolate import splprep, splev
    from sklearn.cluster import DBSCAN

    SAMPLE_INTERVAL_M = 0.5
    LANE_WIDTH_M      = 3.5
    DBSCAN_EPS_M      = 4.0
    MIN_CLUSTER_PTS   = 50

    def _resample(points, interval=SAMPLE_INTERVAL_M):
        if len(points) < 2: return points
        dists = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        cum   = np.concatenate([[0.0], np.cumsum(dists)])
        total = cum[-1]
        if total < 1e-6: return points[:1]
        num_segments = max(1, int(math.ceil(total / interval)))
        tgt = np.linspace(0.0, total, num_segments + 1)
        new = np.zeros((len(tgt), points.shape[1]), dtype=points.dtype)
        for c in range(points.shape[1]):
            new[:, c] = np.interp(tgt, cum, points[:, c])
        return new

    def _smooth(pts2):
        if len(pts2) < 4: return pts2
        try:
            tck, _ = splprep([pts2[:, 0], pts2[:, 1]], s=0.5,
                             k=min(3, len(pts2) - 1))
            u_new = np.linspace(0, 1, max(len(pts2), 20))
            x, y = splev(u_new, tck)
            return np.stack([x, y], axis=-1)
        except Exception:
            return pts2

    def _lateral(poly, w):
        if len(poly) < 2: return poly.copy()
        t = np.diff(poly[:, :2], axis=0)
        t = np.vstack([t, t[-1:]])
        n = np.linalg.norm(t, axis=1, keepdims=True) + 1e-8
        t = t / n
        normals = np.stack([-t[:, 1], t[:, 0]], axis=1)
        r = poly.copy()
        r[:, :2] += normals * w
        return r

    all_pos = []
    for label_dir in scenario_dirs:
        if not os.path.isdir(label_dir): continue
        agent_tracks = defaultdict(list)
        for fname in sorted(os.listdir(label_dir)):
            with open(os.path.join(label_dir, fname)) as fh:
                for line in fh.readlines()[1:]:
                    parts = line.strip().split()
                    if len(parts) < 12: continue
                    try:
                        agent_tracks[parts[10]].append(
                            (float(parts[1]), float(parts[2])))
                    except (ValueError, IndexError):
                        continue
        for pts in agent_tracks.values():
            if len(pts) < 3: continue
            all_pos.append(_resample(np.array(pts, dtype=np.float64)))

    if len(all_pos) < 2:
        return {}

    centroids = np.array([np.mean(s, axis=0) for s in all_pos])
    labels = DBSCAN(eps=DBSCAN_EPS_M, min_samples=3).fit_predict(centroids)
    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        if lbl >= 0:
            clusters[lbl].append(all_pos[i])

    map_features = {}
    fid = 0
    for segs in clusters.values():
        merged = np.vstack(segs)
        if len(merged) < MIN_CLUSTER_PTS: continue
        smoothed = _smooth(merged[:, :2])
        if len(smoothed) < 2: continue
        poly3 = _to_3d(smoothed)
        map_features[fid] = {
            'type': 'LANE_SURFACE_STREET', 'polyline': poly3,
            'speed_limit_mph': None, 'interpolating': None,
            'entry_lanes': [], 'exit_lanes': [],
            'left_neighbor': [], 'right_neighbor': [],
        }
        fid += 1
        for sign in (1.2, -1.2):
            ep = _lateral(poly3, LANE_WIDTH_M * sign)
            map_features[fid] = {'type': 'ROAD_EDGE_BOUNDARY', 'polyline': ep}
            fid += 1
    return map_features
