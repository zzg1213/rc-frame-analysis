#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RC 框架结构内力计算与配筋设计程序
依据: GB 50010-2010, GB 50011-2010, GB 50009-2012
"""

import json
import argparse
import math
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

# ============================================================
# 1. 常量与材料参数
# ============================================================

CONCRETE_PARAMS = {
    "C25": {"fc": 11.9, "ft": 1.27, "Ec": 2.80e4},
    "C30": {"fc": 14.3, "ft": 1.43, "Ec": 3.00e4},
    "C35": {"fc": 16.7, "ft": 1.57, "Ec": 3.15e4},
    "C40": {"fc": 19.1, "ft": 1.71, "Ec": 3.25e4},
}

REBAR_PARAMS = {
    "HRB400": {"fy": 360.0, "fyk": 400.0, "Es": 2.0e5},
    "HRB500": {"fy": 435.0, "fyk": 500.0, "Es": 2.0e5},
}

# xi_b 界限相对受压区高度 (beta1=0.80 for C50及以下)
XI_B_TABLE = {
    ("C30", "HRB400"): 0.518,
    ("C30", "HRB500"): 0.424,
    ("C35", "HRB400"): 0.518,
    ("C40", "HRB400"): 0.518,
    ("C25", "HRB400"): 0.518,
}

REBAR_DIAMETERS = [6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32]
REBAR_AREAS = {
    6: 28.3, 8: 50.3, 10: 78.5, 12: 113.1, 14: 153.9, 16: 201.1,
    18: 254.5, 20: 314.2, 22: 380.1, 25: 490.9, 28: 615.8, 32: 804.2,
}

# 各荷载组合的竖向荷载系数 (cd, cl)
COMBO_VERT_FACTORS = {
    'C1': (1.2,  1.4),
    'C2': (1.35, 0.98),
    'C3': (1.2,  0.6),
    'C4': (1.2,  0.6),
    'C5': (1.0,  0.5),
    'C6': (1.0,  0.5),
}

DEFAULT_CONFIG = {
    "concrete_grade": "C30",
    "rebar_grade": "HRB400",
    "stirrup_grade": "HRB400",
    "floor_dead_load_kPa": 4.0,
    "roof_dead_load_kPa": 5.0,
    "floor_live_load_kPa": 2.0,
    "roof_live_load_kPa": 0.5,
    "wall_line_load_kNm": 6.0,
    "slab_thickness_mm": 100,
    "gamma_concrete": 25.0,
    "intensity": "8(0.20g)",
    "site_category": "II",
    "design_group": 1,
    "Tg": 0.35,
    "damping_ratio": 0.05,
    "alpha_max": 0.16,
    "seismic_grade": 1,
    "cover_beam_mm": 25,
    "cover_column_mm": 30,
}


def load_layout(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_config(path, base_cfg):
    cfg = dict(base_cfg)
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            user = json.load(f)
        cfg.update(user)
    return cfg


def get_material(cfg):
    cg = cfg["concrete_grade"]
    rg = cfg["rebar_grade"]
    sg = cfg["stirrup_grade"]
    c = CONCRETE_PARAMS[cg]
    r = REBAR_PARAMS[rg]
    s = REBAR_PARAMS[sg]
    xi_b = XI_B_TABLE.get((cg, rg), 0.518)
    return {
        "fc": c["fc"], "ft": c["ft"], "Ec": c["Ec"],
        "fy": r["fy"], "Es": r["Es"],
        "fyv": s["fy"],
        "xi_b": xi_b,
        "beta1": 0.80,
    }


# ============================================================
# 2. 二维框架提取
# ============================================================

def parse_node_index(name):
    """N_i_j -> (i, j)"""
    parts = name.split('_')
    return int(parts[1]), int(parts[2])


def parse_beam_key(name):
    """B_i_j_dir -> (i, j, dir)"""
    parts = name.split('_')
    return int(parts[1]), int(parts[2]), int(parts[3])


def parse_column_key(name):
    """C_i_j -> (i, j)"""
    parts = name.split('_')
    return int(parts[1]), int(parts[2])


def detect_slab_panels(layout):
    """检测围合楼板: 网格单元(i,j)四边有梁则存在楼板.
    返回 set of (i, j) 表示左下角为 N_i_j 的网格单元有楼板."""
    story_key = next(iter(layout["beams"]))
    beams = layout["beams"][story_key]

    beam_set_x = set()  # (i, j) 表示 B_i_j_1 存在
    beam_set_y = set()  # (i, j) 表示 B_i_j_2 存在
    for bk in beams:
        i, j, d = parse_beam_key(bk)
        if d == 1:
            beam_set_x.add((i, j))
        else:
            beam_set_y.add((i, j))

    node_indices = set()
    for nk in layout["nodes"]:
        node_indices.add(parse_node_index(nk))

    slab_panels = set()
    for (i, j) in node_indices:
        # 网格单元 (i,j) 的四角: N_i_j, N_{i+1}_j, N_{i+1}_{j+1}, N_i_{j+1}
        if (i+1, j) not in node_indices:
            continue
        if (i, j+1) not in node_indices:
            continue
        if (i+1, j+1) not in node_indices:
            continue
        # 检查四边梁
        bottom = (i, j) in beam_set_x      # B_i_j_1 (底边, X向)
        top = (i, j+1) in beam_set_x       # B_i_{j+1}_1 (顶边, X向)
        left = (i, j) in beam_set_y        # B_i_j_2 (左边, Y向)
        right = (i+1, j) in beam_set_y     # B_{i+1}_j_2 (右边, Y向)
        if bottom and top and left and right:
            slab_panels.add((i, j))

    return slab_panels


def calc_tributary_width_x(beam_i, beam_j, layout, slab_panels):
    """计算X向梁 B_{beam_i}_{beam_j}_1 的从属宽度(mm).
    上方: 网格单元 (beam_i, beam_j)  → 跨距 grid_y[beam_j+1] - grid_y[beam_j]
    下方: 网格单元 (beam_i, beam_j-1) → 跨距 grid_y[beam_j] - grid_y[beam_j-1]
    """
    grid_y = layout["geometry"]["grid_y"]
    width = 0.0
    # 上方
    if (beam_i, beam_j) in slab_panels:
        width += (grid_y[beam_j + 1] - grid_y[beam_j]) / 2.0
    # 下方
    if beam_j > 0 and (beam_i, beam_j - 1) in slab_panels:
        width += (grid_y[beam_j] - grid_y[beam_j - 1]) / 2.0
    return width


def calc_tributary_width_y(beam_i, beam_j, layout, slab_panels):
    """计算Y向梁 B_{beam_i}_{beam_j}_2 的从属宽度(mm).
    右方: 网格单元 (beam_i, beam_j)   → 跨距 grid_x[beam_i+1] - grid_x[beam_i]
    左方: 网格单元 (beam_i-1, beam_j) → 跨距 grid_x[beam_i] - grid_x[beam_i-1]
    """
    grid_x = layout["geometry"]["grid_x"]
    width = 0.0
    # 右方
    if (beam_i, beam_j) in slab_panels:
        width += (grid_x[beam_i + 1] - grid_x[beam_i]) / 2.0
    # 左方
    if beam_i > 0 and (beam_i - 1, beam_j) in slab_panels:
        width += (grid_x[beam_i] - grid_x[beam_i - 1]) / 2.0
    return width


def is_edge_beam_x(beam_i, beam_j, slab_panels):
    """X向梁是否为边梁(至少一侧无楼板)."""
    has_upper = (beam_i, beam_j) in slab_panels
    has_lower = (beam_i, beam_j - 1) in slab_panels if beam_j > 0 else False
    return not (has_upper and has_lower)


def is_edge_beam_y(beam_i, beam_j, slab_panels):
    """Y向梁是否为边梁."""
    has_right = (beam_i, beam_j) in slab_panels
    has_left = (beam_i - 1, beam_j) in slab_panels if beam_i > 0 else False
    return not (has_right and has_left)


def find_connected_components(indices):
    """对一维有序索引列表, 找出连续段. 返回list of list."""
    if not indices:
        return []
    indices = sorted(indices)
    components = [[indices[0]]]
    for k in range(1, len(indices)):
        if indices[k] == indices[k-1] + 1:
            components[-1].append(indices[k])
        else:
            components.append([indices[k]])
    return components


def extract_2d_frames(layout):
    """提取所有X向和Y向2D框架.
    返回 (frames_x, frames_y):
      frames_x[j] = list of sub-frames, 每个 sub-frame = {
        'y_index': j, 'node_indices': [i0,i1,...],
        'beam_keys': [...], 'column_keys': [...]}
      frames_y[i] = 类似
    """
    story_key = next(iter(layout["beams"]))
    beams_data = layout["beams"][story_key]
    col_key = next(iter(layout["columns"]))
    cols_data = layout["columns"][col_key]

    # 收集各轴线上的构件
    x_beams_by_j = defaultdict(set)   # j -> set of i (X向梁起始i)
    y_beams_by_i = defaultdict(set)   # i -> set of j (Y向梁起始j)
    for bk in beams_data:
        i, j, d = parse_beam_key(bk)
        if d == 1:
            x_beams_by_j[j].add(i)
        else:
            y_beams_by_i[i].add(j)

    cols_by_j = defaultdict(set)  # j -> set of i
    cols_by_i = defaultdict(set)  # i -> set of j
    for ck in cols_data:
        i, j = parse_column_key(ck)
        cols_by_j[j].add(i)
        cols_by_i[i].add(j)

    # X向框架: 对每条Y轴线j
    frames_x = {}
    all_j = sorted(set(list(x_beams_by_j.keys()) + list(cols_by_j.keys())))
    for j in all_j:
        beam_is = x_beams_by_j.get(j, set())
        col_is = cols_by_j.get(j, set())
        if not beam_is and not col_is:
            continue
        # 收集所有涉及的节点i索引
        node_is = set(col_is)
        for bi in beam_is:
            node_is.add(bi)
            node_is.add(bi + 1)
        # 连通分量
        all_i_sorted = sorted(node_is)
        components = find_connected_components(all_i_sorted)
        sub_frames = []
        for comp in components:
            comp_set = set(comp)
            bkeys = [f"B_{i}_{j}_1" for i in beam_is if i in comp_set]
            ckeys = [f"C_{i}_{j}" for i in comp_set if i in col_is]
            if not bkeys or not ckeys:
                continue
            sub_frames.append({
                'axis': 'X', 'axis_index': j,
                'node_i_indices': sorted(comp),
                'beam_keys': bkeys,
                'column_keys': ckeys,
            })
        if sub_frames:
            frames_x[j] = sub_frames

    # Y向框架: 对每条X轴线i
    frames_y = {}
    all_i = sorted(set(list(y_beams_by_i.keys()) + list(cols_by_i.keys())))
    for i in all_i:
        beam_js = y_beams_by_i.get(i, set())
        col_js = cols_by_i.get(i, set())
        if not beam_js and not col_js:
            continue
        node_js = set(col_js)
        for bj in beam_js:
            node_js.add(bj)
            node_js.add(bj + 1)
        all_j_sorted = sorted(node_js)
        components = find_connected_components(all_j_sorted)
        sub_frames = []
        for comp in components:
            comp_set = set(comp)
            bkeys = [f"B_{i}_{j}_2" for j in beam_js if j in comp_set]
            ckeys = [f"C_{i}_{j}" for j in comp_set if j in col_js]
            if not bkeys or not ckeys:
                continue
            sub_frames.append({
                'axis': 'Y', 'axis_index': i,
                'node_j_indices': sorted(comp),
                'beam_keys': bkeys,
                'column_keys': ckeys,
            })
        if sub_frames:
            frames_y[i] = sub_frames

    return frames_x, frames_y


# ============================================================
# 3. 荷载计算
# ============================================================

def calc_beam_loads(layout, cfg, slab_panels):
    """计算每根梁的均布恒载g和活载q (kN/m).
    返回 beam_loads[beam_key] = {'g': ..., 'q': ...}
    """
    story_key = next(iter(layout["beams"]))
    beams_data = layout["beams"][story_key]
    sections = layout["sections"]
    num_stories = layout["parameters"]["num_stories"]

    beam_loads = {}  # beam_key -> {story -> {'g':, 'q':}}

    for bk, binfo in beams_data.items():
        i, j, d = parse_beam_key(bk)
        sec = sections[binfo["section"]]
        b_mm = sec["b"]
        h_mm = sec["h"]
        slab_t = cfg["slab_thickness_mm"]

        # 从属宽度 (mm)
        if d == 1:
            trib_w_mm = calc_tributary_width_x(i, j, layout, slab_panels)
            is_edge = is_edge_beam_x(i, j, slab_panels)
        else:
            trib_w_mm = calc_tributary_width_y(i, j, layout, slab_panels)
            is_edge = is_edge_beam_y(i, j, slab_panels)

        trib_w_m = trib_w_mm / 1000.0

        loads_by_story = {}
        for story in range(1, num_stories + 1):
            is_roof = (story == num_stories)
            dead_kPa = cfg["roof_dead_load_kPa"] if is_roof else cfg["floor_dead_load_kPa"]
            live_kPa = cfg["roof_live_load_kPa"] if is_roof else cfg["floor_live_load_kPa"]

            # 板传恒载
            q_slab_dead = dead_kPa * trib_w_m

            # 梁自重 (扣除板厚重叠)
            g_beam = cfg["gamma_concrete"] * b_mm * (h_mm - slab_t) / 1e6

            # 墙体荷载 (边梁加, 屋面不加)
            g_wall = cfg["wall_line_load_kNm"] if (is_edge and not is_roof) else 0.0

            g = q_slab_dead + g_beam + g_wall
            q = live_kPa * trib_w_m

            loads_by_story[story] = {'g': g, 'q': q}

        beam_loads[bk] = loads_by_story

    return beam_loads


def calc_story_weights(layout, cfg, slab_panels, beam_loads):
    """计算各层重力荷载代表值 Gi (N).
    返回 list, Gi[0] = 第1层.

    单位约定:
      - 梁线荷载 g/q: N/mm (数值上等于 kN/m)
      - 混凝土容重: cfg["gamma_concrete"] kN/m³ → 25e-6 N/mm³
      - 面荷载: cfg[*_kPa] kPa = 1e-3 N/mm²
      - 所有长度: mm
      - 输出 Gi: N
    """
    sections = layout["sections"]
    num_stories = layout["parameters"]["num_stories"]
    grid_x = layout["geometry"]["grid_x"]
    grid_y = layout["geometry"]["grid_y"]

    story_key = next(iter(layout["beams"]))
    beams_data = layout["beams"][story_key]
    col_key = next(iter(layout["columns"]))
    cols_data = layout["columns"][col_key]

    # 混凝土容重: kN/m³ → N/mm³ = kN/m³ × 1e-6
    gamma_N_mm3 = cfg["gamma_concrete"] * 1e-6  # N/mm³

    Gi_list = []
    for story in range(1, num_stories + 1):
        is_roof = (story == num_stories)
        # 面荷载: kPa → N/mm²
        dead_N_mm2 = (cfg["roof_dead_load_kPa"] if is_roof else cfg["floor_dead_load_kPa"]) * 1e-3
        live_N_mm2 = (cfg["roof_live_load_kPa"] if is_roof else cfg["floor_live_load_kPa"]) * 1e-3

        # 梁线荷载 (N/mm) × 梁长 (mm) = N
        beam_dead_total = 0.0
        beam_live_total = 0.0
        for bk, binfo in beams_data.items():
            L_mm = binfo["length"]
            bl = beam_loads[bk][story]
            beam_dead_total += bl['g'] * L_mm   # N/mm × mm = N
            beam_live_total += bl['q'] * L_mm

        # 柱自重: N/mm³ × mm³ = N
        col_dead_total = 0.0
        for ck, cinfo in cols_data.items():
            sec = sections[cinfo["section"]]
            col_dead_total += gamma_N_mm3 * sec["b"] * sec["h"] * cinfo["length"]

        # 板面积恒载: N/mm² × mm² = N
        slab_dead = 0.0
        slab_live = 0.0
        for (si, sj) in slab_panels:
            area_mm2 = (grid_x[si + 1] - grid_x[si]) * (grid_y[sj + 1] - grid_y[sj])
            slab_dead += dead_N_mm2 * area_mm2
            slab_live += live_N_mm2 * area_mm2

        total_dead = beam_dead_total + col_dead_total + slab_dead
        total_live = beam_live_total + slab_live

        Gi = total_dead + 0.5 * total_live   # N
        Gi_list.append(Gi)

    return Gi_list


# ============================================================
# 4. 底部剪力法
# ============================================================

def calc_seismic_coefficient(T1, cfg):
    """计算地震影响系数 alpha1."""
    Tg = cfg["Tg"]
    alpha_max = cfg["alpha_max"]
    gamma = 0.9
    eta1 = 0.02
    eta2 = 1.0

    if T1 <= 0.1:
        alpha1 = (0.45 + (1.0 - 0.45) * T1 / 0.1) * alpha_max
    elif T1 <= Tg:
        alpha1 = alpha_max
    elif T1 <= 5 * Tg:
        alpha1 = alpha_max * (Tg / T1) ** gamma
    else:
        alpha1 = alpha_max * (eta2 * 0.2 ** gamma - eta1 * (T1 - 5 * Tg))
        alpha1 = max(alpha1, 0.0)
    return alpha1


def base_shear_method(layout, cfg, Gi_list):
    """底部剪力法计算各层水平力 (N).
    Gi_list: 各层重力荷载代表值 (N)
    返回 seismic_data dict，所有力单位 N."""
    num_stories = layout["parameters"]["num_stories"]
    grid_z = layout["geometry"]["grid_z"]

    T1 = 0.08 * num_stories
    alpha1 = calc_seismic_coefficient(T1, cfg)
    G_eq = 0.85 * sum(Gi_list)     # N
    F_EK = alpha1 * G_eq            # N

    Tg = cfg["Tg"]
    if T1 > 1.4 * Tg:
        delta_n = 0.08 * T1 + 0.07
    else:
        delta_n = 0.0
    delta_Fn = delta_n * F_EK      # N

    # 各层标高 Hi (mm)，Gi·Hi 乘积的量纲在比值中抵消
    Hi_list = [grid_z[s] for s in range(1, num_stories + 1)]   # mm

    sum_GH = sum(Gi_list[k] * Hi_list[k] for k in range(num_stories))  # N·mm

    floor_forces = []
    for k in range(num_stories):
        Fi = Gi_list[k] * Hi_list[k] / sum_GH * (F_EK - delta_Fn)  # N
        floor_forces.append(Fi)
    floor_forces[-1] += delta_Fn

    floor_shears = [0.0] * num_stories
    cumsum = 0.0
    for k in range(num_stories - 1, -1, -1):
        cumsum += floor_forces[k]
        floor_shears[k] = cumsum

    return {
        "T1": round(T1, 4),
        "alpha1": round(alpha1, 6),
        "G_eq": round(G_eq, 0),      # N
        "F_EK": round(F_EK, 0),      # N
        "delta_n": round(delta_n, 4),
        "floor_forces": [round(f, 0) for f in floor_forces],   # N
        "floor_shears": [round(s, 0) for s in floor_shears],   # N
    }


# ============================================================
# 4.5 D值法分配水平力到各榀框架
# ============================================================

def calc_column_D(col_info, beams_at_node, sections, Ec):
    """计算单根柱的侧向刚度D值 (N/mm = kN/m)."""
    sec = sections[col_info["section"]]
    bc = sec["b"]
    hc = sec["h"]
    Ic = bc * hc**3 / 12.0  # mm⁴
    h_col = col_info["length"]  # mm

    ic = Ec * Ic / h_col  # N·mm

    if ic < 1e-10:
        return 0.0

    # 汇交于该节点的梁的线刚度之和
    sum_ib = sum(
        Ec * (sections[binfo["section"]]["b"] * sections[binfo["section"]]["h"]**3 / 12.0) / binfo["length"]
        for binfo in beams_at_node
    )

    K = sum_ib / ic
    alpha = K / (2.0 + K)
    return alpha * 12.0 * Ec * Ic / (h_col ** 3)  # N/mm = kN/m


def calc_frame_D_values(frame_info, layout, cfg):
    """计算一榀框架中所有柱的D值之和."""
    story_key = next(iter(layout["beams"]))
    beams_data = layout["beams"][story_key]
    col_key = next(iter(layout["columns"]))
    cols_data = layout["columns"][col_key]
    sections = layout["sections"]
    Ec = get_material(cfg)["Ec"]

    total_D = 0.0
    for ck in frame_info['column_keys']:
        cinfo = cols_data[ck]
        node_name = cinfo["node"]

        # 找汇交于该节点的梁
        beams_at_node = [
            beams_data[bk] for bk in frame_info['beam_keys']
            if beams_data[bk]["i_node"] == node_name or beams_data[bk]["j_node"] == node_name
        ]

        total_D += calc_column_D(cinfo, beams_at_node, sections, Ec)

    return total_D


def distribute_forces_to_frames(seismic_data, frames_dict, layout, cfg):
    """按D值比例将楼层水平力分配到各榀框架.
    返回 frame_forces[frame_key] = [F1, F2, ...] 各层力."""
    # 计算所有框架的D值
    frame_D = {}
    for key, sub_list in frames_dict.items():
        for idx, sf in enumerate(sub_list):
            fk = f"{key}_{idx}"
            frame_D[fk] = calc_frame_D_values(sf, layout, cfg)

    total_D = sum(frame_D.values())
    if total_D < 1e-10:
        total_D = 1.0

    frame_forces = {}
    for fk, Dv in frame_D.items():
        ratio = Dv / total_D
        frame_forces[fk] = [f * ratio for f in seismic_data["floor_forces"]]

    return frame_forces, frame_D


# ============================================================
# 5. 矩阵位移法 — 2D框架内力分析
# ============================================================


def build_full_frame_model(frame_info, layout, cfg):
    """构建一榀2D框架的完整多层模型.

    2D 坐标系: x(水平, 沿框架方向), z(竖向).
    """
    num_stories = layout["parameters"]["num_stories"]
    grid_x = layout["geometry"]["grid_x"]
    grid_y = layout["geometry"]["grid_y"]
    grid_z = layout["geometry"]["grid_z"]
    sections = layout["sections"]
    story_key = next(iter(layout["beams"]))
    beams_data = layout["beams"][story_key]
    col_key = next(iter(layout["columns"]))
    cols_data = layout["columns"][col_key]
    mat = get_material(cfg)
    Ec = mat["Ec"]

    axis = frame_info['axis']
    if axis == 'X':
        node_span_indices = frame_info['node_i_indices']
        axis_index = frame_info['axis_index']
        # 水平坐标取 grid_x[i]
        horiz_coords = {i: grid_x[i] for i in node_span_indices}
    else:
        node_span_indices = frame_info['node_j_indices']
        axis_index = frame_info['axis_index']
        horiz_coords = {j: grid_y[j] for j in node_span_indices}

    # 构建节点: (楼层, 跨索引) -> node_id
    nodes = []  # list of (x_mm, z_mm)
    node_map = {}  # (story, span_idx) -> node_id

    for span_idx in node_span_indices:
        # 底部 (story=0)
        x = horiz_coords[span_idx]
        z = grid_z[0]  # 0
        nid = len(nodes)
        nodes.append((x, z))
        node_map[(0, span_idx)] = nid

    for story in range(1, num_stories + 1):
        for span_idx in node_span_indices:
            x = horiz_coords[span_idx]
            z = grid_z[story]
            nid = len(nodes)
            nodes.append((x, z))
            node_map[(story, span_idx)] = nid

    # 固支节点 (story=0)
    fixed_nodes = set()
    for span_idx in node_span_indices:
        fixed_nodes.add(node_map[(0, span_idx)])

    # 构建单元
    elements = []
    span_idx_set = set(node_span_indices)

    for story in range(1, num_stories + 1):
        # 柱单元: 从 (story-1, span_idx) 到 (story, span_idx)
        for ck in frame_info['column_keys']:
            ci, cj = parse_column_key(ck)
            if axis == 'X':
                span_idx = ci
            else:
                span_idx = cj

            if span_idx not in span_idx_set:
                continue

            ni = node_map[(story - 1, span_idx)]
            nj = node_map[(story, span_idx)]
            cinfo = cols_data[ck]
            sec = sections[cinfo["section"]]
            A = sec["b"] * sec["h"]
            I = sec["b"] * sec["h"]**3 / 12.0
            L = cinfo["length"]
            xi, zi = nodes[ni]
            xj, zj = nodes[nj]
            k_loc = element_stiffness_local(Ec, A, I, L)
            T_mat = transformation_matrix(xi, zi, xj, zj)
            elements.append({
                'type': 'column', 'ni': ni, 'nj': nj,
                'E': Ec, 'A': A, 'I': I, 'L': L,
                'story': story, 'key': ck,
                'section': cinfo["section"],
                'k_local': k_loc, 'T': T_mat,
            })

        # 梁单元: 同一层相邻节点
        for bk in frame_info['beam_keys']:
            bi, bj, bd = parse_beam_key(bk)
            binfo = beams_data[bk]
            sec = sections[binfo["section"]]
            A = sec["b"] * sec["h"]
            I = sec["b"] * sec["h"]**3 / 12.0
            L = binfo["length"]

            if axis == 'X':
                ni = node_map[(story, bi)]
                nj = node_map[(story, bi + 1)]
            else:
                ni = node_map[(story, bj)]
                nj = node_map[(story, bj + 1)]

            xi, zi = nodes[ni]
            xj, zj = nodes[nj]
            k_loc = element_stiffness_local(Ec, A, I, L)
            T_mat = transformation_matrix(xi, zi, xj, zj)
            elements.append({
                'type': 'beam', 'ni': ni, 'nj': nj,
                'E': Ec, 'A': A, 'I': I, 'L': L,
                'story': story, 'key': bk,
                'section': binfo["section"],
                'k_local': k_loc, 'T': T_mat,
            })

    # DOF编号: 每个非固支节点 3 DOF [u, w, theta]
    n_nodes = len(nodes)
    node_dofs = {}  # node_id -> (dof_u, dof_w, dof_theta) or None
    dof_count = 0
    for nid in range(n_nodes):
        if nid in fixed_nodes:
            node_dofs[nid] = None
        else:
            node_dofs[nid] = (dof_count, dof_count + 1, dof_count + 2)
            dof_count += 3

    return {
        'nodes': nodes,
        'elements': elements,
        'fixed_nodes': fixed_nodes,
        'node_dofs': node_dofs,
        'n_dof': dof_count,
        'node_map': node_map,
        'frame_info': frame_info,
    }


def element_stiffness_local(E, A, I, L):
    """6×6 局部坐标系刚度矩阵 (轴力+弯曲).
    DOF顺序: [u_i, w_i, theta_i, u_j, w_j, theta_j]
    局部坐标: u沿轴向, w垂直轴向."""
    EA_L  = E * A / L
    EI_L3 = E * I / (L ** 3)
    EI_L2 = E * I / (L ** 2)
    EI_L  = E * I / L

    return np.array([
        [ EA_L,         0,          0,      -EA_L,         0,          0      ],
        [ 0,     12*EI_L3,   6*EI_L2,          0,  -12*EI_L3,   6*EI_L2   ],
        [ 0,      6*EI_L2,   4*EI_L,            0,   -6*EI_L2,   2*EI_L    ],
        [-EA_L,        0,          0,       EA_L,         0,          0      ],
        [ 0,    -12*EI_L3,  -6*EI_L2,          0,   12*EI_L3,  -6*EI_L2   ],
        [ 0,      6*EI_L2,   2*EI_L,            0,   -6*EI_L2,   4*EI_L    ],
    ])


def transformation_matrix(x_i, z_i, x_j, z_j):
    """计算坐标变换矩阵 T (6×6).
    从局部坐标到全局坐标: 全局 [X(水平), Z(竖向)].
    局部坐标: u 沿 i->j 方向, w 垂直于 u (逆时针90°)."""
    dx = x_j - x_i
    dz = z_j - z_i
    L = math.sqrt(dx**2 + dz**2)
    c = dx / L  # cos
    s = dz / L  # sin

    T = np.zeros((6, 6))
    # 节点 i
    T[0, 0] = c;  T[0, 1] = s
    T[1, 0] = -s; T[1, 1] = c
    T[2, 2] = 1
    # 节点 j
    T[3, 3] = c;  T[3, 4] = s
    T[4, 3] = -s; T[4, 4] = c
    T[5, 5] = 1
    return T


def assemble_stiffness(model):
    """组装整体刚度矩阵 K."""
    n = model['n_dof']
    K = np.zeros((n, n))

    for elem in model['elements']:
        ni = elem['ni']
        nj = elem['nj']

        k_global = elem['T'].T @ elem['k_local'] @ elem['T']

        # 获取全局DOF索引，None 表示固支（跳过）
        dofs_i = model['node_dofs'][ni]
        dofs_j = model['node_dofs'][nj]
        raw = (list(dofs_i) if dofs_i is not None else [None, None, None]) + \
              (list(dofs_j) if dofs_j is not None else [None, None, None])

        # 过滤有效自由度，用 numpy fancy indexing 批量组装
        valid = [k for k, d in enumerate(raw) if d is not None]
        idx   = [raw[k] for k in valid]
        K[np.ix_(idx, idx)] += k_global[np.ix_(valid, valid)]

    return K


def calc_fixed_end_forces_beam(q, L):
    """均布荷载 q (kN/m) 作用下梁的固端力.
    q 向下为正 (竖向荷载). L 单位 mm.
    返回局部坐标系下的固端力向量 [N_i, V_i, M_i, N_j, V_j, M_j].
    单位: N 和 N·mm (与刚度矩阵一致).
    """
    # q(kN/m) -> q(N/mm): 1 kN/m = 1 N/mm
    q_N_mm = q  # kN/m = N/mm
    V = q_N_mm * L / 2.0        # N
    M = q_N_mm * L**2 / 12.0    # N·mm

    # 固端反力: V向上(w正方向), M_i逆时针为负, M_j顺时针为正
    f_fixed = np.array([0.0, V, -M, 0.0, V, M])
    return f_fixed


def assemble_load_vector(model, layout, cfg, beam_loads, load_case, seismic_forces=None):
    """组装荷载向量 F.
    load_case: 'D', 'L', 'EL', 'ER'
    """
    n = model['n_dof']
    F = np.zeros(n)
    num_stories = layout["parameters"]["num_stories"]

    if load_case in ('D', 'L'):
        for elem in model['elements']:
            if elem['type'] != 'beam':
                continue

            story = elem['story']
            bk = elem['key']

            if bk not in beam_loads:
                continue

            bl = beam_loads[bk][story]
            q = bl['g'] if load_case == 'D' else bl['q']

            if abs(q) < 1e-12:
                continue

            ni = elem['ni']
            nj = elem['nj']

            # 固端力 (局部坐标) -> 全局坐标，使用预计算的 T
            f_fixed_global = elem['T'].T @ calc_fixed_end_forces_beam(q, elem['L'])

            dofs_i = model['node_dofs'][ni]
            dofs_j = model['node_dofs'][nj]
            raw = (list(dofs_i) if dofs_i is not None else [None, None, None]) + \
                  (list(dofs_j) if dofs_j is not None else [None, None, None])
            for r, d in enumerate(raw):
                if d is not None:
                    F[d] += f_fixed_global[r]

    # 地震工况: 水平力
    if load_case in ('EL', 'ER') and seismic_forces is not None:
        node_map = model['node_map']
        frame_info = model['frame_info']
        axis = frame_info['axis']
        if axis == 'X':
            span_indices = frame_info['node_i_indices']
        else:
            span_indices = frame_info['node_j_indices']

        sign = 1.0 if load_case == 'EL' else -1.0

        for story in range(1, num_stories + 1):
            F_story = seismic_forces[story - 1]
            # 均分到该层所有节点
            n_nodes_this_story = len(span_indices)
            f_per_node = sign * F_story / n_nodes_this_story  # N

            for si in span_indices:
                nid = node_map[(story, si)]
                dofs = model['node_dofs'][nid]
                if dofs is not None:
                    F[dofs[0]] += f_per_node  # 水平DOF

    return F


def solve_displacements(K, F):
    """求解 K·d = F."""
    try:
        d = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        d = np.linalg.lstsq(K, F, rcond=None)[0]
    return d


def recover_member_forces(model, d, beam_loads, load_case):
    """回代求单元端力.
    返回 member_forces[elem_index] = {
        'N_i', 'V_i', 'M_i', 'N_j', 'V_j', 'M_j'  (N, N, N·mm)
    }
    """
    results = {}

    for idx, elem in enumerate(model['elements']):
        ni = elem['ni']
        nj = elem['nj']

        # 提取节点位移 (全局坐标)
        d_global = np.zeros(6)
        dofs_i = model['node_dofs'][ni]
        dofs_j = model['node_dofs'][nj]
        if dofs_i is not None:
            d_global[0] = d[dofs_i[0]]
            d_global[1] = d[dofs_i[1]]
            d_global[2] = d[dofs_i[2]]
        if dofs_j is not None:
            d_global[3] = d[dofs_j[0]]
            d_global[4] = d[dofs_j[1]]
            d_global[5] = d[dofs_j[2]]

        # 变换到局部坐标，使用预计算的 T 和 k_local
        d_local = elem['T'] @ d_global

        # 局部端力 = k_local · d_local
        f_local = elem['k_local'] @ d_local

        # 减去固端力 (梁单元有均布荷载时)
        if elem['type'] == 'beam' and load_case in ('D', 'L'):
            bk = elem['key']
            story = elem['story']
            if bk in beam_loads:
                bl = beam_loads[bk][story]
                q = bl['g'] if load_case == 'D' else bl['q']
                if abs(q) > 1e-12:
                    f_fixed = calc_fixed_end_forces_beam(q, elem['L'])
                    f_local = f_local - f_fixed

        # 单位: N (轴力/剪力), N·mm (弯矩)
        results[idx] = {
            'N_i': f_local[0], 'V_i': f_local[1], 'M_i': f_local[2],
            'N_j': f_local[3], 'V_j': f_local[4], 'M_j': f_local[5],
            'type': elem['type'], 'key': elem['key'], 'story': elem['story'],
        }

    return results


def analyze_frame(model, layout, cfg, beam_loads, seismic_forces):
    """对一榀框架进行4个工况的完整分析.
    返回 {load_case: member_forces}."""
    K = assemble_stiffness(model)
    results = {}

    for lc in ['D', 'L', 'EL', 'ER']:
        F = assemble_load_vector(model, layout, cfg, beam_loads, lc, seismic_forces)
        d = solve_displacements(K, F)
        mf = recover_member_forces(model, d, beam_loads, lc)
        results[lc] = {'displacements': d, 'member_forces': mf}

    return results


# ============================================================
# 6. 荷载组合
# ============================================================

def combine_load_cases(analysis_results):
    """6种荷载组合取包络.
    返回 envelope[elem_idx] = {
        'M_neg_max': 最大负弯矩(取绝对值较大的负弯矩),
        'M_pos_max': 最大正弯矩,
        'V_max': 最大剪力,
        'N_max': 最大轴力(压为负),
        'M_with_N': 与最大轴力对应的弯矩,
        'combinations': {C1..C6: forces}
    }
    """
    D = analysis_results['D']['member_forces']
    L = analysis_results['L']['member_forces']
    EL = analysis_results['EL']['member_forces']
    ER = analysis_results['ER']['member_forces']

    combos = {
        'C1': (1.2, 1.4, 0.0, 0.0),
        'C2': (1.35, 0.98, 0.0, 0.0),   # 1.35D + 1.4*0.7L = 1.35D + 0.98L
        'C3': (1.2, 0.6, 1.3, 0.0),
        'C4': (1.2, 0.6, 0.0, 1.3),
        'C5': (1.0, 0.5, 1.3, 0.0),
        'C6': (1.0, 0.5, 0.0, 1.3),
    }

    envelope = {}
    for idx in D:
        elem_combos = {}
        for cname, (cd, cl, cel, cer) in combos.items():
            forces = {}
            for key in ['N_i', 'V_i', 'M_i', 'N_j', 'V_j', 'M_j']:
                val = cd * D[idx][key] + cl * L[idx][key]
                if cel > 0:
                    val += cel * EL[idx][key]
                if cer > 0:
                    val += cer * ER[idx][key]
                forces[key] = val
            elem_combos[cname] = forces

        # 取包络
        all_M_i = [fc['M_i'] for fc in elem_combos.values()]
        all_M_j = [fc['M_j'] for fc in elem_combos.values()]
        all_V_i = [fc['V_i'] for fc in elem_combos.values()]
        all_V_j = [fc['V_j'] for fc in elem_combos.values()]
        all_N_i = [fc['N_i'] for fc in elem_combos.values()]
        all_N_j = [fc['N_j'] for fc in elem_combos.values()]

        # 梁: 端部取负弯矩(绝对值最大), 跨中取正弯矩
        # M_i, M_j 是端部弯矩, 跨中弯矩需要估算
        elem_type = D[idx]['type']

        if elem_type == 'beam':
            M_neg_left = min(all_M_i)  # 最大负弯矩(i端)
            M_neg_right = min(all_M_j)  # j端负弯矩...实际上j端的M_j符号需考虑
            # 对于梁, M_i为左端弯矩, M_j为右端弯矩
            # 跨中弯矩估算: M_mid ≈ qL²/8 - (M_left + M_right)/2 的包络
            # 这里简化: 取各组合中 M_mid 的最大正值
            M_mid_max = -1e30
            for cname, fc in elem_combos.items():
                # 跨中弯矩 = 简支弯矩 - (Mi+Mj)/2 ... 不太准确
                # 使用: M_mid = (M_i + M_j)/2 的相反方向 + 简支梁弯矩
                # 简化处理: 从D/L工况的q计算
                pass
            # 更简单的做法: 对每种组合计算跨中弯矩
            # M_mid = -(M_i + M_j)/2 + q_combo * L² / 8
            # 但组合后的q不容易获得, 用内力关系:
            # 对于均布荷载梁: M(x) = M_i + V_i*x - q*x²/2
            # 跨中 x = L/2: M_mid = M_i + V_i*L/2 - q*L²/8
            # 但组合后V_i已知, 且 V_i - V_j = q*L (平衡)
            # q_combo = (V_i - (-V_j)) / L = (V_i + V_j) / L  (注意V_j方向)
            # 实际上弯矩极值点: V=0处, M最大
            # 简化: M_mid ≈ (V_i * L/2 + M_i) 对每种组合
            elem_L = D[idx].get('L_mm', None)
            for cname, fc in elem_combos.items():
                # V_i 的单位: kN, L 的单位: mm
                # 但这里没有L... 需要从model中获取
                # 暂时用 M_i 和 V_i 估算
                # M_mid 是跨中弯矩, 先取 (M_i + M_j)/2 + |V_i|*L/4 的近似
                # 太不准确, 改用更好的方法
                pass

            envelope[idx] = {
                'type': 'beam',
                'key': D[idx]['key'],
                'story': D[idx]['story'],
                'M_neg_left': min(all_M_i),
                'M_neg_right': min(all_M_j),  # M_j取最小(最大负弯矩)
                'M_pos_left': max(all_M_i),
                'M_pos_right': max(all_M_j),
                'V_max_left': max(abs(v) for v in all_V_i),
                'V_max_right': max(abs(v) for v in all_V_j),
                'combinations': elem_combos,
            }
        else:
            # 柱: 取最大轴力和对应弯矩
            # 找最大压力(N最小, 因为压力为负)
            worst_N = None
            worst_M = None
            worst_combo = None
            for cname, fc in elem_combos.items():
                N = min(fc['N_i'], fc['N_j'])
                M = max(abs(fc['M_i']), abs(fc['M_j']))
                if worst_N is None or N < worst_N:
                    worst_N = N
                    worst_M = M
                    worst_combo = cname
                elif abs(N - worst_N) < 1e-6 and M > worst_M:
                    worst_M = M
                    worst_combo = cname

            # 也取最大弯矩对应的轴力
            max_M_combo = None
            max_M_val = 0
            for cname, fc in elem_combos.items():
                M = max(abs(fc['M_i']), abs(fc['M_j']))
                if M > max_M_val:
                    max_M_val = M
                    max_M_combo = cname

            envelope[idx] = {
                'type': 'column',
                'key': D[idx]['key'],
                'story': D[idx]['story'],
                'N_max': worst_N,
                'M_with_Nmax': worst_M,
                'M_max': max_M_val,
                'N_with_Mmax': min(elem_combos[max_M_combo]['N_i'],
                                   elem_combos[max_M_combo]['N_j']) if max_M_combo else 0,
                'V_max': max(max(abs(v) for v in all_V_i), max(abs(v) for v in all_V_j)),
                'M_top': None,  # 将在后续设置
                'M_bot': None,
                'combinations': elem_combos,
            }

    return envelope


def calc_beam_midspan_moments(envelope, model, beam_loads):
    """计算梁跨中弯矩. 利用内力平衡关系 M(x)=M_i+V_i*x-q*x²/2."""
    for idx, env in envelope.items():
        if env['type'] != 'beam':
            continue
        elem = model['elements'][idx]
        L = elem['L']   # mm
        bk = env['key']
        story = env['story']
        bl = beam_loads.get(bk, {}).get(story, {'g': 0.0, 'q': 0.0})
        g_beam = bl['g']    # N/mm (= kN/m)
        q_live = bl['q']    # N/mm

        best_M_mid = -1e30
        for cname, fc in env['combinations'].items():
            cd, cl = COMBO_VERT_FACTORS.get(cname, (1.2, 1.4))
            q_combo = cd * g_beam + cl * q_live   # N/mm
            V_i = fc['V_i']   # N
            M_i = fc['M_i']   # N·mm
            # M(x) = M_i + V_i*x - q*x²/2, 取跨中 x = L/2
            M_mid = M_i + V_i * (L / 2.0) - q_combo * (L / 2.0) ** 2 / 2.0
            if M_mid > best_M_mid:
                best_M_mid = M_mid

        env['M_pos_mid'] = max(best_M_mid, 0.0)


# ============================================================
# 7. 梁纵筋设计（正截面受弯）
# ============================================================

def _beam_effective_depth(h, cfg):
    """梁有效高度 h0 (mm): 保护层 + 纵筋半径（不考虑箍筋层）."""
    d_bar = 20
    as_ = cfg["cover_beam_mm"] + d_bar / 2.0
    return h - as_, as_


def design_beam_long(M_Nmm, b, h, cfg, mat, position='support'):
    """梁正截面受弯纵筋计算（仅纵向钢筋）.

    M_Nmm  : 设计弯矩 (N·mm), 取绝对值
    position: 'support' 或 'midspan'，影响最小配筋率
    返回 dict: As(mm²), As_prime(mm²), x(mm), section_type, xi
    """
    M = abs(M_Nmm)                  # N·mm
    fc   = mat["fc"]                  # N/mm²
    fy   = mat["fy"]
    ft   = mat["ft"]
    Es   = mat["Es"]
    xi_b = mat["xi_b"]
    alpha1 = 1.0
    eps_cu = 0.0033

    h0, as_ = _beam_effective_depth(h, cfg)
    as_prime = as_                    # 压筋保护层取相同值（对称）

    if h0 <= 0:
        return {'As': 0.0, 'As_prime': 0.0, 'x': 0.0, 'section_type': 'invalid', 'xi': 0.0}

    # 最小配筋率
    seismic_grade = cfg.get("seismic_grade", 1)
    if position == 'support':
        rho_min = max(0.004 if seismic_grade == 1 else 0.003 if seismic_grade == 2 else 0.0025,
                      (0.80 if seismic_grade == 1 else 0.65 if seismic_grade == 2 else 0.55) * ft / fy)
    else:
        rho_min = max(0.003 if seismic_grade == 1 else 0.0025 if seismic_grade == 2 else 0.002,
                      (0.65 if seismic_grade == 1 else 0.55 if seismic_grade == 2 else 0.45) * ft / fy)
    As_min = rho_min * b * h0

    if M < 1e-3:
        return {'As': As_min, 'As_prime': 0.0, 'x': 0.0, 'section_type': 'min_only', 'xi': 0.0}

    alpha_s   = M / (alpha1 * fc * b * h0 ** 2)
    alpha_s_b = xi_b * (1.0 - xi_b / 2.0)   # 单筋截面 alpha_s 界限

    if alpha_s <= alpha_s_b:
        # ── 单筋矩形截面：规范简化公式直接求解 ──────────────────────────────
        xi  = 1.0 - math.sqrt(1.0 - 2.0 * alpha_s)
        x   = xi * h0
        As  = alpha1 * fc * b * x / fy
        As  = max(As, As_min)
        return {
            'As': round(As, 1), 'As_prime': 0.0, 'x': round(x, 2),
            'section_type': 'single_reinforced', 'xi': round(xi, 4),
        }

    else:
        # ── 双筋截面：迭代确认压区高度与压筋应力 ─────────────────────────────
        # 初始取 x = xi_b*h0（充分利用混凝土）
        # 迭代目标：力平衡 alpha1*fc*b*x + As'*sigma_s'(x) = As*fy
        #           弯矩平衡 M = alpha1*fc*b*x*(h0-x/2) + As'*sigma_s'*(h0-as')
        x = xi_b * h0
        As = 0.0
        As_prime = 0.0
        for _ in range(60):
            eps_s_prime  = eps_cu * (x - as_prime) / x if x > 1e-6 else 0.0
            sigma_s_prime = min(Es * eps_s_prime, fy) if eps_s_prime > 1e-9 else 0.0

            M_conc = alpha1 * fc * b * x * (h0 - x / 2.0)
            lever  = h0 - as_prime
            if lever < 1e-6 or sigma_s_prime < 1e-9:
                As_prime = 0.0
            else:
                As_prime = max((M - M_conc) / (sigma_s_prime * lever), 0.0)

            As = (alpha1 * fc * b * x + As_prime * sigma_s_prime) / fy

            # 由力平衡反算 x，看是否与假设一致
            x_new = (As * fy - As_prime * sigma_s_prime) / (alpha1 * fc * b)
            x_new = max(x_new, 2.0 * as_prime)
            if abs(x_new - x) < 0.1:
                x = x_new
                break
            x = 0.5 * (x + x_new)      # 阻尼更新，防止震荡

        As       = max(As, As_min)
        As_prime = max(As_prime, 0.0)
        return {
            'As': round(As, 1), 'As_prime': round(As_prime, 1), 'x': round(x, 2),
            'section_type': 'double_reinforced', 'xi': round(x / h0, 4),
        }


def design_beam_full(elem_idx, envelope, model, cfg):
    """梁纵向钢筋设计（含腰筋，不含箍筋）.
    内力单位: N·mm（弯矩），N（剪力/轴力）."""
    env = envelope[elem_idx]
    elem = model['elements'][elem_idx]
    sections = model.get('sections', None)
    mat = get_material(cfg)

    sec_name = elem['section']
    sec = sections.get(sec_name)
    if sec is None:
        return None

    b = sec['b']
    h = sec['h']

    M_neg_left  = abs(env.get('M_neg_left',  0))   # N·mm
    M_neg_right = abs(env.get('M_neg_right', 0))   # N·mm
    M_pos_mid   = abs(env.get('M_pos_mid',   0))   # N·mm

    res_left  = design_beam_long(M_neg_left,  b, h, cfg, mat, 'support')
    res_right = design_beam_long(M_neg_right, b, h, cfg, mat, 'support')
    res_mid   = design_beam_long(M_pos_mid,   b, h, cfg, mat, 'midspan')

    # 上部纵筋（支座负弯矩）取左右较大值
    top_result = res_left if res_left['As'] >= res_right['As'] else res_right

    # 腰筋
    waist = design_beam_waist(b, h, cfg)

    return {
        'top': top_result,
        'bot_mid': res_mid,
        'waist': waist,
        'b': b, 'h': h,
    }


# ============================================================
# 8. 柱纵筋设计（偏心受压，对称配筋）
# ============================================================

def design_column_long(N_N, M_Nmm, b, h, cfg, mat, col_length):
    """偏心受压纵筋计算（对称配筋，显式迭代求中和轴）.

    对称配筋时轴力平衡方程退化为 N = α₁·fc·b·x（大偏心），
    可直接求 x；小偏心时受拉侧钢筋未屈服，σs 与 x 耦合，
    用二分法在 [ξb·h0, h] 区间迭代满足轴力-弯矩-变形协调.

    N_N     : 轴力 (N, 压力取正值传入)
    M_Nmm   : 弯矩 (N·mm)
    返回 dict: As_one_side(mm²), x(mm), xi, ecc_type, eta, e
    """
    N  = abs(N_N)                      # N
    M  = abs(M_Nmm)                    # N·mm
    fc   = mat["fc"]
    fy   = mat["fy"]
    ft   = mat["ft"]
    Es   = mat["Es"]
    xi_b = mat["xi_b"]
    alpha1 = 1.0
    eps_cu = 0.0033

    cover  = cfg["cover_column_mm"]
    d_bar  = 20
    as_    = cover + d_bar / 2.0      # 纵筋重心距截面近边
    as_p   = as_                       # 对称
    h0     = h - as_

    # 配筋率限值
    rho_min_total = 0.012
    rho_max_side  = 0.012
    As_min = max(rho_min_total * b * h / 2.0, 0.002 * b * h)
    As_max = rho_max_side * b * h

    if N < 1e-3:
        # 接近纯弯，按梁处理最小值
        return {'As_one_side': round(As_min, 1), 'x': 0.0, 'xi': 0.0,
                'ecc_type': 'small_N', 'eta': 1.0, 'e': 0.0}

    # ── P-Δ 放大 ────────────────────────────────────────────────────────
    e0     = M / N
    ea     = max(20.0, h / 30.0)
    ei     = e0 + ea
    zeta_c = min(0.5 * fc * b * h / N, 1.0)
    if ei / h0 > 1e-10:
        eta = 1.0 + (col_length / h) ** 2 / (1300.0 * (ei / h0)) * zeta_c
    else:
        eta = 1.0
    eta = max(eta, 1.0)
    ei_eta = eta * ei
    e  = ei_eta + h / 2.0 - as_       # 轴力至受拉侧钢筋合力点的距离

    # ── 大偏心（ξ ≤ ξb）: 对称配筋时力平衡直接给出 x ─────────────────────
    x_direct = N / (alpha1 * fc * b)
    xi_direct = x_direct / h0

    if xi_direct <= xi_b:
        # ---- 大偏心，直接求解 ----
        x   = max(x_direct, 2.0 * as_p)        # 确保压筋应变 ≥ 0
        As  = (N * e - alpha1 * fc * b * x * (h0 - x / 2.0)) / (fy * (h0 - as_p))
        As  = max(As, As_min)
        As  = min(As, As_max)

        # 验证压筋应力（仅提示，不改变结果）
        eps_s_prime = eps_cu * (x - as_p) / x if x > 1e-6 else 0.0
        sigma_s_prime = min(Es * eps_s_prime, fy) if eps_s_prime > 0 else 0.0

        return {
            'As_one_side': round(max(As, 0.0), 1),
            'x': round(x, 2), 'xi': round(x / h0, 4),
            'ecc_type': 'large_eccentricity',
            'eta': round(eta, 4), 'e': round(e, 2),
            'sigma_s_prime': round(sigma_s_prime, 2),
        }

    else:
        # ── 小偏心（ξ > ξb）: 受拉侧钢筋未屈服，σs 与 x 耦合 ─────────────
        # 弯矩平衡（对受拉钢筋取矩）:
        #   N·e = α₁·fc·b·x·(h0-x/2) + As'·fy·(h0-as')
        #   => As = (N·e - α₁·fc·b·x·(h0-x/2)) / (fy·(h0-as'))
        # 力平衡（对称, As=As'）:
        #   N = α₁·fc·b·x + As·fy - As·σs(x)
        #   => N - α₁·fc·b·x = As·(fy - σs(x))
        # σs(x) = Es·εcu·(h0-x)/x （正=拉，负=压；夹在 [-fy, fy]）
        #
        # 将弯矩方程代入力平衡，得关于 x 的方程 → 二分法求解

        def force_residual(x_try):
            if x_try <= as_p or x_try > h:
                return 1e12
            eps_s    = eps_cu * (h0 - x_try) / x_try
            sigma_s  = max(-fy, min(fy, Es * eps_s))
            Fc       = alpha1 * fc * b * x_try
            M_conc   = Fc * (h0 - x_try / 2.0)
            lever    = h0 - as_p
            if lever < 1e-6:
                return 1e12
            As_try   = max((N * e - M_conc) / (fy * lever), As_min)
            N_calc   = Fc + As_try * fy - As_try * sigma_s   # 对称 As=As'
            return N_calc - N

        x_lo, x_hi = xi_b * h0, min(h - 1.0, h)
        r_lo = force_residual(x_lo)
        r_hi = force_residual(x_hi)

        if r_lo * r_hi > 0:
            x = xi_b * h0        # 符号不变，退化取 ξb 端
        else:
            x = x_lo
            for _ in range(80):
                x_mid = (x_lo + x_hi) / 2.0
                r_mid = force_residual(x_mid)
                if abs(r_mid) < 0.5:  # 收敛阈值 0.5 N
                    x = x_mid
                    break
                if r_lo * r_mid <= 0:
                    x_hi, r_hi = x_mid, r_mid
                else:
                    x_lo, r_lo = x_mid, r_mid
            else:
                x = (x_lo + x_hi) / 2.0

        eps_s   = eps_cu * (h0 - x) / x
        sigma_s = max(-fy, min(fy, Es * eps_s))
        Fc      = alpha1 * fc * b * x
        lever   = h0 - as_p
        As      = max((N * e - Fc * (h0 - x / 2.0)) / (fy * lever), As_min)
        As      = min(As, As_max)

        return {
            'As_one_side': round(max(As, 0.0), 1),
            'x': round(x, 2), 'xi': round(x / h0, 4),
            'ecc_type': 'small_eccentricity',
            'eta': round(eta, 4), 'e': round(e, 2),
            'sigma_s': round(sigma_s, 2),
        }


def design_column_full(elem_idx, envelope, model, cfg):
    """柱纵向钢筋设计（仅纵筋，不含箍筋）."""
    env = envelope[elem_idx]
    elem = model['elements'][elem_idx]
    mat = get_material(cfg)
    sections = model.get('sections', None)

    sec_name = elem['section']
    sec = sections.get(sec_name)
    if sec is None:
        return None

    b = sec['b']
    h = sec['h']
    col_length = elem['L']

    fc = mat["fc"]
    seismic_grade = cfg.get("seismic_grade", 1)
    mu_limit = 0.65 if seismic_grade == 1 else 0.75

    # 取两个控制组合中较大配筋
    res1 = design_column_long(env['N_max'],       env['M_with_Nmax'], b, h, cfg, mat, col_length)
    res2 = design_column_long(env['N_with_Mmax'], env['M_max'],       b, h, cfg, mat, col_length)
    if res1['As_one_side'] >= res2['As_one_side']:
        res = res1
    else:
        res = res2

    N_for_ratio = abs(env['N_max'])   # N
    mu_N = N_for_ratio / (fc * b * h)  # fc N/mm², b·h mm² → 无量纲

    return {
        'As_one_side': res['As_one_side'],
        'As_total':    round(res['As_one_side'] * 2, 1),
        'x':           res['x'],
        'xi':          res['xi'],
        'ecc_type':    res['ecc_type'],
        'eta':         res['eta'],
        'mu_N':        round(mu_N, 3),
        'axial_ratio_ok': mu_N <= mu_limit,
        'b': b, 'h': h,
    }


# ============================================================
# 9. 选筋（仅纵向钢筋）
# ============================================================

def select_longitudinal_rebar(As_required, b, cover, d_stirrup=0, max_types=2,
                               min_d=12, max_d=32):
    """选配纵向钢筋.
    返回 {'bars': '4C20', 'As_actual': xxx, 'n': 4, 'd': 20, 'rows': 1}
    或 {'bars': '4C20+2C18', 'As_actual': xxx, ...}."""
    if As_required <= 0:
        As_required = REBAR_AREAS[min_d] * 2  # 最少2根

    candidates = [d for d in REBAR_DIAMETERS if min_d <= d <= max_d]
    best = None
    best_waste = 1e30

    # 单一直径
    for d in candidates:
        area1 = REBAR_AREAS[d]
        n = max(2, math.ceil(As_required / area1))
        # 偶数优先
        if n % 2 == 1:
            n += 1
        As_actual = n * area1

        # 检查单层排列
        net_width = b - 2 * cover - 2 * d_stirrup
        min_gap = max(25, d)
        needed_width = n * d + (n - 1) * min_gap
        rows = 1
        if needed_width > net_width:
            # 两层
            n_per_row = max(2, int(net_width / (d + min_gap)))
            if n_per_row < 2:
                continue
            rows = math.ceil(n / n_per_row)
            if rows > 2:
                continue

        waste = As_actual - As_required
        if waste >= 0 and waste < best_waste:
            best_waste = waste
            best = {'bars': f"{n}C{d}", 'As_actual': round(As_actual, 1),
                    'n': n, 'diameters': [d], 'rows': rows}

    # 两种直径组合
    for i, d1 in enumerate(candidates):
        for d2 in candidates[i:i+2]:  # 相邻直径
            if d1 == d2:
                continue
            a1 = REBAR_AREAS[d1]
            a2 = REBAR_AREAS[d2]
            for n1 in range(2, 10, 2):
                remaining = As_required - n1 * a1
                if remaining <= 0:
                    break
                n2 = max(2, math.ceil(remaining / a2))
                if n2 % 2 == 1:
                    n2 += 1
                As_actual = n1 * a1 + n2 * a2

                total_n = n1 + n2
                d_max = max(d1, d2)
                net_width = b - 2 * cover - 2 * d_stirrup
                min_gap = max(25, d_max)
                needed_width = total_n * d_max + (total_n - 1) * min_gap
                rows = 1
                if needed_width > net_width:
                    rows = 2
                    if total_n > 12:
                        continue

                waste = As_actual - As_required
                if waste >= 0 and waste < best_waste:
                    best_waste = waste
                    best = {
                        'bars': f"{n1}C{d1}+{n2}C{d2}",
                        'As_actual': round(As_actual, 1),
                        'n': total_n,
                        'diameters': [d1, d2],
                        'rows': rows,
                    }

    if best is None:
        # fallback: 最大直径尽量多根
        d = max_d
        n = max(2, math.ceil(As_required / REBAR_AREAS[d]))
        if n % 2 == 1:
            n += 1
        best = {'bars': f"{n}C{d}", 'As_actual': round(n * REBAR_AREAS[d], 1),
                'n': n, 'diameters': [d], 'rows': 2}

    return best


def select_column_rebar(As_one_side, b, h, cover):
    """选配柱纵筋（对称配筋）.
    As_one_side: 单侧所需面积(mm²)，总面积 = 2 × As_one_side."""
    As_total = 2 * As_one_side
    rebar = select_longitudinal_rebar(As_total, min(b, h), cover,
                                       min_d=16, max_d=32)
    return rebar


def design_beam_waist(b, h, cfg):
    """腰筋需求计算.
    当梁腹板高度 hw = h - 板厚 ≥ 450mm 时需设腰筋（GB 50010-2010 §9.2.13）.
    返回 {'needed': bool, 'hw': mm, 'As_side': mm²} 或 None."""
    slab_t = cfg["slab_thickness_mm"]
    hw = h - slab_t
    if hw < 450:
        return None
    As_side = 0.001 * b * hw   # 单侧: ρsv ≥ 0.1% × b × hw
    return {'needed': True, 'hw': hw, 'As_side': As_side}


def select_waist_rebar(As_side_required, hw):
    """选配腰筋（每侧）.
    As_side_required: mm²  hw: mm (腹板高度)
    返回 {'spec': '2C12/side', 'As_side_actual': mm²}."""
    if As_side_required <= 0:
        return None
    max_spacing = 200   # mm，腰筋间距不宜大于 200mm
    for d in [10, 12, 14, 16]:
        a1 = REBAR_AREAS[d]
        n = max(2, math.ceil(As_side_required / a1))
        # 间距校核：若间距超限则增加根数
        spacing = hw / (n + 1)
        if spacing > max_spacing:
            n = max(n, math.ceil(hw / max_spacing) - 1)
        As_actual = n * a1
        if As_actual >= As_side_required:
            return {'spec': f"{n}C{d}/side", 'As_side_actual': round(As_actual, 1)}
    # fallback
    d, n = 16, max(2, math.ceil(As_side_required / REBAR_AREAS[16]))
    return {'spec': f"{n}C{d}/side", 'As_side_actual': round(n * REBAR_AREAS[d], 1)}


# ============================================================
# 10. 可视化
# ============================================================

def plot_frame_internal_force(model, envelope, layout, force_type, filename):
    """绘制一榀框架的弯矩图或剪力图.
    force_type: 'M' or 'V'.
    内力数据单位: N·mm (弯矩) / N (剪力)；图中标注转换为 kN·m / kN 显示."""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    nodes = model['nodes']
    elements = model['elements']

    for elem in elements:
        ni, nj = elem['ni'], elem['nj']
        xi, zi = nodes[ni]; xj, zj = nodes[nj]
        color = 'blue' if elem['type'] == 'beam' else 'red'
        ax.plot([xi, xj], [zi, zj], color=color, linewidth=2, zorder=1)

    # 确定比例尺（使用原始 N/N·mm 量级）
    max_val = 0
    for idx, env in envelope.items():
        if env['type'] == 'beam':
            vals = ([abs(env.get('M_neg_left', 0)), abs(env.get('M_pos_mid', 0)),
                     abs(env.get('M_neg_right', 0))] if force_type == 'M'
                    else [env.get('V_max_left', 0), env.get('V_max_right', 0)])
        else:
            vals = [env.get('M_max', 0)] if force_type == 'M' else [env.get('V_max', 0)]
        max_val = max(max_val, max(abs(v) for v in vals) if vals else 0)

    grid_x = layout["geometry"]["grid_x"]
    typical_span = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 3600
    scale = typical_span * 0.3 / max_val if max_val > 0 else 1.0

    # 弯矩显示系数: N·mm → kN·m (÷1e6)；剪力: N → kN (÷1e3)
    disp_factor = 1e-6 if force_type == 'M' else 1e-3

    for idx, env in envelope.items():
        elem = elements[idx]
        ni, nj = elem['ni'], elem['nj']
        xi, zi = nodes[ni]; xj, zj = nodes[nj]

        if env['type'] == 'beam' and force_type == 'M':
            M_left  = env.get('M_neg_left',  0)
            M_right = env.get('M_neg_right', 0)
            M_mid   = env.get('M_pos_mid',   0)
            n_pts   = 20
            xs = np.linspace(xi, xj, n_pts)
            zs = np.array([
                zi - (M_left * (1 - t) + M_right * t
                      + 4 * M_mid * t * (1 - t) * (1 if M_mid > 0 else 0)) * scale
                for t in np.linspace(0, 1, n_pts)
            ])
            ax.fill_between(xs, zi, zs, alpha=0.3, color='green')
            ax.plot(xs, zs, color='green', linewidth=0.8)
            ax.annotate(f"{M_left * disp_factor:.2f}", (xi, zi),
                        fontsize=6, color='darkgreen', ha='left',  va='bottom')
            ax.annotate(f"{M_mid * disp_factor:.2f}", ((xi+xj)/2, zi),
                        fontsize=6, color='darkgreen', ha='center', va='top')
            ax.annotate(f"{M_right * disp_factor:.2f}", (xj, zi),
                        fontsize=6, color='darkgreen', ha='right', va='bottom')

        elif env['type'] == 'beam' and force_type == 'V':
            V_left  = env.get('V_max_left',  0)
            V_right = env.get('V_max_right', 0)
            xs = [xi, xi, xj, xj]
            zs = [zi, zi + V_left * scale, zi - V_right * scale, zi]
            ax.fill(xs, zs, alpha=0.3, color='orange')
            ax.plot(xs, zs, color='orange', linewidth=0.8)
            ax.annotate(f"{V_left * disp_factor:.2f}", (xi, zi),
                        fontsize=6, color='darkorange', ha='left')
            ax.annotate(f"{V_right * disp_factor:.2f}", (xj, zi),
                        fontsize=6, color='darkorange', ha='right')

        elif env['type'] == 'column' and force_type == 'M':
            M_max  = env.get('M_max', 0)
            offset = M_max * scale * 0.5
            zm = (zi + zj) / 2
            ax.plot([xi, xi + offset, xi - offset, xi],
                    [zi, zm, zm, zj], color='purple', linewidth=0.8)

    for nid in model['fixed_nodes']:
        x, z = nodes[nid]
        ax.plot(x, z, '^', color='black', markersize=10, zorder=5)

    unit_str = 'kN·m' if force_type == 'M' else 'kN'
    title_map = {'M': f'Bending Moment Diagram ({unit_str})',
                 'V': f'Shear Force Diagram ({unit_str})'}
    frame_info = model['frame_info']
    axis_label = (f"{'X' if frame_info['axis']=='X' else 'Y'}-Frame @ "
                  f"{'Y' if frame_info['axis']=='X' else 'X'}={frame_info['axis_index']}")
    ax.set_title(f"{axis_label} - {title_map[force_type]}")
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_aspect('auto')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_story_shear(seismic_data, layout, filename):
    """绘制楼层剪力分布图."""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    num_stories = len(seismic_data["floor_shears"])
    stories = list(range(1, num_stories + 1))
    # 内力单位为 N，图中显示 kN
    shears_kN = [s / 1000.0 for s in seismic_data["floor_shears"]]
    F_EK_kN   = seismic_data['F_EK'] / 1000.0

    ax.barh(stories, shears_kN, color='steelblue', edgecolor='navy', height=0.6)

    for i, v in enumerate(shears_kN):
        ax.text(v + max(shears_kN) * 0.02, stories[i], f"{v:.1f} kN", va='center', fontsize=8)

    ax.set_xlabel('Story Shear (kN)')
    ax.set_ylabel('Story')
    ax.set_yticks(stories)
    ax.set_title(f"Story Shear Distribution\n"
                 f"T1={seismic_data['T1']}s, alpha1={seismic_data['alpha1']}, "
                 f"F_EK={F_EK_kN:.2f} kN")
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_reinforcement_plan(layout, all_beam_results, all_col_results, filename, story_label=None):
    """绘制平面配筋汇总图."""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    nodes = layout["nodes"]
    story_key = next(iter(layout["beams"]))
    beams_data = layout["beams"][story_key]
    col_key = next(iter(layout["columns"]))
    cols_data = layout["columns"][col_key]

    # 绘制网格线和节点
    for nk, ninfo in nodes.items():
        ax.plot(ninfo['x'], ninfo['y'], 'ko', markersize=4)

    # 绘制梁
    for bk, binfo in beams_data.items():
        ni_name = binfo['i_node']
        nj_name = binfo['j_node']
        xi = nodes[ni_name]['x']
        yi = nodes[ni_name]['y']
        xj = nodes[nj_name]['x']
        yj = nodes[nj_name]['y']
        ax.plot([xi, xj], [yi, yj], 'b-', linewidth=1.5)

        # 标注配筋
        if bk in all_beam_results:
            br = all_beam_results[bk]
            mid_x = (xi + xj) / 2
            mid_y = (yi + yj) / 2
            text = f"{br.get('top_bars', '')}/{br.get('bot_bars', '')}"
            if br.get('stirrup_spec'):
                text += f"\n{br['stirrup_spec']}"
            offset = 200 if binfo['direction'] == 'X' else -400
            ax.annotate(text, (mid_x, mid_y + offset), fontsize=5,
                       ha='center', color='darkblue',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))

    # 绘制柱
    for ck, cinfo in cols_data.items():
        node_name = cinfo['node']
        x = nodes[node_name]['x']
        y = nodes[node_name]['y']
        sec_name = cinfo['section']
        sec = layout['sections'][sec_name]
        b = sec['b']
        h = sec['h']
        rect = patches.Rectangle((x - b/2, y - h/2), b, h,
                                  linewidth=1.5, edgecolor='red', facecolor='lightyellow')
        ax.add_patch(rect)

        if ck in all_col_results:
            cr = all_col_results[ck]
            text = cr.get('rebar_text', '')
            ax.annotate(text, (x, y), fontsize=5, ha='center', va='center', color='darkred')

    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    title = f'Reinforcement Plan - Story {story_label}' if story_label else 'Reinforcement Plan'
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 11. 主函数
# ============================================================

def run_analysis(layout, cfg):
    """执行完整分析流程."""
    print("=" * 60)
    print("RC Frame Analysis - Start")
    print("=" * 60)

    mat = get_material(cfg)
    sections = layout["sections"]
    num_stories = layout["parameters"]["num_stories"]

    # Step 1: 楼板检测
    print("[1/8] Detecting slab panels...")
    slab_panels = detect_slab_panels(layout)
    print(f"  Found {len(slab_panels)} slab panels")

    # Step 2: 提取2D框架
    print("[2/8] Extracting 2D frames...")
    frames_x, frames_y = extract_2d_frames(layout)
    n_fx = sum(len(v) for v in frames_x.values())
    n_fy = sum(len(v) for v in frames_y.values())
    print(f"  X-frames: {n_fx}, Y-frames: {n_fy}")

    # Step 3: 荷载计算
    print("[3/8] Calculating beam loads...")
    beam_loads = calc_beam_loads(layout, cfg, slab_panels)

    # Step 4: 层重力荷载
    print("[4/8] Calculating story weights...")
    Gi_list = calc_story_weights(layout, cfg, slab_panels, beam_loads)
    for i, g in enumerate(Gi_list):
        print(f"  Story {i+1}: Gi = {g:.0f} N  ({g/1000:.1f} kN)")

    # Step 5: 底部剪力法
    print("[5/8] Base shear method...")
    seismic_data = base_shear_method(layout, cfg, Gi_list)
    print(f"  T1={seismic_data['T1']}s, alpha1={seismic_data['alpha1']}, "
          f"F_EK={seismic_data['F_EK']:.0f} N  ({seismic_data['F_EK']/1000:.2f} kN)")

    # Step 6: 分配水平力到各框架
    print("[6/8] Distributing forces to frames...")
    fx_forces_x, fx_D_x = distribute_forces_to_frames(seismic_data, frames_x, layout, cfg)
    fy_forces_y, fy_D_y = distribute_forces_to_frames(seismic_data, frames_y, layout, cfg)

    # Step 7: 内力分析+配筋
    print("[7/8] Frame analysis and design...")
    all_frame_results = {}
    all_beam_reinf = {}
    all_col_reinf = {}

    def process_frames(frames_dict, frame_forces, direction):
        for key, sub_list in frames_dict.items():
            for idx, sf in enumerate(sub_list):
                fk = f"{key}_{idx}"
                print(f"  Analyzing {direction}-frame {fk}...")

                model = build_full_frame_model(sf, layout, cfg)
                model['sections'] = sections

                sf_forces = frame_forces.get(fk, [0.0] * num_stories)

                results = analyze_frame(model, layout, cfg, beam_loads, sf_forces)
                envelope = combine_load_cases(results)
                calc_beam_midspan_moments(envelope, model, beam_loads)

                # 纵筋设计
                frame_beam_reinf = {}
                frame_col_reinf = {}
                for eidx, env in envelope.items():
                    if env['type'] == 'beam':
                        design = design_beam_full(eidx, envelope, model, cfg)
                        if design:
                            bk = env['key']
                            top_res = design['top']
                            mid_res = design['bot_mid']

                            top_rebar = select_longitudinal_rebar(
                                top_res['As'], design['b'], cfg['cover_beam_mm'])
                            bot_rebar = select_longitudinal_rebar(
                                mid_res['As'], design['b'], cfg['cover_beam_mm'])

                            reinf = {
                                'top': {
                                    'As_calc':    top_res['As'],
                                    'As_prime':   top_res['As_prime'],
                                    'section_type': top_res['section_type'],
                                    'xi':         top_res['xi'],
                                    'x_mm':       top_res['x'],
                                    'bars':       top_rebar['bars'],
                                    'As_actual':  top_rebar['As_actual'],
                                },
                                'bot_mid': {
                                    'As_calc':    mid_res['As'],
                                    'As_prime':   mid_res['As_prime'],
                                    'section_type': mid_res['section_type'],
                                    'xi':         mid_res['xi'],
                                    'x_mm':       mid_res['x'],
                                    'bars':       bot_rebar['bars'],
                                    'As_actual':  bot_rebar['As_actual'],
                                },
                            }

                            # 腰筋选配
                            waist_info = design.get('waist')
                            if waist_info:
                                wr = select_waist_rebar(waist_info['As_side'], waist_info['hw'])
                                if wr:
                                    reinf['waist'] = {
                                        'hw_mm':          waist_info['hw'],
                                        'As_side_calc':   round(waist_info['As_side'], 1),
                                        'spec':           wr['spec'],
                                        'As_side_actual': wr['As_side_actual'],
                                    }

                            frame_beam_reinf[bk] = reinf
                            story = env['story']
                            result_key = f"{bk}_story{story:02d}"
                            all_beam_reinf[result_key] = {
                                'beam_key': bk, 'story': story,
                                # 内力包络 (N, N·mm)
                                'envelope': {
                                    'M_neg_left_Nmm':  round(env.get('M_neg_left',  0), 0),
                                    'M_neg_right_Nmm': round(env.get('M_neg_right', 0), 0),
                                    'M_pos_mid_Nmm':   round(env.get('M_pos_mid',   0), 0),
                                    'V_max_left_N':    round(env.get('V_max_left',  0), 0),
                                    'V_max_right_N':   round(env.get('V_max_right', 0), 0),
                                },
                                'reinforcement': reinf,
                                'top_bars': top_rebar['bars'],
                                'bot_bars': bot_rebar['bars'],
                            }

                    elif env['type'] == 'column':
                        design = design_column_full(eidx, envelope, model, cfg)
                        if design:
                            ck = env['key']
                            col_rebar = select_column_rebar(
                                design['As_one_side'], design['b'], design['h'],
                                cfg['cover_column_mm'])

                            story = env['story']
                            result_key = f"{ck}_story{story:02d}"
                            col_result = {
                                'col_key': ck, 'story': story,
                                # 内力包络 (N, N·mm)
                                'envelope': {
                                    'N_max_N':  round(env.get('N_max', 0), 0),
                                    'M_max_Nmm': round(env.get('M_max', 0), 0),
                                    'V_max_N':  round(env.get('V_max', 0), 0),
                                },
                                'reinforcement': {
                                    'As_one_side_calc': design['As_one_side'],
                                    'As_total_calc':    design['As_total'],
                                    'bars':             col_rebar['bars'],
                                    'As_actual':        col_rebar['As_actual'],
                                    'x_mm':             design['x'],
                                    'xi':               design['xi'],
                                    'ecc_type':         design['ecc_type'],
                                    'eta':              design['eta'],
                                },
                                'mu_N': design['mu_N'],
                                'axial_ratio_ok': design['axial_ratio_ok'],
                                'rebar_text': col_rebar['bars'],
                            }
                            mu_limit_val = 0.65 if cfg.get('seismic_grade', 1) == 1 else 0.75
                            if result_key in all_col_reinf:
                                existing = all_col_reinf[result_key]
                                if col_result['reinforcement']['As_one_side_calc'] > existing['reinforcement']['As_one_side_calc']:
                                    all_col_reinf[result_key] = col_result
                                # mu_N 取两方向最大值
                                max_mu = max(all_col_reinf[result_key]['mu_N'], col_result['mu_N'])
                                all_col_reinf[result_key]['mu_N'] = max_mu
                                all_col_reinf[result_key]['axial_ratio_ok'] = max_mu <= mu_limit_val
                            else:
                                all_col_reinf[result_key] = col_result

                all_frame_results[f"{direction}_{fk}"] = {
                    'model': model, 'envelope': envelope,
                    'beam_reinf': frame_beam_reinf,
                    'col_reinf': frame_col_reinf,
                }

    process_frames(frames_x, fx_forces_x, 'X')
    process_frames(frames_y, fy_forces_y, 'Y')

    print("[8/8] Analysis complete.")

    return {
        'seismic': seismic_data,
        'Gi_list': Gi_list,
        'slab_panels': list(slab_panels),
        'frames_x': {str(k): len(v) for k, v in frames_x.items()},
        'frames_y': {str(k): len(v) for k, v in frames_y.items()},
        'all_frame_results': all_frame_results,
        'all_beam_reinf': all_beam_reinf,
        'all_col_reinf': all_col_reinf,
    }


def build_output_json(layout, cfg, analysis):
    """组装输出JSON."""
    output = {
        "__source_layout": "",
        "design_parameters": cfg,
        "seismic": analysis['seismic'],
        "story_weights": [round(g, 1) for g in analysis['Gi_list']],
    }

    # 梁结果
    beams_output = {}
    for rk, rdata in analysis['all_beam_reinf'].items():
        beams_output[rk] = {
            'beam_key': rdata['beam_key'],
            'story': rdata['story'],
            'envelope': rdata['envelope'],
            'reinforcement': rdata['reinforcement'],
        }
    output['beams'] = beams_output

    # 柱结果
    columns_output = {}
    for rk, rdata in analysis['all_col_reinf'].items():
        columns_output[rk] = {
            'col_key': rdata['col_key'],
            'story': rdata['story'],
            'envelope': rdata['envelope'],
            'reinforcement': rdata['reinforcement'],
            'mu_N': rdata['mu_N'],
            'axial_ratio_ok': rdata['axial_ratio_ok'],
        }
    output['columns'] = columns_output

    return output


def check_axial_ratios(analysis):
    """检查所有柱的轴压比, 返回最大 mu_N."""
    max_mu = 0.0
    for rk, rdata in analysis['all_col_reinf'].items():
        mu = rdata['mu_N']
        if mu > max_mu:
            max_mu = mu
    return max_mu


def main():
    parser = argparse.ArgumentParser(description='RC Frame Analysis')
    parser.add_argument('--input', required=True, help='Input layout JSON file')
    parser.add_argument('--config', default=None, help='Optional config JSON file')
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    layout = load_layout(input_path)
    cfg = load_config(args.config, DEFAULT_CONFIG)

    # 轴压比自动调整: 超过0.65则增大柱截面, 重新分析
    mu_limit = 0.65
    max_iter = 10
    for iteration in range(max_iter):
        analysis = run_analysis(layout, cfg)
        max_mu = check_axial_ratios(analysis)
        if max_mu <= mu_limit:
            print(f"  Axial ratio check PASSED: max mu_N = {max_mu:.3f} <= {mu_limit}")
            break
        else:
            # 增大所有柱截面 50mm
            print(f"  Axial ratio EXCEEDED: max mu_N = {max_mu:.3f} > {mu_limit}")
            for sec_name, sec_data in layout["sections"].items():
                if "column" in sec_name:
                    old_b = sec_data["b"]
                    old_h = sec_data["h"]
                    sec_data["b"] = old_b + 50
                    sec_data["h"] = old_h + 50
                    print(f"  Enlarged {sec_name}: {old_b}x{old_h} -> {sec_data['b']}x{sec_data['h']}")
            if iteration == max_iter - 1:
                print(f"  WARNING: Max iterations reached, mu_N={max_mu:.3f} still > {mu_limit}")

    # 输出目录: 在输入文件目录下新建 analysis/layout{n}/ 子文件夹
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    m = re.match(r'(layout\d+)_', input_basename)
    layout_subdir = m.group(1) if m else "unknown"
    output_dir = os.path.join(os.path.dirname(input_path), "analysis", layout_subdir)
    os.makedirs(output_dir, exist_ok=True)

    output_json = build_output_json(layout, cfg, analysis)
    output_json["__source_layout"] = os.path.basename(input_path)

    json_path = os.path.join(output_dir, f"analysis_{input_basename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"JSON output: {json_path}")

    # 可视化
    prefix = os.path.join(output_dir, f"analysis_{input_basename}")

    # 楼层剪力图
    plot_story_shear(analysis['seismic'], layout, f"{prefix}_story_shear.png")
    print(f"Plot: {prefix}_story_shear.png")

    # 逐榀框架弯矩图和剪力图
    for fk, fdata in analysis['all_frame_results'].items():
        model = fdata['model']
        envelope = fdata['envelope']
        safe_fk = fk.replace(' ', '_')

        plot_frame_internal_force(model, envelope, layout, 'M',
                                  f"{prefix}_frame_{safe_fk}_M.png")
        plot_frame_internal_force(model, envelope, layout, 'V',
                                  f"{prefix}_frame_{safe_fk}_V.png")

    # 每层平面配筋图
    num_stories = layout["parameters"]["num_stories"]
    for story in range(1, num_stories + 1):
        beam_plan = {}
        for rk, rdata in analysis['all_beam_reinf'].items():
            if rdata['story'] == story:
                bk = rdata['beam_key']
                beam_plan[bk] = rdata

        col_plan = {}
        for rk, rdata in analysis['all_col_reinf'].items():
            if rdata['story'] == story:
                ck = rdata['col_key']
                col_plan[ck] = rdata

        plot_reinforcement_plan(layout, beam_plan, col_plan,
                                f"{prefix}_reinforcement_story{story:02d}.png",
                                story_label=story)
        print(f"Plot: {prefix}_reinforcement_story{story:02d}.png")

    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
