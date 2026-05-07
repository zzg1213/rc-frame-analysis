#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RC Frame Layout Generator (2D X-Z / X-Y)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    HAS_MPL = True
except Exception:
    HAS_MPL = False


DEFAULT_CONFIG: Dict[str, Any] = {
    "plane": "XY",  # 平面类型："XY" 或 "XZ"
    "layout_type": "layout6",  # ?????"full" ? "layout6"
    "num_spans_range": [5, 5],  # X ????????????
    "num_stories": 8,  # 固定层数（优先于 num_stories_range）
    "num_stories_range": [6, 6],  # 层数随机范围（含端点）
    "num_spans_y_range": [7, 7],  # Y ??????????????
    "span_length_x_mm": None,  # X 向跨长固定值；None 表示从范围采样并重复
    "span_length_y_other_mm": None,  # Y 向其余跨长度；None 表示从范围采样并重复
    "span_length_range_mm": [3300, 6000],  # X 向跨长采样范围（mm）
    "span_length_y_range_mm": [3000, 6000],  # Y 向跨长采样范围（mm）
    "story_height_range_mm": [3000, 4200],  # 层高采样范围（mm，300mm 步长）
    "beam_section_range_mm": {"b": [250, 400], "h": [450, 700]},  # 梁截面范围（mm）
    "column_section_range_mm": {"b": [300, 600], "h": [300, 600]},  # 柱截面范围（mm）
    "column_vary_by_story": False,  # 是否按层变化柱截面
    "max_attempts": 20,  # 生成失败时最大重试次数
}



# 深度合并配置字典（递归覆盖）
def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


# 读取配置文件并应用覆盖
def load_config(path: str | None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        cfg = deep_update(cfg, overrides)
    return cfg


# 创建可复现的随机数生成器
def make_generator(seed: int, device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


# 采样整数（闭区间）
def sample_int(g: torch.Generator, device: torch.device, low: int, high: int) -> int:
    if low == high:
        return int(low)
    return int(torch.randint(low, high + 1, (1,), generator=g, device=device).item())


# 采样浮点数（闭区间）
def sample_float(
    g: torch.Generator, device: torch.device, low: float, high: float
) -> float:
    if math.isclose(low, high):
        return float(low)
    r = torch.rand(1, generator=g, device=device).item()
    return float(low + (high - low) * r)


# 从候选列表中随机选择一个
def sample_choice(g: torch.Generator, device: torch.device, choices: List[Any]) -> Any:
    idx = sample_int(g, device, 0, len(choices) - 1)
    return choices[idx]



# 按指定步长量化长度，并限制在范围内
def quantize_length(value: float, step: float, low: float | None = None, high: float | None = None) -> float:
    if step <= 0:
        raise ValueError("step must be positive.")
    q = round(value / step) * step
    if low is not None and q < low:
        q = math.ceil(low / step) * step
    if high is not None and q > high:
        q = math.floor(high / step) * step
    return float(q)


# 采样奇数整数（用于 Y 向跨数）
def sample_odd_int(
    g: torch.Generator, device: torch.device, low: int, high: int
) -> int:
    if low == high:
        if low % 2 == 0:
            raise ValueError("num_spans_y must be odd.")
        return int(low)
    for _ in range(20):
        val = sample_int(g, device, low, high)
        if val % 2 == 1:
            return val
    # 兜底：在范围内取最近的奇数
    if low % 2 == 1:
        return low
    if high % 2 == 1:
        return high
    raise ValueError("No odd number in the given range for num_spans_y.")


# 由跨度/层高列表生成累计坐标
def cumulative_positions(lengths: List[float]) -> List[float]:
    coords = [0.0]
    total = 0.0
    for L in lengths:
        total += L
        coords.append(total)
    return coords


# 生成平面布局的有效节点集合（用于凹凸布局）
def build_active_nodes(
    num_spans_x: int, num_spans_y: int, layout_type: str | None
) -> set[tuple[int, int]]:
    nodes = {
        (i, j) for j in range(num_spans_y + 1) for i in range(num_spans_x + 1)
    }
    if not layout_type or layout_type == "full":
        return nodes
    if layout_type == "layout6":
        # layout6 对应布局6.png：右侧中部缺口（忽略图内开洞）
        if num_spans_x != 5 or num_spans_y != 7:
            raise ValueError("layout6 requires num_spans_x=5 and num_spans_y=7.")
        # 缺口为 2x2 节点块：i=4..5, j=3..4（j=0 为底部）
        for j in (3, 4):
            for i in (4, 5):
                nodes.discard((i, j))
        return nodes
    raise ValueError(f"unsupported layout_type: {layout_type}")

def format_story_label(story_index: int) -> str:
    return f"{story_index:02d}"


# 解析楼层标签的起止索引
def parse_story_label(label: str) -> Tuple[int, int]:
    if "-" in label:
        start_str, end_str = label.split("-", 1)
        return int(start_str), int(end_str)
    value = int(label)
    return value, value


# 按内容合并连续楼层区间（如 01-05）
def group_story_ranges(
    story_map: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not story_map:
        return {}
    ordered_labels = sorted(story_map.keys(), key=lambda k: parse_story_label(k)[0])
    merged: Dict[str, Dict[str, Dict[str, Any]]] = {}
    run_start = ordered_labels[0]
    run_end = ordered_labels[0]
    run_value = story_map[ordered_labels[0]]
    for label in ordered_labels[1:]:
        value = story_map[label]
        prev_end = parse_story_label(run_end)[1]
        curr_start, curr_end = parse_story_label(label)
        if value == run_value and curr_start == prev_end + 1 and curr_start == curr_end:
            run_end = label
            continue
        start_idx, _ = parse_story_label(run_start)
        end_idx, _ = parse_story_label(run_end)
        if start_idx == end_idx:
            key = format_story_label(start_idx)
        else:
            key = f"{format_story_label(start_idx)}-{format_story_label(end_idx)}"
        merged[key] = run_value
        run_start = label
        run_end = label
        run_value = value
    start_idx, _ = parse_story_label(run_start)
    end_idx, _ = parse_story_label(run_end)
    if start_idx == end_idx:
        key = format_story_label(start_idx)
    else:
        key = f"{format_story_label(start_idx)}-{format_story_label(end_idx)}"
    merged[key] = run_value
    return merged


# 强制合并为完整楼层区间（如 01-06）
def collapse_to_full_range(
    story_map: Dict[str, Dict[str, Dict[str, Any]]], num_stories: int
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not story_map:
        return {}
    first_label = sorted(story_map.keys(), key=lambda k: parse_story_label(k)[0])[0]
    value = story_map[first_label]
    if num_stories <= 1:
        key = format_story_label(1)
    else:
        key = f"{format_story_label(1)}-{format_story_label(num_stories)}"
    return {key: value}


# 由网格坐标生成平面节点字典
def build_nodes_dict(
    x_coords: List[float],
    y_coords: List[float],
    active_nodes: set[tuple[int, int]] | None = None,
) -> Dict[str, Dict[str, float]]:
    nodes_out: Dict[str, Dict[str, float]] = {}
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            if active_nodes is not None and (i, j) not in active_nodes:
                continue
            nodes_out[f"N_{i}_{j}"] = {"x": float(x), "y": float(y)}
    return nodes_out


# 构建截面输出字典及截面 ID 到名称的映射
def build_section_maps(
    sections: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, str]]:
    sections_out: Dict[str, Dict[str, float]] = {}
    id_to_name: Dict[int, str] = {}
    beam_idx = 1
    column_idx = 1
    for sec in sections:
        if sec["type"] == "beam":
            name = f"beam{beam_idx}"
            beam_idx += 1
        else:
            name = f"column{column_idx}"
            column_idx += 1
        sections_out[name] = {"b": float(sec["b"]), "h": float(sec["h"])}
        id_to_name[int(sec["id"])] = name
    return sections_out, id_to_name


# 按层生成柱构件信息（平面布局）
def build_columns_by_story(
    elements: List[Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    section_name_by_id: Dict[int, str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    columns_by_story: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in elements:
        if e["type"] != "column":
            continue
        ni = nodes_by_id[e["ni"]]
        nj = nodes_by_id[e["nj"]]
        story = max(int(ni["story"]), int(nj["story"]))
        story_label = format_story_label(story)
        columns_by_story.setdefault(story_label, {})
        i = int(ni["grid_i"])
        j = int(ni["grid_j"])
        key = f"C_{i}_{j}"
        length = abs(float(nj["z"]) - float(ni["z"]))
        section_name = section_name_by_id.get(int(e["section_id"]), "column1")
        columns_by_story[story_label][key] = {
            "node": f"N_{i}_{j}",
            "direction": "Z",
            "length": length,
            "section": section_name,
            "reinforcement_id": "RC_01",
        }
    return columns_by_story


# 按层生成梁构件信息（平面布局）
def build_beams_by_story(
    elements: List[Dict[str, Any]],
    nodes_by_id: Dict[int, Dict[str, Any]],
    section_name_by_id: Dict[int, str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    beams_by_story: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in elements:
        if e["type"] != "beam":
            continue
        ni = nodes_by_id[e["ni"]]
        nj = nodes_by_id[e["nj"]]
        story = int(ni["story"])
        story_label = format_story_label(story)
        beams_by_story.setdefault(story_label, {})
        dir_flag = e.get("dir")
        if dir_flag == "x":
            if int(ni["grid_i"]) <= int(nj["grid_i"]):
                left, right = ni, nj
            else:
                left, right = nj, ni
            i = int(left["grid_i"])
            j = int(left["grid_j"])
            key = f"B_{i}_{j}_1"
            i_node = f"N_{i}_{j}"
            j_node = f"N_{int(right['grid_i'])}_{int(right['grid_j'])}"
            length = abs(float(right["x"]) - float(left["x"]))
            direction = "X"
        else:
            if int(ni["grid_j"]) <= int(nj["grid_j"]):
                down, up = ni, nj
            else:
                down, up = nj, ni
            i = int(down["grid_i"])
            j = int(down["grid_j"])
            key = f"B_{i}_{j}_2"
            i_node = f"N_{i}_{j}"
            j_node = f"N_{int(up['grid_i'])}_{int(up['grid_j'])}"
            length = abs(float(up["y"]) - float(down["y"]))
            direction = "Y"
        section_name = section_name_by_id.get(int(e["section_id"]), "beam1")
        beams_by_story[story_label][key] = {
            "i_node": i_node,
            "j_node": j_node,
            "direction": direction,
            "length": length,
            "section": section_name,
            "reinforcement_id": "RB_01",
        }
    return beams_by_story


# 计算 XZ 平面节点编号
def node_id(story_index: int, grid_x_index: int, num_spans: int) -> int:
    return story_index * (num_spans + 1) + grid_x_index


# 计算 XY 平面节点编号
def node_id_xy(
    level_index: int, grid_y_index: int, grid_x_index: int, num_spans_x: int, num_spans_y: int
) -> int:
    return (
        level_index * (num_spans_x + 1) * (num_spans_y + 1)
        + grid_y_index * (num_spans_x + 1)
        + grid_x_index
    )


# 选择水平荷载输入类型
def choose_horizontal_input(
    g: torch.Generator, device: torch.device, mode: str
) -> str:
    if mode in ("F_story", "V_base"):
        return mode
    return sample_choice(g, device, ["F_story", "V_base"])


# 按规则分配层间水平力
def distribute_story_forces(
    z_coords: List[float], V_base: float, rule: str, w_power: float
) -> List[float]:
    story_heights = z_coords[1:]
    if not story_heights:
        return []

    if rule == "uniform":
        weights = [1.0 for _ in story_heights]
    elif rule == "w_power":
        weights = [h**w_power for h in story_heights]
    else:
        weights = [h for h in story_heights]

    total = sum(weights)
    if math.isclose(total, 0.0):
        return [0.0 for _ in story_heights]
    return [V_base * w / total for w in weights]


# 构建布局模型（仅输出布局与构件尺寸）
def build_model(
    cfg: Dict[str, Any], base_seed: int, index: int, device: torch.device
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    g = make_generator(base_seed + index, device)
    plane = cfg.get("plane", "XY").upper()

    if plane == "XY":
        num_spans_x = cfg.get("num_spans") or sample_int(
            g, device, cfg["num_spans_range"][0], cfg["num_spans_range"][1]
        )
        num_spans_y = cfg.get("num_spans_y")
        if num_spans_y is None:
            num_spans_y = sample_odd_int(
                g, device, cfg["num_spans_y_range"][0], cfg["num_spans_y_range"][1]
            )
        elif num_spans_y % 2 == 0:
            raise ValueError("num_spans_y must be odd.")

        span_lengths_x = cfg.get("span_lengths")
        if not span_lengths_x:
            if cfg.get("span_length_x_mm") is None:
                span_x = sample_float(
                    g, device, cfg["span_length_range_mm"][0], cfg["span_length_range_mm"][1]
                )
            else:
                span_x = float(cfg["span_length_x_mm"])
            span_x = quantize_length(
                span_x,
                300.0,
                cfg["span_length_range_mm"][0],
                cfg["span_length_range_mm"][1],
            )
            span_lengths_x = [span_x for _ in range(num_spans_x)]
        else:
            if cfg.get("num_spans") is not None and len(span_lengths_x) != num_spans_x:
                raise ValueError("span_lengths length must match num_spans.")
            if len({round(float(v), 6) for v in span_lengths_x}) != 1:
                raise ValueError("span_lengths must be uniform in X direction.")
            for v in span_lengths_x:
                if float(v) % 300.0 != 0:
                    raise ValueError("span_lengths must be multiples of 300 mm.")
            num_spans_x = len(span_lengths_x)

        span_lengths_y = cfg.get("span_lengths_y")
        if not span_lengths_y:
            y_low = cfg["span_length_y_range_mm"][0]
            y_high = cfg["span_length_y_range_mm"][1]
            if y_high - y_low < 300.0:
                raise ValueError("span_length_y_range_mm must be at least 300mm wide.")
            span_y_value = cfg.get("span_length_y_other_mm")
            if span_y_value is None:
                span_y_value = cfg.get("span_length_y_mid_mm")
            if span_y_value is None:
                span_y = sample_float(
                    g,
                    device,
                    y_low,
                    y_high,
                )
            else:
                span_y = float(span_y_value)
            span_y = quantize_length(
                span_y,
                300.0,
                y_low,
                y_high,
            )
            span_lengths_y = [span_y for _ in range(num_spans_y)]
        else:
            if cfg.get("num_spans_y") is not None and len(span_lengths_y) != num_spans_y:
                raise ValueError("span_lengths_y length must match num_spans_y.")
            num_spans_y = len(span_lengths_y)
            if num_spans_y % 2 == 0:
                raise ValueError("span_lengths_y length must be odd.")
            span_lengths_y = [float(v) for v in span_lengths_y]
            if len({round(v, 6) for v in span_lengths_y}) != 1:
                raise ValueError("span_lengths_y must be uniform in Y direction.")
            for v in span_lengths_y:
                if v % 300.0 != 0:
                    raise ValueError("span_lengths_y must be multiples of 300 mm.")

        if any(L <= 0 for L in span_lengths_x) or any(L <= 0 for L in span_lengths_y):
            raise ValueError("Invalid span length (<= 0).")

        num_stories = cfg.get("num_stories") or sample_int(
            g, device, cfg["num_stories_range"][0], cfg["num_stories_range"][1]
        )
        story_heights = cfg.get("story_heights")
        if not story_heights:
            height = quantize_length(
                sample_float(
                    g,
                    device,
                    cfg["story_height_range_mm"][0],
                    cfg["story_height_range_mm"][1],
                ),
                300.0,
                cfg["story_height_range_mm"][0],
                cfg["story_height_range_mm"][1],
            )
            story_heights = [height for _ in range(num_stories)]
        else:
            if cfg.get("num_stories") is not None and len(story_heights) not in (1, num_stories):
                raise ValueError("story_heights length must be 1 or match num_stories.")
            for v in story_heights:
                if float(v) % 300.0 != 0:
                    raise ValueError("story_heights must be multiples of 300 mm.")
            if len(story_heights) == 1:
                story_heights = [float(story_heights[0]) for _ in range(num_stories)]
            else:
                first_val = float(story_heights[0])
                if any(not math.isclose(float(v), first_val) for v in story_heights):
                    raise ValueError("story_heights must be uniform for all stories.")
                num_stories = len(story_heights)

        x_coords = cumulative_positions(span_lengths_x)
        y_coords = cumulative_positions(span_lengths_y)
        z_coords = cumulative_positions(story_heights)

        layout_type = cfg.get("layout_type", "full")
        active_nodes = build_active_nodes(num_spans_x, num_spans_y, layout_type)

        num_spans = num_spans_x
        num_stories = num_stories
    else:
        raise ValueError("当前仅支持 XY 平面布局输出。")

    nodes: List[Dict[str, Any]] = []
    if plane == "XY":
        for level in range(num_stories + 1):
            z_val = z_coords[level]
            for j in range(num_spans_y + 1):
                for i in range(num_spans_x + 1):
                    if (i, j) not in active_nodes:
                        continue
                    nid = node_id_xy(level, j, i, num_spans_x, num_spans_y)
                    nodes.append(
                        {
                            "id": nid,
                            "x": x_coords[i],
                            "y": y_coords[j],
                            "z": z_val,
                            "story": level,
                            "grid_i": i,
                            "grid_j": j,
                        }
                    )
    else:
        for j in range(num_stories + 1):
            for i in range(num_spans + 1):
                nid = node_id(j, i, num_spans)
                nodes.append(
                    {
                        "id": nid,
                        "x": x_coords[i],
                        "y": 0.0,
                        "z": z_coords[j],
                        "story": j,
                        "grid_i": i,
                        "grid_j": j,
                    }
                )

    beam_b = quantize_length(
        sample_float(
            g, device, cfg["beam_section_range_mm"]["b"][0], cfg["beam_section_range_mm"]["b"][1]
        ),
        50.0,
        cfg["beam_section_range_mm"]["b"][0],
        cfg["beam_section_range_mm"]["b"][1],
    )
    beam_h = quantize_length(
        sample_float(
            g, device, cfg["beam_section_range_mm"]["h"][0], cfg["beam_section_range_mm"]["h"][1]
        ),
        50.0,
        cfg["beam_section_range_mm"]["h"][0],
        cfg["beam_section_range_mm"]["h"][1],
    )
    column_sections: List[Dict[str, Any]] = []
    col_low = max(
        cfg["column_section_range_mm"]["b"][0], cfg["column_section_range_mm"]["h"][0]
    )
    col_high = min(
        cfg["column_section_range_mm"]["b"][1], cfg["column_section_range_mm"]["h"][1]
    )
    if col_low > col_high:
        raise ValueError("column_section_range_mm b/h ranges do not overlap for square columns.")
    if cfg.get("column_vary_by_story", False):
        for _ in range(num_stories):
            col_b = quantize_length(sample_float(g, device, col_low, col_high), 50.0, col_low, col_high)
            col_h = col_b
            column_sections.append({"b": col_b, "h": col_h})
    else:
        col_b = quantize_length(sample_float(g, device, col_low, col_high), 50.0, col_low, col_high)
        col_h = col_b
        # 不按层变化时只保留一个柱截面，避免输出多余 column2..columnN
        column_sections = [{"b": col_b, "h": col_h}]

    sections: List[Dict[str, Any]] = []
    sections.append({"id": 1, "type": "beam", "b": beam_b, "h": beam_h})
    for idx, sec in enumerate(column_sections, start=1):
        sections.append({"id": 100 + idx, "type": "column", "b": sec["b"], "h": sec["h"]})

    elements: List[Dict[str, Any]] = []
    eid = 1
    if plane == "XY":
    # 柱：每个有效网格点的竖向构件
        for level in range(1, num_stories + 1):
            section_id = 100 + level if cfg.get("column_vary_by_story", False) else 100 + 1
            for j in range(num_spans_y + 1):
                for i in range(num_spans_x + 1):
                    if (i, j) not in active_nodes:
                        continue
                    ni = node_id_xy(level - 1, j, i, num_spans_x, num_spans_y)
                    nj = node_id_xy(level, j, i, num_spans_x, num_spans_y)
                    elements.append(
                        {
                            "id": eid,
                            "type": "column",
                            "ni": ni,
                            "nj": nj,
                            "section_id": section_id,
                        }
                    )
                    eid += 1
    # 梁：每层沿 X/Y 方向布置（相邻两点均为有效节点）
        for level in range(1, num_stories + 1):
            for j in range(num_spans_y + 1):
                for i in range(num_spans_x):
                    if (i, j) not in active_nodes or (i + 1, j) not in active_nodes:
                        continue
                    ni = node_id_xy(level, j, i, num_spans_x, num_spans_y)
                    nj = node_id_xy(level, j, i + 1, num_spans_x, num_spans_y)
                    elements.append(
                        {
                            "id": eid,
                            "type": "beam",
                            "dir": "x",
                            "ni": ni,
                            "nj": nj,
                            "section_id": 1,
                        }
                    )
                    eid += 1
            for i in range(num_spans_x + 1):
                for j in range(num_spans_y):
                    if (i, j) not in active_nodes or (i, j + 1) not in active_nodes:
                        continue
                    ni = node_id_xy(level, j, i, num_spans_x, num_spans_y)
                    nj = node_id_xy(level, j + 1, i, num_spans_x, num_spans_y)
                    elements.append(
                        {
                            "id": eid,
                            "type": "beam",
                            "dir": "y",
                            "ni": ni,
                            "nj": nj,
                            "section_id": 1,
                        }
                    )
                    eid += 1
    else:
        # 柱
        for story in range(1, num_stories + 1):
            section_id = 100 + story if cfg.get("column_vary_by_story", False) else 100 + 1
            for i in range(num_spans + 1):
                ni = node_id(story - 1, i, num_spans)
                nj = node_id(story, i, num_spans)
                elements.append(
                    {
                        "id": eid,
                        "type": "column",
                        "ni": ni,
                        "nj": nj,
                        "section_id": section_id,
                    }
                )
                eid += 1
        # 梁
        for story in range(1, num_stories + 1):
            for i in range(num_spans):
                ni = node_id(story, i, num_spans)
                nj = node_id(story, i + 1, num_spans)
                elements.append(
                    {
                        "id": eid,
                        "type": "beam",
                        "ni": ni,
                        "nj": nj,
                        "section_id": 1,
                    }
                )
                eid += 1

    nodes_by_id = {n["id"]: n for n in nodes}
    nodes_out = build_nodes_dict(x_coords, y_coords, active_nodes=active_nodes)
    sections_out, section_name_by_id = build_section_maps(sections)
    columns_raw = build_columns_by_story(elements, nodes_by_id, section_name_by_id)
    beams_raw = build_beams_by_story(elements, nodes_by_id, section_name_by_id)
    columns_out = collapse_to_full_range(columns_raw, num_stories)
    beams_out = collapse_to_full_range(beams_raw, num_stories)

    model_id = f"m{base_seed}_{index:04d}" if base_seed is not None else f"m{index:04d}"

    model = {
        "parameters": {
            "structure_type": "RC_Frame",
            "num_stories": num_stories,
        },
        "geometry": {
            "grid_x": x_coords,
            "grid_y": y_coords,
            "grid_z": z_coords,
        },
        "nodes": nodes_out,
        "sections": sections_out,
        "columns": columns_out,
        "beams": beams_out,
    }

    num_beams = sum(len(v) for v in beams_raw.values())
    num_columns = sum(len(v) for v in columns_raw.values())
    summary = {
        "model_id": model_id,
        "plane": plane,
        "num_spans": num_spans_x,
        "num_spans_y": num_spans_y,
        "num_stories": num_stories,
        "num_nodes": len(nodes_out),
        "num_elements": num_beams + num_columns,
        "num_beams": num_beams,
        "num_columns": num_columns,
    }
    return model, summary


# 绘制平面布局示意图（仅 XY）
def plot_layout(
    model: Dict[str, Any],
    out_path: str,
    plot_rooms: bool = True,
    dpi: int = 180,
) -> None:
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for visualization.")

    grid_x = model["geometry"]["grid_x"]
    grid_y = model["geometry"]["grid_y"]
    nodes = model["nodes"]
    beams_by_story = model.get("beams", {})
    story_keys = sorted(beams_by_story.keys(), key=lambda k: parse_story_label(k)[0])
    beams = beams_by_story[story_keys[0]] if story_keys else {}

    fig, ax = plt.subplots(figsize=(8, 5))

    if plot_rooms:
        for j in range(len(grid_y) - 1):
            for i in range(len(grid_x) - 1):
                k00 = f"N_{i}_{j}"
                k10 = f"N_{i + 1}_{j}"
                k01 = f"N_{i}_{j + 1}"
                k11 = f"N_{i + 1}_{j + 1}"
                if not all(k in nodes for k in (k00, k10, k01, k11)):
                    continue
                x0, x1 = grid_x[i], grid_x[i + 1]
                y0, y1 = grid_y[j], grid_y[j + 1]
                rect = Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="#f0f3f6",
                    edgecolor="none",
                    alpha=0.7,
                )
                ax.add_patch(rect)

    for beam in beams.values():
        ni = nodes[beam["i_node"]]
        nj = nodes[beam["j_node"]]
        ax.plot([ni["x"], nj["x"]], [ni["y"], nj["y"]], color="#2a6fdb", linewidth=2.0)

    xs = [n["x"] for n in nodes.values()]
    ys = [n["y"] for n in nodes.values()]
    ax.scatter(xs, ys, s=18, color="#111111", zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    title = "RC Frame Plan Layout"
    if story_keys:
        title = f"{title} - {story_keys[0]}"
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# 主入口：解析参数并生成布局
def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_outdir = os.path.join(base_dir, "out")
    parser = argparse.ArgumentParser(description="2D RC frame layout generator")
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="number of models to generate (default: 1)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=default_outdir,
        help="output directory (default: ./out under this script folder)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--config", type=str, default=None, help="JSON config to override defaults"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing files"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] torch device = {device}")
    if args.seed is None:
        base_seed = int(time.time())
    else:
        base_seed = args.seed

    attempts = 0
    generated = 0
    while generated < args.n and attempts < cfg.get("max_attempts", 20):
        try:
            model, summary = build_model(cfg, base_seed, generated, device)
        except Exception as exc:
            attempts += 1
            print(f"[WARN] generation failed: {exc} (attempt {attempts})")
            continue

        model_id = summary["model_id"]
        layout_prefix = f"layout6_"
        json_path = os.path.join(args.outdir, f"{layout_prefix}{model_id}.json")
        png_path = os.path.join(args.outdir, f"{layout_prefix}{model_id}_layout.png")

        if not args.overwrite and (os.path.exists(json_path) or os.path.exists(png_path)):
            print(f"[SKIP] {model_id} exists (use --overwrite to replace)")
            generated += 1
            continue

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)

        if cfg.get("visualize", {}).get("enabled", True):
            if HAS_MPL:
                plot_layout(
                    model,
                    png_path,
                    plot_rooms=cfg.get("visualize", {}).get("plot_rooms", True),
                    dpi=cfg.get("visualize", {}).get("dpi", 180),
                )
            else:
                print("[WARN] matplotlib not available; skipping PNG visualization.")

        if summary["plane"] == "XY":
            print(
                f"[OK] {model_id} | spans_x={summary['num_spans']} spans_y={summary['num_spans_y']} "
                f"stories={summary['num_stories']} nodes={summary['num_nodes']} "
                f"elements={summary['num_elements']} beams={summary['num_beams']} "
                f"columns={summary['num_columns']}"
            )
        else:
            print(
                f"[OK] {model_id} | spans={summary['num_spans']} stories={summary['num_stories']} "
                f"nodes={summary['num_nodes']} elements={summary['num_elements']} "
                f"beams={summary['num_beams']} columns={summary['num_columns']}"
            )
        generated += 1

    if generated < args.n:
        print(f"[WARN] only generated {generated} models after {attempts} attempts.")


if __name__ == "__main__":
    main()
