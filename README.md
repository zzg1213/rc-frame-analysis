# RC Frame Analysis

钢筋混凝土框架结构数据集生成工具。项目流程包括平面布局生成、荷载与地震作用计算、二维框架内力分析、梁柱配筋设计，以及 JSON/PNG 数据集样本输出。

## 环境

推荐使用已有的 `torch_cuda` 环境运行。为避免修改环境内的软件包，直接在仓库根目录设置 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "$PWD\src"
```

主要运行依赖为 `torch`、`numpy`、`matplotlib`。本仓库不自动安装这些依赖，避免改动既有计算环境。

## 快速开始

生成指定布局类型：

```powershell
$env:PYTHONPATH = "$PWD\src"
& "C:\Project\anaconda3\envs\torch_cuda\python.exe" -m rc_frame_analysis.generate --layout 1 --n 1 --seed 42 --outdir out
```

对布局 JSON 做内力计算和配筋：

```powershell
$env:PYTHONPATH = "$PWD\src"
& "C:\Project\anaconda3\envs\torch_cuda\python.exe" -m rc_frame_analysis.analyze --input out/layout1_m42_0000.json
```

一键生成并分析：

```powershell
$env:PYTHONPATH = "$PWD\src"
& "C:\Project\anaconda3\envs\torch_cuda\python.exe" -m rc_frame_analysis.pipeline --layout 1 --n 1 --seed 42 --outdir out
```

## 输出

布局生成阶段输出：

- `layout{n}_*.json`：结构布局、网格、节点、梁、柱、截面和楼层信息。
- `layout{n}_*_layout.png`：首层平面布局图。

分析阶段输出到输入 JSON 同级的 `analysis/layout{n}/`：

- `analysis_layout{n}_*.json`：地震作用、层重、梁柱内力包络和配筋结果。
- `*_story_shear.png`：楼层剪力图。
- `*_frame_*_M.png` / `*_frame_*_V.png`：各榀框架弯矩图和剪力图。
- `*_reinforcement_story*.png`：楼层配筋平面图。

## 项目结构

```text
src/rc_frame_analysis/
  generate.py          # 布局生成 CLI
  analyze.py           # 内力计算与配筋 CLI
  pipeline.py          # 生成 + 分析流水线 CLI
  analysis_core.py     # 结构分析、荷载组合、配筋和绘图核心逻辑
  layouts/             # 1-9 号布局生成器
configs/layouts/       # 布局类型说明
docs/method.md         # 数据集生成流程和原理
examples/              # 少量示例样本
```

详细方法说明见 [docs/method.md](docs/method.md)。
