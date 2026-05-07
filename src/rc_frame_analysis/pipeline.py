from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .generate import build_parser as build_generate_parser
from .generate import generate_layouts


def build_parser() -> argparse.ArgumentParser:
    parser = build_generate_parser()
    parser.description = "Generate layouts and run RC frame analysis in one command"
    parser.add_argument(
        "--analysis-config",
        type=str,
        default=None,
        help="optional JSON config for analysis and reinforcement design",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generated = generate_layouts(args)
    layout_jsons = [path for path in generated if path.suffix.lower() == ".json"]

    for layout_json in layout_jsons:
        cmd = [
            sys.executable,
            "-m",
            "rc_frame_analysis.analyze",
            "--input",
            str(Path(layout_json).resolve()),
        ]
        if args.analysis_config:
            cmd += ["--config", str(Path(args.analysis_config).resolve())]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
