from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .layouts import resolve_generator


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        candidate = Path(f"{path}_{i:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to create unique temp dir under {path}")


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 1000):
        candidate = parent / f"{stem}_{i:02d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to allocate filename for {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate RC frame layout JSON/PNG files")
    parser.add_argument("--layout", type=int, default=1, help="layout id (1-9)")
    parser.add_argument("--n", type=int, default=1, help="number of models to generate")
    parser.add_argument("--outdir", type=str, default="out", help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--config", type=str, default=None, help="JSON config override")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    return parser


def generate_layouts(args: argparse.Namespace) -> list[Path]:
    if args.layout < 1 or args.layout > 9:
        raise ValueError("layout must be between 1 and 9")

    project_dir = Path.cwd()
    generator = resolve_generator(args.layout)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (project_dir / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_outdir = unique_dir(outdir / f"_tmp_layout{args.layout}_{timestamp}")
    temp_outdir.mkdir(parents=True, exist_ok=False)

    cmd = [
        sys.executable,
        str(generator),
        "--n",
        str(args.n),
        "--outdir",
        str(temp_outdir),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        cmd += ["--config", str(config_path)]

    if args.overwrite:
        cmd.append("--overwrite")

    subprocess.run(cmd, check=True, cwd=project_dir)

    moved: list[Path] = []
    for src in sorted(temp_outdir.iterdir()):
        if src.is_dir():
            continue

        dest = outdir / src.name
        if dest.exists():
            if args.overwrite:
                dest.unlink()
            else:
                dest = next_available_path(dest)
        shutil.move(str(src), str(dest))
        moved.append(dest)

    try:
        temp_outdir.rmdir()
    except OSError:
        pass

    print(f"[OK] moved {len(moved)} files to {outdir}")
    return moved


def main() -> None:
    args = build_parser().parse_args()
    generate_layouts(args)


if __name__ == "__main__":
    main()
