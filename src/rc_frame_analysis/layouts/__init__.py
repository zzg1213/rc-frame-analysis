from __future__ import annotations

from pathlib import Path


VALID_LAYOUTS = tuple(range(1, 10))


def resolve_generator(layout: int) -> Path:
    if layout not in VALID_LAYOUTS:
        raise ValueError("layout must be between 1 and 9")

    generator = Path(__file__).resolve().parent / f"generator{layout}.py"
    if not generator.exists():
        raise FileNotFoundError(f"generator not found: {generator}")
    return generator
