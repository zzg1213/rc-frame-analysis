from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RC frame dataset tools. Use the generate, analyze, or pipeline "
            "modules for actual work."
        )
    )
    parser.print_help()


if __name__ == "__main__":
    main()
