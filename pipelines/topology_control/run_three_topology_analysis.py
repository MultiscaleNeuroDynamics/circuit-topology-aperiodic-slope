# -*- coding: utf-8 -*-
"""只读m=0.99 REAL/ER/DP统计流程的小型控制器。 / Small controller for the read-only m=0.99 REAL/ER/DP statistics pipeline."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Additional arguments are forwarded to analyze_three_topologies.py, "
            "including --project-root and explicit LFP/input/output roots."
        ),
    )
    _, forwarded = parser.parse_known_args()
    script = Path(__file__).resolve().parent / "analyze_three_topologies.py"
    cmd = [sys.executable, str(script), *forwarded]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
