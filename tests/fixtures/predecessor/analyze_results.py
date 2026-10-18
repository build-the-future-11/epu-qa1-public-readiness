#!/usr/bin/env python3
"""Regenerate EPU-QA1 tables from a completed, hash-verified run."""
import argparse
import json
from pathlib import Path

from run_study import regenerate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = regenerate(args.receipt, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
