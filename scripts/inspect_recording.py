#!/usr/bin/env python3
"""Validate a local trace without printing conversation contents."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from recording import validate_recording

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    try:
        result = validate_recording(args.recording)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot inspect recording: {exc}\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["valid"] else 1)
