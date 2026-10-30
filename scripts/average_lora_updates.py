"""Construct or validate an equal average of effective LoRA updates without training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.mitigation.lora_average import (  # noqa: E402
    build_average,
    merge_average,
    validate_average,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "validate", "merge"])
    parser.add_argument("--adapter", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--average-dir", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = build_average(args.adapter, args.output_dir)
    elif args.command == "validate":
        result = validate_average(args.output_dir, args.adapter)
    else:
        if not args.average_dir:
            parser.error("merge requires --average-dir")
        result = merge_average(args.average_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ["schema", "binding_sha256", "manifest_file"]},
                     indent=2))


if __name__ == "__main__":
    main()
