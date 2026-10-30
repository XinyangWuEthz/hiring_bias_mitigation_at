"""Refuse to start the patching pipeline on a machine that would silently do the wrong thing.

Checks, and writes to reports/interp/env.json:
  * the output root exists and carries the drive marker (or HBM_ALLOW_LOCAL_OUTPUT=1), and
    HF_HOME is not on an unmounted path -- the repo's .env points both at an external drive
  * library versions, visible GPUs and their free memory
  * which GDN kernels are importable: without `fla` the torch reference path runs, which is
    correct but slow, and the two paths must be compared once (E0) before mixing artifacts

    HBM_OUTPUT_ROOT=/home/samoilenko/hbm-local HF_HOME=/home/samoilenko/hbm-local/hf-cache \\
        uv run python scripts/interp/00_env_check.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hiring_bias_mitigation.interp.io import kernel_info, report_dir  # noqa: E402
from hiring_bias_mitigation.utils.config import output_root, require_output_root  # noqa: E402


def main() -> None:
    import peft
    import torch
    import transformers

    problems = []
    root = output_root()
    try:
        require_output_root()
    except SystemExit as exc:
        problems.append(str(exc).splitlines()[0])
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if not hf_home.parent.exists():
        problems.append(f"HF_HOME {hf_home} sits on a path that does not exist (unmounted drive?)")

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({"index": i, "name": torch.cuda.get_device_name(i),
                         "free_gb": round(free / 2**30, 1), "total_gb": round(total / 2**30, 1)})
    else:
        problems.append("CUDA is not available")

    kernels = kernel_info()
    report = {
        "output_root": str(root),
        "hf_home": str(hf_home),
        "disk_free_gb": round(shutil.disk_usage(root if root.exists() else Path.home()).free
                              / 2**30, 1),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "gpus": gpus,
        "kernels": kernels,
        "gdn_path": "fla" if kernels["fla"] else "torch reference (slow)",
        "problems": problems,
    }
    out = report_dir() / "env.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not kernels["fla"]:
        print("\nwarning: flash-linear-attention is not installed; GDN layers run the torch "
              "reference path. `uv pip install flash-linear-attention` before the margin pass.")
    if problems:
        raise SystemExit(f"{len(problems)} problem(s); see above")


if __name__ == "__main__":
    main()
