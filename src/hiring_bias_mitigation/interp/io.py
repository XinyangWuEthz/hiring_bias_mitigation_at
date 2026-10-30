"""Where interpretability artifacts live, and the metadata every one of them carries.

Heavy artifacts (margins, caches, manifests) go to the output drive under `outputs/interp/`;
small summaries meant for the repository go to `reports/interp/`. Every artifact gets a
`.meta.json` sidecar naming the code revision, the model spec and the numerical path it was
produced on: a patching effect of 0.1 logits is only comparable with another one produced on
the same kernels and dtype, and the GDN kernels in particular switch silently between the
torch fallback and `fla` depending on what happens to be installed.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ..utils.config import REPO_ROOT, load_yaml, resolve_output_path

CONFIG_DIR = REPO_ROOT / "configs" / "interp"
REPORT_DIR = REPO_ROOT / "reports" / "interp"


def interp_dir(*parts: str) -> Path:
    path = Path(resolve_output_path("outputs/interp")).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_dir(*parts: str) -> Path:
    path = REPORT_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or CONFIG_DIR / "data.yaml")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def kernel_info() -> dict:
    """Which implementation the GDN layers will actually run on."""
    # The kernel module itself, not the package: `flash-linear-attention` installed without
    # its `fla-core` dependency imports as `fla` but has no `fla.ops`.
    info = {}
    for name, module in (("fla", "fla.ops.gated_delta_rule"), ("causal_conv1d", "causal_conv1d")):
        try:
            __import__(module)
            info[name] = True
        except ImportError:
            info[name] = False
    return info


def write_meta(artifact: str | Path, **fields) -> Path:
    import torch
    import transformers

    meta = {
        "artifact": str(artifact),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "kernels": kernel_info(),
        **fields,
    }
    path = Path(f"{artifact}.meta.json")
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path
