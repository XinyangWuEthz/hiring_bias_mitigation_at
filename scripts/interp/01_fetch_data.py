"""Download everything the patching pipeline reads, to the paths the rest of the repo expects.

    model     Qwen/Qwen3.5-9B (text weights, config, tokenizer), the SFT-v1 adapter and the
              merged SFT-v3 release (01b_extract_adapter.py recovers its adapter)
    pool      synthetic-data raw/en_invariant.parquet -> artifacts/semisynthetic-v1/raw/
              (the discovery grid's job-CV pairs) and sft_train.parquet (which pairs SFT saw)
    audit     responses-dataset runs for the 9B-en baseline and SFT-v1 adapter -> outputs/raw/
              (E0 compares the HF margins against these decisions). SFT-v3's run is not
              published; 02_margins.py falls back to its release's evaluation.json

Idempotent: files already present are not fetched again. The DPO adapter is not on the Hub;
it is passed to the later scripts by path.

    uv run python scripts/interp/01_fetch_data.py [--skip-model]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hiring_bias_mitigation.interp.extract import SFT_V3, SFT_V3_REVISION  # noqa: E402
from hiring_bias_mitigation.interp.spec import load_spec_set  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("interp.fetch")

SYNTHETIC = "Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data"
RESPONSES = "Stereotypes-in-LLMs/hiring-bias-mitigation-responses"
POOL_FILES = {"data/raw/en_invariant.parquet": "raw/en_invariant.parquet",
              "data/sft_train.parquet": "sft_train.parquet"}
AUDIT_RUNS = ("Qwen3.5-9B--en--baseline", "Qwen3.5-9B--en--sft--adapter--en_only")


def _copy(repo: str, filename: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    if dest.exists():
        log.info("present: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo, filename, repo_type="dataset")
    shutil.copyfile(src, dest)
    log.info("wrote %s", dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-model", action="store_true",
                        help="fetch only the data files (the model is ~19 GB)")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    specs = load_spec_set()
    if not args.skip_model:
        # The checkpoint also ships a vision tower; it is downloaded because the safetensors
        # shards mix the two, and skipped at load time by Qwen3_5ForCausalLM.
        path = snapshot_download(specs.base_model, allow_patterns=[
            "*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer*"])
        log.info("base model at %s", path)
        log.info("SFT-v1 adapter at %s", snapshot_download(specs.adapter_paths["sft_v1"]))
        log.info("SFT-v3 merged at %s", snapshot_download(SFT_V3, revision=SFT_V3_REVISION))

    pool_root = Path(resolve_output_path("artifacts/semisynthetic-v1"))
    for remote, local in POOL_FILES.items():
        _copy(SYNTHETIC, remote, pool_root / local)

    raw = Path(resolve_output_path("outputs/raw"))
    for run in AUDIT_RUNS:
        _copy(RESPONSES, f"runs/{run}.parquet", raw / f"{run}.parquet")
        _copy(RESPONSES, f"runs/{run}.meta.json", raw / f"{run}.meta.json")


if __name__ == "__main__":
    main()
