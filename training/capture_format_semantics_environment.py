#!/usr/bin/env python3
"""Capture software, hardware, repository, and base-model provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from paths import (
    PROJECT_ROOT,
    EXPERIMENTS_DIR,
    BASE_MODEL,
    LLAMAFACTORY_DIR,
    LLAMAFACTORY_PYTHON,
    VLLM_PYTHON,
)


EXPERIMENT_DIR = EXPERIMENTS_DIR / "format_semantics_multiseed"
ENV_PYTHONS = {
    "llamafactory": Path(LLAMAFACTORY_PYTHON),
    "vllm_eval": Path(VLLM_PYTHON),
}
PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "vllm",
    "bert-score",
    "sentence-transformers",
    "scikit-learn",
    "scipy",
)


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr.strip()}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(python: Path) -> dict:
    code = (
        "import importlib.metadata,json; names="
        + repr(PACKAGES)
        + "; out={};"
        + "\nfor n in names:\n"
        + "  try: out[n]=importlib.metadata.version(n)\n"
        + "  except importlib.metadata.PackageNotFoundError: out[n]=None\n"
        + "print(json.dumps(out))"
    )
    output = run([str(python), "-c", code])
    return json.loads(output)


def git_state(root: Path) -> dict:
    return {
        "path": str(root),
        "commit": run(["git", "rev-parse", "HEAD"], root),
        "branch": run(["git", "branch", "--show-current"], root),
        "dirty": bool(run(["git", "status", "--porcelain"], root)),
    }


def main() -> None:
    gpu_csv = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    config_path = Path(BASE_MODEL) / "config.json"
    payload = {
        "platform": platform.platform(),
        "gpu_rows": gpu_csv.splitlines(),
        "nvidia_smi": run(["nvidia-smi"]),
        "repositories": {
            "project": git_state(PROJECT_ROOT),
            "llama_factory": git_state(LLAMAFACTORY_DIR),
        },
        "environments": {
            name: {
                "python": str(python),
                "python_version": run([str(python), "--version"]),
                "packages": package_versions(python),
            }
            for name, python in ENV_PYTHONS.items()
        },
        "base_model": {
            "path": str(BASE_MODEL),
            "config_sha256": sha256(config_path),
            "config": json.loads(config_path.read_text(encoding="utf-8")),
        },
    }
    output_path = EXPERIMENT_DIR / "environment_manifest.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
