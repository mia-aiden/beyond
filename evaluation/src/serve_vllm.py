from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from io_utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_ROOT = PROJECT_ROOT / "evaluation"


def build_server_command(config: dict[str, Any]) -> list[str]:
    executable = shutil.which("vllm")
    if executable is None:
        raise RuntimeError(
            "The vllm executable was not found. Install vLLM in the CUDA server environment."
        )

    model_path = str(config.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Set model_path in evaluation/configs/vllm.yaml.")

    served_model_name = str(config.get("served_model_name", "")).strip()
    if not served_model_name:
        raise ValueError("Set served_model_name in evaluation/configs/vllm.yaml.")

    command = [
        executable,
        "serve",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        str(config.get("host", "0.0.0.0")),
        "--port",
        str(int(config.get("port", 8000))),
        "--dtype",
        str(config.get("dtype", "auto")),
        "--tensor-parallel-size",
        str(int(config.get("tensor_parallel_size", 1))),
        "--gpu-memory-utilization",
        str(float(config.get("gpu_memory_utilization", 0.9))),
        "--max-model-len",
        str(int(config.get("max_model_len", 4096))),
    ]
    if bool(config.get("trust_remote_code", False)):
        command.append("--trust-remote-code")

    chat_template = str(config.get("chat_template", "")).strip()
    if chat_template:
        command.extend(["--chat-template", chat_template])

    server_api_key = str(config.get("server_api_key", "")).strip()
    if server_api_key:
        command.extend(["--api-key", server_api_key])

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the configured vLLM server.")
    parser.add_argument(
        "--config",
        type=Path,
        default=EVAL_ROOT / "configs" / "vllm.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        command = build_server_command(load_yaml(args.config))
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    display_command = command.copy()
    if "--api-key" in display_command:
        key_index = display_command.index("--api-key") + 1
        display_command[key_index] = "***"

    print("Starting vLLM server:")
    print(" ".join(display_command), flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
