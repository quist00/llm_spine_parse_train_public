#!/usr/bin/env python3
"""
Campus-oriented wrapper for tools/start_vllm.py.

Purpose:
- Reuse the proven NRP launcher logic without modifying it.
- Provide campus-safe defaults for host, port, adapter path, and HF cache.
- Keep startup reproducible via docker_campus/.env.campus values.

Usage:
  python tools/start_vllm_campus.py
  python tools/start_vllm_campus.py --dry-run
  python tools/start_vllm_campus.py --adapter-path /data/checkpoints/my_adapter/adapter_model
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _set_default_env() -> None:
    # Shared campus data root in-container; matches docker_campus compose mount.
    data_root = os.environ.get("CAMPUS_DATA_ROOT", "/data")

    os.environ.setdefault("VLLM_HOST", "0.0.0.0")
    os.environ.setdefault("VLLM_PORT", "8000")
    os.environ.setdefault("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    os.environ.setdefault("VLLM_ADAPTER_NAME", "spine_adapter")
    os.environ.setdefault(
        "VLLM_ADAPTER_PATH",
        f"{data_root}/checkpoints/qwen2_5_vl_lora_512Res/adapter_model",
    )

    # vGPU-safe defaults: disable NCCL transport paths that often fail with
    # "CUDA driver error: operation not supported" on shared/virtualized GPUs.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # vGPU fix: disable PyTorch expandable_segments (cuMemCreate) which is
    # unsupported on shared/virtual GPU drivers — causes "operation not supported".
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    # flashinfer cache root; flashinfer derives .cache/flashinfer beneath this base.
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/data/vllm_cache")
    # Compatibility alias for older wrappers/tools that inspect this variable directly.
    os.environ.setdefault("FLASHINFER_WORKSPACE_DIR", "/data/vllm_cache/.cache/flashinfer")
    # Prefer Torch SDPA backend on campus vGPU to avoid FlashInfer init issues.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
    # The ViT/MM encoder path has its own backend selector in vLLM.
    os.environ.setdefault("VLLM_MM_ENCODER_ATTN_BACKEND", "TORCH_SDPA")

    # Keep large downloaded weights on persistent mount instead of ephemeral layers.
    os.environ.setdefault("HF_HOME", "/home/jovyan/.hf_home")


def main() -> None:
    _set_default_env()

    repo_root = Path(__file__).resolve().parents[1]
    base_launcher = repo_root / "tools" / "start_vllm.py"

    if not base_launcher.exists():
        print(f"Base launcher not found: {base_launcher}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(base_launcher), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(repo_root), env=os.environ.copy()))


if __name__ == "__main__":
    main()
