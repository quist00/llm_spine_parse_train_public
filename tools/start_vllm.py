#!/usr/bin/env python3
"""
Adaptive vLLM launcher for Qwen2.5-VL + LoRA serving.

Performs GPU compatibility and VRAM preflight checks, then
auto-tunes launch parameters for whichever GPU node was assigned.
Fails fast with actionable diagnostics on incompatible hardware.

Usage (from ~/llm_train/):
    python tools/start_vllm.py
    python tools/start_vllm.py --dry-run
    python tools/start_vllm.py --adapter-path checkpoints/my_run/adapter_model

Environment variable overrides:
    VLLM_MODEL           HuggingFace model ID (default: Qwen/Qwen2.5-VL-7B-Instruct)
    START_VLLM_DTYPE     Model dtype passed to vLLM (default: float16)
    START_VLLM_MAX_MODEL_LEN
                         Maximum model sequence length (default: auto from VRAM tier)
    START_VLLM_GPU_MEM_UTIL
                         Override gpu_memory_utilization (0.0–1.0). Bypasses the
                         preflight safety guard — use when the guard over-reduces
                         utilization due to pre-load vs post-load VRAM difference.
    START_VLLM_LIMIT_MM_PER_PROMPT
                         Raw JSON string for --limit-mm-per-prompt.
                         Default: {"image": 1, "video": 0, "audio": 0}
    VLLM_ADAPTER_PATH    LoRA adapter directory path
    VLLM_ADAPTER_NAME    LoRA alias name exposed by the API (default: spine_adapter)
    VLLM_MM_ENCODER_ATTN_BACKEND
                         Optional vLLM multimodal encoder attention backend override
    VLLM_ENABLE_AUTO_TOOL_CHOICE
                         Enable OpenAI-compatible auto tool choice in vLLM.
                         Must be paired with VLLM_TOOL_CALL_PARSER.
    VLLM_TOOL_CALL_PARSER
                         vLLM tool-call parser name (example: pythonic, hermes).
    VLLM_PORT            Port to bind (default: 8000)
    VLLM_HOST            Host to bind (default: 127.0.0.1)
    HF_HOME              HuggingFace cache root
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path.home() / ".cache"))
os.environ.setdefault("VLLM_CACHE_DIR", str(Path.home() / ".cache" / "vllm"))
# ---------------------------------------------------------------------------
# Defaults — override via env or CLI args
# ---------------------------------------------------------------------------
DEFAULTS = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "adapter_path": "../training_output/qwen2_5_vl_lora_full/adapter_model",
    "adapter_name": "spine_adapter",
    "mm_encoder_attn_backend": "",
    "port": 8000,
    "host": "127.0.0.1",
    "dtype": "float16",
    "limit_mm_per_prompt": '{"image": 1, "video": 0, "audio": 0}',
    "enable_auto_tool_choice": False,
    "tool_call_parser": "",
    "max_loras": 1,
    "max_lora_rank": 64,
}

# VRAM tier table: (min_gb_exclusive, max_gb_inclusive) → (util, max_model_len, enforce_eager)
# enforce_eager=True avoids CUDAGraph compilation failures on older/smaller GPUs
VRAM_TIERS = [
    (0,  14,  0.80, 1536, True),
    (14, 20,  0.82, 2048, True),
    (20, 26,  0.85, 2048, True),   # A10 22GB lands here
    (26, 44,  0.90, 4096, False),  # A100 40GB
    (44, 999, 0.92, 8192, False),  # A100 80GB
]

MIN_COMPUTE_CAPABILITY = (7, 0)  # Volta; hard vLLM minimum

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fail(msg: str, hint: str = "") -> None:
    print(f"\n[PREFLIGHT FAIL] {msg}", file=sys.stderr)
    if hint:
        print(f"  → {hint}", file=sys.stderr)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"[PREFLIGHT WARN] {msg}")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def resolve_user_path(raw_path: str, repo_root: Path) -> Path:
    """Resolve user-provided paths consistently across notebook and terminal CWDs."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (repo_root / path).resolve()


def configure_hf_cache_env() -> str:
    """Set Hugging Face to use the default cache location."""
    
    # Default HF cache location on this system
    default_cache = str(Path.home() / ".cache" / "huggingface")
    
    os.environ["HF_HOME"] = default_cache
    os.environ["HF_HUB_CACHE"] = str(Path(default_cache) / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(default_cache) / "hub")
    
    ok(f"HF cache configured (default): {default_cache}")
    ok(f"HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")
    
    return default_cache


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def check_nvidia_smi() -> dict:
    """Return GPU info dict or fail fast."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            fail(
                "nvidia-smi failed — no GPU visible in this pod.",
                "Request a GPU-enabled node profile on NRP.",
            )
        line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        total_mb = int(parts[1])
        free_mb = int(parts[2])
        cap_str = parts[3]                        # e.g. "8.6"
        cap = tuple(int(x) for x in cap_str.split("."))
        ok(f"GPU: {name}  total={total_mb // 1024}GB  free={free_mb // 1024}GB  sm={cap_str}")
        return {"name": name, "total_mb": total_mb, "free_mb": free_mb, "cap": cap, "cap_str": cap_str}
    except FileNotFoundError:
        fail(
            "nvidia-smi not found.",
            "This container may not have NVIDIA drivers mounted. Check your pod spec.",
        )


def check_compute_capability(cap: tuple, cap_str: str) -> None:
    if cap < MIN_COMPUTE_CAPABILITY:
        fail(
            f"GPU compute capability {cap_str} is below the vLLM minimum (sm_70 / Volta).",
            f"Request a newer GPU — A10 (sm_86), T4 (sm_75), or A100 (sm_80) all work.",
        )
    ok(f"Compute capability {cap_str} ≥ sm_70 — vLLM minimum satisfied.")


def check_torch_cuda() -> None:
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            fail(
                "PyTorch reports CUDA unavailable.",
                "Verify NVIDIA runtime is mounted and nvidia-smi works.",
            )
        # Minimal kernel smoke test — same op that triggered your colleague's error
        _ = torch.zeros(8, dtype=torch.bool, device="cuda")
        ok(f"PyTorch {torch.__version__} CUDA {torch.version.cuda} — kernel smoke test passed.")
    except ImportError:
        fail("PyTorch not installed.", "pip install torch")
    except RuntimeError as exc:
        fail(
            f"PyTorch CUDA kernel error: {exc}",
            "GPU/driver/torch binary mismatch. Try a different node or rebuild the image.",
        )


def check_vllm_extensions(cap_str: str) -> None:
    try:
        import vllm  # noqa: PLC0415
        ok(f"vLLM {vllm.__version__} imported.")
        # Attempt to load compiled extensions — fails on arch mismatches
        import vllm._C  # type: ignore[import-untyped]  # noqa: PLC0415, F401
        ok("vLLM CUDA extensions loaded successfully.")
    except ImportError as exc:
        if "vllm._C" in str(exc) or "_C" in str(exc):
            fail(
                f"vLLM CUDA extension failed to load on this GPU (sm_{cap_str.replace('.', '')}).",
                (
                    "This wheel was not compiled for this GPU architecture.\n"
                    "  Options:\n"
                    "  1. Request a different node profile (A100 / A10 preferred).\n"
                    "  2. Reinstall vLLM matching your CUDA version:\n"
                    "       pip install vllm --force-reinstall\n"
                    "  3. Check vLLM compatibility matrix: https://docs.vllm.ai"
                ),
            )
        fail(f"vLLM import failed: {exc}", "pip install vllm")


def check_adapter_path(adapter_path: Path) -> None:
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not (adapter_path / f).exists()]
    if not adapter_path.exists():
        fail(
            f"Adapter directory not found: {adapter_path}",
            "Check --adapter-path or VLLM_ADAPTER_PATH. Run 'ls checkpoints/' to see available adapters.",
        )
    if missing:
        fail(
            f"Adapter directory exists but is missing required files: {missing}",
            f"Path checked: {adapter_path}",
        )
    ok(f"Adapter path valid: {adapter_path}")


def compute_vram_profile(total_mb: int, free_mb: int) -> dict:
    total_gb = total_mb / 1024
    free_gb = free_mb / 1024

    # Pick tier
    profile = None
    for (lo, hi, util, max_len, eager) in VRAM_TIERS:
        if lo < total_gb <= hi:
            profile = {"util": util, "max_model_len": max_len, "enforce_eager": eager}
            break
    if profile is None:
        profile = VRAM_TIERS[-1][2:]
        profile = {"util": VRAM_TIERS[-1][2], "max_model_len": VRAM_TIERS[-1][3], "enforce_eager": VRAM_TIERS[-1][4]}

    # Safety guard: if util * total would exceed free - 300MB headroom, reduce util
    headroom_mb = 300
    safe_util = (free_mb - headroom_mb) / total_mb
    if safe_util < profile["util"]:
        warn(
            f"Free VRAM ({free_gb:.1f}GB) is tight. "
            f"Reducing utilization from {profile['util']} → {safe_util:.2f} to preserve {headroom_mb}MB headroom."
        )
        profile["util"] = round(max(safe_util, 0.60), 2)

    ok(
        f"VRAM profile: total={total_gb:.1f}GB  free={free_gb:.1f}GB  "
        f"util={profile['util']}  max_model_len={profile['max_model_len']}  "
        f"enforce_eager={profile['enforce_eager']}"
    )
    return profile


# ---------------------------------------------------------------------------
# Kill stale server
# ---------------------------------------------------------------------------

def kill_existing_server() -> None:
    result = subprocess.run(
        ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
        capture_output=True,
    )
    if result.returncode == 0:
        print("[INFO] Killed existing vLLM process — waiting 3s for cleanup...")
        time.sleep(3)
    else:
        print("[INFO] No existing vLLM process found.")


# ---------------------------------------------------------------------------
# Build and launch vLLM command
# ---------------------------------------------------------------------------

def build_command(cfg: dict) -> list:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["model"],
        "--download-dir", cfg["download_dir"],
        "--host", cfg["host"],
        "--port", str(cfg["port"]),
        "--trust-remote-code",
    ]
    if cfg["adapter_path"]:
        cmd += [
            "--enable-lora",
            "--max-loras", str(cfg["max_loras"]),
            "--max-lora-rank", str(cfg["max_lora_rank"]),
            "--lora-modules", f"{cfg['adapter_name']}={cfg['adapter_path']}",
        ]
    cmd += [
        "--dtype", cfg["dtype"],
        "--max-model-len", str(cfg["max_model_len"]),
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", str(cfg["util"]),
        "--limit-mm-per-prompt", cfg["limit_mm_per_prompt"],
    ]
    if cfg["mm_encoder_attn_backend"]:
        cmd += [
            "--mm-encoder-attn-backend",
            cfg["mm_encoder_attn_backend"],
        ]
    if cfg["enable_auto_tool_choice"]:
        cmd.append("--enable-auto-tool-choice")
        cmd += ["--tool-call-parser", cfg["tool_call_parser"]]
    if cfg["enforce_eager"]:
        cmd.append("--enforce-eager")
    return cmd


def launch(cfg: dict, dry_run: bool, detach: bool) -> None:
    env = os.environ.copy()
    hf_home = cfg.get("hf_home", "")
    if hf_home:
        env["HF_HOME"] = hf_home
        env["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        env["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "hub")
    env.setdefault("XDG_CACHE_HOME", "/data/vllm_cache")
    env.setdefault("FLASHINFER_WORKSPACE_BASE", env["XDG_CACHE_HOME"])
    # Compatibility alias: some setups inspect this directly even though
    # flashinfer computes paths from FLASHINFER_WORKSPACE_BASE.
    env.setdefault("FLASHINFER_WORKSPACE_DIR", f"{env['FLASHINFER_WORKSPACE_BASE']}/.cache/flashinfer")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Ensure cache directories exist before importing/initializing backends.
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["FLASHINFER_WORKSPACE_DIR"]).mkdir(parents=True, exist_ok=True)

    cmd = build_command(cfg)
    log_path = Path(os.environ.get("VLLM_LOG_PATH", "/tmp/vllm.log"))

    print(f"\n{'='*60}")
    print("vLLM LAUNCH SUMMARY")
    print(f"{'='*60}")
    print(f"  GPU      : {cfg['gpu_name']}")
    print(f"  Model    : {cfg['model']}")
    print(f"  DType    : {cfg['dtype']}")
    print(f"  Adapter  : {cfg['adapter_name']} → {cfg['adapter_path']}")
    print(f"  Util     : {cfg['util']}  max_len={cfg['max_model_len']}  eager={cfg['enforce_eager']}")
    print(f"  MM Limit : {cfg['limit_mm_per_prompt']}")
    if cfg["enable_auto_tool_choice"]:
        print(f"  Tools    : auto-choice enabled (parser={cfg['tool_call_parser']})")
    if cfg["mm_encoder_attn_backend"]:
        print(f"  MM Attn  : {cfg['mm_encoder_attn_backend']}")
    print(f"  Endpoint : http://{cfg['host']}:{cfg['port']}/v1")
    print(f"  Log      : {log_path}")
    print(f"  Cache    : {env['XDG_CACHE_HOME']}")
    print(f"  FlashInf : {env['FLASHINFER_WORKSPACE_DIR']}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Command that would be executed:")
        print("  " + " \\\n  ".join(cmd))
        return

    if detach:
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,   # equivalent to nohup — survives terminal close
        )
        print(f"[INFO] vLLM started with PID {proc.pid}  (session-leader, survives terminal close)")
        print(f"[INFO] Monitor: tail -f {log_path}")
        print(f"[INFO] Check:   curl -s http://{cfg['host']}:{cfg['port']}/v1/models")
        return

    print("[INFO] Launching vLLM in foreground (container-safe mode).")
    print(f"[INFO] Check: curl -s http://{cfg['host']}:{cfg['port']}/v1/models")
    # Foreground mode keeps this process as PID 1 in Docker so the container stays healthy.
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        sys.exit(rc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive vLLM launcher with GPU preflight checks."
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULTS["model"]))
    parser.add_argument(
        "--adapter-path",
        default=os.environ.get("VLLM_ADAPTER_PATH") or "",
    )
    parser.add_argument(
        "--adapter-name",
        default=os.environ.get("VLLM_ADAPTER_NAME", DEFAULTS["adapter_name"]),
    )
    parser.add_argument(
        "--mm-encoder-attn-backend",
        default=os.environ.get(
            "VLLM_MM_ENCODER_ATTN_BACKEND",
            DEFAULTS["mm_encoder_attn_backend"],
        ),
        help="Optional vLLM multimodal encoder attention backend override.",
    )
    parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        default=os.environ.get("VLLM_ENABLE_AUTO_TOOL_CHOICE", "").lower() in {"1", "true", "yes"},
        help="Enable vLLM auto tool choice for OpenAI-compatible clients.",
    )
    parser.add_argument(
        "--tool-call-parser",
        default=os.environ.get("VLLM_TOOL_CALL_PARSER", DEFAULTS["tool_call_parser"]),
        help="vLLM tool-call parser name (required when auto tool choice is enabled).",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", DEFAULTS["port"])))
    parser.add_argument("--host", default=os.environ.get("VLLM_HOST", DEFAULTS["host"]))
    parser.add_argument("--dtype", default=os.environ.get("START_VLLM_DTYPE", DEFAULTS["dtype"]))
    parser.add_argument(
        "--limit-mm-per-prompt",
        default=os.environ.get("START_VLLM_LIMIT_MM_PER_PROMPT", DEFAULTS["limit_mm_per_prompt"]),
        help="Raw JSON string for vLLM --limit-mm-per-prompt.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override auto-calculated max_model_len from VRAM tier (set via START_VLLM_MAX_MODEL_LEN env).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command without launching")
    parser.add_argument(
        "--detach",
        action="store_true",
        default=os.environ.get("VLLM_DETACH", "").lower() in {"1", "true", "yes"},
        help="Launch vLLM in background and write logs to VLLM_LOG_PATH (default: foreground)",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    hf_home = configure_hf_cache_env()

    print("=" * 60)
    print("vLLM PREFLIGHT CHECKS")
    print("=" * 60)

    # --- Checks ---
    gpu_info = check_nvidia_smi()
    check_compute_capability(gpu_info["cap"], gpu_info["cap_str"])
    check_torch_cuda()
    check_vllm_extensions(gpu_info["cap_str"])

    adapter_path = resolve_user_path(args.adapter_path, repo_root) if args.adapter_path else None
    if adapter_path is not None:
        check_adapter_path(adapter_path)

    vram_profile = compute_vram_profile(gpu_info["total_mb"], gpu_info["free_mb"])

    print("\n[OK] All preflight checks passed.\n")

    # --- Build config ---
    cfg = {
        "gpu_name": gpu_info["name"],
        "model": args.model,
        "adapter_path": str(adapter_path) if adapter_path else "",
        "adapter_name": args.adapter_name,
        "mm_encoder_attn_backend": args.mm_encoder_attn_backend,
        "enable_auto_tool_choice": args.enable_auto_tool_choice,
        "tool_call_parser": args.tool_call_parser,
        "port": args.port,
        "host": args.host,
        "dtype": args.dtype,
        "limit_mm_per_prompt": args.limit_mm_per_prompt,
        "max_loras": DEFAULTS["max_loras"],
        "max_lora_rank": DEFAULTS["max_lora_rank"],
        "download_dir": str(Path(hf_home) / "hub"),
        "hf_home": hf_home,
        **vram_profile,
    }
    # Allow explicit override of max_model_len (useful for multimodal models with heavy KV cache overhead)
    if args.max_model_len is not None:
        cfg["max_model_len"] = args.max_model_len
    # Check env override (for backward compat with shell scripts)
    env_override = os.environ.get("START_VLLM_MAX_MODEL_LEN")
    if env_override:
        try:
            cfg["max_model_len"] = int(env_override)
        except ValueError:
            warn(f"Invalid START_VLLM_MAX_MODEL_LEN='{env_override}' (not an integer), using tier default {cfg['max_model_len']}")

    # Allow explicit override of gpu_memory_utilization — useful when the preflight safety guard
    # over-reduces utilization because it anchors on pre-load free VRAM rather than post-load headroom.
    gpu_util_override = os.environ.get("START_VLLM_GPU_MEM_UTIL")
    if gpu_util_override:
        try:
            util_val = float(gpu_util_override)
            if 0.0 < util_val <= 1.0:
                startup_max_util = max((gpu_info["free_mb"] - 512) / gpu_info["total_mb"], 0.60)
                if util_val > startup_max_util:
                    warn(
                        "START_VLLM_GPU_MEM_UTIL "
                        f"{util_val} exceeds startup-safe util {startup_max_util:.2f} "
                        f"(free={gpu_info['free_mb']/1024:.2f}GB). Clamping to {startup_max_util:.2f}."
                    )
                    util_val = round(startup_max_util, 2)
                warn(f"START_VLLM_GPU_MEM_UTIL={util_val} overrides preflight-calculated util={cfg['util']}")
                cfg["util"] = util_val
            else:
                warn(f"Invalid START_VLLM_GPU_MEM_UTIL='{gpu_util_override}' (must be 0<val<=1), keeping {cfg['util']}")
        except ValueError:
            warn(f"Invalid START_VLLM_GPU_MEM_UTIL='{gpu_util_override}' (not a float), keeping {cfg['util']}")

    if cfg["enable_auto_tool_choice"] and not cfg["tool_call_parser"]:
        fail(
            "Auto tool choice was enabled but no tool-call parser was configured.",
            "Set VLLM_TOOL_CALL_PARSER (for example: pythonic) alongside VLLM_ENABLE_AUTO_TOOL_CHOICE=1.",
        )

    # --- Kill stale server and launch ---
    kill_existing_server()
    launch(cfg, dry_run=args.dry_run, detach=args.detach)


if __name__ == "__main__":
    main()
