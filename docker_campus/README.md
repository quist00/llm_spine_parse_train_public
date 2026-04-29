# Campus Docker Runtime (Isolated from NRP)

This folder is intentionally separate from the existing `docker/` setup so the NRP flow remains unchanged.

## Goals

- Keep `docker/` untouched.
- Reuse the same image built from `docker/Dockerfile`.
- Run on campus plain Docker + NVIDIA runtime.
- Persist large assets on host `/data` instead of baking into images.
- Support shared operations by multiple users.

## Files

- `docker-compose.campus.yml`: campus runtime compose file.
- `.env.campus`: campus environment values for this runtime.
- `bootstrap_data_dirs.sh`: one-time host setup for shared `/data` directories.
- `../tools/start_vllm_campus.py`: campus wrapper that calls `tools/start_vllm.py` with campus defaults.

## Quick Start

1. One-time shared data setup on host:

```bash
bash docker_campus/bootstrap_data_dirs.sh /data/llm_spine_parse llmtrain
```

2. Edit env file:

```bash
vi docker_campus/.env.campus
```

3. Build image using existing NRP-safe Dockerfile (no changes here):

```bash
docker build -f docker/Dockerfile -t llm-spine-parse:dev .
```

### Publish to Docker Hub as v1.3 (recommended for campus pull-based deploy)

If your campus host pulls from Docker Hub, publish a tagged image:

```bash
docker login
docker buildx create --name multiarch --use --bootstrap 2>/dev/null || docker buildx use multiarch

docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t <dockerhub-username>/llm-spine-parse:v1.3 \
  -t <dockerhub-username>/llm-spine-parse:latest \
  --push \
  .

docker buildx imagetools inspect <dockerhub-username>/llm-spine-parse:v1.3
```

Then set `CAMPUS_IMAGE` in your env file, for example:

```bash
CAMPUS_IMAGE=<dockerhub-username>/llm-spine-parse:v1.3
```

4. Start Jupyter container:

```bash
docker compose \
  --env-file docker_campus/.env.campus \
  -f docker_campus/docker-compose.campus.yml \
  up -d llm-spine-campus
```

5. Start vLLM container for batch workloads (Qwen2.5-VL + LoRA adapter):

```bash
docker compose \
  --env-file docker_campus/.env.campus \
  -f docker_campus/docker-compose.campus.yml \
  up -d vllm-campus
```

6. Validate service on port 8001:

```bash
docker exec -it llm-spine-campus nvidia-smi
curl -s http://localhost:8001/v1/models | cat
```

### View logs for the service:

```bash
docker logs -f vllm-campus
```

### Request examples

To Qwen (batch) with adapter:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "spine_adapter", "messages": [{"role": "user", "content": "Analyze this spine image..."}], "temperature": 0.7}'
```

## Shared Data Layout

Host path defaults to `/data/llm_spine_parse`:

- `hf_home/`: Hugging Face cache and downloaded model files.
- `checkpoints/`: adapters and training checkpoints.
- `input/`: dataset files.
- `img/`: image assets.
- `logs/`: runtime logs.

Container view:

- `/home/jovyan/.hf_home` -> host `hf_home/`
- `/data/checkpoints` -> host `checkpoints/`
- `/data/input` -> host `input/`
- `/data/img` -> host `img/`
- `/data/logs` -> host `logs/`

## Multi-User Notes

Use a shared UNIX group (example: `llmtrain`) and ensure both users are in that group. The bootstrap script sets setgid permissions so new files retain group ownership.

If host permission policy is strict, ask admin to ensure:

- both users belong to the same shared group
- `/data/llm_spine_parse` is writable by that group

## vLLM Campus Launcher

`tools/start_vllm_campus.py` does not replace or modify `tools/start_vllm.py`. It only sets campus defaults and then delegates to the existing launcher.

Default campus overrides:

- `VLLM_HOST=0.0.0.0`
- `VLLM_PORT=8000`
- `VLLM_ADAPTER_PATH=/data/checkpoints/qwen2_5_vl_lora_512Res/adapter_model`
- `HF_HOME=/home/jovyan/.hf_home`
- `NCCL_P2P_DISABLE=1`
- `NCCL_IB_DISABLE=1`
- `NCCL_SHM_DISABLE=1`
- `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1`
- `CUDA_VISIBLE_DEVICES=0`

These compatibility defaults are intended for campus/shared vGPU environments where vLLM can fail with CUDA driver errors during engine initialization.

You can still override with env vars or CLI flags.

## Troubleshooting

- If you are unsure which Dockerfile to use for campus, use `docker/Dockerfile` for now. It is the same known-good image path used by your working NRP flow and keeps behavior aligned across environments.
- Create a dedicated `docker_campus/Dockerfile` only if campus hardware/runtime compatibility cannot be resolved with compose/env settings.
- If `docker login` fails, authenticate first before pulling/pushing private images.
- If GPU is not detected, verify host NVIDIA runtime and run `nvidia-smi` on host and inside container.
- If adapter is not found, set `VLLM_ADAPTER_PATH` in `.env.campus` to the actual mounted path.
- If permissions fail on `/data`, re-run bootstrap and verify group membership.
