# LLM Spine Parse Train - Docker Build & Deployment Guide

## Architecture

This Docker image is designed to work in **two environments**:

1. **Campus (Nautilus JupyterHub):** Runs with `jupyterhub-singleuser` (inherited from base image)
2. **Home/Local:** Runs with Jupyter Lab via docker-compose or direct `docker run`

**Base image:** `quay.io/jupyter/pytorch-notebook:latest`
- ✅ Already includes proper JupyterHub entrypoint
- ✅ PyTorch and CUDA stack pre-installed via the base image
- ✅ Non-root jovyan user (uid 1000)
- ✅ Tested with JupyterHub 4.x

### Path Convention

Use `/home/jovyan/llm_train` as the canonical project root inside this container image.
All training, adapter, and notebook commands in this guide assume that path.

---

## Quick Start

### Prerequisites
- **Docker** 20.10+
- **Docker Compose** 2.0+ (for local testing)
- **Docker Buildx** (for pushing `linux/amd64` images from Apple Silicon)
- **NVIDIA Docker Runtime** (for GPU support)
  ```bash
  # Optional host-level GPU runtime check
  docker run --rm --runtime=nvidia nvidia/cuda:12.1.0-base nvidia-smi
  ```

---

## Local Development (Mac/Linux/Windows)

### 1. Build locally
```bash
cd ~/path/to/llm_spine_parse_train

docker build -f docker/Dockerfile -t llm-spine-parse:dev .
```

On Apple Silicon, that default build is typically `linux/arm64`. That is fine for local iteration, but the image you push for Nautilus or other non-ARM Linux/Jupyter GPU environments should be built as `linux/amd64`.

### 2. Run with Jupyter Lab (simple)
```bash
# Interactive Jupyter Lab on GPU, no authentication
docker run --rm --runtime=nvidia \
  -p 8888:8888 \
  -v $(pwd):/home/jovyan/llm_train \
  --gpus all \
  -e GRANT_SUDO=yes \
  llm-spine-parse:dev \
  start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''
```

Then open: `http://localhost:8888` in your browser.



**Access:**
- Jupyter Lab: `http://localhost:8888` (no password)
- vLLM API: `http://localhost:8001` (when `vllm-server` is running)

---

## Publishing to Docker Hub (for Nautilus JupyterHub)

### 1. Create Docker Hub account
- Go to https://hub.docker.com → Sign Up
- Create repo: `llm-spine-parse` (or similar)

### 2. Build & tag for registry
```bash
# Replace `yourusername` with your actual Docker Hub username
docker buildx create --name multiarch --use --bootstrap
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t yourusername/llm-spine-parse:latest \
  -t yourusername/llm-spine-parse:v1.0 \
  --cache-from type=registry,ref=yourusername/llm-spine-parse:buildcache \
  --cache-to type=registry,ref=yourusername/llm-spine-parse:buildcache,mode=max \
  --push \
  .
```

First build vs subsequent builds:
- If `--cache-from ...:buildcache` fails with "importing cache manifest" on a new repo/tag, run one build without `--cache-from` to seed cache.
- After that first successful push, use both `--cache-from` and `--cache-to` for faster rebuilds.

### 2b. Why cache-from/cache-to matters

`buildx` often runs on a builder that does not keep your local pip/layer cache between runs. Without explicit cache export/import, repeated builds can re-download dependencies and re-run expensive layers.

- `--cache-to ...:buildcache` exports reusable build cache to your registry.
- `--cache-from ...:buildcache` imports that cache on the next build.
- Combined with the Dockerfile pip cache mount, this reduces repeated dependency download time significantly.

The first cached build is usually still slow; subsequent builds are where you get the speedup.

### 3. Login & push
```bash
docker login
```

### 4. Verify on registry
```bash
# Should show linux/amd64 in the manifest
docker buildx imagetools inspect yourusername/llm-spine-parse:latest
```

### 5. Run Published Image on Non-JupyterHub Runtime (optional)

Use this when running directly on your own VM/runtime instead of Nautilus JupyterHub.

```bash
docker pull yourusername/llm-spine-parse:latest

docker run --rm --gpus all \
  -p 8888:8888 \
  -v /path/to/repo:/home/jovyan/llm_train \
  yourusername/llm-spine-parse:latest \
  start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''
```

---

## Using in Nautilus JupyterHub (CSUF Campus)

**This image is intended to work on campus without Dockerfile changes.**

The base image (`quay.io/jupyter/pytorch-notebook`) already includes the proper `jupyterhub-singleuser` entrypoint that Nautilus expects.

### Steps:

1. **Build & push to Docker Hub** (see Publishing section above)

2. **Log in** to Nautilus JupyterHub at your campus instance

3. **Create Server:**
   - Adjust resources (e.g., 4 CPUs, 32GB RAM, 1 GPU)
   - **Notebook Container Image** → Select **"Other"**
   - **Paste:** `docker.io/yourusername/llm-spine-parse:latest`
   - Click **Start Server**

4. **Your environment launches** with:
   - ✅ Jupyter Lab pre-configured
  - ✅ All ML/training + LoRA adapter dependencies ready
   - ✅ GPU access (if requested)
  - ✅ Python, PyTorch, and CUDA provided through the selected base image

---

## Container Paths & Volume Mounts

**Inside container:**
- `/home/jovyan/llm_train` – Your project root (training scripts, configs, notebooks)
- `/home/jovyan/.jupyter` – Jupyter config (auto-persisted on campus)

### Repo Notebook Bootstrap

The image bundles notebooks from `notebooks/*.ipynb` and copies them into a runtime notebooks folder on startup.

- Preferred destination (when project directory is present): `/home/jovyan/llm_train/notebooks`
- Fallback destination: `/home/jovyan/notebooks`

Copy behavior is non-destructive: existing notebook files in the destination are not overwritten.

Current bundled smoke-test notebook:
- `notebooks/vllm_adapter_image_smoke_test.ipynb`

**Local mappings (docker-compose):**
```yaml
volumes:
  - ..:/home/jovyan/llm_train       # Project root (code + data + checkpoints)
```

---

## Recommended Host Mappings

**Host mappings (recommended):**
```bash
docker run --rm --runtime=nvidia --gpus all \
  -p 8888:8888 \
  -v $(pwd):/home/jovyan/llm_train \
  llm-spine-parse:dev \
  start-notebook.sh --NotebookApp.token=''
```

---

## Training Workflows
### caching HF and improve loading
``` bash
cd ~/llm_train

# Persistent Hugging Face cache (survives pod restarts if home is persistent)
export HF_HOME=~/llm_train/.hf_home
mkdir -p "$HF_HOME"

# Faster checkpoint shard loading
export HF_ENABLE_PARALLEL_LOADING=true
export HF_PARALLEL_LOADING_WORKERS=8

```

### Workflow 1: Train LoRA adapter
Run from a host terminal (local Docker runtime):
```bash
docker run --rm --runtime=nvidia \
  -v $(pwd):/home/jovyan/llm_train \
  --gpus all \
  llm-spine-parse:dev \
  python /home/jovyan/llm_train/train_qwen_vl_lora_qlora.py --profile nrp_a100
```

### Workflow 1b: Full dataset run in JupyterHub terminal
Run inside a Jupyter terminal on campus (from the repo root):
```bash
cd ~/llm_train

export RUN_NAME_FULL=qwen2_5_vl_lora_full_$(date +%Y%m%d_%H%M%S)
export OUT_DIR_FULL=checkpoints/$RUN_NAME_FULL

python train_qwen_vl_lora_qlora.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --profile nrp_a100_32gb \
  --data_file input/512/train.jsonl \
  --eval_data_file input/512/eval.jsonl \
  --output_dir "$OUT_DIR_FULL" \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --early_stopping_patience 3
```

### Workflow 2: Test existing LoRA adapter
Run from a host terminal (local Docker runtime):
```bash
docker run --rm --runtime=nvidia \
  -v $(pwd):/home/jovyan/llm_train \
  --gpus all \
  llm-spine-parse:dev \
  python /home/jovyan/llm_train/test_lora_inference.py
```

### Workflow 3: Interactive adapter checks in Jupyter
Run from a host terminal, then execute checks in notebook cells or a Jupyter terminal:
```bash
docker run --rm --runtime=nvidia \
  -v $(pwd):/home/jovyan/llm_train \
  --gpus all \
  -p 8888:8888 \
  llm-spine-parse:dev \
  start-notebook.sh --NotebookApp.token=''
# Then run adapter validation scripts from notebook terminal/cells.
```

### Workflow 4: vLLM server with base model + LoRA adapter
Run from a host terminal:
```bash
# Start both services
docker-compose -f docker/docker-compose.yml up -d

# Check vLLM logs
docker-compose -f docker/docker-compose.yml logs -f vllm-server

# Example OpenAI-compatible request
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spine_adapter",
    "messages": [{"role": "user", "content": "Return JSON: {\"title\":\"...\",\"author\":\"...\",\"call_no\":\"...\"}"}],
    "max_tokens": 128
  }'
```

The compose service mounts your workspace and loads the adapter from:
`/home/jovyan/llm_train/checkpoints/qwen2_5_vl_lora/adapter_model`

If your adapter lives elsewhere, update the `--lora-modules` path in `docker/docker-compose.yml`.

---

## GPU Debugging

### Check GPU inside container
```bash
docker run --rm --runtime=nvidia --gpus all llm-spine-parse:dev nvidia-smi
```

### Verify CUDA + PyTorch
```bash
docker run --rm --runtime=nvidia --gpus all llm-spine-parse:dev \
  python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Current device: {torch.cuda.current_device()}')
print(f'Device name: {torch.cuda.get_device_name()}')
  "
```

---



---

## Image Versioning & Maintenance

### Tag naming convention
```bash
# Development (local testing)
docker build -t llm-spine-parse:dev .

# Release / deployment image for non-ARM Linux
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t yourusername/llm-spine-parse:v1.0 \
  -t yourusername/llm-spine-parse:v1.2 \
  --cache-from type=registry,ref=yourusername/llm-spine-parse:buildcache \
  --cache-to type=registry,ref=yourusername/llm-spine-parse:buildcache,mode=max \
  --push \
  .
```

### Update dependencies
If you update `docker/requirements.txt`:
```bash
# Rebuild without cache to get fresh dependencies
docker build --no-cache -f docker/Dockerfile -t llm-spine-parse:dev .
```

---

## Next Steps

1. **Test locally:**
   ```bash
   docker build -f docker/Dockerfile -t llm-spine-parse:dev .
   docker run --rm --runtime=nvidia --gpus all -p 8888:8888 llm-spine-parse:dev
   ```

2. **Push to Docker Hub:**
   ```bash
   # If this is your first cached build, run once without --cache-from.
   docker buildx build \
     --platform linux/amd64 \
     -f docker/Dockerfile \
     -t yourusername/llm-spine-parse:v1.0 \
     -t yourusername/llm-spine-parse:latest \
     --cache-from type=registry,ref=yourusername/llm-spine-parse:buildcache \
     --cache-to type=registry,ref=yourusername/llm-spine-parse:buildcache,mode=max \
     --push \
     .
   ```

3. **Test on Nautilus:** Use the image URL (docker.io/yourusername/llm-spine-parse:latest) in JupyterHub "Other" option

4. **Iterate:** Update Dockerfile → rebuild → push → test on campus infra

---

## Resources

- [Docker Docs](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [JupyterHub DockerSpawner](https://jupyterhub-dockerspawner.readthedocs.io/)
- [CSUF Nautilus Policy](https://csuf.screenstepslive.com/m/129011/l/1958774-how-to-use-the-other-image-option-in-jupyterhub)
