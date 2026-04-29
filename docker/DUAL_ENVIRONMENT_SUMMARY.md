# Docker Setup Summary - Dual-Environment Support

## ✅ What Changed

Your Docker setup now works in **both environments** using a single image:

### **1. Base Image Switch**
- **Before:** `nvidia/cuda:12.1.0-devel-ubuntu22.04` (manual setup)
- **After:** `quay.io/jupyter/pytorch-notebook:latest` (Jupyter-ready)

**Benefits:**
- ✅ Already includes `jupyterhub-singleuser` entrypoint (CSUF requirement)
- ✅ PyTorch and CUDA stack pre-installed via the base image
- ✅ Non-root jovyan user configured
- ✅ Proper Jupyter Lab setup

### **2. CMD Strategy**
- **Campus (Nautilus):** Uses inherited `jupyterhub-singleuser` → auto-detected by DockerSpawner
- **Home:** Override with `start-notebook.sh` via docker-compose → launches Jupyter Lab

### **3. Path Updates**
All paths changed from `/workspace` → `/home/jovyan/llm_train` to match this project layout:
```yaml
volumes:
  - ..:/home/jovyan/llm_train      # Your project (code + data + checkpoints)
```

---

## 📋 Testing Checklist

### Local (Before Pushing to Docker Hub)

1. **Build:**
   ```bash
  cd ~/path/to/llm_spine_parse_train
   docker build -f docker/Dockerfile -t llm-spine-parse:dev .
   ```
  On Apple Silicon, this local build is usually `linux/arm64`. Use `buildx --platform linux/amd64` for the image you push to Docker Hub.

2. **Test Jupyter:**
   ```bash
   docker run --rm --runtime=nvidia --gpus all -p 8888:8888 \
     -v $(pwd):/home/jovyan/llm_train \
     llm-spine-parse:dev \
     start-notebook.sh --NotebookApp.token=''
   ```
   Visit: http://localhost:8888

3. **Test GPU:**
   ```bash
   docker run --rm --runtime=nvidia --gpus all llm-spine-parse:dev \
     python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

4. **Test training script:**
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   docker-compose exec llm-spine-parse \
     python /home/jovyan/llm_train/train_qwen_vl_lora_qlora.py --profile home_dev --num_samples 10
   ```

---

## 🚀 Deployment Steps

### 1. Push to Docker Hub
```bash
# Login once
docker login

# Build and push non-ARM Linux image for Nautilus / Jupyter GPU hosts
docker buildx create --name multiarch --use --bootstrap
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t yourusername/llm-spine-parse:v1.0 \
  -t yourusername/llm-spine-parse:latest \
  --push \
  .

# Verify manifest
docker buildx imagetools inspect yourusername/llm-spine-parse:latest
```

### 2. Test on Campus (Nautilus JupyterHub)
1. Log into Nautilus JupyterHub
2. Create Server → Select resources (e.g., 1 GPU, 32GB RAM)
3. Notebook Container Image → Select **"Other"**
4. Paste: `docker.io/yourusername/llm-spine-parse:latest`
5. Start Server
6. Verify:
   - Jupyter Lab opens
   - Run: `!nvidia-smi` in a notebook
   - Import torch and check CUDA

---

## 🏠 vs 🏫 Comparison

| Feature | Home (docker-compose) | Campus (Nautilus) |
|---------|----------------------|-------------------|
| **Base Command** | `start-notebook.sh` (override) | `jupyterhub-singleuser` (inherited) |
| **Authentication** | None (local dev) | Campus SSO |
| **Multiple Services** | ✅ Jupyter + vLLM separate | ❌ Single container only |
| **Port Exposure** | ✅ 8888, 8000, 8001, etc. | ❌ Only 8888 (Jupyter) |
| **vLLM Server** | Separate service in compose | Manual launch inside Jupyter |
| **Data Location** | Inside mounted project repo | Campus-managed persistence |
| **Resource Limits** | Docker resource constraints | JupyterHub scheduler |

---

## 🔧 Common Tasks

### Run Training on Campus
```python
# In a Jupyter terminal
cd ~/llm_train
python train_qwen_vl_lora_qlora.py --profile nrp_a100
```

### Run vLLM on Campus (Manual)
```python
# In a Jupyter terminal (background process)
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 64 \
  --lora-modules spine_adapter=~/llm_train/checkpoints/qwen2_5_vl_lora/adapter_model \
  --host 127.0.0.1 --port 8000 > vllm.log 2>&1 &

# Access from another notebook
import requests
response = requests.post(
  "http://127.0.0.1:8000/v1/completions",
  json={"model": "spine_adapter", "prompt": "Test prompt", "max_tokens": 64},
)
```

### Run vLLM at Home (Separate Service)
```bash
docker-compose up vllm-server
# API available at http://localhost:8001
```

---

## ✅ CSUF Compliance Check

| Requirement | Status | Notes |
|-------------|--------|-------|
| Python 3.6+ | ✅ Pass | Python 3.11 from base image |
| `notebook` package | ✅ Pass | Included in base image |
| JupyterHub 4.x | ✅ Pass | jupyterhub==4.0.2 installed |
| Proper CMD | ✅ Pass | Inherited from base (jupyterhub-singleuser) |
| Non-root user | ✅ Pass | jovyan (uid 1000) |
| Public registry | ⏳ Pending | Push to Docker Hub |

**Verdict:** Ready for campus deployment after pushing to Docker Hub!

