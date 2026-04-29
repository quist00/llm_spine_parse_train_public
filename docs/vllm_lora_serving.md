# vLLM LoRA Serving Guide

Serve trained LoRA adapters on quantized base models with vLLM.

## Why This Approach

**Traditional pipeline problems:**
- Merge LoRA → base requires 24GB+ RAM
- Re-quantize merged model requires AWQ/GPTQ tooling (version conflicts, format mismatches)
- Results in large (~7GB) static model files
- Can't swap adapters without re-serving

**vLLM LoRA solution:**
- Serve official pre-quantized base (Qwen/Qwen2.5-VL-7B-Instruct-AWQ, ~5GB)
- Load tiny LoRA adapters dynamically (~50-200MB)
- Swap multiple adapters without restart
- Fits 12GB VRAM with full vision capability
- No merge/quantization tooling needed

---

## Prerequisites

```bash
# Install vLLM (in main venv or dedicated serving venv)
pip install vllm

# Verify your adapter structure
ls checkpoints/qwen2_5_vl_lora/adapter_model/
# Should contain:
#   adapter_config.json
#   adapter_model.safetensors (or adapter_model.bin)
```

---

## Basic Serving

### Start vLLM Server with LoRA Adapter

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --trust-remote-code \
  --enable-lora \
  --lora-modules default=checkpoints/qwen2_5_vl_lora/adapter_model \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 1536 \
  --max-num-seqs 1 \
  --swap-space 2 \
  --enforce-eager
```

**Key flags explained:**
- `--model Qwen/Qwen2.5-VL-7B-Instruct-AWQ`: Official AWQ base (auto-downloads from HF on first run)
- `--enable-lora`: Enable LoRA adapter loading
- `--lora-modules default=/path/to/adapter_model`: Load adapter with name "default"
- `--gpu-memory-utilization 0.85`: Reserve 15% VRAM for overhead (tune for your GPU)
- `--max-model-len 1536`: Context length (reduce if OOM)
- `--max-num-seqs 1`: Concurrent sequences (increase for higher throughput if VRAM allows)
- `--enforce-eager`: Disable CUDA graph for lower VRAM (slightly slower)

### Verify Server Started

```bash
# Check health
curl http://localhost:8000/health

# List available models (should show base + "default" adapter)
curl http://localhost:8000/v1/models | jq
```

Expected output:
```json
{
  "data": [
    {
      "id": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
      "object": "model",
      "root": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    },
    {
      "id": "default",
      "object": "model",
      "parent": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
      "root": "/path/to/your/adapter_model"
    }
  ]
}
```

✅ If you see both models, adapter loaded successfully!

---

## Testing Inference

### Text-Only Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "Extract book titles from: The Great Gatsby by F. Scott Fitzgerald, 1984 by George Orwell"
      }
    ],
    "max_tokens": 200,
    "temperature": 0.1
  }'
```

### Vision + Text Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://your-bookshelf-image.jpg"
            }
          },
          {
            "type": "text",
            "text": "Extract the book information from this spine image and return as JSON with keys: title, author, call_no."
          }
        ]
      }
    ],
    "max_tokens": 300,
    "temperature": 0.1
  }'
```

**Image URL formats supported:**
- `https://...` (public URL)
- `data:image/jpeg;base64,...` (base64-encoded)
- `file:///path/to/image.jpg` (local file path)

---

## Python Client Example

### OpenAI-Compatible Client

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require real key
)

# Helper to encode local image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Text-only request
response = client.chat.completions.create(
    model="default",  # Use your adapter
    messages=[
        {
            "role": "user",
            "content": "Extract book titles: The Catcher in the Rye, Moby Dick"
        }
    ],
    max_tokens=200,
    temperature=0.1
)
print(response.choices[0].message.content)

# Vision request with local image
image_b64 = encode_image("img/segments/spine_001.jpg")
response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": "Extract book info as JSON: {title, author, call_no}"
                }
            ]
        }
    ],
    max_tokens=300,
    temperature=0.1
)
print(response.choices[0].message.content)
```

---

## Advanced: Multiple Adapters

Serve multiple LoRA adapters simultaneously and switch between them per request:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --trust-remote-code \
  --enable-lora \
  --lora-modules \
    512res=checkpoints/qwen2_5_vl_lora_512Res/adapter_model,\
    fullres=checkpoints/qwen2_5_vl_lora_fullres_v2/adapter_model \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 1536
```

Then select adapter per request:
```bash
# Use 512res adapter
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "512res", "messages": [...]}'

# Use fullres adapter
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "fullres", "messages": [...]}'
```

---

## Memory Tuning

Adjust for different VRAM sizes:

### 12GB GPU (Tight Fit)
```bash
--gpu-memory-utilization 0.85 \
--max-model-len 1536 \
--max-num-seqs 1 \
--swap-space 2 \
--enforce-eager
```

### 16GB GPU (Comfortable)
```bash
--gpu-memory-utilization 0.90 \
--max-model-len 2048 \
--max-num-seqs 2 \
--swap-space 4
```

### 24GB GPU (Full Context)
```bash
--gpu-memory-utilization 0.95 \
--max-model-len 4096 \
--max-num-seqs 4 \
--swap-space 8
```

**If you hit OOM:**
1. Reduce `--max-model-len`
2. Lower `--gpu-memory-utilization`
3. Add `--enforce-eager` if not already set
4. Reduce `--max-num-seqs` to 1

---

## Production Deployment

### Docker Container

```dockerfile
FROM vllm/vllm-openai:latest

COPY checkpoints/qwen2_5_vl_lora/adapter_model /app/adapter_model

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server", \
  "--model", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", \
  "--trust-remote-code", \
  "--enable-lora", \
  "--lora-modules", "default=/app/adapter_model", \
  "--host", "0.0.0.0", \
  "--port", "8000"]
```

Build and run:
```bash
docker build -t vllm-lora-spine .
docker run --gpus all -p 8000:8000 vllm-lora-spine
```

---

## References

- vLLM LoRA docs: https://docs.vllm.ai/en/latest/models/lora.html
- Qwen2.5-VL official AWQ: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ
- OpenAI API compatibility: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
