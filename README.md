# LLM Spine Parse Train (QLoRA + vLLM Deployment)

Train Qwen2.5-VL vision-language model with QLoRA for book spine extraction, then deploy with vLLM.

**Three hardware profiles supported:**
- **Home GPU** (12GB VRAM): Fast iteration and testing
- **NRP A100 24GB**: Production training, memory-optimized
- **NRP A100 32GB**: Production training, highest quality

**Pipeline:** QLoRA Training → Test Adapters → vLLM Deployment

---

## Contents
- `notebook.ipynb` – Data prep: generate JSONL with base64 images
- `notebooks/` – Repo-managed Jupyter notebooks shipped with the Docker image
- `train_config.py` – Training profiles (home_dev, nrp_a100, nrp_a100_32gb) and shared config
- `train_qwen_vl_lora_qlora.py` – QLoRA trainer with profile support and evaluation
- `tools/split_jsonl.py` – Split JSONL into train/eval/test sets
- `tools/resize_segments.py` – Resize images for memory optimization
- `img/segments/` – Source images (gitignored except README)
- `input/train.jsonl`, `input/eval.jsonl`, `input/test.jsonl` – Split training data

## Training Data Format (JSONL)
Each line is a JSON object:
```json
{
  "prompt": "Extract the book information from this spine image and return it as JSON with keys: title, author, call_no.",
  "completion": "{\"title\": \"...\", \"author\": \"...\", \"call_no\": \"...\"}",
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install PyTorch with CUDA (check https://pytorch.org/get-started/locally/)
# For CUDA 12.1 (works with driver 13.x):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## NRP & Docker
Use custom image with: docker.io/quist99/llm-spine-parse:v1.2

## Training Profiles

Three profiles optimize for different hardware. **Key insight:** Profile selection is primarily about fitting in VRAM, not quality. The 32GB profile delivers better quality, not just speed.

### Profile: `home_dev` (Auto-selected for <20GB VRAM)
- **Use case:** Fast iteration, testing on subset of data
- Batch size: 1, Gradient accumulation: 16 (effective batch = 16)
- Gradient checkpointing: ON (saves ~2GB VRAM)
- Max length: 512 tokens
- LoRA rank: 8 (smaller adapters)
- Default: 100 samples, 1 epoch
- **Quality note:** Lower LoRA rank = less expressive fine-tuning

### Profile: `nrp_a100` (Auto-selected for 20-28GB VRAM)
- **Use case:** 24GB A100 or similar, production training when 32GB unavailable
- Batch size: 1, Gradient accumulation: 16 (effective batch = 16)
- Gradient checkpointing: ON (required for 24GB)
- Max length: 512 tokens (reduced to fit in VRAM)
- LoRA rank: 8 (matches home_dev for consistent quality)
- Default: Full dataset, 3 epochs
- **Quality note:** Same as home_dev—this is memory-optimized, not quality-improved. Speed/iteration gains from full dataset training, not better adapters.

### Profile: `nrp_a100_32gb` (Auto-selected for ≥28GB VRAM)
- **Use case:** 32GB A100, production training with highest quality
- Batch size: 1, Gradient accumulation: 16 (effective batch = 16)
- Gradient checkpointing: OFF (not needed with 32GB)
- Max length: 1024 tokens (full context window)
- LoRA rank: 16 (double home_dev, more expressive fine-tuning)
- Default: Full dataset, 3 epochs

### Profile Selection Strategy

| Hardware | Profile | Best For | Quality vs Home |
|----------|---------|----------|-----------------|
| <20GB | `home_dev` | Testing, iteration | Baseline |
| 20-28GB | `nrp_a100` | Full training when 32GB unavailable | Same as home (memory-optimized) |
| ≥28GB | `nrp_a100_32gb` | Full training | **Better** (2x LoRA rank + longer context) |

---

## Full Pipeline (Home or NRP)

### Fast Path: Train → Serve with LoRA (⭐ Recommended for 12GB)


**Step 1: QLoRA Training** (main venv)
```bash
source venv/bin/activate

python train_qwen_vl_lora_qlora.py \
  --profile home_dev \
  --data_file input/train.jsonl \
  --eval_data_file input/eval.jsonl \
  --evaluation_strategy epoch
```
**Outputs:** `checkpoints/qwen2_5_vl_lora/adapter_model/` (~50-200MB)


**Step 2: Deploy with vLLM + LoRA** (no merge/AWQ needed!)
```bash
pip install vllm

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
  --enforce-eager
```

**Step 3: Test serving:**
```bash
curl http://localhost:8000/v1/models  # Verify adapter loaded as "default"
```

---

### Step 0: Data Preparation (Optional)
Split your JSONL into train/eval/test sets and optionally resize images:

```bash
# Split data (80/10/10)
python tools/split_jsonl.py \
  --input llm_training_data_multimodal.jsonl \
  --train-out input/train.jsonl \
  --eval-out input/eval.jsonl \
  --test-out input/test.jsonl \
  --eval-size 0.1 \
  --test-size 0.1 \
  --seed 42

# Optional: Resize images for 12GB GPU training (reduces VRAM, allows longer context)
python tools/resize_segments.py \
  --input-dir img/segments \
  --output-dir img/segments_512 \
  --max-size 512 \
  --workers 4
```

Then update your notebook or data generator to point to the resized images.

### Step 1: QLoRA Training (main venv)
```bash
source venv/bin/activate

# Recommended: With evaluation set for best-model selection and early stopping
python train_qwen_vl_lora_qlora.py \
  --data_file input/train.jsonl \
  --eval_data_file input/eval.jsonl \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --early_stopping_patience 3

# Profile-specific examples (all with eval)
# Home GPU (12GB): Fast test with small sample
python train_qwen_vl_lora_qlora.py \
  --profile home_dev \
  --data_file input/train.jsonl \
  --eval_data_file input/eval.jsonl \
  --evaluation_strategy epoch

# NRP 24GB: Full training with eval
python train_qwen_vl_lora_qlora.py \
  --profile nrp_a100 \
  --data_file input/train.jsonl \
  --eval_data_file input/eval.jsonl \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --early_stopping_patience 3

# NRP 32GB: Full training, highest quality
python train_qwen_vl_lora_qlora.py \
  --profile nrp_a100_32gb \
  --data_file input/train.jsonl \
  --eval_data_file input/eval.jsonl \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --early_stopping_patience 3


```


**Test inference with your adapter:**
```bash
# Text completion (use model="default" to invoke your adapter)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "Extract book titles: The Great Gatsby, 1984, Brave New World"
      }
    ],
    "max_tokens": 200,
    "temperature": 0.1
  }'

# Vision completion with your adapter
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
              "url": "https://your-image-url.jpg"
            }
          },
          {
            "type": "text",
            "text": "Extract the book information from this spine image."
          }
        ]
      }
    ],
    "max_tokens": 300,
    "temperature": 0.1
  }'
```

Python client (OpenAI-compatible):
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require real key
)

# Use model="default" to invoke your trained adapter
response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file:///path/to/spine.jpg"}},
                {"type": "text", "text": "Extract book info and return as JSON."}
            ]
        }
    ],
    max_tokens=300,
    temperature=0.1
)

print(response.choices[0].message.content)
```
---


