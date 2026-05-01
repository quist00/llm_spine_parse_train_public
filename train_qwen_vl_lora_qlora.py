"""
QLoRA fine-tuning for Qwen2.5-VL with memory-efficient training.
Trains both vision and text encoders with 4-bit quantization.

Supports environment profiles for different hardware (home GPU vs NRP A100).

Usage:
  # Auto-detect GPU and suggest profile
  python train_qwen_vl_lora_qlora.py --model_id Qwen/Qwen2.5-VL-7B-Instruct
  
  # Explicit profile for home GPU (12GB VRAM)
  python train_qwen_vl_lora_qlora.py --profile home_dev
  
  # Explicit profile for NRP A100 (32GB VRAM)
  python train_qwen_vl_lora_qlora.py --profile nrp_a100
"""
# **CRITICAL:** Set CUDA memory allocator BEFORE torch import to avoid fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import base64
import io
import json
import math
import inspect
from pathlib import Path
from typing import Dict, Any, List

def get_device_and_dtype():
    """
    Determine device and dtype based on available hardware.
    MPS (Apple Silicon) takes priority, then CUDA, then CPU with warning.
    
    Returns:
        tuple: (device, dtype, use_4bit)
    """
    try:
        # Check MPS first (Apple Silicon) - no 4-bit support
        if torch.backends.mps.is_available():
            dtype = torch.bfloat16 if hasattr(torch.backends.mps, 'is_bf16_supported') and torch.backends.mps.is_bf16_supported() else torch.float16
            print(f"🔍 Detected MPS (Apple Silicon) with {dtype} support")
            return (torch.device("mps"), dtype, False)
        
        # Check CUDA (NVIDIA GPU) - supports 4-bit quantization
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"🔍 Detected CUDA GPU ({props.name}) with {vram_gb:.1f}GB VRAM, using {dtype}")
            return (torch.device("cuda"), dtype, True)
        
    except Exception:
        pass
    
    # Fallback to CPU (with warning)
    print("⚠️  No GPU detected - falling back to CPU. Training will be very slow!")
    return (torch.device("cpu"), torch.float32, False)


def get_device_and_dtype():
    """
    Determine device and dtype based on available hardware.
    MPS (Apple Silicon) takes priority, then CUDA, then CPU with warning.
    
    Returns:
        tuple: (device, dtype, use_4bit)
    """
    try:
        # Check MPS first (Apple Silicon) - no 4-bit support
        if torch.backends.mps.is_available():
            dtype = torch.bfloat16 if hasattr(torch.backends.mps, 'is_bf16_supported') and torch.backends.mps.is_bf16_supported() else torch.float16
            print(f"🔍 Detected MPS (Apple Silicon) with {dtype} support")
            return (torch.device("mps"), dtype, False)
        
        # Check CUDA (NVIDIA GPU) - supports 4-bit quantization
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"🔍 Detected CUDA GPU ({props.name}) with {vram_gb:.1f}GB VRAM, using {dtype}")
            return (torch.device("cuda"), dtype, True)
        
    except Exception:
        pass
    
    # Fallback to CPU (with warning)
    print("⚠️  No GPU detected - falling back to CPU. Training will be very slow!")
    return (torch.device("cpu"), torch.float32, False)


import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers.data.data_collator import DataCollatorWithPadding

from train_config import (
    TrainingConfig,
    load_training_data,
    print_training_summary,
    detect_profile,
    load_profile,
    apply_profile_to_config,
    TRAINING_PROFILES,
)


class QwenVLDataCollator:
    """Custom collator for Qwen2.5-VL. Optimized for batch_size=1 with variable image sizes."""
    
    def __init__(self, tokenizer, padding=True, pad_to_multiple_of=None, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
    
    def __call__(self, batch):
        """Collate batch: ensures proper tensor dimensions."""
        if not batch:
            return {}
        
        # For batch_size=1, ensure shapes match model expectations
        if len(batch) == 1:
            item = batch[0]
            result = {}
            
            # Text tensors: prefer [B, L]; if [L], add batch dim
            for key in ["input_ids", "attention_mask", "labels"]:
                if key in item:
                    tensor = item[key]
                    if tensor.dim() == 1:
                        result[key] = tensor.unsqueeze(0)
                    elif tensor.dim() == 2:
                        result[key] = tensor
                    else:
                        result[key] = tensor
            
            # pixel_values: expect [B, T, C, H, W]; normalize shapes
            if "pixel_values" in item:
                pv = item["pixel_values"]
                if pv.dim() == 5:  # [B, T, C, H, W]
                    result["pixel_values"] = pv
                elif pv.dim() == 4:  # [T, C, H, W]
                    result["pixel_values"] = pv.unsqueeze(0)
                elif pv.dim() == 3:  # [C, H, W] -> assume T=1
                    result["pixel_values"] = pv.unsqueeze(0).unsqueeze(0)
                else:
                    # Unexpected: best-effort add missing dims
                    result["pixel_values"] = pv
            
            # image_grid_thw: keep as [B,3] tensor (required for model indexing)
            if "image_grid_thw" in item:
                grid = item["image_grid_thw"]
                if isinstance(grid, torch.Tensor):
                    if grid.dim() == 2 and grid.shape[-1] == 3:
                        # Already [B,3]
                        result["image_grid_thw"] = grid
                    elif grid.dim() == 1 and grid.numel() == 3:
                        # [3,] -> [1,3]
                        result["image_grid_thw"] = grid.unsqueeze(0)
                    else:
                        # Try to reshape to [B,3]
                        result["image_grid_thw"] = grid.reshape(-1, 3)
                elif isinstance(grid, (list, tuple)):
                    # Convert to tensor [B,3]
                    if len(grid) == 3 and not isinstance(grid[0], (list, tuple)):
                        # (3,) -> [1,3]
                        result["image_grid_thw"] = torch.tensor([[int(grid[0]), int(grid[1]), int(grid[2])]], dtype=torch.int64)
                    else:
                        # List of tuples -> [B,3]
                        result["image_grid_thw"] = torch.tensor([list(g) for g in grid], dtype=torch.int64)
                else:
                    result["image_grid_thw"] = grid
            
            return result
        
        # For batch_size > 1, pad text and handle images
        # Extract and pad text
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            ids = item["input_ids"].squeeze() if item["input_ids"].dim() > 1 else item["input_ids"]
            mask = item["attention_mask"].squeeze() if item["attention_mask"].dim() > 1 else item["attention_mask"]
            lbls = item["labels"].squeeze() if item["labels"].dim() > 1 else item["labels"]
            
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lbls)
        
        # Pad to max length
        max_length = max(len(ids) for ids in input_ids)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for ids, mask, lbls in zip(input_ids, attention_mask, labels):
            padding_length = max_length - len(ids)
            padded_input_ids.append(torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=ids.dtype)]))
            padded_attention_mask.append(torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)]))
            padded_labels.append(torch.cat([lbls, torch.full((padding_length,), -100, dtype=lbls.dtype)]))
        
        result = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }
        
        # For pixel_values: stack if same size, otherwise take first (fallback)
        pixel_vals = [item["pixel_values"] for item in batch if "pixel_values" in item]
        if pixel_vals:
            try:
                # Normalize each to [B=1, T, C, H, W], then concat on B
                normed = []
                for pv in pixel_vals:
                    if pv.dim() == 5:
                        normed.append(pv)
                    elif pv.dim() == 4:
                        normed.append(pv.unsqueeze(0))
                    elif pv.dim() == 3:
                        normed.append(pv.unsqueeze(0).unsqueeze(0))
                    else:
                        normed.append(pv)
                result["pixel_values"] = torch.cat(normed, dim=0)
            except RuntimeError:
                # Variable sizes - use first sample only (batch_size should be 1 for variable sizes)
                pv = pixel_vals[0]
                if pv.dim() == 5:
                    result["pixel_values"] = pv
                elif pv.dim() == 4:
                    result["pixel_values"] = pv.unsqueeze(0)
                elif pv.dim() == 3:
                    result["pixel_values"] = pv.unsqueeze(0).unsqueeze(0)
                else:
                    result["pixel_values"] = pv
        
        # Stack image_grid_thw -> keep as [B,3] tensor
        grid_vals = [item["image_grid_thw"] for item in batch if "image_grid_thw" in item]
        if grid_vals:
            # Normalize all to tensors, then stack
            grid_tensors = []
            for g in grid_vals:
                if isinstance(g, torch.Tensor):
                    if g.dim() == 2 and g.shape[-1] == 3:
                        grid_tensors.append(g)
                    elif g.dim() == 1 and g.numel() == 3:
                        grid_tensors.append(g.unsqueeze(0))
                    else:
                        grid_tensors.append(g.reshape(-1, 3))
                elif isinstance(g, (list, tuple)):
                    if len(g) == 3 and not isinstance(g[0], (list, tuple)):
                        grid_tensors.append(torch.tensor([[int(g[0]), int(g[1]), int(g[2])]], dtype=torch.int64))
                    else:
                        grid_tensors.append(torch.tensor([list(x) for x in g], dtype=torch.int64))
                else:
                    grid_tensors.append(g)
            
            if grid_tensors:
                try:
                    result["image_grid_thw"] = torch.cat(grid_tensors, dim=0)
                except:
                    result["image_grid_thw"] = grid_tensors[0]
        
        return result


def b64_to_pil(data_uri: str, example_idx: int) -> Image.Image:
    """Convert base64 data URI to PIL Image with clearer error messages."""
    if not data_uri:
        raise ValueError(f"Example {example_idx} is missing 'image' data (got empty/None)")
    if not isinstance(data_uri, str):
        raise TypeError(
            f"Example {example_idx} field 'image' must be a string, got {type(data_uri)}"
        )

    if data_uri.startswith("data:image/"):
        data_uri = data_uri.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(data_uri)
    except Exception as exc:  # pragma: no cover - safety guard for bad data
        raise ValueError(f"Example {example_idx} image base64 decode failed: {exc}") from exc

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - safety guard for corrupt bytes
        raise ValueError(f"Example {example_idx} image bytes could not be opened: {exc}") from exc

    return img


class QwenVLLoRADataset(Dataset):
    """Dataset for Qwen2.5-VL LoRA fine-tuning using processor for text+image."""
    
    def __init__(self, examples: List[Dict[str, Any]], processor):
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex.get("prompt", "") or ""
        completion = ex.get("completion", "") or ""
        image_pil = b64_to_pil(ex.get("image", ""), idx)

        # Qwen2.5-VL expects an explicit image placeholder token in the text.
        image_token = getattr(self.processor, "image_token", None) or "<image>"
        text_with_image = f"{prompt}\n{image_token}\nAnswer: {completion}"
        
        # Use processor to jointly prepare text + image; let it produce tensors with proper dims
        proc = self.processor(
            text=text_with_image,
            images=image_pil,
            padding=False,
            return_tensors="pt",
        )

        # Processor returns batch-first tensors; keep shapes consistent
        input_ids = proc["input_ids"].squeeze(0) if proc["input_ids"].dim() == 2 else proc["input_ids"]
        attention_mask = proc["attention_mask"].squeeze(0) if proc["attention_mask"].dim() == 2 else proc["attention_mask"]
        pixel_values = proc["pixel_values"]  # [B=1, T, C, H, W] for a single sample
        
        # Qwen2.5-VL requires image_grid_thw for vision position embeddings
        result = {
            "pixel_values": pixel_values,  # Keep as 2D tensor
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
        
        # Add image_grid_thw if present (required for vision positional encoding)
        # Keep as [B=1, 3] to simplify collation
        if "image_grid_thw" in proc:
            grid_thw_raw = proc["image_grid_thw"]
            result["image_grid_thw"] = grid_thw_raw  # tensor [1,3]
        
        return result


def setup_model(model_id: str, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """
    Setup LoRA configuration for vision-language model.
    Supports both CUDA (with 4-bit quantization) and MPS (without quantization).
    
    Args:
        model_id: Hugging Face model ID
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        
    Returns:
        tuple: (model, tokenizer, processor, device, dtype, use_4bit)
    """
    print("[1/3] Setting up model configuration...")
    
    # Determine device and whether to use 4-bit quantization
    device, dtype, use_4bit = get_device_and_dtype()
    
    # Load tokenizer and processor.
    # Pin to slow processor to avoid behavior drift from fast-processor default changes.
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    # Transformers is migrating from AutoModelForVision2Seq -> AutoModelForImageTextToText.
    # Prefer the new class when available while staying compatible with older releases.
    model_cls = getattr(transformers, "AutoModelForImageTextToText", AutoModelForVision2Seq)
    
    # Load model
    print(f"Loading model {model_id} on {device} with dtype={dtype}, use_4bit={use_4bit}...")
    
    if use_4bit:
        # CUDA path with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        model = model_cls.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # Prepare 4-bit model for LoRA gradient computation
        model = prepare_model_for_kbit_training(model)
    else:
        # MPS or CPU path without quantization
        try:
            model = model_cls.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        except TypeError:
            # Fallback for older transformers versions
            model = model_cls.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        model.to(device)
    
    # Setup LoRA on both vision and text components
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            # Text attention layers
            "q_proj", "v_proj",
            # MLP layers  
            "gate_proj", "up_proj", "down_proj",
            # Vision components (if available in model)
            "linear_q", "linear_v",
        ],
    )
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer, processor, device, dtype, use_4bit


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-VL with environment profiles")
    
    # Profile selection
    parser.add_argument("--profile", type=str, choices=list(TRAINING_PROFILES.keys()),
                        help="Training profile (home_dev, nrp_a100). Auto-detects if not specified.")
    
    # Model and data
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model ID from Hugging Face")
    parser.add_argument("--data_file", type=str, default="llm_training_data_multimodal.jsonl",
                        help="Path to JSONL training data")
    parser.add_argument("--eval_data_file", type=str, default=None,
                        help="Optional path to JSONL evaluation data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen2_5_vl_lora",
                        help="Output directory for checkpoints")
    
    # Override profile settings (optional)
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Number of training epochs (overrides profile)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size per GPU (overrides profile)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Gradient accumulation steps (overrides profile)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Max sequence length (overrides profile)")
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None,
                        help="Enable gradient checkpointing (overrides profile)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides profile)")
    parser.add_argument("--lora_r", type=int, default=None,
                        help="LoRA rank (overrides profile)")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha (overrides profile)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max training samples (overrides profile, for testing)")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Warmup steps (overrides profile)")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N steps (overrides profile)")
    parser.add_argument("--logging_steps", type=int, default=None,
                        help="Log every N steps (overrides profile)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override total training steps (for debugging)")
    # Evaluation controls
    parser.add_argument("--evaluation_strategy", type=str, choices=["no", "steps", "epoch"], default=None,
                        help="Evaluation strategy when eval data is provided. Defaults to 'epoch' if eval data is set.")
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="Run evaluation every N steps when evaluation_strategy=steps")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Eval batch size per device (default 1 for VL models)")
    parser.add_argument("--load_best_model_at_end", action="store_true",
                        help="Load the best model at end based on eval metric")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss",
                        help="Metric to select best model (default: eval_loss)")
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of evaluation calls with no improvement after which training will be stopped early")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load training data
    print(f"\n{'='*60}")
    print("QWEN2.5-VL QLORA FINE-TUNING PIPELINE")
    print(f"{'='*60}\n")
    
    # Determine profile
    if args.profile:
        profile_name = args.profile
        print(f"✓ Using explicit profile: {profile_name}")
    else:
        profile_name = detect_profile()
        print(f"✓ Auto-detected profile: {profile_name}")
    
    # Load profile and create config
    profile = load_profile(profile_name)
    config = TrainingConfig(
        data_file=args.data_file,
        output_dir=args.output_dir,
    )
    config = apply_profile_to_config(config, profile)
    
    # Apply CLI overrides
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_length is not None:
        config.max_length = args.max_length
    if args.gradient_checkpointing is not None:
        config.gradient_checkpointing = args.gradient_checkpointing
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.lora_r is not None:
        config.lora_r = args.lora_r
    if args.lora_alpha is not None:
        config.lora_alpha = args.lora_alpha
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.save_steps is not None:
        config.save_steps = args.save_steps
    if args.logging_steps is not None:
        config.logging_steps = args.logging_steps
    
    config.lora_dropout = args.lora_dropout
    
    # Load examples
    examples = load_training_data(config.data_file, config.max_samples)
    print(f"\n✓ Loaded {len(examples)} training examples")
    eval_examples = None
    if args.eval_data_file:
        try:
            eval_examples = load_training_data(args.eval_data_file, None)
            print(f"✓ Loaded {len(eval_examples)} eval examples from {args.eval_data_file}")
        except Exception as e:
            print(f"⚠️  Failed to load eval data '{args.eval_data_file}': {e}. Continuing without eval.")
            eval_examples = None
    
    # Print summary
    print_training_summary(config, len(examples))
    
    # Setup model with LoRA (supports both CUDA 4-bit and MPS)
    model, tokenizer, processor, device, dtype, use_4bit = setup_model(
        args.model_id,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    # PyTorch >=2.9 requires explicit checkpoint behavior; also ensure inputs can carry grads
    # when using gradient checkpointing with PEFT/QLoRA.
    if config.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
    
    # Create dataset
    print("\n[2/3] Preparing dataset...")
    train_dataset = QwenVLLoRADataset(examples, processor)
    eval_dataset = None
    if eval_examples:
        eval_dataset = QwenVLLoRADataset(eval_examples, processor)

    # Quick sanity peek at first example to verify shapes/keys
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("Sample keys:", list(sample.keys()))
        for k, v in sample.items():
            try:
                print(f"  {k}: shape {tuple(v.shape)} dtype {v.dtype}")
            except Exception:
                print(f"  {k}: type {type(v)}")
    else:
        print("Dataset is empty after loading examples; nothing to train.")
    
    # Setup training arguments with optional evaluation
    evaluation_strategy = "no"
    eval_steps = None
    if eval_dataset is not None:
        evaluation_strategy = args.evaluation_strategy if args.evaluation_strategy is not None else "epoch"
        eval_steps = args.eval_steps

    save_strategy = "steps"
    if evaluation_strategy in {"steps", "epoch"}:
        save_strategy = evaluation_strategy

    load_best = bool(args.load_best_model_at_end and eval_dataset is not None)
    metric_best = args.metric_for_best_model if eval_dataset is not None else None
    greater_is_better = None
    if metric_best is not None:
        name = metric_best.lower()
        greater_is_better = not ("loss" in name or "perplex" in name)

    # Determine training optimizers based on device
    if use_4bit:
        optimizer = "paged_adamw_8bit"  # CUDA with 4-bit
    else:
        optimizer = "adamw_torch"  # MPS or CPU
    
    # Build TrainingArguments with compatibility fallback for older Transformers
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    ta_kwargs = dict(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        save_strategy=save_strategy,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=optimizer,
        bf16=dtype == torch.bfloat16 and device.type == "cuda",
        fp16=dtype == torch.float16,
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_pin_memory=(device.type != "mps"),  # MPS doesn't benefit from pin memory
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        seed=config.seed,
    )

    # Optional/Versioned args
    if "per_device_eval_batch_size" in ta_params:
        ta_kwargs["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    if "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = evaluation_strategy
    elif "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = evaluation_strategy
    if "eval_steps" in ta_params:
        ta_kwargs["eval_steps"] = eval_steps
    if "load_best_model_at_end" in ta_params:
        ta_kwargs["load_best_model_at_end"] = load_best
    if "metric_for_best_model" in ta_params and metric_best is not None:
        ta_kwargs["metric_for_best_model"] = metric_best
    if "greater_is_better" in ta_params and greater_is_better is not None:
        ta_kwargs["greater_is_better"] = greater_is_better
    if config.gradient_checkpointing and "gradient_checkpointing_kwargs" in ta_params:
        ta_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = TrainingArguments(**ta_kwargs)
    
    # Create custom data collator for vision + text features
    data_collator = QwenVLDataCollator(tokenizer=tokenizer)
    
    # Create trainer
    print("[3/3] Setting up trainer...")
    callbacks = []
    if eval_dataset is not None and args.early_stopping_patience is not None and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )
    trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    
    # Train
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    # Display device information
    if use_4bit:
        print(f"Device map: {model.hf_device_map}")
    else:
        print(f"Training device: {device}")
    print("First trainable parameter names (sample):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("  ", name)
            break
    trainer.train()

    # Final evaluation (if eval set provided)
    if eval_dataset is not None:
        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}\n")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if "eval_loss" in metrics and math.isfinite(metrics["eval_loss"]):
            try:
                metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                pass
        print("Eval metrics:", {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()})
        metrics_path = Path(config.output_dir) / "metrics_eval.json"
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"✓ Saved eval metrics to {metrics_path}")
        except Exception as e:
            print(f"⚠️  Failed to save eval metrics: {e}")
    
    # Save final model and adapters
    print(f"\n{'='*60}")
    print("SAVING ADAPTERS")
    print(f"{'='*60}\n")
    
    # Save adapter weights only
    adapter_path = Path(config.output_dir) / "adapter_model"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    print(f"✓ Saved LoRA adapters to {adapter_path}")
    
    # Save tokenizer
    tokenizer.save_pretrained(str(adapter_path))
    print(f"✓ Saved tokenizer to {adapter_path}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
   

if __name__ == "__main__":
    main()
