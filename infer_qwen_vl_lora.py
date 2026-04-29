"""
Inference helper: load base Qwen2.5-VL and LoRA adapter, then run VLM generation
with an image and a prompt.

Example:
  python infer_qwen_vl_lora.py \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_dir checkpoints/qwen2_5_vl_lora/lora_adapter \
    --image /path/to/spine.jpg \
    --prompt "Extract the book information (title, author, call_no) as JSON."
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id")
    ap.add_argument("--adapter_dir", type=str, required=True, help="Path to LoRA adapter directory")
    ap.add_argument("--image", type=str, required=True, help="Path to input image")
    ap.add_argument("--prompt", type=str, default="Extract the book information (title, author, call_no) as JSON.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quant for inference")
    return ap.parse_args()


def main():
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    use_4bit = not args.no_4bit

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

    print(f"Loading tokenizer/processor: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    processor = AutoProcessor.from_pretrained(args.base_model)

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        quantization_config=bnb_config,
    )

    print(f"Attaching LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    image = Image.open(str(img_path)).convert("RGB")
    prompt = args.prompt

    # Build inputs for VLM
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Generating...")
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    output = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print("\n=== MODEL OUTPUT ===\n")
    print(output)


if __name__ == "__main__":
    main()
