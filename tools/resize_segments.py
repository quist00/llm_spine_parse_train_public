#!/usr/bin/env python3
"""
Resize all images in a directory to a maximum longest side and save to a new directory.

Example:
  python tools/resize_segments.py \
    --input-dir img/segments \
    --output-dir img/segments_512 \
    --max-size 512

Notes:
- Preserves aspect ratio; scales so max(height, width) == max_size.
- Processes common image formats (.jpg, .jpeg, .png, .webp, .bmp, .tif, .tiff).
- Skips non-image files; creates output directories as needed.
- Preserves original format/extension (PNGs stay PNG, JPEGs stay JPEG).
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from PIL import Image

# Pillow configuration: avoid DecompressionBomb warnings for reasonable sizes
Image.MAX_IMAGE_PIXELS = None


def resize_image(img: Image.Image, max_size: int) -> Image.Image:
    """
    Resize while preserving aspect ratio so the longest side equals max_size.
    Pads images to ensure both dimensions are at least 28px (Qwen2.5-VL requirement).
    """
    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        scale = 1.0
    else:
        scale = max_size / float(longest)
    
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    
    # Qwen2.5-VL requires both dimensions >= 28 pixels
    MIN_DIM = 28
    w_final, h_final = resized.size
    
    # Pad width if needed
    if w_final < MIN_DIM:
        target_w = MIN_DIM
        pad_left = (target_w - w_final) // 2
        pad_right = target_w - w_final - pad_left
        padded = Image.new(resized.mode, (target_w, h_final), color=(255, 255, 255) if resized.mode == "RGB" else (255, 255, 255, 255))
        padded.paste(resized, (pad_left, 0))
        resized = padded
        w_final = target_w
    
    # Pad height if needed
    if h_final < MIN_DIM:
        target_h = MIN_DIM
        pad_top = (target_h - h_final) // 2
        pad_bottom = target_h - h_final - pad_top
        padded = Image.new(resized.mode, (w_final, target_h), color=(255, 255, 255) if resized.mode == "RGB" else (255, 255, 255, 255))
        padded.paste(resized, (0, pad_top))
        resized = padded
    
    return resized


def process_image(src: Path, dst: Path, max_size: int) -> Tuple[bool, str]:
    """Resize one image; returns (success, message). Keeps original format."""
    fmt_map = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }
    suffix = src.suffix.lower()
    fmt = fmt_map.get(suffix, "PNG")

    try:
        with Image.open(src) as img:
            if fmt == "JPEG":
                img = img.convert("RGB")
            else:
                img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")

            resized = resize_image(img, max_size)
            dst.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = {}
            if fmt == "JPEG":
                save_kwargs["quality"] = 95

            resized.save(dst, format=fmt, **save_kwargs)
        return True, f"ok: {src.name} -> {dst.name}"
    except Exception as exc:  # pragma: no cover
        return False, f"fail: {src} ({exc})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Resize images to a maximum longest side and copy to output directory.")
    parser.add_argument("--input-dir", required=True, help="Source directory containing images")
    parser.add_argument("--output-dir", required=True, help="Destination directory for resized images")
    parser.add_argument("--max-size", type=int, default=512, help="Maximum pixels for the longest side (default: 512)")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be processed without writing output")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    src_dir = Path(args.input_dir)
    dst_dir = Path(args.output_dir)
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Source directory not found or not a directory: {src_dir}")
        return 1

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print(f"No image files found in {src_dir}")
        return 0

    print(f"Found {len(files)} images. Max size: {args.max_size}. Output: {dst_dir}")
    if args.dry_run:
        for f in files:
            rel = f.relative_to(src_dir)
            print(f"would process: {rel}")
        return 0

    # Parallel processing with progress bar
    def process_one(f: Path) -> Tuple[bool, str]:
        rel = f.relative_to(src_dir)
        dst = dst_dir / rel
        return process_image(f, dst, args.max_size)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_one, files),
            total=len(files),
            desc="Resizing",
            unit="img"
        ))

    success = sum(1 for ok, _ in results if ok)
    fail = sum(1 for ok, _ in results if not ok)
    
    # Print failures
    for ok, msg in results:
        if not ok:
            print(msg)

    print(f"Done. Success: {success}, Fail: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
