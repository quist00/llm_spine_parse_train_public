#!/usr/bin/env python3
"""
Convert all HEIC/HEIF images in a directory to PNG, preserving folder structure.

Example:
  python tools/convertHeicToPng.py \
    --input-dir img/heic \
    --output-dir img/output/png \
    --workers 4

Notes:
- Uses `pillow-heif` to open HEIC/HEIF files via Pillow.
- Converts to RGB and saves optimized PNGs (reasonable compression).
- Preserves directory structure relative to `--input-dir`.
- Skips files when output is newer unless `--force` is provided.
- Supports parallel conversion with a progress bar.
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import pillow_heif
from tqdm import tqdm


# Register HEIF plugin for Pillow
pillow_heif.register_heif_opener()

# Pillow configuration: allow large images
Image.MAX_IMAGE_PIXELS = None


def convert_one(src: Path, dst: Path) -> Tuple[bool, str]:
    """Convert a single HEIC/HEIF image to PNG.
    Returns (success, message).
    """
    try:
        with Image.open(src) as img:
            # Convert to RGB to ensure 8-bit/channel PNG
            if img.mode != "RGB":
                img = img.convert("RGB")

            dst.parent.mkdir(parents=True, exist_ok=True)
            # Optimize PNG size with reasonable compression
            img.save(dst, format="PNG", optimize=True, compress_level=6)
        return True, f"ok: {src.name} -> {dst.name}"
    except Exception as exc:  # pragma: no cover
        return False, f"fail: {src} ({exc})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert HEIC/HEIF images to PNG, preserving folder structure."
    )
    parser.add_argument("--input-dir", required=True, help="Source directory containing .heic/.heif files")
    parser.add_argument("--output-dir", required=True, help="Destination directory for converted PNGs")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be processed without writing output")
    parser.add_argument("--force", action="store_true", help="Force reconversion even if output is newer")
    args = parser.parse_args()

    src_dir = Path(args.input_dir)
    dst_dir = Path(args.output_dir)
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Source directory not found or not a directory: {src_dir}")
        return 1

    exts = {".heic", ".heif"}
    files = [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print(f"No HEIC/HEIF files found in {src_dir}")
        return 0

    print(f"Found {len(files)} images. Output: {dst_dir}")

    if args.dry_run:
        for f in files:
            rel = f.relative_to(src_dir)
            out = (dst_dir / rel).with_suffix(".png")
            print(f"would convert: {rel} -> {out.relative_to(dst_dir)}")
        return 0

    def process_one(f: Path) -> Tuple[bool, str]:
        rel = f.relative_to(src_dir)
        dst = (dst_dir / rel).with_suffix(".png")

        if not args.force and dst.exists():
            try:
                if f.stat().st_mtime <= dst.stat().st_mtime:
                    return True, f"skip (up-to-date): {rel}"
            except Exception:
                pass

        return convert_one(f, dst)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_one, files),
            total=len(files),
            desc="Converting",
            unit="img"
        ))

    success = sum(1 for ok, _ in results if ok)
    fail = sum(1 for ok, _ in results if not ok)

    # Print failures and skips
    for ok, msg in results:
        if not ok or msg.startswith("skip"):
            print(msg)

    print(f"Done. Success: {success}, Fail: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
import os
from PIL import Image
import pillow_heif

# Register HEIF plugin
pillow_heif.register_heif_opener()


def convert_heic_to_png(input_dir, output_dir):
    """Converts all .heic files in input_dir to .png files in a matching output_dir structure with optimized size."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.heic'):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_dir, output_filename)

            try:
                with Image.open(input_path) as img:
                    # Convert to RGB if needed (8 bits/channel)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Save optimized PNG (like macOS Preview)
                    img.save(output_path, format="PNG", optimize=True, compress_level=6)
                print(f"✅ Converted: {filename} -> {output_filename}")
            except Exception as e:
                print(f"❌ Failed to convert {filename}: {e}")


if __name__ == '__main__':
    # Paths for input and output
    base_dir = "img"
    input_subfolders = ["heic"]

    for subfolder in input_subfolders:
        input_dir = os.path.join(base_dir, subfolder)
        output_dir = os.path.join(base_dir, "output", "png")

        convert_heic_to_png(input_dir, output_dir)

    print(f"\n✅ Done converting all HEIC images for {input_subfolders} and stored in 'img/output'.")