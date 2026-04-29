#!/usr/bin/env python3
"""
Split a JSONL dataset into train/eval/test subsets.

Usage examples:
    # Fractions
    python tools/split_jsonl.py \
        --input llm_training_data_multimodal.jsonl \
        --train-out input/train.jsonl \
        --eval-out input/eval.jsonl \
        --test-out input/test.jsonl \
        --eval-size 0.1 --test-size 0.1 \
        --seed 42

    # Absolute counts (e.g., 2700 total => eval 270, test 270)
    python tools/split_jsonl.py \
        --input llm_training_data_multimodal.jsonl \
        --train-out input/train.jsonl \
        --eval-out input/eval.jsonl \
        --test-out input/test.jsonl \
        --eval-size 270 --test-size 270 \
        --seed 42

Notes:
- `--eval-size` and `--test-size` accept a float fraction (0<frac<1) or an integer count (>=1).
- Invalid JSON lines are skipped with a warning.
- Shuffling is enabled by default for a fair split.
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON on line {i}: {e}")
    return items


def _resolve_size(size_val: str, total: int, label: str) -> int:
    """Convert size string to count. Accepts float fraction or int count."""
    try:
        if "." in str(size_val):
            size = float(size_val)
        else:
            size = float(int(size_val))
    except Exception as exc:  # pragma: no cover - input validation
        raise ValueError(f"Invalid --{label}-size value: {size_val}") from exc

    if size <= 0:
        raise ValueError(f"--{label}-size must be > 0 (fraction or count)")

    if 0 < size < 1:
        count = int(round(total * size))
    else:
        count = int(size)

    return count


def split_items(
    items: List[Dict[str, Any]],
    eval_size: str,
    test_size: str,
    seed: int,
    shuffle: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(items)
    if n == 0:
        return [], [], []

    eval_count = _resolve_size(eval_size, n, "eval")
    test_count = _resolve_size(test_size, n, "test")

    if eval_count + test_count >= n:
        raise ValueError("Eval + test size must leave at least 1 example for train")

    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(items)

    eval_items = items[:eval_count]
    test_items = items[eval_count:eval_count + test_count]
    train_items = items[eval_count + test_count:]
    return train_items, eval_items, test_items


def write_jsonl(path: Path, items: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Split JSONL into train/eval/test")
    p.add_argument("--input", required=True, help="Path to input JSONL")
    p.add_argument("--train-out", required=True, help="Output path for train JSONL")
    p.add_argument("--eval-out", required=True, help="Output path for eval JSONL")
    p.add_argument("--test-out", required=True, help="Output path for test JSONL")
    p.add_argument("--eval-size", default="0.1", help="Eval size as float fraction or integer count (default 0.1)")
    p.add_argument("--test-size", default="0.1", help="Test size as float fraction or integer count (default 0.1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    p.add_argument("--no-shuffle", action="store_true", help="Disable shuffling before split")
    args = p.parse_args()

    in_path = Path(args.input)
    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    test_out = Path(args.test_out)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"Reading: {in_path}")
    items = read_jsonl(in_path)
    n = len(items)
    print(f"Loaded {n} examples")
    if n < 3:
        raise ValueError("Need at least 3 examples to split into train/eval/test")

    train_items, eval_items, test_items = split_items(
        items,
        eval_size=args.eval_size,
        test_size=args.test_size,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    print(f"Writing train: {train_out} ({len(train_items)})")
    write_jsonl(train_out, train_items)

    print(f"Writing eval:  {eval_out} ({len(eval_items)})")
    write_jsonl(eval_out, eval_items)

    print(f"Writing test:  {test_out} ({len(test_items)})")
    write_jsonl(test_out, test_items)

    print("Done.")


if __name__ == "__main__":
    main()
