#!/usr/bin/env python3
"""
Filter Hendrycks MATH train JSONL to pure-numeric ground_truth only, then draw
a fixed random subset (seed=42): 3k train + 200 test.

Outputs under data/ by default:
  - math_numeric.jsonl       (all filtered rows)
  - math_numeric_3k.jsonl    (3000 train)
  - math_numeric_200.jsonl   (200 test, disjoint from 3k)

Usage:
    python scripts/filter_math_numeric.py
    python scripts/filter_math_numeric.py --input raw/foo.jsonl --out-dir data
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

PURE_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_pure_numeric_ground_truth(row: Dict[str, Any]) -> bool:
    g = row.get("ground_truth")
    if not isinstance(g, str):
        return False
    return bool(PURE_NUM.fullmatch(g))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=root / "raw" / "hendrycks-MATH-benchmark_train.jsonl",
        help="Source JSONL",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data",
        help="Directory for output JSONL files",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--test-size", type=int, default=200)
    args = p.parse_args()

    total_sample = args.train_size + args.test_size
    all_rows = load_jsonl(args.input)
    filtered = [r for r in all_rows if is_pure_numeric_ground_truth(r)]

    rng = random.Random(args.seed)
    pool = filtered.copy()
    rng.shuffle(pool)
    if len(pool) < total_sample:
        raise SystemExit(
            f"Need at least {total_sample} pure-numeric rows, got {len(pool)}"
        )
    chosen = pool[:total_sample]
    train = chosen[: args.train_size]
    test = chosen[args.train_size : args.train_size + args.test_size]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(args.out_dir / "math_numeric.jsonl", filtered)
    write_jsonl(args.out_dir / "math_numeric_3k.jsonl", train)
    write_jsonl(args.out_dir / "math_numeric_200.jsonl", test)

    print(f"Input rows:        {len(all_rows)}")
    print(f"Pure-numeric:      {len(filtered)}  -> {args.out_dir / 'math_numeric.jsonl'}")
    print(
        f"Sample {total_sample} (seed={args.seed}): "
        f"{args.train_size} train, {args.test_size} test"
    )
    print(f"  train -> {args.out_dir / 'math_numeric_3k.jsonl'}")
    print(f"  test  -> {args.out_dir / 'math_numeric_200.jsonl'}")


if __name__ == "__main__":
    main()
