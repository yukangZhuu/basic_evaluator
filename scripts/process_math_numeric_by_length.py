#!/usr/bin/env python3
"""
Filter data/math_numeric.jsonl by question/solution length, then random split.

Rules (applied in order for reporting):
  1. Drop rows with question length > 1200
  2. Drop rows with solution length > 2000
  3. Drop rows with solution length < 100

Then sample with fixed seed: 3000 train + 200 test (disjoint).

Outputs under data/:
  - math_numeric_processed.jsonl
  - math_numeric_processed_3k.jsonl
  - math_numeric_processed_200.jsonl
  - math_numeric_processed_report.jsonl

Usage:
    python scripts/process_math_numeric_by_length.py
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def qlen(row: Dict[str, Any]) -> int:
    return len(row.get("question") or "")


def slen(row: Dict[str, Any]) -> int:
    return len(row.get("solution") or "")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_report(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "math_numeric.jsonl",
    )
    p.add_argument("--out-dir", type=Path, default=root / "data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--q-max", type=int, default=1200, help="max question chars (inclusive)")
    p.add_argument("--s-max", type=int, default=2000, help="max solution chars (inclusive)")
    p.add_argument("--s-min", type=int, default=100, help="min solution chars (inclusive)")
    args = p.parse_args()

    report_path = args.out_dir / "math_numeric_processed_report.jsonl"
    if report_path.exists():
        report_path.unlink()

    total_need = args.train_size + args.test_size
    rows = load_jsonl(args.input)
    n0 = len(rows)

    append_report(
        report_path,
        {
            "type": "run",
            "utc_iso": datetime.now(timezone.utc).isoformat(),
            "input_path": str(args.input.resolve()),
            "input_row_count": n0,
            "thresholds": {
                "question_max_chars_inclusive": args.q_max,
                "solution_max_chars_inclusive": args.s_max,
                "solution_min_chars_inclusive": args.s_min,
            },
            "sample": {
                "seed": args.seed,
                "train_size": args.train_size,
                "test_size": args.test_size,
            },
        },
    )

    after_q: List[Dict[str, Any]] = []
    for r in rows:
        if qlen(r) <= args.q_max:
            after_q.append(r)
    n1 = len(after_q)
    append_report(
        report_path,
        {
            "type": "filter_step",
            "order": 1,
            "rule": f"question length <= {args.q_max}",
            "removed_count": n0 - n1,
            "remaining_count": n1,
        },
    )

    after_smax: List[Dict[str, Any]] = []
    for r in after_q:
        if slen(r) > args.s_max:
            continue
        after_smax.append(r)
    n2 = len(after_smax)
    append_report(
        report_path,
        {
            "type": "filter_step",
            "order": 2,
            "rule": f"solution length <= {args.s_max}",
            "removed_count": n1 - n2,
            "remaining_count": n2,
        },
    )

    filtered: List[Dict[str, Any]] = []
    for r in after_smax:
        if slen(r) < args.s_min:
            continue
        filtered.append(r)
    n3 = len(filtered)
    append_report(
        report_path,
        {
            "type": "filter_step",
            "order": 3,
            "rule": f"solution length >= {args.s_min}",
            "removed_count": n2 - n3,
            "remaining_count": n3,
        },
    )

    out_all = args.out_dir / "math_numeric_processed.jsonl"
    write_jsonl(out_all, filtered)
    append_report(
        report_path,
        {
            "type": "output",
            "path": str(out_all.resolve()),
            "row_count": n3,
            "role": "full_filtered",
        },
    )

    if n3 < total_need:
        append_report(
            report_path,
            {
                "type": "error",
                "message": f"Need {total_need} rows after filter, got {n3}",
            },
        )
        raise SystemExit(
            f"After filters only {n3} rows remain; need {total_need} for train+test."
        )

    rng = random.Random(args.seed)
    pool = filtered.copy()
    rng.shuffle(pool)
    train = pool[: args.train_size]
    test = pool[args.train_size : args.train_size + args.test_size]

    out_train = args.out_dir / "math_numeric_processed_3k.jsonl"
    out_test = args.out_dir / "math_numeric_processed_200.jsonl"
    write_jsonl(out_train, train)
    write_jsonl(out_test, test)

    train_idx = {r.get("index") for r in train}
    test_idx = {r.get("index") for r in test}
    overlap = train_idx & test_idx
    if overlap:
        append_report(report_path, {"type": "error", "message": f"train/test index overlap: {overlap}"})
        raise SystemExit("train/test overlap")

    append_report(
        report_path,
        {
            "type": "output",
            "path": str(out_train.resolve()),
            "row_count": len(train),
            "role": "train_sample",
        },
    )
    append_report(
        report_path,
        {
            "type": "output",
            "path": str(out_test.resolve()),
            "row_count": len(test),
            "role": "test_sample",
        },
    )

    append_report(
        report_path,
        {
            "type": "summary",
            "input_row_count": n0,
            "filtered_row_count": n3,
            "removed_total": n0 - n3,
            "train_row_count": len(train),
            "test_row_count": len(test),
            "sample_seed": args.seed,
        },
    )

    print(f"Input:     {n0}  -> filtered: {n3}  -> {out_all}")
    print(f"Sample:    {len(train)} train, {len(test)} test (seed={args.seed})")
    print(f"  train -> {out_train}")
    print(f"  test  -> {out_test}")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
