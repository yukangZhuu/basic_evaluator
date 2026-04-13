#!/usr/bin/env python3
"""Download GPQA-Diamond (test split) from Hugging Face and write JSONL for the evaluator.

Uses HF mirror by default when HF_ENDPOINT is unset:
  export HF_ENDPOINT=https://hf-mirror.com
"""
import json
import os
import sys

# Prefer mirror for users in regions with slow huggingface.co
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset  # noqa: E402


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo_root, "data", "gpqa_diamond_test.jsonl")

    print(f"Loading fingertap/GPQA-Diamond (test) ...")
    ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    print(f"  rows: {len(ds)}, columns: {ds.column_names}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            rec = {
                "index": i,
                "question": row["question"],
                "answer": row["answer"].strip().upper()
                if isinstance(row["answer"], str)
                else str(row["answer"]).strip().upper(),
            }
            if rec["answer"] not in ("A", "B", "C", "D"):
                print(f"WARNING: row {i} unexpected answer {rec['answer']!r}", file=sys.stderr)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} lines to {out_path}")


if __name__ == "__main__":
    main()
