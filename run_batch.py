#!/usr/bin/env python3
"""
Batch evaluation runner — run multiple (model, benchmark) combinations
in sequence without stopping to edit main.py.

Usage:
    python run_batch.py

Edit the JOBS list below to define your evaluation matrix.
Each job can override any Config field; unspecified fields keep
the defaults from main.py's Config class.
"""
import os
import sys
import time
from datetime import datetime

# ── Define your evaluation jobs here ─────────────────────────────────────────

JOBS = [
    # ── Example: same model, multiple benchmarks ──
    # {
    #     "MODEL_PATH": "../models/Qwen3-1.7B",
    #     "BENCHMARK_DATA_PATH": "./data/aime24_bench_schema.jsonl",
    #     "BENCHMARK_TYPE": "verl_aligned",
    #     "TEMPERATURE": 1,
    #     "REPEAT_N": 32,
    # },
    # {
    #     "MODEL_PATH": "../models/Qwen3-1.7B",
    #     "BENCHMARK_DATA_PATH": "./data/aime25_bench_schema.jsonl",
    #     "BENCHMARK_TYPE": "verl_aligned",
    #     "TEMPERATURE": 1,
    #     "REPEAT_N": 32,
    # },

    # ── Example: same benchmark, multiple models ──
    # {
    #     "MODEL_PATH": "../models/model_step_400",
    #     "BENCHMARK_DATA_PATH": "./data/math500_bench_schema.jsonl",
    #     "BENCHMARK_TYPE": "math500_bench_schema",
    # },
    # {
    #     "MODEL_PATH": "../models/model_step_800",
    #     "BENCHMARK_DATA_PATH": "./data/math500_bench_schema.jsonl",
    #     "BENCHMARK_TYPE": "math500_bench_schema",
    # },

    # ── Your actual jobs ──
    # {
    #     "MODEL_PATH": "../models/ttn2k_unsolvable_hint_dapo_8xpro6000_global_step_1200",
    #     "BENCHMARK_DATA_PATH": "./data/aime24_bench_schema.jsonl",
    #     "BENCHMARK_TYPE": "verl_aligned",
    #     "TEMPERATURE": 1,
    #     "REPEAT_N": 32,
    # },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/ttn_test_200.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "TEMPERATURE": 1,
        "REPEAT_N": 8,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/math500_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "TEMPERATURE": 1,
        "REPEAT_N": 4,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/amc23_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "TEMPERATURE": 1,
        "REPEAT_N": 8,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "TEMPERATURE": 1,
        "REPEAT_N": 8,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime25_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "TEMPERATURE": 1,
        "REPEAT_N": 8,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "TEMPERATURE": 1,
        "REPEAT_N": 8,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/gpqa_diamond_test.jsonl",
        "BENCHMARK_TYPE": "verl_gpqa_diamond",
        "TEMPERATURE": 1,
        "REPEAT_N": 4,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/scibench_train.jsonl",
        "BENCHMARK_TYPE": "verl_scibench",
        "TEMPERATURE": 1,
        "REPEAT_N": 4,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/mmlu_pro_test.jsonl",
        "BENCHMARK_TYPE": "verl_mmlu_pro",
        "TEMPERATURE": 1,
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 1,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 1,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 2,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 4,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 8,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 16,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 32,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 64,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 128,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
    {
        "MODEL_PATH": "../models/ttn2k_unsolvable_adaptive_hint_global_step_1200",
        "BENCHMARK_DATA_PATH": "./data/aime24_25_26_bench_schema.jsonl",
        "BENCHMARK_TYPE": "verl_aligned",
        "PASS_N": 256,
        "TEMPERATURE": 1,   # 或 0，看你要不要采样
        "REPEAT_N": 1,
    },
]

# ── Runner logic ─────────────────────────────────────────────────────────────

def run_batch():
    from main import Config, main as run_main

    # Snapshot original Config values so each job starts from a clean base
    defaults = {k: v for k, v in vars(Config).items()
                if not k.startswith('_') and k.isupper()}

    total_jobs = len(JOBS)
    results = []

    print("╔" + "═" * 68 + "╗")
    print(f"║  Batch Runner: {total_jobs} job(s) queued" + " " * (68 - 34 - len(str(total_jobs))) + "║")
    print("╚" + "═" * 68 + "╝")

    batch_start = time.time()

    for idx, job in enumerate(JOBS, 1):
        # Reset to defaults, then apply job overrides
        for k, v in defaults.items():
            setattr(Config, k, v)
        for k, v in job.items():
            if not hasattr(Config, k):
                print(f"  WARNING: unknown config key {k!r}, skipping")
                continue
            setattr(Config, k, v)

        model_name = os.path.basename(Config.MODEL_PATH.rstrip('/'))
        bench_name = os.path.splitext(os.path.basename(Config.BENCHMARK_DATA_PATH))[0]

        print(f"\n{'━' * 70}")
        print(f"  Job {idx}/{total_jobs}")
        print(f"  Model:     {model_name}")
        print(f"  Benchmark: {bench_name}  (type: {Config.BENCHMARK_TYPE})")
        print(f"  Temp: {Config.TEMPERATURE}  Repeat: {Config.REPEAT_N}  PassN: {Config.PASS_N}")
        print(f"{'━' * 70}")

        job_start = time.time()
        try:
            run_main()
            status = "OK"
        except Exception as e:
            print(f"\n  ERROR in job {idx}: {e}")
            status = f"FAILED: {e}"
        job_elapsed = time.time() - job_start

        results.append({
            "job": idx,
            "model": model_name,
            "benchmark": bench_name,
            "benchmark_type": Config.BENCHMARK_TYPE,
            "output_dir": Config.OUTPUT_DIR,
            "status": status,
            "elapsed_min": round(job_elapsed / 60, 1),
        })

    # ── Final summary ────────────────────────────────────────────────────
    batch_elapsed = time.time() - batch_start

    print(f"\n{'╔' + '═' * 68 + '╗'}")
    print(f"║  Batch Complete: {total_jobs} job(s) in {batch_elapsed / 60:.1f} min" +
          " " * max(0, 68 - 40 - len(str(total_jobs)) - len(f"{batch_elapsed / 60:.1f}")) + "║")
    print(f"{'╚' + '═' * 68 + '╝'}")

    for r in results:
        tag = "✓" if r["status"] == "OK" else "✗"
        print(f"  {tag} Job {r['job']}: {r['model']} × {r['benchmark']}  "
              f"[{r['elapsed_min']} min]  {r['status']}")
        if r["status"] == "OK":
            print(f"    → {r['output_dir']}")

    # Save batch log
    log_path = f"./outputs/batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({"jobs": results, "total_elapsed_min": round(batch_elapsed / 60, 1)},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  Batch log: {log_path}")


if __name__ == "__main__":
    run_batch()
