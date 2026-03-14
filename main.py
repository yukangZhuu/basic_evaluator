import json
import os
os.environ['OMP_NUM_THREADS'] = '1'


class Config:
    # -------------------------------------------------------------------------
    # 4× A800 (80GB) 推荐配置：Qwen3 1.7B + teacher_traces_new Pass@32
    #
    # 1.7B 模型只占 ~3.4GB 显存，TP 多卡通信开销大于收益。
    # 正确做法：DP=4 → 4 个进程各占 1 卡，各跑一份完整模型，吞吐 ≈ 4×。
    # -------------------------------------------------------------------------
    MODEL_PATH = "../models/Qwen3-1.7B"
    OUTPUT_DIR = None  # auto-generated

    BENCHMARK_DATA_PATH = "./data/teacher_traces_new.jsonl"
    BENCHMARK_TYPE = "teacher_traces_new"

    THINKING_MODE = True

    TENSOR_PARALLEL_SIZE = 1   # 每个副本占 1 卡
    DATA_PARALLEL_SIZE = 4     # 4 个独立进程，各在 1 张 GPU 上推理
    GPU_MEMORY_UTILIZATION = 0.95
    MAX_MODEL_LEN = 8192

    USE_PARALLEL = True
    # max_num_seqs 限制 vLLM 调度器同时处理的序列数，确保永不超出 KV cache → 零抢占。
    # 单卡 ~70GB KV cache，worst-case 8192 tokens × 48KB/token ≈ 400MB/seq → ~170 max。
    # 设 160 留余量。BATCH_SIZE 可以大一些（vLLM 自动排队，不会抢占）。
    MAX_NUM_SEQS = 160
    BATCH_SIZE = 64

    MAX_TOKENS = 8192
    TEMPERATURE = 0.6
    TOP_P = 0.95
    STOP_TOKENS = None

    MAX_SAMPLE = None  # None = 全量
    PASS_N = 32


def _print_config():
    print("=" * 60)
    print("LLM Evaluator - Starting Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model Path:      {Config.MODEL_PATH}")
    print(f"  Benchmark:       {Config.BENCHMARK_TYPE}  ({Config.BENCHMARK_DATA_PATH})")
    print(f"  Thinking Mode:   {Config.THINKING_MODE}")
    print(f"  Pass N:          {Config.PASS_N}")
    print(f"  TP: {Config.TENSOR_PARALLEL_SIZE}  DP: {Config.DATA_PARALLEL_SIZE}")
    print(f"  Batch Size:      {Config.BATCH_SIZE}")
    print(f"  Max Samples:     {Config.MAX_SAMPLE if Config.MAX_SAMPLE is not None else 'All'}")
    print(f"  Output Dir:      {Config.OUTPUT_DIR}")


def _get_gpu_ids(dp_size: int):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        ids = [int(x.strip()) for x in visible.split(",") if x.strip()]
    else:
        ids = list(range(dp_size))
    if len(ids) < dp_size:
        raise RuntimeError(
            f"DATA_PARALLEL_SIZE={dp_size} but only {len(ids)} GPUs visible "
            f"(CUDA_VISIBLE_DEVICES={visible or 'not set'})")
    return ids[:dp_size]


def _worker_config():
    return {
        'model_path': Config.MODEL_PATH,
        'benchmark_type': Config.BENCHMARK_TYPE,
        'data_path': Config.BENCHMARK_DATA_PATH,
        'thinking_mode': Config.THINKING_MODE,
        'tensor_parallel_size': Config.TENSOR_PARALLEL_SIZE,
        'gpu_memory_utilization': Config.GPU_MEMORY_UTILIZATION,
        'max_model_len': Config.MAX_MODEL_LEN,
        'enable_thinking': Config.THINKING_MODE,
        'batch_size': Config.BATCH_SIZE,
        'max_tokens': Config.MAX_TOKENS,
        'temperature': Config.TEMPERATURE,
        'top_p': Config.TOP_P,
        'stop': Config.STOP_TOKENS,
        'n_samples': Config.PASS_N,
        'max_samples': Config.MAX_SAMPLE,
        'output_dir': Config.OUTPUT_DIR,
        'max_num_seqs': Config.MAX_NUM_SEQS,
    }


# ── Progress monitor ─────────────────────────────────────────────────────────

def _monitor_progress(procs, output_dir, dp_size, config):
    """Poll shard files and show unified progress until all workers finish."""
    import time as _time

    max_samples = config.get('max_samples')

    # Compute total expected samples (same logic as dp_worker shard split)
    from adaptors.adaptor_factory import AdaptorFactory
    adaptor = AdaptorFactory.create_adaptor(
        config['benchmark_type'], config['data_path'], config['thinking_mode'])
    total = len(adaptor.load_benchmark_data())
    if max_samples and max_samples > 0:
        total = min(total, max_samples)
    del adaptor

    start = _time.time()
    last_done = 0

    while True:
        # Count completed items across all shards
        done = 0
        for rank in range(dp_size):
            path = os.path.join(output_dir, f"_shard{rank}_results.jsonl")
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        done += sum(1 for _ in f)
                except OSError:
                    pass

        elapsed = _time.time() - start
        pct = done / total * 100 if total > 0 else 0
        speed = done / elapsed if elapsed > 0 else 0
        remaining = total - done
        eta_min = (remaining / speed / 60) if speed > 0 else 0

        bar_len = 30
        filled = int(bar_len * done / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_len - filled)

        print(f"\r  [{bar}] {done}/{total} ({pct:.1f}%) | "
              f"{speed:.1f} samples/s | ETA: {eta_min:.1f} min   ",
              end='', flush=True)

        last_done = done

        # Check if all workers have exited
        if all(not p.is_alive() for p in procs):
            break

        _time.sleep(5)

    # Final line
    done = 0
    for rank in range(dp_size):
        path = os.path.join(output_dir, f"_shard{rank}_results.jsonl")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    done += sum(1 for _ in f)
            except OSError:
                pass
    elapsed = _time.time() - start
    pct = done / total * 100 if total > 0 else 0
    print(f"\r  [{'█' * bar_len}] {done}/{total} ({pct:.1f}%) | "
          f"Total: {elapsed / 60:.1f} min                        ")

    for p in procs:
        p.join()


# ── Data-parallel path (DP > 1) ─────────────────────────────────────────────

def run_data_parallel():
    from multiprocessing import Process
    from core.dp_worker import dp_worker_main, load_jsonl
    from datetime import datetime

    dp_size = Config.DATA_PARALLEL_SIZE
    gpu_ids = _get_gpu_ids(dp_size)
    config = _worker_config()

    print(f"\n  Launching {dp_size} workers on GPUs {gpu_ids} ...")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    procs = []
    for rank in range(dp_size):
        p = Process(target=dp_worker_main,
                    args=(rank, gpu_ids[rank], dp_size, config))
        p.start()
        procs.append(p)

    # Unified progress monitor: poll shard files every 5s
    _monitor_progress(procs, Config.OUTPUT_DIR, dp_size, config)

    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        print(f"\nERROR: Workers {failed} exited with errors. "
              "Fix the issue and re-run — completed shards will be resumed.")
        return

    # ── Merge shard files ────────────────────────────────────────────────
    print("\nMerging shard results ...")
    all_results = []
    all_pass_rates = []
    for rank in range(dp_size):
        all_results.extend(load_jsonl(
            os.path.join(Config.OUTPUT_DIR, f"_shard{rank}_results.jsonl")))
        all_pass_rates.extend(load_jsonl(
            os.path.join(Config.OUTPUT_DIR, f"_shard{rank}_pass_rates.jsonl")))

    results_path = os.path.join(Config.OUTPUT_DIR, "results.jsonl")
    pass_path = os.path.join(Config.OUTPUT_DIR,
                             f"per_question_pass_PASS{Config.PASS_N}.jsonl")

    with open(results_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(pass_path, 'w', encoding='utf-8') as f:
        for r in all_pass_rates:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # ── Summary report ───────────────────────────────────────────────────
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get('is_correct'))
    accuracy = correct / total * 100 if total > 0 else 0

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': total,
        'correct_samples': correct,
        'incorrect_samples': total - correct,
        'accuracy': correct / total if total > 0 else 0,
        'accuracy_percentage': accuracy,
        'data_parallel_size': dp_size,
        'pass_n': Config.PASS_N,
    }
    ts = report['timestamp'].replace(':', '-').replace('.', '-')
    report_path = os.path.join(Config.OUTPUT_DIR, f"report_{ts}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Total Samples:   {total}")
    print(f"  Correct:         {correct}")
    print(f"  Accuracy:        {accuracy:.2f}%")
    print(f"\n  Results:    {results_path}")
    print(f"  Pass rates: {pass_path}")
    print(f"  Report:     {report_path}")
    print("=" * 60)


# ── Single-process path (DP = 1) ────────────────────────────────────────────

def run_single_process():
    from core.evaluator import Evaluator
    from adaptors.adaptor_factory import AdaptorFactory

    print("\nInitializing adaptor ...")
    adaptor = AdaptorFactory.create_adaptor(
        Config.BENCHMARK_TYPE, Config.BENCHMARK_DATA_PATH, Config.THINKING_MODE)

    print("Initializing evaluator ...")
    evaluator = Evaluator(
        model_path=Config.MODEL_PATH,
        tensor_parallel_size=Config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=Config.GPU_MEMORY_UTILIZATION,
        max_model_len=Config.MAX_MODEL_LEN,
        use_parallel=Config.USE_PARALLEL,
        batch_size=Config.BATCH_SIZE,
        enable_thinking=Config.THINKING_MODE
    )

    print("\nRunning evaluation ...")
    evaluation_result = evaluator.evaluate(
        adaptor=adaptor, max_tokens=Config.MAX_TOKENS,
        temperature=Config.TEMPERATURE, top_p=Config.TOP_P,
        stop=Config.STOP_TOKENS, max_samples=Config.MAX_SAMPLE,
        n_samples=Config.PASS_N, output_dir=Config.OUTPUT_DIR
    )

    report = evaluation_result['report']

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Total Samples:   {report['total_samples']}")
    print(f"  Correct:         {report['correct_samples']}")
    print(f"  Accuracy:        {report['accuracy_percentage']:.2f}%")

    ts = report['timestamp'].replace(':', '-').replace('.', '-')
    report_path = os.path.join(Config.OUTPUT_DIR, f"report_{ts}.json")
    evaluator.save_report(report, report_path)

    print(f"\n  Results:    {os.path.join(Config.OUTPUT_DIR, 'results.jsonl')}")
    print(f"  Pass rates: {os.path.join(Config.OUTPUT_DIR, f'per_question_pass_PASS{Config.PASS_N}.jsonl')}")
    print(f"  Report:     {report_path}")
    print("=" * 60)

    evaluator.cleanup()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    model_name = os.path.basename(Config.MODEL_PATH.rstrip('/'))
    Config.OUTPUT_DIR = f"./outputs/{Config.BENCHMARK_TYPE}/{model_name}_PASS{Config.PASS_N}"

    _print_config()

    if not os.path.exists(Config.MODEL_PATH):
        print(f"\nError: Model path does not exist: {Config.MODEL_PATH}")
        return
    if not os.path.exists(Config.BENCHMARK_DATA_PATH):
        print(f"\nError: Benchmark data path does not exist: {Config.BENCHMARK_DATA_PATH}")
        return

    if Config.DATA_PARALLEL_SIZE > 1:
        run_data_parallel()
    else:
        run_single_process()


if __name__ == "__main__":
    main()
