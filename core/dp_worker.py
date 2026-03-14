"""
Data-parallel worker: each instance runs in its own subprocess on a single GPU.
Processes a contiguous shard of the dataset with incremental saving + resume.
"""
import json
import os
import time
from typing import List, Dict, Any


def dp_worker_main(rank: int, gpu_id: int, dp_size: int, config: dict):
    """Entry point for one DP worker process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = "1"

    from adaptors.adaptor_factory import AdaptorFactory
    from core.parallel_inference import ParallelInference

    adaptor = AdaptorFactory.create_adaptor(
        config['benchmark_type'], config['data_path'], config['thinking_mode']
    )
    all_data = adaptor.load_benchmark_data()
    max_samples = config.get('max_samples')
    if max_samples and max_samples > 0:
        all_data = all_data[:max_samples]

    total = len(all_data)
    chunk = (total + dp_size - 1) // dp_size
    start = rank * chunk
    end = min(start + chunk, total)
    shard_data = all_data[start:end]

    if not shard_data:
        print(f"[Worker {rank}] No data assigned, exiting.")
        return

    shard_prompts = adaptor.format_prompts_batch(shard_data)
    system_prompt = adaptor.system_prompt

    print(f"[Worker {rank}] GPU {gpu_id}: samples {start + 1}-{end} "
          f"({len(shard_data)} items)")

    engine = ParallelInference(
        model_path=config['model_path'],
        tensor_parallel_size=config.get('tensor_parallel_size', 1),
        gpu_memory_utilization=config.get('gpu_memory_utilization', 0.95),
        max_model_len=config.get('max_model_len', 8192),
        enable_thinking=config.get('enable_thinking', False),
        max_num_seqs=config.get('max_num_seqs', 256),
    )

    n_samples = config['n_samples']
    batch_size = config['batch_size']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, f"_shard{rank}_results.jsonl")
    pass_path = os.path.join(output_dir, f"_shard{rank}_pass_rates.jsonl")

    done = _count_valid_lines(results_path)
    remaining = len(shard_data) - done
    total_batches = (remaining + batch_size - 1) // batch_size

    if done > 0:
        print(f"[Worker {rank}] Resuming: {done}/{len(shard_data)} already done")
    if done >= len(shard_data):
        print(f"[Worker {rank}] Shard already complete.")
        engine.cleanup()
        return

    cumulative_time = 0.0

    for batch_idx, b_start in enumerate(range(done, len(shard_data), batch_size)):
        b_end = min(b_start + batch_size, len(shard_data))
        bp = shard_prompts[b_start:b_end]
        bd = shard_data[b_start:b_end]

        result = engine.generate_single_batch_with_metrics(
            prompts=bp,
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            stop=config.get('stop'),
            system_prompt=system_prompt,
            n=n_samples
        )

        bm = result['metrics']
        cumulative_time += bm['batch_time']

        batch_eval = _evaluate_batch(result['results'], bd, adaptor, n_samples)
        batch_pass = _compute_pass_rates(batch_eval, bd, n_samples)

        _append_jsonl(batch_eval, results_path)
        _append_jsonl(batch_pass, pass_path)

    engine.cleanup()
    print(f"[Worker {rank}] Finished. Total inference: {cumulative_time:.1f}s")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _evaluate_batch(inference_results: List[Dict], batch_data: List[Dict],
                    adaptor, n_samples: int) -> List[Dict[str, Any]]:
    results = []
    for r, item in zip(inference_results, batch_data):
        if n_samples > 1:
            texts = r['generated_text']
            if not isinstance(texts, list):
                texts = [texts]
            item_results = []
            for txt in texts:
                ma = adaptor.extract_answer(txt)
                gt = adaptor.get_ground_truth(item)
                ok = adaptor.verify_answer(ma, gt)
                item_results.append({
                    'model_output': txt, 'model_answer': ma,
                    'ground_truth': gt, 'is_correct': ok
                })
            results.append({
                'question': adaptor.get_question(item),
                'model_output': texts,
                'model_answer': [x['model_answer'] for x in item_results],
                'ground_truth': adaptor.get_ground_truth(item),
                'is_correct': any(x['is_correct'] for x in item_results),
                'pass_n_results': item_results
            })
        else:
            txt = r['generated_text']
            ma = adaptor.extract_answer(txt)
            gt = adaptor.get_ground_truth(item)
            ok = adaptor.verify_answer(ma, gt)
            results.append({
                'question': adaptor.get_question(item),
                'model_output': txt, 'model_answer': ma,
                'ground_truth': gt, 'is_correct': ok,
            })
    return results


def _compute_pass_rates(eval_results: List[Dict], batch_data: List[Dict],
                        n_samples: int) -> List[Dict[str, Any]]:
    rates = []
    for r, item in zip(eval_results, batch_data):
        idx = item.get('index')
        if n_samples > 1:
            pc = sum(1 for x in r['pass_n_results'] if x['is_correct'])
        else:
            pc = 1 if r.get('is_correct', False) else 0
        rates.append({
            'index': idx,
            'question': r['question'],
            'pass_count': pc,
            'pass_rate': round(pc / n_samples, 6) if n_samples > 0 else 0.0,
        })
    return rates


def _append_jsonl(items: List[Dict], path: str):
    with open(path, 'a', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _count_valid_lines(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                break
    return count


def load_jsonl(path: str) -> List[Dict]:
    items = []
    if not path or not os.path.exists(path):
        return items
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return items
