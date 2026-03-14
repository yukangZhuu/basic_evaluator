import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from .model_inference import ModelInference
from .parallel_inference import ParallelInference


class Evaluator:
    """Single-process evaluator with incremental batch saving and resume."""

    def __init__(self, model_path: str, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 4096,
                 use_parallel: bool = True, batch_size: int = 32, enable_thinking: bool = False):
        self.model_path = model_path
        self.use_parallel = use_parallel
        self.batch_size = batch_size
        self.enable_thinking = enable_thinking

        if use_parallel:
            self.inference_engine = ParallelInference(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enable_thinking=enable_thinking
            )
        else:
            self.inference_engine = ModelInference(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enable_thinking=enable_thinking
            )

    def evaluate(self, adaptor, max_tokens: int = 2048,
                temperature: float = 0.0, top_p: float = 1.0,
                stop: List[str] = None, max_samples: int = None,
                n_samples: int = 1, output_dir: Optional[str] = None) -> Dict[str, Any]:
        data = adaptor.load_benchmark_data()
        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]

        prompts = adaptor.format_prompts_batch(data)
        system_prompt = getattr(adaptor, 'system_prompt', None)
        total_samples = len(data)

        results_path = None
        pass_rates_path = None
        start_idx = 0

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_path = os.path.join(output_dir, "results.jsonl")
            pass_rates_path = os.path.join(output_dir, f"per_question_pass_PASS{n_samples}.jsonl")
            start_idx = self._count_valid_lines(results_path)
            if start_idx >= total_samples:
                print(f"All {total_samples} samples already processed.")
                all_eval = self._load_jsonl(results_path)[:total_samples]
                all_pass = self._load_jsonl(pass_rates_path)[:total_samples]
                report = self._generate_report(all_eval, self._empty_metrics())
                return {'results': all_eval, 'metrics': {}, 'report': report,
                        'per_question_pass_rates': all_pass, 'pass_k': n_samples}
            if start_idx > 0:
                print(f"  Resuming: {start_idx}/{total_samples} done, "
                      f"continuing from sample {start_idx + 1}...")

        remaining = total_samples - start_idx
        total_batches = (remaining + self.batch_size - 1) // self.batch_size
        print(f"Starting evaluation on {total_samples} samples with Pass@{n_samples} "
              f"({remaining} remaining, {total_batches} batches of {self.batch_size})...")

        all_evaluation_results = []
        all_pass_rates = []
        total_tokens = 0
        total_inference_time = 0.0

        for batch_idx, batch_start in enumerate(range(start_idx, total_samples, self.batch_size)):
            batch_end = min(batch_start + self.batch_size, total_samples)
            batch_prompts = prompts[batch_start:batch_end]
            batch_data = data[batch_start:batch_end]

            print(f"\n  Batch {batch_idx + 1}/{total_batches} "
                  f"(samples {batch_start + 1}-{batch_end} / {total_samples})...")

            result = self.inference_engine.generate_single_batch_with_metrics(
                prompts=batch_prompts, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, stop=stop,
                system_prompt=system_prompt, n=n_samples
            )

            bm = result['metrics']
            total_tokens += bm['total_tokens']
            total_inference_time += bm['batch_time']

            batch_eval = self._evaluate_batch_results(result['results'], batch_data, adaptor, n_samples)
            batch_pass = self._compute_batch_pass_rates(batch_eval, batch_data, n_samples)

            all_evaluation_results.extend(batch_eval)
            all_pass_rates.extend(batch_pass)

            if output_dir:
                self._append_jsonl(batch_eval, results_path)
                self._append_jsonl(batch_pass, pass_rates_path)

            processed_so_far = batch_end - start_idx
            remaining_now = total_samples - batch_end
            speed = processed_so_far / total_inference_time if total_inference_time > 0 else 0
            eta_min = (remaining_now / speed / 60) if speed > 0 else 0
            print(f"    Time: {bm['batch_time']:.1f}s | Tokens/s: {bm['tokens_per_second']:.0f} | "
                  f"Progress: {batch_end}/{total_samples} | ETA: {eta_min:.1f} min")

        if start_idx > 0 and output_dir:
            all_evaluation_results = self._load_jsonl(results_path)[:total_samples]
            all_pass_rates = self._load_jsonl(pass_rates_path)[:total_samples]

        inference_metrics = {
            'total_time': total_inference_time,
            'average_time_per_request': total_inference_time / total_samples if total_samples > 0 else 0,
            'total_tokens': total_tokens,
            'tokens_per_second': total_tokens / total_inference_time if total_inference_time > 0 else 0,
            'requests_per_second': total_samples / total_inference_time if total_inference_time > 0 else 0,
        }
        report = self._generate_report(all_evaluation_results, inference_metrics)

        return {
            'results': all_evaluation_results, 'metrics': inference_metrics,
            'report': report, 'per_question_pass_rates': all_pass_rates, 'pass_k': n_samples,
        }

    # ── Batch helpers ────────────────────────────────────────────────────────

    def _evaluate_batch_results(self, inference_results, batch_data, adaptor, n_samples):
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
                    item_results.append({'model_output': txt, 'model_answer': ma,
                                         'ground_truth': gt, 'is_correct': ok})
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
                results.append({'question': adaptor.get_question(item), 'model_output': txt,
                                'model_answer': ma, 'ground_truth': gt, 'is_correct': ok})
        return results

    def _compute_batch_pass_rates(self, eval_results, batch_data, n_samples):
        rates = []
        for r, item in zip(eval_results, batch_data):
            idx = item.get('index')
            if n_samples > 1:
                pc = sum(1 for x in r['pass_n_results'] if x['is_correct'])
            else:
                pc = 1 if r.get('is_correct', False) else 0
            rates.append({
                'index': idx, 'question': r['question'],
                'pass_count': pc, 'pass_rate': round(pc / n_samples, 6) if n_samples > 0 else 0.0,
            })
        return rates

    # ── File I/O ─────────────────────────────────────────────────────────────

    @staticmethod
    def _append_jsonl(items, path):
        with open(path, 'a', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _count_valid_lines(path):
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

    @staticmethod
    def _load_jsonl(path):
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

    @staticmethod
    def _empty_metrics():
        return {'total_time': 0, 'average_time_per_request': 0,
                'total_tokens': 0, 'tokens_per_second': 0, 'requests_per_second': 0}

    # ── Report ───────────────────────────────────────────────────────────────

    def _generate_report(self, results, metrics):
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r['is_correct'])
        accuracy = correct_samples / total_samples if total_samples > 0 else 0
        error_analysis = self._analyze_errors([r for r in results if not r['is_correct']])
        return {
            'timestamp': datetime.now().isoformat(),
            'total_samples': total_samples,
            'correct_samples': correct_samples,
            'incorrect_samples': total_samples - correct_samples,
            'accuracy': accuracy, 'accuracy_percentage': accuracy * 100,
            'inference_metrics': {
                'total_time': metrics.get('total_time', 0),
                'average_time_per_request': metrics.get('average_time_per_request', 0),
                'total_tokens': metrics.get('total_tokens', 0),
                'tokens_per_second': metrics.get('tokens_per_second', 0),
                'requests_per_second': metrics.get('requests_per_second', 0),
            },
            'error_analysis': error_analysis
        }

    def _analyze_errors(self, incorrect_results):
        if not incorrect_results:
            return {'total_errors': 0, 'error_types': {}, 'sample_errors': []}
        error_types = {}
        for result in incorrect_results:
            ma = result.get('model_answer', '')
            if isinstance(ma, list):
                ma = ma[0] if ma else ""
            ma = str(ma).strip()
            gt = str(result.get('ground_truth', '')).strip()
            if not ma:
                et = 'no_answer'
            elif ma == gt:
                et = 'format_mismatch'
            else:
                et = 'incorrect_answer'
            error_types[et] = error_types.get(et, 0) + 1
        sample_errors = []
        for r in incorrect_results[:10]:
            q = r['question']
            if len(q) > 200:
                q = q[:200] + '...'
            mo = r.get('model_output', '')
            if isinstance(mo, list):
                mo = f"[First of {len(mo)} outputs]\n" + str(mo[0])
            if isinstance(mo, str) and len(mo) > 500:
                mo = mo[:500] + '...'
            ma = r.get('model_answer', '')
            if isinstance(ma, list):
                ma = str(ma)
            sample_errors.append({'question': q, 'model_answer': ma,
                                  'ground_truth': r.get('ground_truth', ''), 'model_output': mo})
        return {'total_errors': len(incorrect_results), 'error_types': error_types,
                'sample_errors': sample_errors}

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_report(self, report, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {output_path}")

    def cleanup(self):
        self.inference_engine.cleanup()
