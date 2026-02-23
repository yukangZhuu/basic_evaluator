import json
import os
from typing import Dict, Any, List
from datetime import datetime
from .model_inference import ModelInference
from .parallel_inference import ParallelInference


class Evaluator:
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
                stop: List[str] = None, max_samples: int = None) -> Dict[str, Any]:
        data = adaptor.load_benchmark_data()
        
        # Apply max_samples limit if specified
        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]
            
        prompts = adaptor.format_prompts_batch(data)
        
        print(f"Starting evaluation on {len(data)} samples...")
        
        # Get system prompt from adaptor if available
        system_prompt = getattr(adaptor, 'system_prompt', None)
        
        if self.use_parallel:
            result = self.inference_engine.generate_batch_parallel_with_metrics(
                prompts=prompts,
                batch_size=self.batch_size,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                system_prompt=system_prompt
            )
        else:
            result = self.inference_engine.generate_batch_with_metrics(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                system_prompt=system_prompt
            )
        
        model_outputs = [r['generated_text'] for r in result['results']]
        inference_metrics = result['metrics']
        
        evaluation_results = adaptor.evaluate_batch(model_outputs, data)
        
        report = self._generate_report(evaluation_results, inference_metrics)
        
        return {
            'results': evaluation_results,
            'metrics': inference_metrics,
            'report': report
        }

    def _generate_report(self, results: List[Dict[str, Any]], 
                        metrics: Dict[str, Any]) -> Dict[str, Any]:
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r['is_correct'])
        accuracy = correct_samples / total_samples if total_samples > 0 else 0
        
        incorrect_results = [r for r in results if not r['is_correct']]
        
        error_analysis = self._analyze_errors(incorrect_results)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': total_samples,
            'correct_samples': correct_samples,
            'incorrect_samples': total_samples - correct_samples,
            'accuracy': accuracy,
            'accuracy_percentage': accuracy * 100,
            'inference_metrics': {
                'total_time': metrics.get('total_time', 0),
                'average_time_per_request': metrics.get('average_time_per_request', 0),
                'total_tokens': metrics.get('total_tokens', 0),
                'tokens_per_second': metrics.get('tokens_per_second', 0),
                'requests_per_second': metrics.get('requests_per_second', 0),
            },
            'error_analysis': error_analysis
        }
        
        return report

    def _analyze_errors(self, incorrect_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not incorrect_results:
            return {
                'total_errors': 0,
                'error_types': {},
                'sample_errors': []
            }
        
        error_types = {}
        
        for result in incorrect_results:
            model_answer = result.get('model_answer', '').strip()
            ground_truth = result.get('ground_truth', '').strip()
            
            if not model_answer:
                error_type = 'no_answer'
            elif model_answer == ground_truth:
                error_type = 'format_mismatch'
            else:
                error_type = 'incorrect_answer'
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        sample_errors = [
            {
                'question': r['question'][:200] + '...' if len(r['question']) > 200 else r['question'],
                'model_answer': r['model_answer'],
                'ground_truth': r['ground_truth'],
                'model_output': r['model_output'][:500] + '...' if len(r['model_output']) > 500 else r['model_output']
            }
            for r in incorrect_results[:10]
        ]
        
        return {
            'total_errors': len(incorrect_results),
            'error_types': error_types,
            'sample_errors': sample_errors
        }

    def save_report(self, report: Dict[str, Any], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved to {output_path}")

    def save_detailed_results(self, results: List[Dict[str, Any]], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Detailed results saved to {output_path}")

    def cleanup(self):
        self.inference_engine.cleanup()
