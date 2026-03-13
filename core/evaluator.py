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
                stop: List[str] = None, max_samples: int = None,
                n_samples: int = 1) -> Dict[str, Any]:
        data = adaptor.load_benchmark_data()
        
        # Apply max_samples limit if specified
        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]
            
        prompts = adaptor.format_prompts_batch(data)
        
        print(f"Starting evaluation on {len(data)} samples with Pass@{n_samples}...")
        
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
                system_prompt=system_prompt,
                n=n_samples
            )
        else:
            result = self.inference_engine.generate_batch_with_metrics(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                system_prompt=system_prompt,
                n=n_samples
            )
        
        if n_samples > 1:
            evaluation_results = []
            for r, item in zip(result['results'], data):
                generated_texts = r['generated_text']
                # generated_texts should be a list when n > 1
                if not isinstance(generated_texts, list):
                    generated_texts = [generated_texts]
                
                item_results = []
                for txt in generated_texts:
                    model_answer = adaptor.extract_answer(txt)
                    ground_truth = adaptor.get_ground_truth(item)
                    is_correct = adaptor.verify_answer(model_answer, ground_truth)
                    item_results.append({
                        'model_output': txt,
                        'model_answer': model_answer,
                        'ground_truth': ground_truth,
                        'is_correct': is_correct
                    })
                
                # Best of N: Correct if any is correct
                is_correct_any = any(res['is_correct'] for res in item_results)
                
                evaluation_results.append({
                    'question': adaptor.get_question(item),
                    'model_output': generated_texts,
                    'model_answer': [res['model_answer'] for res in item_results],
                    'ground_truth': adaptor.get_ground_truth(item),
                    'is_correct': is_correct_any,
                    'pass_n_results': item_results
                })
        else:
            model_outputs = [r['generated_text'] for r in result['results']]
            evaluation_results = adaptor.evaluate_batch(model_outputs, data)
        
        inference_metrics = result['metrics']
        
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
            model_answer = result.get('model_answer', '')
            if isinstance(model_answer, list):
                # For Pass@N, if it's incorrect, it means ALL were incorrect.
                # We just take the first one for simple type analysis, or maybe we should analyze all?
                # For simplicity, we analyze the first answer.
                model_answer = model_answer[0] if model_answer else ""
            
            model_answer = str(model_answer).strip()
            ground_truth = str(result.get('ground_truth', '')).strip()
            
            if not model_answer:
                error_type = 'no_answer'
            elif model_answer == ground_truth:
                # This case shouldn't happen if verify_answer is correct, but possible with format mismatch
                error_type = 'format_mismatch'
            else:
                error_type = 'incorrect_answer'
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        sample_errors = []
        for r in incorrect_results[:10]:
            question = r['question']
            if len(question) > 200:
                question = question[:200] + '...'
                
            model_answer = r.get('model_answer', '')
            if isinstance(model_answer, list):
                model_answer = str(model_answer)
                
            model_output = r.get('model_output', '')
            if isinstance(model_output, list):
                # Show first output + count
                model_output = f"[First of {len(model_output)} outputs]\n" + str(model_output[0])
            
            if len(model_output) > 500:
                model_output = model_output[:500] + '...'
                
            sample_errors.append({
                'question': question,
                'model_answer': model_answer,
                'ground_truth': r.get('ground_truth', ''),
                'model_output': model_output
            })
        
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
