import asyncio
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
import time


class ParallelInference:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 4096):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.llm = None
        self._initialize_model()

    def _initialize_model(self):
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True
        )

    async def generate_batch_async(self, prompts: List[str],
                                   max_tokens: int = 2048,
                                   temperature: float = 0.0,
                                   top_p: float = 1.0,
                                   stop: Optional[List[str]] = None) -> List[str]:
        loop = asyncio.get_event_loop()
        
        def sync_generate():
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop
            )
            outputs = self.llm.generate(prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]
        
        results = await loop.run_in_executor(None, sync_generate)
        return results

    def generate_batch_parallel(self, prompts: List[str],
                                batch_size: int = 32,
                                max_tokens: int = 2048,
                                temperature: float = 0.0,
                                top_p: float = 1.0,
                                stop: Optional[List[str]] = None) -> List[str]:
        all_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            results = self._generate_batch_sync(batch, max_tokens, temperature, top_p, stop)
            all_results.extend(results)
        
        return all_results

    def _generate_batch_sync(self, prompts: List[str],
                            max_tokens: int = 2048,
                            temperature: float = 0.0,
                            top_p: float = 1.0,
                            stop: Optional[List[str]] = None) -> List[str]:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def generate_batch_parallel_with_metrics(self, prompts: List[str],
                                            batch_size: int = 32,
                                            max_tokens: int = 2048,
                                            temperature: float = 0.0,
                                            top_p: float = 1.0,
                                            stop: Optional[List[str]] = None) -> Dict[str, Any]:
        all_results = []
        all_metrics = []
        total_start_time = time.time()
        total_tokens = 0
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_start_time = time.time()
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop
            )
            
            outputs = self.llm.generate(batch, sampling_params)
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            batch_results = []
            batch_tokens = 0
            
            for output in outputs:
                generated_text = output.outputs[0].text
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(output.outputs[0].token_ids)
                batch_tokens += completion_tokens
                total_tokens += completion_tokens
                
                batch_results.append({
                    'generated_text': generated_text,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                })
            
            all_results.extend(batch_results)
            all_metrics.append({
                'batch_time': batch_time,
                'batch_size': len(batch),
                'batch_tokens': batch_tokens
            })
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        metrics = {
            'total_time': total_time,
            'average_time_per_request': total_time / len(prompts),
            'total_tokens': total_tokens,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'requests_per_second': len(prompts) / total_time if total_time > 0 else 0,
            'batch_metrics': all_metrics
        }
        
        return {
            'results': all_results,
            'metrics': metrics
        }

    def cleanup(self):
        if self.llm is not None:
            del self.llm
