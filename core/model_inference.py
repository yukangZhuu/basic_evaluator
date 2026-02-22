import time
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import torch


class ModelInference:
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

    def generate_batch(self, prompts: List[str], 
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
        results = [output.outputs[0].text for output in outputs]
        return results

    def generate_batch_with_metrics(self, prompts: List[str],
                                    max_tokens: int = 2048,
                                    temperature: float = 0.0,
                                    top_p: float = 1.0,
                                    stop: Optional[List[str]] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results = []
        total_tokens = 0
        
        for output in outputs:
            generated_text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            total_tokens += completion_tokens
            
            results.append({
                'generated_text': generated_text,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            })
        
        metrics = {
            'total_time': total_time,
            'average_time_per_request': total_time / len(prompts),
            'total_tokens': total_tokens,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'requests_per_second': len(prompts) / total_time if total_time > 0 else 0
        }
        
        return {
            'results': results,
            'metrics': metrics
        }

    def generate_single(self, prompt: str,
                       max_tokens: int = 2048,
                       temperature: float = 0.0,
                       top_p: float = 1.0,
                       stop: Optional[List[str]] = None) -> str:
        results = self.generate_batch([prompt], max_tokens, temperature, top_p, stop)
        return results[0] if results else ""

    def cleanup(self):
        if self.llm is not None:
            del self.llm
            torch.cuda.empty_cache()
