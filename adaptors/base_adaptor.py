from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json


class BaseAdaptor(ABC):
    def __init__(self, data_path: str, thinking_mode: bool = False):
        self.data_path = data_path
        self.thinking_mode = thinking_mode
        self.data = self._load_data()
        self.system_prompt = self._get_system_prompt()

    @abstractmethod
    def _load_data(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def format_prompt(self, item: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def extract_answer(self, model_output: str) -> str:
        pass

    @abstractmethod
    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        pass

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        return item.get('answer', '')

    def load_benchmark_data(self) -> List[Dict[str, Any]]:
        return self.data

    def format_prompts_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        return [self.format_prompt(item) for item in items]

    def evaluate_batch(self, model_outputs: List[str], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for model_output, item in zip(model_outputs, items):
            model_answer = self.extract_answer(model_output)
            ground_truth = self.get_ground_truth(item)
            is_correct = self.verify_answer(model_answer, ground_truth)
            
            results.append({
                'question': item.get('question', ''),
                'model_output': model_output,
                'model_answer': model_answer,
                'ground_truth': ground_truth,
                'is_correct': is_correct
            })
        return results
