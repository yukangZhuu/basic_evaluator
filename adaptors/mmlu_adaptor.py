import json
from typing import Dict, Any, List
from .base_adaptor import BaseAdaptor


class MMLUAdaptor(BaseAdaptor):
    def _load_data(self) -> List[Dict[str, Any]]:
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _get_system_prompt(self) -> str:
        if self.thinking_mode:
            return "You are a knowledgeable assistant. Please think step by step and select the correct answer from the given options."
        else:
            return "You are a knowledgeable assistant. Select the correct answer from the given options."

    def format_prompt(self, item: Dict[str, Any]) -> str:
        question = item.get('question', '')
        choices = item.get('choices', [])
        
        choices_text = ""
        for i, choice in enumerate(choices):
            choices_text += f"{chr(65 + i)}. {choice}\n"
        
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += f"Options:\n{choices_text}"
        prompt += "Answer:"
        
        return prompt

    def extract_answer(self, model_output: str) -> str:
        output = model_output.strip()
        
        import re
        answer_pattern = r'Answer:\s*([A-D])'
        match = re.search(answer_pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        for char in output:
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()
        
        return ""

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        return model_answer.upper() == ground_truth.upper()
