import json
from typing import Dict, Any, List
from .base_adaptor import BaseAdaptor


class GSM8KAdaptor(BaseAdaptor):
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
            return "You are a helpful mathematical assistant. Please think step by step to solve the problem and provide the final answer in the format '#### <number>'."
        else:
            return "You are a helpful mathematical assistant. Solve the following problem and provide the final answer in the format '#### <number>'."

    def format_prompt(self, item: Dict[str, Any]) -> str:
        question = item.get('question', '')
        return f"{self.system_prompt}\n\nQuestion: {question}\n\nAnswer:"

    def extract_answer(self, model_output: str) -> str:
        output = model_output.strip()
        
        if '####' in output:
            answer_part = output.split('####')[-1].strip()
            answer_part = answer_part.split('\n')[0].strip()
            return answer_part
        
        import re
        number_pattern = r'[-+]?\d*\.?\d+'
        numbers = re.findall(number_pattern, output)
        if numbers:
            return numbers[-1]
        
        return ""

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        try:
            model_num = float(model_answer.replace(',', '').strip())
            truth_num = float(ground_truth.replace(',', '').strip())
            return abs(model_num - truth_num) < 1e-6
        except (ValueError, AttributeError):
            return model_answer.strip() == ground_truth.strip()
