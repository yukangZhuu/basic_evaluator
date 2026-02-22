import json
from typing import Dict, Any, List
from .base_adaptor import BaseAdaptor


class Math500Adaptor(BaseAdaptor):
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
            return "You are an expert mathematician. Please think step by step to solve the problem and provide the final answer in a clear format."
        else:
            return "You are an expert mathematician. Solve the following problem and provide the final answer."

    def format_prompt(self, item: Dict[str, Any]) -> str:
        question = item.get('question', item.get('problem', ''))
        return f"{self.system_prompt}\n\nProblem: {question}\n\nSolution:"

    def extract_answer(self, model_output: str) -> str:
        output = model_output.strip()
        
        import re
        answer_patterns = [
            r'Answer:\s*([^\n]+)',
            r'answer\s*[:=]\s*([^\n]+)',
            r'Final\s+answer\s*[:=]\s*([^\n]+)',
            r'####\s*([^\n]+)',
            r'\$\$\s*([^\$]+)\s*\$\$',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                answer = answer.replace('$', '').strip()
                return answer
        
        lines = output.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.lower().startswith(('step', 'solution', 'because', 'therefore', 'thus')):
                return line
        
        return ""

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        model_answer = model_answer.strip()
        ground_truth = ground_truth.strip()
        
        try:
            model_num = float(model_answer.replace(',', '').replace('$', '').strip())
            truth_num = float(ground_truth.replace(',', '').replace('$', '').strip())
            return abs(model_num - truth_num) < 1e-6
        except (ValueError, AttributeError):
            pass
        
        model_lower = model_answer.lower()
        truth_lower = ground_truth.lower()
        
        if model_lower == truth_lower:
            return True
        
        import re
        model_clean = re.sub(r'[^\w\s-]', '', model_lower)
        truth_clean = re.sub(r'[^\w\s-]', '', truth_lower)
        
        return model_clean == truth_clean
