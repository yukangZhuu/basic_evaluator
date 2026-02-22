import json
import re
from typing import Dict, Any, List
from .base_adaptor import BaseAdaptor

try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


class TeacherTracesAdaptor(BaseAdaptor):
    def _load_data(self) -> List[Dict[str, Any]]:
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _get_system_prompt(self) -> str:
        return "You are an expert mathematician with strong problem-solving skills. Carefully analyze mathematical problems and provide step-by-step reasoning to arrive at the correct answer."

    def format_prompt(self, item: Dict[str, Any]) -> str:
        question = item.get('question', '')
        user_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += f"{user_prompt}\n\n"
        prompt += "Answer:"
        
        return prompt

    def extract_answer(self, model_output: str) -> str:
        output = model_output.strip()
        
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        boxed_matches = re.findall(boxed_pattern, output)
        if boxed_matches:
            return boxed_matches[-1].strip()
        
        boxed_pattern_alt = r'boxed\{([^}]*)\}'
        boxed_matches_alt = re.findall(boxed_pattern_alt, output)
        if boxed_matches_alt:
            return boxed_matches_alt[-1].strip()
        
        answer_pattern = r'(?:answer|Answer|ANSWER)\s*[:=]\s*([^\n]+)'
        answer_match = re.search(answer_pattern, output)
        if answer_match:
            return answer_match.group(1).strip()
        
        lines = output.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.lower().startswith(('step', 'solution', 'because', 'therefore', 'thus', 'first', 'next', 'finally')):
                return line
        
        return ""

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        if MATH_VERIFY_AVAILABLE:
            return self._verify_answer_with_math_verify(ground_truth, model_answer)
        else:
            return self._verify_answer_fallback(ground_truth, model_answer)

    def _verify_answer_with_math_verify(self, ground_truth: str, model_answer: str) -> bool:
        if self._is_empty(ground_truth) or self._is_empty(model_answer):
            return False
        
        gt_clean = re.sub(r'\s+', '', ground_truth.strip())
        ma_clean = re.sub(r'\s+', '', model_answer.strip())
        if gt_clean == ma_clean:
            return True
        
        if self._is_number(ground_truth) and self._is_number(model_answer):
            if self._compare_numbers(ground_truth, model_answer):
                return True
        
        try:
            parsed_gt = parse(ground_truth, extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
                StringExtractionConfig()
            ])
            parsed_ma = parse(model_answer, extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
                StringExtractionConfig()
            ])
            if parsed_gt and parsed_ma:
                result = verify(parsed_gt, parsed_ma)
                if result:
                    return True
        except Exception as e:
            pass
        
        try:
            mcq_answers = ['A', 'B', 'C', 'D', 'E']
            config = StringExtractionConfig(strings=tuple(mcq_answers))
            parsed_gt = parse(ground_truth, extraction_config=[config])
            parsed_ma = parse(model_answer, extraction_config=[config])
            if parsed_gt and parsed_ma:
                result = verify(parsed_gt, parsed_ma)
                if result:
                    return True
        except Exception as e:
            pass
        
        try:
            parsed_gt = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
            parsed_ma = parse(model_answer, extraction_config=[ExprExtractionConfig()])
            if parsed_gt and parsed_ma:
                result = verify(parsed_gt, parsed_ma)
                if result:
                    return True
        except Exception as e:
            pass
        
        try:
            wrapped_gt = self._wrap_latex(ground_truth)
            wrapped_ma = self._wrap_latex(model_answer)
            
            parsed_gt = parse(wrapped_gt, extraction_config=[LatexExtractionConfig()])
            parsed_ma = parse(wrapped_ma, extraction_config=[LatexExtractionConfig()])
            
            if parsed_gt and parsed_ma:
                result = verify(parsed_gt, parsed_ma)
                if result:
                    return True
        except Exception as e:
            pass
        
        try:
            numbers_gt = re.findall(r'[-+]?\d*\.?\d+', ground_truth)
            numbers_ma = re.findall(r'[-+]?\d*\.?\d+', model_answer)
            if numbers_gt and numbers_ma:
                if numbers_gt == numbers_ma:
                    return True
        except Exception as e:
            pass
        
        return False

    def _verify_answer_fallback(self, ground_truth: str, model_answer: str) -> bool:
        if self._is_empty(ground_truth) or self._is_empty(model_answer):
            return False
        
        gt_clean = re.sub(r'\s+', '', ground_truth.strip())
        ma_clean = re.sub(r'\s+', '', model_answer.strip())
        if gt_clean == ma_clean:
            return True
        
        if self._is_number(ground_truth) and self._is_number(model_answer):
            if self._compare_numbers(ground_truth, model_answer):
                return True
        
        try:
            numbers_gt = re.findall(r'[-+]?\d*\.?\d+', ground_truth)
            numbers_ma = re.findall(r'[-+]?\d*\.?\d+', model_answer)
            if numbers_gt and numbers_ma:
                if numbers_gt == numbers_ma:
                    return True
        except Exception as e:
            pass
        
        return False

    def _is_empty(self, text: str) -> bool:
        return not text or text.strip() == ""

    def _is_number(self, text: str) -> bool:
        try:
            float(text.strip())
            return True
        except ValueError:
            return False

    def _compare_numbers(self, num1: str, num2: str) -> bool:
        try:
            n1 = float(num1.strip())
            n2 = float(num2.strip())
            return abs(n1 - n2) < 1e-6
        except ValueError:
            return False

    def _wrap_latex(self, text: str) -> str:
        text = text.strip()
        if not text.startswith('$'):
            text = f'${text}$'
        return text
