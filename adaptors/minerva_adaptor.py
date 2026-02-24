import json
import re
from typing import Dict, Any, List
from .base_adaptor import BaseAdaptor

try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


class MinervaAdaptor(BaseAdaptor):
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
            return "You are an expert mathematician with strong problem-solving skills. Analyze mathematical problems efficiently with clear, concise reasoning. Focus on key insights and avoid unnecessary elaboration. Provide step-by-step reasoning but keep it focused and to the point."
        else:
            return "You are an expert mathematician. Solve the problem and put your final answer within \boxed{}."

    def format_prompt(self, item: Dict[str, Any]) -> str:
        problem = item.get('problem', '')
        
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Problem: {problem}\n\n"
        
        if self.thinking_mode:
            prompt += "Provide a clear, step-by-step solution. Focus on key mathematical insights and avoid unnecessary elaboration.\n"
            prompt += "After your reasoning, put your final answer within \\boxed{} tags. For example, If the answer is 42, write \\boxed{42}.\n\n"
            prompt += "Solution:"
        else:
            prompt += "Solve the problem and put your final answer within \\boxed{} tags.\n"
            prompt += "Answer:"
        
        return prompt

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        # Minerva数据集的答案在answer字段中
        answer = item.get('answer', '')
        return answer

    def extract_answer(self, model_output: str) -> str:
        output = model_output.strip()
        
        boxed_start = r'\boxed{'
        start_idx = output.rfind(boxed_start)
        
        if start_idx == -1:
            return ""
        
        content_start = start_idx + len(boxed_start)
        brace_count = 0
        content_end = -1
        
        for i in range(content_start, len(output)):
            char = output[i]
            if char == '{':
                brace_count += 1
            elif char == '}':
                if brace_count == 0:
                    content_end = i
                    break
                else:
                    brace_count -= 1
        
        if content_end == -1:
            return ""
        
        return output[content_start:content_end].strip()

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        if MATH_VERIFY_AVAILABLE:
            return self._verify_answer_with_math_verify(ground_truth, model_answer)
        else:
            print(f"!!!Error: math-verify library not available!!!")
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
        except Exception:
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
        except Exception:
            pass
        
        try:
            parsed_gt = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
            parsed_ma = parse(model_answer, extraction_config=[ExprExtractionConfig()])
            if parsed_gt and parsed_ma:
                result = verify(parsed_gt, parsed_ma)
                if result:
                    return True
        except Exception:
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
        except Exception:
            pass
        
        try:
            numbers_gt = re.findall(r'[-+]?\d*\.?\d+', ground_truth)
            numbers_ma = re.findall(r'[-+]?\d*\.?\d+', model_answer)
            if numbers_gt and numbers_ma:
                if numbers_gt == numbers_ma:
                    return True
        except Exception:
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
        except Exception:
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

    def _wrap_latex(self, text: str) -> bool:
        return f"\\({text}\\)"
