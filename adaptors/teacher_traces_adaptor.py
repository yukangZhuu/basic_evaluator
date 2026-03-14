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
        if self.thinking_mode:
            return "You are an expert mathematician with strong problem-solving skills. Carefully analyze mathematical problems and provide step-by-step reasoning to arrive at the correct answer."
        else:
            return "You are an expert mathematician. Solve the problem and put your final answer within \boxed{}."

    def format_prompt(self, item: Dict[str, Any]) -> str:
        question = item.get('question', '')

        prompt = f"Question: {question}\n\n"
        prompt += ("Use this format:\n"
                   "<think>\n"
                   "[Your original reasoning process here, showing how YOU would reach this solution]\n"
                   "</think>\n\n"
                   "\\boxed{answer}\n")

        return prompt

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        return item.get('ground_truth', '')

    def get_question(self, item: Dict[str, Any]) -> str:
        return item.get('question', '')

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
        """
        Verification cascade (stops at first match):
        1. Exact string match (after whitespace normalization)
        2. Numeric comparison (float, tolerance 1e-6)
        3. math_verify with all extraction configs (LaTeX, Expr, String)
        4. math_verify with boxed-wrapped LaTeX parsing
        """
        if not model_answer or not model_answer.strip() or not ground_truth or not ground_truth.strip():
            return False

        gt_clean = re.sub(r'\s+', '', ground_truth.strip())
        ma_clean = re.sub(r'\s+', '', model_answer.strip())
        if gt_clean == ma_clean:
            return True

        try:
            gt_f = float(ground_truth.strip())
            ma_f = float(model_answer.strip())
            if abs(gt_f - ma_f) < 1e-6:
                return True
        except (ValueError, OverflowError):
            pass

        if not MATH_VERIFY_AVAILABLE:
            raise RuntimeError("math-verify library is required but not installed. Run: pip install math-verify")

        try:
            configs = [LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()]
            parsed_gt = parse(ground_truth, extraction_config=configs)
            parsed_ma = parse(model_answer, extraction_config=configs)
            if parsed_gt and parsed_ma and verify(parsed_gt, parsed_ma):
                return True
        except Exception:
            pass

        try:
            wrapped_gt = f'\\boxed{{{ground_truth.strip()}}}'
            wrapped_ma = f'\\boxed{{{model_answer.strip()}}}'
            parsed_gt = parse(wrapped_gt, extraction_config=[LatexExtractionConfig()])
            parsed_ma = parse(wrapped_ma, extraction_config=[LatexExtractionConfig()])
            if parsed_gt and parsed_ma and verify(parsed_gt, parsed_ma):
                return True
        except Exception:
            pass

        return False
