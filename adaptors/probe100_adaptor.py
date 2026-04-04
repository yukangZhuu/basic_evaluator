import json
import re
from typing import Dict, Any, List, Optional
from .base_adaptor import BaseAdaptor

try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


class Probe100Adaptor(BaseAdaptor):
    """
    Adaptor for probe100 guided evaluation.

    Each question is expanded into multiple variants based on g_levels and
    guidance_modes.  Prompts are returned as raw, fully-formatted strings
    (including <|im_start|> tokens) so the inference engine must NOT apply
    a chat template on top.
    """

    raw_prompts = True  # signal to inference engine

    DEFAULT_G_LEVELS = [0.25, 0.5, 0.75, 1.0]
    DEFAULT_GUIDANCE_MODES = ['prefix', 'hint']

    def __init__(self, data_path: str, thinking_mode: bool = False,
                 g_levels: Optional[List[float]] = None,
                 guidance_modes: Optional[List[str]] = None,
                 max_raw_samples: Optional[int] = None):
        self.g_levels = g_levels if g_levels is not None else self.DEFAULT_G_LEVELS
        self.guidance_modes = guidance_modes if guidance_modes is not None else self.DEFAULT_GUIDANCE_MODES
        self.max_raw_samples = max_raw_samples
        super().__init__(data_path, thinking_mode)

    # ── Data loading & expansion ─────────────────────────────────────────

    def _load_data(self) -> List[Dict[str, Any]]:
        raw_data: List[Dict[str, Any]] = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))

        if self.max_raw_samples is not None and self.max_raw_samples > 0:
            raw_data = raw_data[:self.max_raw_samples]

        expanded: List[Dict[str, Any]] = []
        for item in raw_data:
            steps = item.get('steps', [])
            for g in self.g_levels:
                n_steps = round(len(steps) * g)
                selected = steps[:n_steps]
                for mode in self.guidance_modes:
                    expanded.append({
                        **item,
                        'g_level': g,
                        'guidance_mode': mode,
                        'guidance_steps_used': n_steps,
                        'selected_steps': selected,
                    })
        return expanded

    def _get_system_prompt(self) -> str:
        return ("You are an expert mathematician with strong problem-solving skills. "
                "Think step by step.")

    # ── Prompt construction ──────────────────────────────────────────────

    def format_prompt(self, item: Dict[str, Any]) -> str:
        mode = item['guidance_mode']
        if mode == 'prefix':
            return self._format_prefix(item)
        return self._format_hint(item)

    def _format_prefix(self, item: Dict[str, Any]) -> str:
        """Teacher-prefix mode: steps are injected as the start of <think>."""
        question = item.get('question', '')
        selected_steps = item.get('selected_steps', [])
        steps_text = '\n'.join(selected_steps)

        prompt = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{question}\n"
            f"Please reason step by step to solve this problem.\n"
            f"Use this format:\n"
            f"<think>\n"
            f"[Your reasoning process here, showing how YOU would reach the solution]\n"
            f"</think>\n"
            f"\\boxed{{answer}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n"
            f"{steps_text}\n"
        )
        return prompt

    def _format_hint(self, item: Dict[str, Any]) -> str:
        """Teacher-hint mode: steps are provided as reference in user message."""
        question = item.get('question', '')
        selected_steps = item.get('selected_steps', [])
        steps_text = '\n'.join(selected_steps)

        prompt = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{question}\n\n"
            f"Below are some initial reasoning steps that may help you:\n"
            f"{steps_text}\n\n"
            f"Please solve the problem step by step. You should use the provided "
            f"steps as a reference, but do NOT just copy them. Instead, reconstruct "
            f"the complete reasoning process in your own words, starting from the "
            f"beginning, and continue the reasoning to find the final answer.\n"
            f"Use this format:\n"
            f"<think>\n"
            f"[Your reasoning process here, showing how YOU would reach the solution]\n"
            f"</think>\n"
            f"\\boxed{{answer}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n"
        )
        return prompt

    # ── Metadata ─────────────────────────────────────────────────────────

    def get_variant_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'g_level': item.get('g_level'),
            'guidance_mode': item.get('guidance_mode'),
            'guidance_steps_used': item.get('guidance_steps_used'),
        }

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        return item.get('ground_truth', '')

    def get_question(self, item: Dict[str, Any]) -> str:
        return item.get('question', '')

    # ── Answer extraction & verification (same as TeacherTracesAdaptor) ──

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
            raise RuntimeError("math-verify library is required but not installed. "
                               "Run: pip install math-verify")

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
