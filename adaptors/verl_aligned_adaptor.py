"""
Adaptor that reproduces the EXACT prompt construction used in verl GRPO training,
so that evaluation results are directly comparable to training metrics.

Key differences from TeacherTracesAdaptor:
  - System prompt matches verl's SYSTEM_PROMPT exactly
  - User content uses verl's _build_standard_user_content format exactly
  - Prompts are marked as raw (raw_prompts=True) so the inference engine
    skips apply_chat_template — we do it ourselves with enable_thinking=False
    and manually append "<think>\n", matching verl's _build_prompt.
"""
import json
import re
from typing import Any, Dict, List

from .base_adaptor import BaseAdaptor

try:
    from math_verify import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        StringExtractionConfig,
        parse,
        verify,
    )
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

# ── Exact copies from verl training code ─────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert mathematician with strong problem-solving skills. "
    "Think step by step."
)

_FORMAT_BLOCK = (
    "Use this format:\n"
    "<think>\n"
    "[Your reasoning process here, showing how YOU would reach the solution]\n"
    "</think>\n"
    "\\boxed{answer}"
)


def _build_standard_user_content(question: str) -> str:
    return (
        f"{question}\n"
        f"Please reason step by step to solve this problem.\n"
        f"{_FORMAT_BLOCK}"
    )


class VerlAlignedAdaptor(BaseAdaptor):
    """
    Produces raw prompts identical to verl GRPO training.
    The inference engine receives fully-formatted token strings
    (with chat template already applied + ``<think>\\n`` appended).
    """

    raw_prompts = True

    def __init__(self, data_path: str, thinking_mode: bool = False):
        self._tokenizer = None
        super().__init__(data_path, thinking_mode)

    def _load_data(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer
        from main import Config
        self._tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_PATH, trust_remote_code=True
        )

    def format_prompt(self, item: Dict[str, Any]) -> str:
        """
        Build the exact same token string verl feeds to the model:
          1. messages = [system, user]
          2. base_text = apply_chat_template(messages, enable_thinking NOT set)
          3. full_text = base_text + "<think>\\n"
        """
        self._ensure_tokenizer()

        question = item.get("question", "")
        user_content = _build_standard_user_content(question)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        base_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return f"{base_text}<think>\n"

    def get_question(self, item: Dict[str, Any]) -> str:
        return item.get("question", "")

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        return item.get("ground_truth", "")

    # ── Answer extraction (same as TeacherTracesAdaptor) ─────────────────

    def extract_answer(self, model_output: str) -> str:
        output = model_output.strip()
        boxed_start = r"\boxed{"
        start_idx = output.rfind(boxed_start)
        if start_idx == -1:
            return ""
        content_start = start_idx + len(boxed_start)
        brace_count = 0
        content_end = -1
        for i in range(content_start, len(output)):
            char = output[i]
            if char == "{":
                brace_count += 1
            elif char == "}":
                if brace_count == 0:
                    content_end = i
                    break
                brace_count -= 1
        if content_end == -1:
            return ""
        return output[content_start:content_end].strip()

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        if (
            not model_answer or not model_answer.strip()
            or not ground_truth or not ground_truth.strip()
        ):
            return False

        gt_clean = re.sub(r"\s+", "", ground_truth.strip())
        ma_clean = re.sub(r"\s+", "", model_answer.strip())
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
            raise RuntimeError(
                "math-verify library is required but not installed. "
                "Run: pip install math-verify"
            )

        try:
            configs = [
                LatexExtractionConfig(),
                ExprExtractionConfig(),
                StringExtractionConfig(),
            ]
            parsed_gt = parse(ground_truth, extraction_config=configs)
            parsed_ma = parse(model_answer, extraction_config=configs)
            if parsed_gt and parsed_ma and verify(parsed_gt, parsed_ma):
                return True
        except Exception:
            pass

        try:
            wrapped_gt = f"\\boxed{{{ground_truth.strip()}}}"
            wrapped_ma = f"\\boxed{{{model_answer.strip()}}}"
            parsed_gt = parse(wrapped_gt, extraction_config=[LatexExtractionConfig()])
            parsed_ma = parse(wrapped_ma, extraction_config=[LatexExtractionConfig()])
            if parsed_gt and parsed_ma and verify(parsed_gt, parsed_ma):
                return True
        except Exception:
            pass

        return False
