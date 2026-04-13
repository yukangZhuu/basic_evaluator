import json
import re
from typing import Any, Dict, List, Optional

from .base_adaptor import BaseAdaptor


_VALID = frozenset("ABCD")


class GPQADiamondAdaptor(BaseAdaptor):
    """
    GPQA-Diamond: multiple-choice science QA (test split).
    Each record: question (full stem + labeled options), answer (ground-truth letter A–D).
    """

    def _load_data(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    def _get_system_prompt(self) -> str:
        if self.thinking_mode:
            return (
                "You are an expert in biology, chemistry, and physics with strong reasoning skills. "
                "Carefully analyze each multiple-choice question and provide step-by-step reasoning "
                "to arrive at the correct option. "
            )
        return (
            "You are an expert in biology, chemistry, and physics. "
            "Answer each multiple-choice question. "
        )

    def format_prompt(self, item: Dict[str, Any]) -> str:
        question = item.get("question", "").strip()
        prompt = f"Question: {question}\n\n"
        prompt += (
            "Use this format:\n"
            "<think>\n"
            "[Your original reasoning process here, showing how YOU would reach this solution]\n"
            "</think>\n\n"
            "\\boxed{answer}\n\n"
            "Important: inside \\boxed{} use only a single character A, B, C, or D (the option letter). "
            "Do not write anything after the closing brace of \\boxed{}.\n"
        )
        return prompt

    def get_question(self, item: Dict[str, Any]) -> str:
        return item.get("question", "")

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        a = item.get("answer", "")
        if isinstance(a, str):
            a = a.strip().upper()
        else:
            a = str(a).strip().upper()
        return a if a in _VALID else ""

    def get_variant_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"benchmark": "gpqa_diamond"}
        if "index" in item:
            meta["index"] = item["index"]
        return meta

    @staticmethod
    def _extract_boxed_letter(text: str) -> Optional[str]:
        """Parse last \\boxed{...} and return a single A–D if present."""
        token = r"\boxed{"
        start_idx = text.rfind(token)
        if start_idx == -1:
            return None
        content_start = start_idx + len(token)
        depth = 0
        end = -1
        for i in range(content_start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                if depth == 0:
                    end = i
                    break
                depth -= 1
        if end == -1:
            return None
        inner = text[content_start:end].strip()
        if not inner:
            return None
        letter = inner[0].upper()
        return letter if letter in _VALID else None

    @staticmethod
    def _letter_from_patterns(text: str) -> Optional[str]:
        """Fallback: explicit 'Answer:' / 'Choice:' lines, then last isolated A–D."""
        t = text.strip()
        # Last explicit markers (case-insensitive)
        for pat in (
            r"(?:^|\n)\s*(?:final\s+)?(?:answer|choice|option)\s*[:：]?\s*\(?([ABCDabcd])\)?\s*(?:\n|$)",
            r"(?:^|\n)\s*\(?([ABCDabcd])\)\s*(?:is\s+correct|is\s+the\s+answer)\s*[.!]?\s*$",
        ):
            matches = list(re.finditer(pat, t, flags=re.IGNORECASE | re.MULTILINE))
            if matches:
                letter = matches[-1].group(1).upper()
                if letter in _VALID:
                    return letter

        # Lines that are only a single letter
        for line in reversed(t.splitlines()):
            s = line.strip()
            if len(s) == 1 and s.upper() in _VALID:
                return s.upper()

        # Last word-boundary A–D (often appears at end of CoT)
        found = list(re.finditer(r"\b([ABCD])\b", t, flags=re.IGNORECASE))
        if found:
            return found[-1].group(1).upper()
        return None

    def extract_answer(self, model_output: str) -> str:
        if not model_output:
            return ""
        boxed = self._extract_boxed_letter(model_output)
        if boxed:
            return boxed
        fb = self._letter_from_patterns(model_output)
        return fb or ""

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        if not ground_truth or ground_truth not in _VALID:
            return False
        ma = (model_answer or "").strip().upper()
        if not ma:
            return False
        if ma in _VALID:
            return ma == ground_truth
        # If model echoed extra text but starts with a valid letter
        if ma[0] in _VALID:
            return ma[0] == ground_truth
        return False
