import json
import re
from typing import Any, Dict, List, Optional

from .base_adaptor import BaseAdaptor

# Dynamically build valid set from each item's options (most are A–D, some A–E or A–C)
_ABCDE = frozenset("ABCDE")


class ARCChallengeAdaptor(BaseAdaptor):
    """
    ARC-Challenge: multiple-choice science QA (test split).
    JSONL schema produced by download_arc_challenge.py:
        question  – the stem
        options   – [{label: "A", text: "..."}, ...]
        answer    – ground-truth letter (normalised to A/B/C/D/E)
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
                "You are an expert in science and reasoning. "
                "Read the multiple-choice question carefully and reason step by step. "
                "Then state the correct option letter."
            )
        return (
            "You are an expert in science and reasoning. "
            "Answer each multiple-choice question by selecting exactly one option letter."
        )

    def format_prompt(self, item: Dict[str, Any]) -> str:
        stem = (item.get("question") or "").strip()
        options = item.get("options", [])
        option_lines = "\n".join(f"{o['label']}. {o['text']}" for o in options)
        valid_letters = ", ".join(o["label"] for o in options)
        prompt = f"Question: {stem}\n\n{option_lines}\n\n"
        prompt += (
            "Use this format:\n"
            "<think>\n"
            "[Your original reasoning process here, showing how YOU would reach this solution]\n"
            "</think>\n\n"
            "\\boxed{answer}\n"
            f"Important: inside \\boxed{{}} use only a single character ({valid_letters}). "
            "Do not put the full option wording or any other text inside the braces. "
            "Do not write anything after the closing brace of \\boxed{}.\n"
        )
        return prompt

    def get_question(self, item: Dict[str, Any]) -> str:
        return item.get("question", "")

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        a = (item.get("answer") or "").strip().upper()
        return a if a in _ABCDE else ""

    def get_variant_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"benchmark": "arc_challenge"}
        for k in ("index", "id"):
            if k in item and item[k] is not None:
                meta[k] = item[k]
        return meta

    # ── Answer extraction ────────────────────────────────────────────────

    @staticmethod
    def _valid_labels(item: Optional[Dict[str, Any]] = None) -> frozenset:
        """Return the set of valid option letters (default A–E superset)."""
        return _ABCDE

    @staticmethod
    def _extract_boxed_letter(text: str) -> Optional[str]:
        """Parse last \\boxed{...} and return a single A–E letter if present."""
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
        return letter if letter in _ABCDE else None

    @staticmethod
    def _letter_from_patterns(text: str) -> Optional[str]:
        """Fallback: 'Answer: X', standalone letter line, last word-boundary A–E."""
        t = text.strip()
        for pat in (
            r"(?:^|\n)\s*(?:final\s+)?(?:answer|choice|option)\s*[:：]?\s*\(?([A-Ea-e])\)?\s*(?:\n|$)",
            r"(?:^|\n)\s*\(?([A-Ea-e])\)\s*(?:is\s+correct|is\s+the\s+answer)\s*[.!]?\s*$",
        ):
            matches = list(re.finditer(pat, t, flags=re.IGNORECASE | re.MULTILINE))
            if matches:
                letter = matches[-1].group(1).upper()
                if letter in _ABCDE:
                    return letter
        for line in reversed(t.splitlines()):
            s = line.strip()
            if len(s) == 1 and s.upper() in _ABCDE:
                return s.upper()
        found = list(re.finditer(r"\b([A-E])\b", t))
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
        if not ground_truth or ground_truth not in _ABCDE:
            return False
        ma = (model_answer or "").strip().upper()
        if not ma:
            return False
        if ma in _ABCDE:
            return ma == ground_truth
        if ma[0] in _ABCDE:
            return ma[0] == ground_truth
        return False
