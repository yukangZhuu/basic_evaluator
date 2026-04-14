"""MMLU-Pro adaptor with verl-aligned prompt construction."""
import json
import re
import string
from typing import Any, Dict, List, Optional

from .base_adaptor import BaseAdaptor
from .verl_aligned_adaptor import VerlPromptMixin
from .mmlu_pro_adaptor import MMLUProAdaptor

_ALL_LETTERS = frozenset(string.ascii_uppercase[:10])  # A–J


class VerlMMLUProAdaptor(VerlPromptMixin, BaseAdaptor):
    """
    MMLU-Pro with verl-aligned prompt.
    Stem and options are merged into a single question text for the verl prompt template.
    Answer extraction & verification reused from MMLUProAdaptor.
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

    def _get_question_text(self, item: Dict[str, Any]) -> str:
        """Merge stem + options into one string for the verl user content."""
        stem = (item.get("question") or "").strip()
        options = item.get("options", [])
        option_lines = "\n".join(f"{o['label']}. {o['text']}" for o in options)
        return f"{stem}\n\n{option_lines}"

    def get_question(self, item: Dict[str, Any]) -> str:
        return item.get("question", "")

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        a = (item.get("answer") or "").strip().upper()
        return a if a in _ALL_LETTERS else ""

    def get_variant_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"benchmark": "mmlu_pro"}
        for k in ("index", "question_id", "category", "src"):
            if k in item and item[k] is not None:
                meta[k] = item[k]
        return meta

    def extract_answer(self, model_output: str) -> str:
        if not model_output:
            return ""
        boxed = MMLUProAdaptor._extract_boxed_letter(model_output)
        if boxed:
            return boxed
        fb = MMLUProAdaptor._letter_from_patterns(model_output)
        return fb or ""

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        if not ground_truth or ground_truth not in _ALL_LETTERS:
            return False
        ma = (model_answer or "").strip().upper()
        if not ma:
            return False
        if ma in _ALL_LETTERS:
            return ma == ground_truth
        if ma[0] in _ALL_LETTERS:
            return ma[0] == ground_truth
        return False
