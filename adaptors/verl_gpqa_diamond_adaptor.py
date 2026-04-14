"""GPQA-Diamond adaptor with verl-aligned prompt construction."""
import json
import re
from typing import Any, Dict, List, Optional

from .base_adaptor import BaseAdaptor
from .verl_aligned_adaptor import VerlPromptMixin
from .gpqa_diamond_adaptor import GPQADiamondAdaptor

_VALID = frozenset("ABCD")


class VerlGPQADiamondAdaptor(VerlPromptMixin, BaseAdaptor):
    """
    GPQA-Diamond with verl-aligned prompt.
    question field already contains stem + options (A/B/C/D).
    Answer extraction & verification reused from GPQADiamondAdaptor.
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

    def extract_answer(self, model_output: str) -> str:
        if not model_output:
            return ""
        boxed = GPQADiamondAdaptor._extract_boxed_letter(model_output)
        if boxed:
            return boxed
        fb = GPQADiamondAdaptor._letter_from_patterns(model_output)
        return fb or ""

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        if not ground_truth or ground_truth not in _VALID:
            return False
        ma = (model_answer or "").strip().upper()
        if not ma:
            return False
        if ma in _VALID:
            return ma == ground_truth
        if ma[0] in _VALID:
            return ma[0] == ground_truth
        return False
