"""SciBench adaptor with verl-aligned prompt construction."""
import json
from typing import Any, Dict, List

from .base_adaptor import BaseAdaptor
from .verl_aligned_adaptor import VerlPromptMixin, _extract_boxed
from .scibench_adaptor import SciBenchAdaptor


class VerlSciBenchAdaptor(VerlPromptMixin, SciBenchAdaptor):
    """
    SciBench with verl-aligned prompt.
    Inherits SciBenchAdaptor for _load_data, get_ground_truth, extract_answer,
    verify_answer (multi-layer numeric), get_variant_metadata.
    Overrides prompt construction via VerlPromptMixin.
    """

    def _get_question_text(self, item: Dict[str, Any]) -> str:
        problem = (item.get("problem_text") or "").strip()
        unit = (item.get("unit") or "").strip()
        if unit:
            return (
                f"{problem}\n"
                f"Give a numerical answer (units: {unit})."
            )
        return problem
