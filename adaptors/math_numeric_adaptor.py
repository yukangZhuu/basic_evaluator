from typing import Dict, Any

from .teacher_traces_adaptor import TeacherTracesAdaptor


class MathNumericAdaptor(TeacherTracesAdaptor):
    """
    Hendrycks MATH-style numeric subset (question + ground_truth, same schema as teacher traces).
    Prompt and verification behavior are identical to TeacherTracesAdaptor; only adds metadata
    for reporting (subject, level, unique_id).
    """

    def get_variant_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "index": item.get("index"),
            "subject": item.get("subject"),
            "level": item.get("level"),
            "unique_id": item.get("unique_id"),
        }
