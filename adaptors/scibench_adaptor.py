import json
import math
import re
from typing import Any, Dict, List, Optional

from .base_adaptor import BaseAdaptor

try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

# Qwen-style thinking tags (match teacher_traces / gpqa adaptors)
_OT = "<" + "think" + ">"
_CT = "</" + "think" + ">"

_FLOAT_RE = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][+-]?\d+)?"
    r"|[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
)


class SciBenchAdaptor(BaseAdaptor):
    """
    SciBench (xw27/scibench): numerical answers in ``answer_number`` (string, may include sign,
    decimals, commas, Unicode minus). Multi-layer verification: normalized string, float with
    tolerances, optional math_verify, and fallback numeric token extraction from messy outputs.
    """

    # Relative / absolute tolerance for physical constants–style answers
    _RTOL = 1e-4
    _ATOL = 1e-6

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
                "You are an expert in chemistry, physics, and related quantitative sciences. "
                "Solve each problem carefully with clear reasoning, then give a single numerical "
                "final answer in \\boxed{} (number only when possible, matching the expected units)."
            )
        return (
            "You are an expert in chemistry, physics, and related quantitative sciences. "
            "Solve each problem and put only the numerical result in \\boxed{}."
        )

    def format_prompt(self, item: Dict[str, Any]) -> str:
        problem = (item.get("problem_text") or "").strip()
        unit = (item.get("unit") or "").strip()
        uline = ""
        if unit:
            uline = (
                f"\n\nGive a numerical answer using units consistent with this specification "
                f"(same physical quantity and dimension): {unit}\n"
            )
        prompt = f"Question: {problem}{uline}\n\n"
        prompt += (
            "Use this format:\n"
            f"{_OT}\n"
            "[Your original reasoning process here, showing how YOU would reach this solution]\n"
            f"{_CT}\n\n"
            "\\boxed{answer}\n"
            "Put only the numerical value inside \\boxed{} (you may include a leading + or - sign). "
            "Do not put units inside \\boxed{} if the number alone is sufficient.\n"
        )
        return prompt

    def get_question(self, item: Dict[str, Any]) -> str:
        return item.get("problem_text", "")

    def get_ground_truth(self, item: Dict[str, Any]) -> str:
        return (item.get("answer_number") or "").strip()

    def get_variant_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"benchmark": "scibench"}
        for k in ("index", "source", "problemid", "unit"):
            if k in item and item[k] is not None:
                meta[k] = item[k]
        return meta

    @staticmethod
    def _unicode_minus_to_ascii(s: str) -> str:
        return (
            s.replace("\u2212", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("−", "-")
        )

    @classmethod
    def _scrub_for_compare(cls, s: str) -> str:
        """Normalize whitespace, unicode minus, commas (thousands), wrapping $, thin spaces."""
        if not s:
            return ""
        t = cls._unicode_minus_to_ascii(s.strip())
        t = t.strip("$").strip()
        t = t.replace("\\,", "").replace("~", "")
        t = re.sub(r"\s+", "", t)
        t = t.replace(",", "")
        return t

    @classmethod
    def _try_parse_float(cls, s: str) -> Optional[float]:
        """Parse a float from a scrubbed or partially messy numeric string."""
        if not s:
            return None
        t = cls._scrub_for_compare(s)
        if not t:
            return None
        try:
            return float(t)
        except ValueError:
            pass
        m = _FLOAT_RE.search(t)
        if not m:
            return None
        frag = m.group(0).replace(",", "")
        try:
            return float(frag)
        except ValueError:
            return None

    @classmethod
    def _floats_match(cls, a: float, b: float) -> bool:
        if math.isnan(a) or math.isnan(b):
            return False
        if math.isinf(a) or math.isinf(b):
            return a == b
        return math.isclose(a, b, rel_tol=cls._RTOL, abs_tol=cls._ATOL)

    def extract_answer(self, model_output: str) -> str:
        output = (model_output or "").strip()
        token = r"\boxed{"
        start_idx = output.rfind(token)
        if start_idx == -1:
            return ""
        content_start = start_idx + len(token)
        depth = 0
        content_end = -1
        for i in range(content_start, len(output)):
            c = output[i]
            if c == "{":
                depth += 1
            elif c == "}":
                if depth == 0:
                    content_end = i
                    break
                depth -= 1
        if content_end == -1:
            return ""
        return output[content_start:content_end].strip()

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        """
        Layers (early exit on success):
        1. Normalized string equality (unicode minus, commas, whitespace, $).
        2. Float equality with rtol/abs tolerance on fully parsed strings.
        3. Float equality after extracting the first numeric token from model output
           (handles trailing unit text or LaTeX outside the number).
        4. math_verify (LaTeX / expr / string) if available.
        5. math_verify on \\boxed-wrapped strings.
        """
        if not ground_truth or not ground_truth.strip():
            return False
        if not model_answer or not model_answer.strip():
            return False

        gt_s = self._scrub_for_compare(ground_truth)
        ma_s = self._scrub_for_compare(model_answer)
        if gt_s and ma_s and gt_s == ma_s:
            return True

        gt_f = self._try_parse_float(ground_truth)
        ma_f = self._try_parse_float(model_answer)
        if gt_f is not None and ma_f is not None and self._floats_match(gt_f, ma_f):
            return True

        if gt_f is not None:
            for m in _FLOAT_RE.finditer(self._unicode_minus_to_ascii(model_answer)):
                try:
                    cand = float(m.group(0).replace(",", ""))
                except ValueError:
                    continue
                if self._floats_match(gt_f, cand):
                    return True

        if not MATH_VERIFY_AVAILABLE:
            return False

        try:
            configs = [LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()]
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
