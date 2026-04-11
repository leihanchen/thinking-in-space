import re
from typing import List

from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen25vl import Qwen25VL


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@register_model("qwen25vl_reasoning")
class Qwen25VLReasoning(Qwen25VL):
    def __init__(self, *args, **kwargs):
        self.output_mode = str(kwargs.pop("output_mode", "answer_only")).strip().lower()
        self.tag_type = str(kwargs.pop("tag_type", "ds")).strip().lower()
        if self.tag_type not in {"ds", "qwen"}:
            raise ValueError(
                f"Unsupported tag_type '{self.tag_type}'. Use 'ds' or 'qwen'."
            )
        self.fallback_to_full = _to_bool(kwargs.pop("fallback_to_full", True))
        self.remove_think_if_no_answer = _to_bool(
            kwargs.pop("remove_think_if_no_answer", True)
        )

        # Encourage stable structured output for reasoning-tuned checkpoints.
        self.enforce_reasoning_format = _to_bool(
            kwargs.pop("enforce_reasoning_format", True)
        )
        default_reasoning_system_prompt = (
            "You are a helpful assistant. For every response, strictly use this format: "
            "<think>your reasoning</think><answer>final answer only</answer>. "
            "Do not omit or rename these tags."
        )
        self.system_prompt = (
            str(kwargs.pop("system_prompt", default_reasoning_system_prompt)).strip()
            if self.enforce_reasoning_format
            else str(kwargs.pop("system_prompt", "")).strip()
        )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _normalize_reasoning_tags(text: str) -> str:
        normalized = text
        normalized = re.sub(r"<\\\s*think\s*>", "</think>", normalized, flags=re.I)
        normalized = re.sub(r"<\\\s*answer\s*>", "</answer>", normalized, flags=re.I)
        # Normalize common qwen boxed answer variants, e.g. "\\boxed{...}" or "\boxed {...}".
        normalized = re.sub(r"\\{2,}boxed\s*\{", r"\\boxed{", normalized)
        normalized = re.sub(r"\\boxed\s*\{", r"\\boxed{", normalized)
        return normalized

    def _extract_answer(self, text: str) -> str:
        normalized = self._normalize_reasoning_tags(text)

        if self.tag_type == "ds":
            answer = self._extract_ds_answer(normalized)
        else:
            answer = self._extract_qwen_answer(normalized)

        if answer:
            return answer

        if self.remove_think_if_no_answer:
            stripped = self._remove_reasoning_markup(normalized)
            if stripped:
                return stripped

        if self.fallback_to_full:
            return normalized.strip()

        return ""

    @staticmethod
    def _remove_reasoning_markup(text: str) -> str:
        stripped = re.sub(
            r"<\s*think\s*>.*?<\s*/\s*think\s*>",
            "",
            text,
            flags=re.I | re.S,
        )
        stripped = re.sub(r"<\s*/?\s*answer\s*>", "", stripped, flags=re.I)
        return stripped.strip()

    @staticmethod
    def _extract_ds_answer(text: str) -> str:
        matches = re.findall(
            r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>",
            text,
            flags=re.I | re.S,
        )
        for candidate in reversed(matches):
            candidate = candidate.strip()
            if candidate:
                return candidate
        return ""

    @staticmethod
    def _extract_qwen_answer(text: str) -> str:
        candidate = Qwen25VLReasoning._extract_last_boxed_content(text)
        if candidate:
            return candidate
        return ""

    @staticmethod
    def _extract_last_boxed_content(text: str) -> str:
        marker = r"\boxed{"
        idx = 0
        last_content = ""
        while True:
            start = text.find(marker, idx)
            if start == -1:
                break

            i = start + len(marker)
            depth = 1
            content_start = i
            while i < len(text) and depth > 0:
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                i += 1

            if depth == 0:
                content = text[content_start : i - 1].strip()
                if content:
                    last_content = content

            idx = start + len(marker)

        return last_content

    def generate_until(self, requests) -> List[str]:
        # Base model generation returns raw decoded responses.
        raw_outputs = super().generate_until(requests)

        # Keep a copy for debugging/inspection without changing return type.
        self.last_raw_outputs = raw_outputs

        if self.output_mode == "full_text":
            self.last_processed_outputs = raw_outputs
            return raw_outputs

        if self.output_mode == "answer_only":
            processed_outputs = [self._extract_answer(output) for output in raw_outputs]
            self.last_processed_outputs = processed_outputs
            return processed_outputs

        raise ValueError(
            f"Unsupported output_mode '{self.output_mode}'. Use 'answer_only' or 'full_text'."
        )
