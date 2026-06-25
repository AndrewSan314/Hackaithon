from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class AnswerResult:
    qid: str
    answer: str
    route: str
    solver: str
    raw_text: str = ""
    confidence: float = 0.0
    fallback_used: bool = False


def valid_label(answer: str, choices: list[str]) -> bool:
    if not answer:
        return False
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]
    return str(answer).strip().upper() in labels


def parse_answer_text(text: str, choices: list[str]) -> str | None:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]
    pattern = rf"<answer>\s*([{re.escape(labels)}])\s*</answer>"
    matches = re.findall(pattern, str(text), flags=re.I)
    if matches:
        return matches[-1].upper()
    loose = re.findall(rf"\b([{re.escape(labels)}])\b", str(text).upper())
    return loose[-1] if loose else None
