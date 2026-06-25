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
    normalized = str(answer).strip().upper()
    labels = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)])
    return len(normalized) == 1 and normalized in labels


def parse_answer_text(text: str, choices: list[str]) -> str | None:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]
    pattern = rf"<answer>\s*([{re.escape(labels)}])\s*</answer>"
    matches = re.findall(pattern, str(text), flags=re.I)
    if matches:
        return matches[-1].upper()
    explicit_patterns = [
        rf"(?:FINAL\s*ANSWER|ANSWER|OPTION|CHOICE|LABEL)\s*[:=\-]?\s*([{re.escape(labels)}])\b",
        rf"\b([{re.escape(labels)}])\s*(?:IS\s+)?(?:CORRECT|BEST|CLOSEST)\b",
    ]
    hits: list[str] = []
    upper_text = str(text).upper()
    for explicit in explicit_patterns:
        hits.extend(re.findall(explicit, upper_text))
    return hits[-1].upper() if hits else None
