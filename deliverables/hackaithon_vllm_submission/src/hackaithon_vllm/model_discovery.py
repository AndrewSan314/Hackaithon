from __future__ import annotations

import json
from pathlib import Path


def has_model_weights(model_dir: Path) -> bool:
    return any(model_dir.glob(pattern) for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"))


def load_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_models(model_root: Path, work_dir: Path, prefer_keywords: tuple[str, ...] = ("qwen3.5", "qwen3_5", "qwen", "9b", "awq")) -> list[dict]:
    candidates = []
    for root in (model_root, Path("/kaggle/input"), work_dir / "hf_models"):
        if not root.exists():
            continue
        for config_path in root.rglob("config.json"):
            model_dir = config_path.parent
            if not has_model_weights(model_dir):
                continue
            config = load_json_safe(config_path)
            text = " ".join(
                [
                    str(model_dir),
                    str(config.get("model_type", "")),
                    str(config.get("_name_or_path", "")),
                    " ".join(config.get("architectures", []) or []),
                ]
            ).lower()
            score = 100 if "qwen" in text else 0
            score += sum(20 for keyword in prefer_keywords if keyword.lower() in text)
            if "awq" in text:
                score += 15
            if "gptq" in text:
                score += 10
            if "causallm" in text:
                score += 10
            candidates.append({"score": score, "path": model_dir, "search_text": text})
    return sorted(candidates, key=lambda item: (item["score"], str(item["path"])), reverse=True)


def select_model(candidates: list[dict], select: str = "auto") -> Path:
    if not candidates:
        raise FileNotFoundError("No local model with config.json and weights was found.")
    if str(select).isdigit():
        return Path(candidates[int(select)]["path"])
    if select and select not in {"auto", "auto_or_hf"}:
        keyword = str(select).lower()
        matches = [item for item in candidates if keyword in str(item["path"]).lower()]
        if matches:
            return Path(matches[0]["path"])
    return Path(candidates[0]["path"])
