from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LawAdminRAGState:
    db_dir: Path
    chunks: list[dict]
    embeddings: np.ndarray
    domain_index: dict
    metadata: dict


def load_law_admin_vector_db(path: Path) -> LawAdminRAGState:
    if not path.exists():
        raise FileNotFoundError(path)
    chunks_path = path / "chunks.jsonl"
    embeddings_path = path / "embeddings.fp16.npy"
    domain_index_path = path / "domain_index.json"
    metadata_path = path / "metadata.json"
    for required in (chunks_path, embeddings_path, domain_index_path):
        if not required.exists():
            raise FileNotFoundError(required)

    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    embeddings = np.load(embeddings_path, mmap_mode="r")
    domain_index = json.loads(domain_index_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if "known_law_admin_facts" not in domain_index:
        raise ValueError("rag_vector_db_final must contain known_law_admin_facts.")
    if len(chunks) != int(embeddings.shape[0]):
        raise ValueError(f"Vector DB mismatch: chunks={len(chunks)} embeddings={embeddings.shape}")
    return LawAdminRAGState(path, chunks, embeddings, domain_index, metadata)


def known_law_admin_range(state: LawAdminRAGState) -> tuple[int, int]:
    info = state.domain_index["known_law_admin_facts"]
    return int(info["start"]), int(info["end"])


def build_law_admin_rag_context(question: str, choices: list[str], snippets: list[dict]) -> str:
    rendered = []
    for index, chunk in enumerate(snippets, 1):
        rendered.append(
            f"[{index}] {chunk.get('section', '')} | answer={chunk.get('answer', '')}\n"
            f"{chunk.get('text', '')}"
        )
    return (
        "Retrieved Vietnamese law/public-administration reference snippets. "
        "Use these snippets as the primary evidence. If a snippet states a known answer, trust it.\n\n"
        + "\n\n".join(rendered)
        + "\n\nQuestion:\n"
        + str(question)
    )
