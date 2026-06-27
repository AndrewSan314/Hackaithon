from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHECKPOINT = "ckpt12_internal_rag"


@dataclass(frozen=True)
class PipelineConfig:
    data_path: Path | None
    work_dir: Path
    pred_path: Path
    model_root: Path
    model_select: str
    bge_model_dir: Path
    adapter_root: Path
    adapter_dir: Path | None
    law_admin_vector_db_dir: Path
    checkpoint_to_run: str
    limit: int | None

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            data_path=Path(os.environ["DATA_PATH"]) if os.environ.get("DATA_PATH") else None,
            work_dir=Path(os.environ.get("WORK_DIR", "/output")),
            pred_path=Path(os.environ.get("PRED_PATH", "/output/pred.csv")),
            model_root=Path(os.environ.get("MODEL_ROOT", "/models")),
            model_select=os.environ.get("MODEL_SELECT", "4b"),
            bge_model_dir=Path(os.environ.get("BGE_MODEL_DIR", "/bge/bge-m3")),
            adapter_root=Path(os.environ.get("ADAPTER_ROOT", "/adapters")),
            adapter_dir=Path(os.environ["ADAPTER_DIR"]) if os.environ.get("ADAPTER_DIR") else None,
            law_admin_vector_db_dir=Path(os.environ.get("LAW_ADMIN_VECTOR_DB_DIR", "/rag/rag_vector_db_final")),
            checkpoint_to_run=os.environ.get("CHECKPOINT_TO_RUN", DEFAULT_CHECKPOINT),
            limit=int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None,
        )

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "PipelineConfig":
        parser = argparse.ArgumentParser(description="Run ckpt12 Hackaithon MCQ inference.")
        parser.add_argument("--data", dest="data_path")
        parser.add_argument("--output", dest="pred_path")
        parser.add_argument("--work-dir", dest="work_dir")
        parser.add_argument("--model-root", dest="model_root")
        parser.add_argument("--model-select", dest="model_select")
        parser.add_argument("--bge-model-dir", dest="bge_model_dir")
        parser.add_argument("--adapter-root", dest="adapter_root")
        parser.add_argument("--adapter-dir", dest="adapter_dir")
        parser.add_argument("--rag-db-dir", dest="law_admin_vector_db_dir")
        parser.add_argument("--checkpoint", dest="checkpoint_to_run")
        parser.add_argument("--limit", type=int)
        args = parser.parse_args(argv)

        base = cls.from_env()
        return cls(
            data_path=Path(args.data_path) if args.data_path else base.data_path,
            work_dir=Path(args.work_dir) if args.work_dir else base.work_dir,
            pred_path=Path(args.pred_path) if args.pred_path else base.pred_path,
            model_root=Path(args.model_root) if args.model_root else base.model_root,
            model_select=args.model_select or base.model_select,
            bge_model_dir=Path(args.bge_model_dir) if args.bge_model_dir else base.bge_model_dir,
            adapter_root=Path(args.adapter_root) if args.adapter_root else base.adapter_root,
            adapter_dir=Path(args.adapter_dir) if args.adapter_dir else base.adapter_dir,
            law_admin_vector_db_dir=(
                Path(args.law_admin_vector_db_dir)
                if args.law_admin_vector_db_dir
                else base.law_admin_vector_db_dir
            ),
            checkpoint_to_run=args.checkpoint_to_run or base.checkpoint_to_run,
            limit=args.limit if args.limit is not None else base.limit,
        )

    def apply_to_env(self) -> None:
        if self.data_path is not None:
            os.environ["DATA_PATH"] = str(self.data_path)
        os.environ["WORK_DIR"] = str(self.work_dir)
        os.environ["PRED_PATH"] = str(self.pred_path)
        os.environ["MODEL_ROOT"] = str(self.model_root)
        os.environ["MODEL_SELECT"] = str(self.model_select)
        os.environ["BGE_MODEL_DIR"] = str(self.bge_model_dir)
        os.environ["ADAPTER_ROOT"] = str(self.adapter_root)
        if self.adapter_dir is not None:
            os.environ["ADAPTER_DIR"] = str(self.adapter_dir)
        os.environ["LAW_ADMIN_VECTOR_DB_DIR"] = str(self.law_admin_vector_db_dir)
        os.environ["CHECKPOINT_TO_RUN"] = self.checkpoint_to_run
        if self.limit is not None:
            os.environ["LIMIT"] = str(self.limit)
        else:
            os.environ.pop("LIMIT", None)
