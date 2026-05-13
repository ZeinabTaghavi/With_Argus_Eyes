from __future__ import annotations

import glob
import html
import io
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import joblib
import numpy as np

from with_argus_eyes.training.helpers import stub_name
from with_argus_eyes.utils.embeddings import build_retriever


TextMode = Literal["context", "canonical", "span"]

DEFAULT_RETRIEVERS = (
    "contriever",
    "qwen3",
    "jina",
    "bge-m3",
    "reason-embed",
    "nv-embed",
    "gritlm",
    "reasonir",
)


def _default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ArgusTextConfig:
    """Configuration for raw-text entity extraction and ARGUS risk scoring."""

    retriever: str = "contriever"
    language: str = "en"
    ner_model: str = "dslim/bert-base-NER"
    risk_threshold: float = 0.3
    ner_threshold: float = 0.5
    order: int | str = 800
    k: int = 50
    text_mode: TextMode = "context"
    batch_size: int = 64
    max_length: int = 1048
    workspace_root: str | Path | None = None
    analysis_out_dir: str | Path = Path("outputs") / "12_risk_outputs"
    results_tag: str = ""
    models_dir: str | Path | None = None
    model_artifact: str | Path | None = None
    device: str | None = None

    @property
    def root(self) -> Path:
        return Path(self.workspace_root).expanduser().resolve() if self.workspace_root else _default_workspace_root()


@dataclass(frozen=True)
class EntityMention:
    """A named entity mention extracted from user-provided text."""

    text: str
    entity_type: str
    start: int
    end: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.text,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "ner_score": self.score,
        }


@dataclass(frozen=True)
class EntityRiskResult:
    """A table-ready ARGUS result for one extracted entity."""

    entity: str
    entity_type: str
    start: int
    end: int
    ner_score: float
    risk_score: float
    above_threshold: bool
    retriever: str
    model_artifact: str
    context: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "ner_score": self.ner_score,
            "risk_score": self.risk_score,
            "above_threshold": self.above_threshold,
            "retriever": self.retriever,
            "model_artifact": self.model_artifact,
            "context": self.context,
        }


def available_retrievers() -> tuple[str, ...]:
    """Return retrievers that are exposed by the notebook workflow."""

    return DEFAULT_RETRIEVERS


def _resolve_path(path: str | Path, root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else root / value


def _build_default_results_tag(retriever: str, order: int | str, k: int) -> str:
    base = stub_name(retriever, f"ratio_unrelevant_below_k_{k}")
    return f"{base}_o_{order}_k_{k}"


def _pick_best_artifact(model_dir: Path) -> Path | None:
    if not model_dir.is_dir():
        return None

    all_joblib = sorted(model_dir.glob("*.joblib"))
    if not all_joblib:
        return None

    mlp_best = [path for path in all_joblib if path.name.startswith("mlp_best_")]
    baseline_best = [path for path in all_joblib if path.name.startswith("baseline_best_")]
    ranked = mlp_best or baseline_best or all_joblib
    ranked.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return ranked[0]


def _available_model_dirs(config: ArgusTextConfig) -> list[Path]:
    out_dir = _resolve_path(config.analysis_out_dir, config.root)
    pattern = out_dir / f"{config.retriever}_ratio_unrelevant_below_k_*_o_{config.order}_k_{config.k}*" / "models"
    return sorted(Path(path) for path in glob.glob(str(pattern)))


def _available_retriever_model_dirs(config: ArgusTextConfig) -> list[Path]:
    out_dir = _resolve_path(config.analysis_out_dir, config.root)
    pattern = out_dir / f"{config.retriever}_ratio_unrelevant_below_k_*" / "models"
    return sorted(Path(path) for path in glob.glob(str(pattern)))


def resolve_model_artifact(config: ArgusTextConfig) -> Path:
    """Resolve the trained model artifact that matches the notebook config."""

    root = config.root
    if config.model_artifact:
        artifact = _resolve_path(config.model_artifact, root)
        if not artifact.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact}")
        return artifact

    candidate_dirs: list[Path] = []
    if config.models_dir:
        candidate_dirs.append(_resolve_path(config.models_dir, root))
    else:
        out_dir = _resolve_path(config.analysis_out_dir, root)
        tag = config.results_tag or _build_default_results_tag(config.retriever, config.order, config.k)
        candidate_dirs.append(out_dir / tag / "models")
        for match in _available_model_dirs(config):
            if match not in candidate_dirs:
                candidate_dirs.append(match)

    for model_dir in candidate_dirs:
        artifact = _pick_best_artifact(model_dir)
        if artifact:
            return artifact

    available = _available_model_dirs(config) or _available_retriever_model_dirs(config)
    available_text = "\n".join(f"  - {path.parent.name}" for path in available) or "  (none)"
    tried = "\n".join(f"  - {path}" for path in candidate_dirs) or "  (none)"
    raise FileNotFoundError(
        "No trained ARGUS model artifact was found for "
        f"retriever={config.retriever!r}, order={config.order!r}, k={config.k!r}.\n"
        f"Tried:\n{tried}\n"
        f"Available matching result folders:\n{available_text}"
    )


@lru_cache(maxsize=4)
def _get_ner_pipeline(model_name: str, device: str | None) -> Callable[[str], list[dict[str, Any]]]:
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise ImportError(
            "Named-entity extraction requires transformers. Install project requirements first."
        ) from exc

    kwargs: dict[str, Any] = {
        "task": "ner",
        "model": model_name,
        "aggregation_strategy": "simple",
    }
    if device is not None:
        kwargs["device"] = int(device) if str(device).lstrip("-").isdigit() else device
    return pipeline(**kwargs)


def _entity_from_raw(raw: dict[str, Any]) -> EntityMention | None:
    text = str(raw.get("word") or raw.get("text") or "").strip()
    if not text:
        return None

    start = raw.get("start")
    end = raw.get("end")
    if start is None or end is None:
        return None

    entity_type = str(raw.get("entity_group") or raw.get("entity") or raw.get("label") or "ENTITY")
    if "-" in entity_type:
        entity_type = entity_type.rsplit("-", 1)[-1]

    return EntityMention(
        text=text,
        entity_type=entity_type,
        start=int(start),
        end=int(end),
        score=float(raw.get("score", 0.0)),
    )


def _dedupe_entities(entities: Iterable[EntityMention]) -> list[EntityMention]:
    seen: set[tuple[int, int, str]] = set()
    deduped: list[EntityMention] = []
    for entity in entities:
        key = (entity.start, entity.end, entity.text.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def extract_entities(
    text: str,
    config: ArgusTextConfig | None = None,
    *,
    ner_pipeline: Callable[[str], list[dict[str, Any]]] | None = None,
) -> list[EntityMention]:
    """Extract named entities from raw text using the configured NER model."""

    config = config or ArgusTextConfig()
    if not text or not text.strip():
        return []

    pipe = ner_pipeline or _get_ner_pipeline(config.ner_model, config.device)
    raw_entities = pipe(text)
    entities = []
    for raw in raw_entities:
        entity = _entity_from_raw(raw)
        if entity is not None and entity.score >= config.ner_threshold:
            entities.append(entity)
    return _dedupe_entities(entities)


def _canonicalize(label: str, context: str) -> str:
    label = (label or "").strip()
    context = (context or "").strip()
    if label and label.lower() not in context.lower():
        return f"{label} : {context}"
    return context


def _texts_for_embedding(text: str, entities: Sequence[EntityMention], text_mode: TextMode) -> list[str]:
    if text_mode == "context":
        return [text for _ in entities]
    if text_mode == "canonical":
        return [_canonicalize(entity.text, text) for entity in entities]
    if text_mode == "span":
        return [text for _ in entities]
    raise ValueError("text_mode must be one of: context, canonical, span")


def _predict_scores(artifact_path: Path, vectors: np.ndarray) -> np.ndarray:
    artifact = _load_joblib_artifact(artifact_path)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        scaler = artifact.get("scaler")
    else:
        model = artifact
        scaler = None

    model_input = scaler.transform(vectors) if scaler is not None else vectors
    return np.asarray(model.predict(model_input), dtype=float).reshape(-1)


def _torch_load_cpu(buffer: bytes) -> Any:
    import torch

    kwargs = {"map_location": torch.device("cpu")}
    try:
        return torch.load(io.BytesIO(buffer), weights_only=False, **kwargs)
    except TypeError:  # pragma: no cover - older torch compatibility
        return torch.load(io.BytesIO(buffer), **kwargs)


def _load_joblib_artifact(artifact_path: Path) -> Any:
    _install_legacy_utils_aliases()
    try:
        return joblib.load(artifact_path)
    except RuntimeError as exc:
        if "Attempting to deserialize object on a CUDA device" not in str(exc):
            raise

        try:
            import torch
        except ImportError:
            raise exc

        original_loader = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = _torch_load_cpu
        try:
            return joblib.load(artifact_path)
        finally:
            torch.storage._load_from_bytes = original_loader


def _alias_module(old: str, new: str) -> None:
    try:
        module = import_module(new)
    except ImportError:
        return
    sys.modules.setdefault(old, module)


def _install_legacy_utils_aliases() -> None:
    """Support model artifacts pickled with the old top-level utils namespace."""

    _alias_module("utils", "with_argus_eyes.utils")
    for submodule in (
        "embeddings",
        "models",
        "models.mlp",
        "models.baselines",
        "models.metrics",
        "models.train_utils",
        "risk",
        "risk.scores",
        "plots",
    ):
        _alias_module(f"utils.{submodule}", f"with_argus_eyes.utils.{submodule}")


def score_entities(
    text: str,
    entities: Sequence[EntityMention | dict[str, Any]],
    config: ArgusTextConfig | None = None,
    *,
    retriever: Any | None = None,
    model_artifact: str | Path | None = None,
) -> list[EntityRiskResult]:
    """Embed and score extracted entities with the configured ARGUS model."""

    config = config or ArgusTextConfig()
    if not text or not text.strip() or not entities:
        return []

    normalized_entities = [
        entity
        if isinstance(entity, EntityMention)
        else EntityMention(
            text=str(entity["entity"]),
            entity_type=str(entity.get("entity_type", "ENTITY")),
            start=int(entity.get("start", 0)),
            end=int(entity.get("end", 0)),
            score=float(entity.get("ner_score", entity.get("score", 0.0))),
        )
        for entity in entities
    ]

    if not normalized_entities:
        return []

    selected_artifact = _resolve_path(model_artifact, config.root) if model_artifact else resolve_model_artifact(config)
    selected_retriever = retriever or build_retriever(config.retriever, device=config.device)

    if config.text_mode == "span":
        vectors = selected_retriever.encode_spans(
            [text for _ in normalized_entities],
            [entity.text for entity in normalized_entities],
            batch_size=config.batch_size,
            max_length=config.max_length,
        )
    else:
        embed_texts = _texts_for_embedding(text, normalized_entities, config.text_mode)
        vectors = selected_retriever.encode_texts(
            embed_texts,
            batch_size=config.batch_size,
            max_length=config.max_length,
        )

    x = np.asarray(vectors, dtype=np.float32)
    scores = _predict_scores(selected_artifact, x)
    if scores.shape[0] != len(normalized_entities):
        raise RuntimeError(
            f"Prediction length mismatch: predicted={scores.shape[0]} expected={len(normalized_entities)}"
        )

    return [
        EntityRiskResult(
            entity=entity.text,
            entity_type=entity.entity_type,
            start=entity.start,
            end=entity.end,
            ner_score=entity.score,
            risk_score=float(score),
            above_threshold=bool(float(score) >= config.risk_threshold),
            retriever=config.retriever,
            model_artifact=str(selected_artifact),
            context=text,
        )
        for entity, score in zip(normalized_entities, scores)
    ]


def analyze_text(
    text: str,
    config: ArgusTextConfig | None = None,
    *,
    ner_pipeline: Callable[[str], list[dict[str, Any]]] | None = None,
    retriever: Any | None = None,
    model_artifact: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run the full raw-text -> entities -> ARGUS scores workflow."""

    config = config or ArgusTextConfig()
    entities = extract_entities(text, config, ner_pipeline=ner_pipeline)
    if not entities:
        return []
    results = score_entities(
        text,
        entities,
        config,
        retriever=retriever,
        model_artifact=model_artifact,
    )
    return [result.as_dict() for result in results]


def highlight_entities(text: str, results: Sequence[dict[str, Any] | EntityRiskResult]) -> str:
    """Return HTML with entities highlighted by risk-threshold status."""

    spans: list[tuple[int, int, str, bool, float]] = []
    for row in results:
        data = row.as_dict() if isinstance(row, EntityRiskResult) else row
        spans.append(
            (
                int(data["start"]),
                int(data["end"]),
                str(data["entity"]),
                bool(data["above_threshold"]),
                float(data["risk_score"]),
            )
        )
    spans.sort(key=lambda item: (item[0], item[1]))

    pieces: list[str] = []
    cursor = 0
    for start, end, label, risky, score in spans:
        if start < cursor:
            continue
        pieces.append(html.escape(text[cursor:start]))
        color = "#f7b2ad" if risky else "#b9dfc6"
        border = "#a33b32" if risky else "#2f7d45"
        pieces.append(
            '<mark style="'
            f"background:{color}; border-bottom:2px solid {border}; padding:0 0.15rem;"
            f'" title="{html.escape(label)} risk={score:.3f}">'
            f"{html.escape(text[start:end])}</mark>"
        )
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)
