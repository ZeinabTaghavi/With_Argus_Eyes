"""Notebook-friendly inference helpers for ARGUS text risk scoring."""

from .text_risk import (
    ArgusTextConfig,
    EntityMention,
    EntityRiskResult,
    analyze_text,
    available_retrievers,
    extract_entities,
    highlight_entities,
    resolve_model_artifact,
    score_entities,
)

__all__ = [
    "ArgusTextConfig",
    "EntityMention",
    "EntityRiskResult",
    "analyze_text",
    "available_retrievers",
    "extract_entities",
    "highlight_entities",
    "resolve_model_artifact",
    "score_entities",
]
