from __future__ import annotations
from typing import Optional
from .base import BaseRetriever
from .contriever_encoder import ContrieverRetriever
from .reasonir_encoder import ReasonIRRetriever
from .qwen3_encoder import Qwen3Retriever
from .jina_encoder import JinaRetriever
from .bge_m3_encoder import BGEM3Retriever
from .reason_embed_encoder import ReasonEmbedRetriever
from .nv_embed_encoder import NVEmbedRetriever
from .gritlm_encoder import GritLMRetriever

def build_retriever(
    name: str,
    *,
    device: Optional[str] = None,
    reasonir_instruction: str = "",
    qwen3_instruction: str = "",
) -> BaseRetriever:
    key = name.strip().lower()
    if key == "contriever":
        return ContrieverRetriever(device=device)
    if key == "reasonir":
        return ReasonIRRetriever(device=device, instruction=reasonir_instruction)
    if key == "qwen3":
        return Qwen3Retriever(device=device, instruction=qwen3_instruction)
    if key == "jina":
        return JinaRetriever(device=device)
    if key in ("bge-m3", "bge_m3", "bge"):
        return BGEM3Retriever(device=device)
    if key in ("reason-embed", "reasonembed", "reason"):
        return ReasonEmbedRetriever(device=device)
    if key in ("nv-embed", "nv_embed", "nv-embed-v2"):
        return NVEmbedRetriever(device=device)
    if key in ("gritlm", "gritlm-v2", "grit"):
        return GritLMRetriever(device=device)  # new case
    raise ValueError(f"Unknown retriever: {name}")