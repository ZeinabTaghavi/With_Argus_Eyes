# utils/embeddings/reason_embed_encoder.py
from __future__ import annotations
from typing import List, Sequence, Optional
import re
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from .base import BaseRetriever
from .device import choose_device

class ReasonEmbedRetriever(BaseRetriever):
    """
    Retriever wrapper for the ReasonEmbed model (Qwen3-based).
    Provides encode_texts and encode_spans with windowed span pooling.
    """
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        # pick GPU or CPU if no device specified
        self.device = torch.device(device) if device else choose_device()
        # default model name from env or fallback to known HF repo
        if model_name is None:
            model_name = os.environ.get(
                "REASONEMBED_MODEL_NAME",
                "hanhainebula/reason-embed-basic-qwen3-4b-0928",
            )
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            output_hidden_states=False,
        )
        try:
            if self.device.type == "cuda":
                self.model = self.model.to(self.device, dtype=torch.float16)
            else:
                self.model = self.model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            self.model = self.model.to("cpu")
        self.model.eval()

    @torch.no_grad()
    def _mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # mean‐pool over non‐padded tokens
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        max_length: int = 2048,
    ) -> List[np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        out_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)
            last_hidden = out.last_hidden_state
            pooled = self._mean_pool(last_hidden, enc["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            out_vecs.extend(normalized.detach().float().cpu().numpy())
        return out_vecs

    @torch.inference_mode()
    def encode_spans(
        self,
        texts: Sequence[str],
        phrases: Sequence[str],
        *,
        occurrence_index: int = 0,
        max_length: int = 1024,
        batch_size: int = 1,
        case_insensitive: bool = True,
        **kwargs,
    ) -> List[np.ndarray]:
        flags = re.IGNORECASE if case_insensitive else 0
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(phrases, str):
            phrases = [phrases]
        assert len(texts) == len(phrases), "texts and phrases must have same length"
        model_max = getattr(self.model.config, "max_position_embeddings", None)
        if not model_max or model_max <= 0:
            model_max = getattr(self.tokenizer, "model_max_length", max_length or 1024)
        window_limit = min(model_max, max_length) if max_length is not None else model_max

        span_vecs: List[np.ndarray] = []
        for text, phrase in zip(texts, phrases):
            if not phrase:
                raise ValueError("Empty phrase")
            matches = list(re.finditer(re.escape(phrase), text, flags=flags))
            if not matches:
                raise ValueError(f"Phrase not found: {phrase!r}")
            m = matches[min(occurrence_index, len(matches)-1)]
            c0, c1 = m.start(), m.end()
            enc_full = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            input_ids_full = enc_full["input_ids"].to(self.device)
            attn_full = enc_full["attention_mask"].to(self.device)
            offsets_full = enc_full["offset_mapping"][0].tolist()
            span_token_idxs = [
                i
                for i, (a, b) in enumerate(offsets_full)
                if not (a == b == 0) and max(a, c0) < min(b, c1)
            ]
            if not span_token_idxs:
                raise ValueError("Span not covered by tokenization")
            span_min = min(span_token_idxs)
            span_max = max(span_token_idxs)
            span_len = span_max - span_min + 1
            window_size = min(window_limit, max(span_len, 1))
            half = max((window_size - span_len) // 2, 0)
            start_idx = max(0, span_min - half)
            end_idx = start_idx + window_size
            T_full = input_ids_full.shape[1]
            if end_idx > T_full:
                end_idx = T_full
                start_idx = max(0, end_idx - window_size)
            if start_idx > span_min or end_idx <= span_max:
                start_idx = max(0, span_min)
                end_idx = min(T_full, span_min + window_size)

            input_ids = input_ids_full[:, start_idx:end_idx]
            attn_mask = attn_full[:, start_idx:end_idx]
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(input_ids=input_ids, attention_mask=attn_mask)
            else:
                out = self.model(input_ids=input_ids, attention_mask=attn_mask)
            token_reps = out.last_hidden_state[0]  # (T_window, H)
            local_span_idxs = [
                i - start_idx
                for i in span_token_idxs
                if start_idx <= i < end_idx
            ]
            if not local_span_idxs:
                raise ValueError("Span not covered by window")
            span_repr = token_reps[local_span_idxs].mean(0)
            span_repr = torch.nn.functional.normalize(span_repr, p=2, dim=0)
            span_vecs.append(span_repr.detach().float().cpu().numpy())
        return span_vecs