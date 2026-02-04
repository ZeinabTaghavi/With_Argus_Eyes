# utils/embeddings/nv_embed_encoder.py
from __future__ import annotations
from typing import List, Sequence, Optional
import re
import os
import numpy as np
import torch
from transformers import AutoModel

from .base import BaseRetriever
from .device import choose_device


class NVEmbedRetriever(BaseRetriever):
    """
    Retriever wrapper for NVIDIA's NV-Embed-v2 model.

    It loads the custom NVEmbedModel via AutoModel (trust_remote_code=True)
    and uses its `_do_encode` helper to produce sentence embeddings.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        # Choose device (GPU if available, otherwise CPU) 
        self.device = torch.device(device) if device else choose_device()

        # Allow override via env; default to official NV-Embed-v2 model
        if model_name is None:
            model_name = os.environ.get("NVEMBED_MODEL_NAME", "nvidia/nv-embed-v2")
        self.model_name = model_name

        # Load NVEmbedModel via AutoModel; this uses modeling_nvembed.NVEmbedModel
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Move everything to device
        try:
            if self.device.type == "cuda":
                self.model = self.model.to(self.device, dtype=torch.float16)
            else:
                self.model = self.model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            # If GPU OOM, fall back to CPU
            self.model = self.model.to("cpu")
            self.device = torch.device("cpu")

        self.model.eval()

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 2,
        max_length: int = 2048,
    ) -> List[np.ndarray]:
        """
        Encode arbitrary texts into NV-Embed sentence embeddings.

        Returns:
            List[np.ndarray] where each array has shape (D,)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Use NVEmbedModel._do_encode for batching & pooling.
        # We request numpy output for easy downstream handling.
        embs = self.model._do_encode(
            prompts=list(texts),
            batch_size=batch_size,
            instruction="",
            max_length=max_length,
            num_workers=0,
            return_numpy=True,
        )  # (N, D) np.ndarray or torch.Tensor

        if isinstance(embs, torch.Tensor):
            embs = embs.detach().cpu().numpy()

        # L2-normalize row-wise
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        embs = embs / norms

        return [embs[i] for i in range(embs.shape[0])]

    @torch.inference_mode()
    def encode_spans(
        self,
        texts: Sequence[str],
        phrases: Sequence[str],
        *,
        occurrence_index: int = 0,
        max_length: int = 2048,
        batch_size: int = 1,  # kept for API compatibility, not used
        case_insensitive: bool = True,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Encode spans within each text by:

        1. Finding the phrase (label) in the raw string via regex.
        2. Tokenizing the *full* text (with EOS) using NV-Embed's tokenizer and
           getting `offset_mapping`.
        3. Mapping the character span of the phrase to token indices.
        4. Selecting a token window around that span if the text is longer than
           the model's context window.
        5. Calling NVEmbedModel.forward on that window, with a `pool_mask` that
           is 1 *only* on the label tokens and 0 elsewhere.

        The model thus sees the full window context, but the final embedding
        is pooled only over the label span.
        """
        flags = re.IGNORECASE if case_insensitive else 0
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(phrases, str):
            phrases = [phrases]
        assert len(texts) == len(phrases), "texts and phrases must have same length"

        # Determine effective maximum window length
        text_cfg = getattr(self.model.config, "text_config", None)
        model_max = None
        if text_cfg is not None:
            model_max = getattr(text_cfg, "max_position_embeddings", None)
        if not model_max or model_max <= 0:
            # fallback to tokenizer model max or provided max_length
            model_max = getattr(self.model.tokenizer, "model_max_length", max_length or 4096)
        window_limit = min(model_max, max_length) if max_length is not None else model_max

        span_vecs: List[np.ndarray] = []
        tokenizer = self.model.tokenizer  # set in NVEmbedModel.__init__

        for text, phrase in zip(texts, phrases):
            if not phrase:
                raise ValueError("Empty phrase for encode_spans")

            # 1) Find character span of the phrase in the text
            matches = list(re.finditer(re.escape(phrase), text, flags=flags))
            if not matches:
                raise ValueError(f"Phrase not found in text: {phrase!r}")
            m = matches[min(occurrence_index, len(matches) - 1)]
            c0, c1 = m.start(), m.end()

            # 2) Tokenize full text (+ EOS) with offsets
            # We mimic NVEmbed's encode: text + eos, add_special_tokens=True
            full_text = text + tokenizer.eos_token
            enc_full = tokenizer(
                full_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            input_ids_full = enc_full["input_ids"].to(self.device)          # (1, T)
            attn_full = enc_full["attention_mask"].to(self.device)          # (1, T)
            offsets_full = enc_full["offset_mapping"][0].tolist()           # [(start, end), ...]

            # 3) Map char-span -> token indices for the label
            span_token_idxs = [
                i
                for i, (a, b) in enumerate(offsets_full)
                if not (a == b == 0) and max(a, c0) < min(b, c1)
            ]
            if not span_token_idxs:
                raise ValueError("Span not covered by tokenization for NV-Embed")

            span_min = min(span_token_idxs)
            span_max = max(span_token_idxs)
            span_len = span_max - span_min + 1

            # 4) Select window around the span (respecting model_max & max_length)
            window_size = min(window_limit, max(span_len, 1))
            half = max((window_size - span_len) // 2, 0)

            start_idx = max(0, span_min - half)
            end_idx = start_idx + window_size
            T_full = input_ids_full.shape[1]

            if end_idx > T_full:
                end_idx = T_full
                start_idx = max(0, end_idx - window_size)

            # ensure the window still fully contains the span
            if start_idx > span_min or end_idx <= span_max:
                start_idx = max(0, span_min)
                end_idx = min(T_full, span_min + window_size)

            input_ids = input_ids_full[:, start_idx:end_idx]   # (1, W)
            attn_mask = attn_full[:, start_idx:end_idx]        # (1, W)

            # 5) Build pool_mask: 1 only on label tokens, 0 elsewhere
            local_span_idxs = [
                i - start_idx
                for i in span_token_idxs
                if start_idx <= i < end_idx
            ]
            if not local_span_idxs:
                raise ValueError("Span not fully covered in selected window for NV-Embed")

            pool_mask = torch.zeros_like(attn_mask)
            for li in local_span_idxs:
                pool_mask[0, li] = 1

            # Call NVEmbedModel.forward directly with label-specific pool_mask
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                pool_mask=pool_mask,
                return_dict=True,
            )
            vec = out["sentence_embeddings"].squeeze(0)  # (D,)

            # L2-normalize (NVEmbed may already normalize, but keep consistent)
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
            span_vecs.append(vec.detach().float().cpu().numpy())

        return span_vecs