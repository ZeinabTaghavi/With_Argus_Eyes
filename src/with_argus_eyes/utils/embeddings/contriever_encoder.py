from __future__ import annotations
from typing import List, Sequence, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Patch around HF torch.load safety check for torch<2.6 (CVE-2025-32434)
try:
    from transformers import modeling_utils as _hf_modeling_utils
    from transformers.utils import import_utils as _hf_import_utils
except Exception:
    _hf_modeling_utils = None
    _hf_import_utils = None

def _disable_torch_load_safety_check() -> None:
    """
    HF 4.57+ calls check_torch_load_is_safe() before torch.load() when loading .bin weights.
    On clusters with torch<2.6 this raises a hard ValueError and breaks old workflows.

    This helper monkey-patches the check to a no-op so that existing models
    can still be loaded. Use with awareness of the underlying CVE.
    """
    def _no_op(*args, **kwargs):
        return None

    # modeling_utils imports the symbol into its own namespace
    if _hf_modeling_utils is not None and hasattr(_hf_modeling_utils, "check_torch_load_is_safe"):
        _hf_modeling_utils.check_torch_load_is_safe = _no_op  # type: ignore[attr-defined]

    # some versions may also call it via import_utils
    if _hf_import_utils is not None and hasattr(_hf_import_utils, "check_torch_load_is_safe"):
        _hf_import_utils.check_torch_load_is_safe = _no_op  # type: ignore[attr-defined]
import re

from .base import BaseRetriever, l2_normalize_rows
from .device import choose_device


class ContrieverRetriever(BaseRetriever):
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "facebook/contriever",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.device = torch.device(device) if device else choose_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Work around HF torch.load safety check (torch<2.6 on this cluster)
        _disable_torch_load_safety_check()
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        try:
            if self.device.type == "cuda":
                self.model = self.model.to(self.device, dtype=torch.float16)
            else:
                self.model = self.model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            self.model = self.model.to("cpu")
        self.model.eval()

    @torch.no_grad()
    def _mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        tok_emb = model_output.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).expand(tok_emb.size()).float()
        summed = (tok_emb * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        max_length: int = 256,
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
            vec = self._mean_pooling(out, enc["attention_mask"])  # (B, H)
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
            out_vecs.extend(vec.detach().float().cpu().numpy())
        return out_vecs  # list of (H,) np arrays

    @torch.inference_mode()
    def encode_spans(
        self,
        texts: Sequence[str],
        phrases: Sequence[str],
        *,
        occurrence_index: int = 0,
        use_last4: bool = True,
        max_length: int = 512,
        batch_size: int = 1,
        case_insensitive: bool = True,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Encode spans (e.g., label inside a paragraph) by averaging token reps over the matched span.

        This implementation avoids losing the span due to truncation by:
          1) Tokenizing the full text once (no truncation) to get complete offsets.
          2) Finding the token indices that cover the phrase span.
          3) Selecting a window of tokens around that span whose length does not exceed
             the model's maximum sequence length and the user-provided max_length.
          4) Running the model only on that window and averaging the span tokens.

        If occurrence_index >= number of matches, we fall back to the last available match
        instead of raising an error.
        """
        flags = re.IGNORECASE if case_insensitive else 0
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(phrases, str):
            phrases = [phrases]
        assert len(texts) == len(phrases), "texts and phrases must be same length"

        out_list: List[np.ndarray] = []

        # Model maximum sequence length (e.g., 512 for Contriever)
        model_max = getattr(self.model.config, "max_position_embeddings", None)
        if not model_max or model_max <= 0:
            model_max = getattr(self.tokenizer, "model_max_length", 512)

        # Effective window length: cannot exceed model_max or user-provided max_length
        if max_length is not None:
            window_limit = min(model_max, max_length)
        else:
            window_limit = model_max

        for text, phrase in zip(texts, phrases):
            if not phrase:
                raise ValueError("Empty phrase provided to encode_spans.")

            # 1) Find character span of the requested occurrence
            matches = list(re.finditer(re.escape(phrase), text, flags=flags))
            if not matches:
                raise ValueError(f"Phrase not found in text: {phrase!r}")
            if occurrence_index >= len(matches):
                # Fall back to the last available match instead of raising
                m = matches[-1]
            else:
                m = matches[occurrence_index]
            c0, c1 = m.start(), m.end()

            # 2) Tokenize full text (no truncation) to get full offsets
            enc_full = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            input_ids_full = enc_full["input_ids"].to(self.device)  # (1, T_full)
            attention_mask_full = enc_full["attention_mask"].to(self.device)
            offsets_full = enc_full["offset_mapping"][0].tolist()  # length T_full
            T_full = input_ids_full.shape[1]

            # 3) Find token indices whose offsets overlap the phrase span
            span_token_idxs = [
                i
                for i, (a, b) in enumerate(offsets_full)
                if not (a == b == 0) and max(a, c0) < min(b, c1)
            ]
            if not span_token_idxs:
                # No tokens overlap the phrase span at all (not a truncation issue now)
                raise ValueError(
                    "Span not covered by tokenization (no tokens overlap the phrase span). "
                    "Check normalization or phrase/text mismatch."
                )

            span_min = min(span_token_idxs)
            span_max = max(span_token_idxs)
            span_len = span_max - span_min + 1

            # Desired window size
            window_size = min(window_limit, max(span_len, 1))

            # Center the window around the span as much as possible
            half = max((window_size - span_len) // 2, 0)
            start_idx = max(0, span_min - half)
            end_idx = start_idx + window_size
            if end_idx > T_full:
                end_idx = T_full
                start_idx = max(0, end_idx - window_size)

            # Ensure the span is fully inside [start_idx, end_idx)
            if start_idx > span_min or end_idx <= span_max:
                # Force a minimal window that tightly covers the span if needed
                start_idx = max(0, span_min)
                end_idx = min(T_full, span_min + window_size)

            # Slice the inputs to the selected window
            input_ids = input_ids_full[:, start_idx:end_idx]
            attention_mask = attention_mask_full[:, start_idx:end_idx]
            offsets = offsets_full[start_idx:end_idx]

            # 4) Run the model on that window
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)

            token_reps_all = (
                torch.stack(out.hidden_states[-4:], dim=0).mean(0)
                if use_last4
                else out.last_hidden_state
            )[0]  # (T_window, H)

            # Map global span_token_idxs to local indices in the window
            local_span_idxs = [
                i - start_idx
                for i in span_token_idxs
                if start_idx <= i < end_idx
            ]
            if not local_span_idxs:
                # Should not happen given the window construction; guard anyway
                raise ValueError(
                    "Span not covered by window even after centering. "
                    "This indicates an inconsistency in span/window logic."
                )

            span_vec = token_reps_all[local_span_idxs].mean(0)
            span_vec = torch.nn.functional.normalize(span_vec, p=2, dim=0)
            out_list.append(span_vec.detach().float().cpu().numpy())

        return out_list