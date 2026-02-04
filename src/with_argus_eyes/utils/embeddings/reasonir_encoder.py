from __future__ import annotations
from typing import List, Sequence, Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import re

from .base import BaseRetriever, l2_normalize_rows
from .device import choose_device

class ReasonIRRetriever(BaseRetriever):
    def __init__(self, device: Optional[str] = None, instruction: str = "", **kwargs):
        super().__init__(device=device, **kwargs)
        self.device = torch.device(device) if device else choose_device()
        self.instruction = instruction
        self.model = AutoModel.from_pretrained("reasonir/ReasonIR-8B", trust_remote_code=True, device_map="auto")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("reasonir/ReasonIR-8B", trust_remote_code=True)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 64, max_length: int = 256) -> List[np.ndarray]:
        if isinstance(texts, str): texts = [texts]
        outs: List[np.ndarray] = []
        # ReasonIR supports a batched .encode, but fall back per-item if needed
        i = 0
        while i < len(texts):
            batch = texts[i:i+batch_size]
            try:
                v = self.model.encode(batch, instruction=self.instruction)  # type: ignore
                arr = v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)
                arr = l2_normalize_rows(arr)
                outs.extend(list(arr))
                i += len(batch)
            except Exception:
                for t in batch:
                    v = self.model.encode(t, instruction=self.instruction)
                    vv = v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)
                    vv = vv[None, :] if vv.ndim == 1 else vv
                    vv = l2_normalize_rows(vv)[0]
                    outs.append(vv)
                i += len(batch)
        return outs

        @torch.inference_mode()
        def encode_spans(
            self,
            texts: Sequence[str],
            phrases: Sequence[str],
            *,
            occurrence_index: int = 0,
            max_length: int = 2048,
            batch_size: int = 1,
            case_insensitive: bool = True,
            **kwargs,
        ) -> List[np.ndarray]:
            """
            Encode spans (e.g., label inside a paragraph) by averaging token reps over the matched span.

            This mirrors the ContrieverRetriever.encode_spans interface, but uses the ReasonIR-8B model.
            We use the last_hidden_state only (ReasonIR is already instruction-tuned at the model level).
            """
            flags = re.IGNORECASE if case_insensitive else 0
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(phrases, str):
                phrases = [phrases]
            assert len(texts) == len(phrases), "texts and phrases must be same length"

            out_list: List[np.ndarray] = []

            # Figure out where the model’s parameters live (e.g. cuda:0 with device_map="auto")
            model_device = next(self.model.parameters()).device

            for text, phrase in zip(texts, phrases):
                if not phrase:
                    raise ValueError("Empty phrase provided to encode_spans.")
                matches = list(re.finditer(re.escape(phrase), text, flags=flags))
                if not matches:
                    raise ValueError(f"Phrase not found in text: {phrase!r}")
                if occurrence_index >= len(matches):
                    raise ValueError(
                        f"occurrence_index {occurrence_index} out of range (found {len(matches)} matches)."
                    )
                m = matches[occurrence_index]
                c0, c1 = m.start(), m.end()

                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                offsets = enc["offset_mapping"][0].tolist()

                # *** FIX: move inputs to the same device as the model ***
                input_ids = input_ids.to(model_device)
                attention_mask = attention_mask.to(model_device)

                # With device_map='auto', HF will handle sharding internally, but inputs
                # must still start on one of the model's devices (not CPU).
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Use token representations from last_hidden_state
                token_reps = out.last_hidden_state[0]  # (T, H)

                idxs = [
                    i
                    for i, (a, b) in enumerate(offsets)
                    if not (a == b == 0) and max(a, c0) < min(b, c1)
                ]
                if not idxs:
                    raise ValueError(
                        "Span not covered by window (likely truncated). "
                        "Shorten text or move phrase earlier."
                    )

                span_vec = token_reps[idxs].mean(0)  # (H,)
                span_vec = torch.nn.functional.normalize(span_vec, p=2, dim=0)
                out_list.append(span_vec.detach().float().cpu().numpy())

            return out_list