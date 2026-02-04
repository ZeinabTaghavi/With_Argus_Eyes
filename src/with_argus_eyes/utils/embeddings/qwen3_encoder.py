from __future__ import annotations
from typing import List, Sequence, Optional
import numpy as np
import torch
import re
from transformers import AutoModel, AutoTokenizer

from .base import BaseRetriever, l2_normalize_rows
from .device import choose_device

class Qwen3Retriever(BaseRetriever):
    def __init__(self, device: Optional[str] = None, instruction: str = "", **kwargs):
        super().__init__(device=device, **kwargs)
        self.device_str = device if device else choose_device()
        self.instruction = instruction
        
        print(f"[Qwen3Retriever] Loading model 'Qwen/Qwen3-Embedding-8B' with device_map='auto'...")
        
        # 1. Load Model with Auto-Sharding & bfloat16
        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-8B", 
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-8B", 
            trust_remote_code=True
        )

    @torch.no_grad()
    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool token representations over non-masked positions.
        """
        # 1. Cast to float32 for summation precision
        last_hidden_state = last_hidden_state.float()

        # 2. SAFETY NET: Replace NaNs or Infs with 0.0 to prevent crash
        last_hidden_state = torch.nan_to_num(last_hidden_state, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Standard mean pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        
        return summed / counts

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 16, max_length: int = 512) -> List[np.ndarray]:
        if isinstance(texts, str): texts = [texts]
        outs: List[np.ndarray] = []
        
        # Identify the correct device for inputs (first layer's device)
        input_device = next(self.model.parameters()).device

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            enc = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            )
            
            # Move inputs to the correct entry device
            enc = {k: v.to(input_device) for k, v in enc.items()}

            # Run Model with bfloat16 autocast
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(**enc)

            v = self._mean_pooling(out.last_hidden_state, enc["attention_mask"])
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            outs.extend(v.detach().float().cpu().numpy())
            
        return outs

    @torch.inference_mode()
    def encode_spans(
        self,
        texts: Sequence[str],
        phrases: Sequence[str],
        *,
        occurrence_index: int = 0,
        max_length: int = 512,
        batch_size: int = 1,  # Keep as 1 for span alignment safety
        case_insensitive: bool = True,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Encode spans using the same safe logic (device_map aware + bfloat16).
        """
        flags = re.IGNORECASE if case_insensitive else 0
        if isinstance(texts, str): texts = [texts]
        if isinstance(phrases, str): phrases = [phrases]
        assert len(texts) == len(phrases), "texts and phrases must be same length"

        out_list: List[np.ndarray] = []
        input_device = next(self.model.parameters()).device # Get correct device

        for text, phrase in zip(texts, phrases):
            # 1. Find the phrase in text
            if not phrase:
                # Fallback for empty phrase
                out_list.append(np.zeros(self.model.config.hidden_size, dtype=np.float32))
                continue
                
            matches = list(re.finditer(re.escape(phrase), text, flags=flags))
            if not matches:
                 # Fallback for missing phrase (prevents crash)
                print(f"Warning: Phrase '{phrase}' not found. Returning zero vector.")
                out_list.append(np.zeros(self.model.config.hidden_size, dtype=np.float32))
                continue

            # Select occurrence
            m = matches[-1] if occurrence_index >= len(matches) else matches[occurrence_index]
            c0, c1 = m.start(), m.end()

            # 2. Tokenize
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            
            # Move to correct device
            input_ids = enc["input_ids"].to(input_device)
            attention_mask = enc["attention_mask"].to(input_device)
            offsets = enc["offset_mapping"][0].tolist()

            # 3. Identify tokens corresponding to the span
            idxs = [
                i for i, (a, b) in enumerate(offsets)
                if not (a == b == 0) and max(a, c0) < min(b, c1)
            ]
            
            if not idxs:
                # Fallback if tokenizer swallowed the span (e.g. truncated)
                out_list.append(np.zeros(self.model.config.hidden_size, dtype=np.float32))
                continue

            # 4. Run Model (SAFE MODE: bfloat16)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 5. Extract and Average Span Tokens
            token_reps = out.last_hidden_state[0] # (T, H)
            
            # Cast to float32 before averaging to prevent overflow
            span_vec = token_reps[idxs].float().mean(0)
            
            # Sanitize just in case
            span_vec = torch.nan_to_num(span_vec, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize
            span_vec = torch.nn.functional.normalize(span_vec, p=2, dim=0)
            out_list.append(span_vec.detach().cpu().numpy())

        return out_list

    # Aliases
    def _encode_texts(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        return np.array(self.encode_texts(texts, **kwargs))
    
    def _encode_batch(self, batch: Sequence[str]) -> np.ndarray:
        return np.array(self.encode_texts(batch))