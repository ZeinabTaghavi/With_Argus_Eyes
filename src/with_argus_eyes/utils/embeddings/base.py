from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence
import numpy as np

def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return x / n

class BaseRetriever(ABC):
    """
    Common interface for all retrievers.
    - encode_texts(texts) -> List[np.ndarray]
    - encode_spans(texts, phrases, ...) -> List[np.ndarray]  (optional; default falls back to encode_texts)
    All encoders MUST return L2-normalized embeddings.
    """

    def __init__(self, device: Optional[str] = None, **kwargs):
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def encode_texts(self, texts: Sequence[str], batch_size: int = 32, max_length: int = 256) -> List[np.ndarray]:
        ...

    def encode_spans(
        self,
        texts: Sequence[str],
        phrases: Sequence[str],
        *,
        occurrence_index: int = 0,
        batch_size: int = 8,
        max_length: int = 512,
        **kwargs,
    ) -> List[np.ndarray]:
        # Default: ignore spans and just encode full texts
        return self.encode_texts(texts, batch_size=batch_size, max_length=max_length)