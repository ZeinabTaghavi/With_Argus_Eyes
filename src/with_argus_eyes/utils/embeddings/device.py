from __future__ import annotations
import torch

def choose_device(min_free_gb: float = 1.0) -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        best_idx, best_free = None, -1
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except Exception:
                free = 0
            if free > best_free:
                best_free, best_idx = free, i
        if best_idx is not None and best_free >= int(min_free_gb * 1024 ** 3):
            torch.cuda.set_device(best_idx)
            torch.backends.cudnn.benchmark = True
            return torch.device(f"cuda:{best_idx}")
    return torch.device("cpu")