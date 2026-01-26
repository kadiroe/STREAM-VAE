from __future__ import annotations
import copy
import torch

def get_gpu(cuda: bool = True) -> torch.device:
    """
    Returns best available device.
    Preference: CUDA > MPS > CPU
    """
    if cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class EarlyStoppingTorch:
    """
    Simple early-stopping helper that keeps best (lowest) loss in-memory.
    Compatible with call pattern: early_stopping(val_loss, model)
    """
    def __init__(self, save_path=None, patience: int = 10, min_delta: float = 0.0):
        self.save_path = save_path
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module):
        score = float(val_loss)
        if self.best is None or score < (self.best - self.min_delta):
            self.best = score
            self.counter = 0
            self.best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if self.save_path:
                torch.save(self.best_state, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model: torch.nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state, strict=True)
