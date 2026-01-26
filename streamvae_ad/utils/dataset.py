from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

class ReconstructDataset(Dataset):
    """
    Sliding-window dataset for reconstruction/likelihood-based AD.
    Returns:
      x_win: [win, F]
      dummy: 0 (placeholder, kept for compatibility)
    """
    def __init__(self, data: np.ndarray, window_size: int = 100):
        super().__init__()
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        self.data = data
        self.window_size = int(window_size)
        self.n = len(self.data)
        self.m = max(0, self.n - self.window_size + 1)

    def __len__(self):
        return self.m

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        return torch.from_numpy(x), 0
