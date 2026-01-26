from __future__ import annotations
import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 2024, deterministic: bool = True):
    """
    Seed Python, NumPy and PyTorch. Optionally enable deterministic behavior.
    Notes:
      - Full determinism depends on platform/device and chosen ops.
      - SDPA/flash attention and some GPU kernels may still have nondeterminism.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism flags (best-effort)
    if deterministic:
        # cuDNN (CUDA)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        # Deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

        # CUDA cublas (helps determinism on some matmul paths)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    return seed

def dataloader_worker_init_fn(worker_id: int):
    """
    Worker init that derives a unique seed for each worker from the main PyTorch seed.
    """
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
