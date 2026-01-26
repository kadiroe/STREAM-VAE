from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch

from streamvae_ad.models.streamvae import StreamVAE
from streamvae_ad.utils.seed import seed_everything, dataloader_worker_init_fn

def load_csv(path: str):
    df = pd.read_csv(path).dropna()
    if "Label" in df.columns:
        y = df["Label"].astype(int).to_numpy()
        X = df.drop(columns=["Label"]).to_numpy(dtype=float)
    else:
        y = None
        X = df.to_numpy(dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="datasets/TSB-AD-M",
                    help="CSV with columns [features..., Label] or just features.")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--deterministic", action="store_true", help="Enable best-effort deterministic mode.")
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--win_size", type=int, default=100)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--validation_size", type=float, default=0.2)
    ap.add_argument("--target_kl", type=float, default=100.0)
    ap.add_argument("--event_l1_weight", type=float, default=1e-3)
    args = ap.parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)

    X, y = load_csv(args.csv)
    model = StreamVAE(
        win_size=args.win_size,
        feats=X.shape[1],
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        validation_size=args.validation_size,
        target_kl=args.target_kl,
        event_l1_weight=args.event_l1_weight,
    )

    # Patch DataLoader workers seeding via global torch generator
    # (StreamVAE internally constructs DataLoaders; we seed globally here.)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    model.fit(X)
    scores = model.decision_function(X)
    print(f"Scores shape: {scores.shape}, min={scores.min():.4g}, max={scores.max():.4g}")

    if y is not None:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            print(f"AUC-ROC: {roc_auc_score(y, scores):.4f}")
            print(f"AUC-PR : {average_precision_score(y, scores):.4f}")
        except Exception as e:
            print("Could not compute sklearn metrics:", e)

if __name__ == "__main__":
    main()
