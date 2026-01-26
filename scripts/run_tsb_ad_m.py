# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


from streamvae_ad.models.streamvae import StreamVAE
from streamvae_ad.utils.seed import seed_everything


try:
    from TSB_AD.evaluation.metrics import get_metrics
except Exception as e:
    raise ImportError(
        "TSB-AD is required for this runner (metrics + length-rank + wrapper). "
        "Install it with: pip install TSB-AD"
    ) from e

try:
    from TSB_AD.model_wrapper import (
        find_length_rank,
        Semisupervise_AD_Pool,
        Unsupervise_AD_Pool,
        run_Semisupervise_AD,
        run_Unsupervise_AD,
    )
except Exception as e:
    raise ImportError(
        "Could not import TSB_AD.model_wrapper.* (needed for find_length_rank and wrapper pools). "
        "Make sure TSB-AD is installed correctly."
    ) from e

try:
    from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
except Exception:
    Optimal_Multi_algo_HP_dict = {}

TR_RE = re.compile(r"_tr_(\d+)_", re.IGNORECASE)


def infer_train_len_from_name(filename: str, n: int) -> int:
    """
    Prefer parsing standard TSB-AD filename pattern: *_tr_<N>_*.csv
    Fall back to 50% if not available.
    """
    m = TR_RE.search(filename)
    if m:
        tr = int(m.group(1))
        return max(1, min(tr, n))
    return max(1, n // 2)


def read_file_list(file_list_csv: str) -> list[str]:
    df = pd.read_csv(file_list_csv)
    if "file_name" not in df.columns:
        raise ValueError(f"Expected column 'file_name' in file list CSV: {file_list_csv}")
    return df["file_name"].astype(str).tolist()


def collect_all_csvs(dataset_dir: str) -> list[str]:
    p = pathlib.Path(dataset_dir)
    return [x.name for x in sorted(p.rglob("*.csv")) if x.is_file()]


def load_series(csv_path: str):
    df = pd.read_csv(csv_path).dropna()
    if "Label" not in df.columns:
        raise ValueError(f"Expected 'Label' column in {csv_path}")
    label = df["Label"].astype(int).to_numpy()
    data = df.iloc[:, 0:-1].values.astype(float)
    if data.ndim == 1:
        data = data[:, None]
    return data, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSB-AD-style benchmark runner (StreamVAE + wrapper support)")
    parser.add_argument("--dataset_dir", type=str, default="datasets/TSB-AD-M/",
                        help="Folder containing TSB-AD-M CSV files.")
    parser.add_argument("--file_list", type=str, default="datasets/TSB-AD-M-Eva.csv",
                        help="Optional: CSV file with column 'file_name' (e.g., TSB-AD-M-Eva.csv). "
                             "If omitted, runs on all CSVs found under dataset_dir.")
    parser.add_argument("--score_dir", type=str, default="eval/score/multi/",
                        help="Where to save per-file anomaly scores as .npy.")
    parser.add_argument("--save_dir", type=str, default="eval/metrics/multi/",
                        help="Where to save aggregated per-file metrics as a CSV.")
    parser.add_argument("--save", action="store_true", help="Write metrics CSV (default off unless set).")

    parser.add_argument("--AD_Name", type=str, default="StreamVAE",
                        help="Detector name. Use 'StreamVAE' for your method, or any name supported by TSB-AD wrapper.")

    # Seeding / determinism
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--deterministic", action="store_true",
                        help="Best-effort deterministic mode (can slow down).")

    # StreamVAE hyperparams (used only when AD_Name == StreamVAE)
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=100.0)
    parser.add_argument("--event_l1_weight", type=float, default=1e-3)

    args = parser.parse_args()

    # --- seeding ---
    seed_everything(args.seed, deterministic=args.deterministic)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("CUDA available:", torch.cuda.is_available())
    print("cuDNN version:", torch.backends.cudnn.version())

    # --- dirs / logging ---
    models_dir = pathlib.Path("saved_models") / args.AD_Name
    models_dir.mkdir(parents=True, exist_ok=True)

    target_dir = pathlib.Path(args.score_dir) / args.AD_Name
    target_dir.mkdir(parents=True, exist_ok=True)

    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(target_dir / f"000_run_{args.AD_Name}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # --- file list ---
    if args.file_list and os.path.exists(args.file_list):
        file_list = read_file_list(args.file_list)
    else:
        file_list = collect_all_csvs(args.dataset_dir)

    # --- HP dict (only used for wrapper models) ---
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict.get(args.AD_Name, {})

    save_path = pathlib.Path(args.save_dir) / f"{args.AD_Name}.csv"
    already_existing = pd.read_csv(save_path) if save_path.exists() else None

    write_rows = []
    start_all = time.time()

    for filename in tqdm(file_list, desc=f"Processing {args.AD_Name}"):
        stem = filename.split(".")[0]
        score_path = target_dir / f"{stem}.npy"
        if score_path.exists():
            continue

        file_path = os.path.join(args.dataset_dir, filename)
        if not os.path.exists(file_path):
            logging.error(f"Missing file: {file_path}")
            continue

        print(f"Processing: {filename} by {args.AD_Name}")

        try:
            data, label = load_series(file_path)
        except Exception as e:
            logging.error(f"Failed to load {filename}: {e}")
            continue

        n = len(label)
        feats = data.shape[1]

        # --- length-rank window (TSB-AD protocol) ---
        try:
            slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        except Exception:
            slidingWindow = args.win_size  # fallback

        # --- train length ---
        train_len = infer_train_len_from_name(filename, n)
        data_train = data[:train_len, :]

        eps = 1e-8
        mu = data_train.mean(axis=0, keepdims=True)
        sd = data_train.std(axis=0, keepdims=True)
        sd = np.where(sd == 0, eps, sd)

        data_n = (data - mu) / sd
        data_train_n = data_n[:train_len, :]

        model_obj = None
        t0 = time.time()

        try:
            # --- dispatch ---
            if args.AD_Name in Semisupervise_AD_Pool:
                # TSB-AD wrapper (semi-supervised = train segment given)
                output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)

            elif args.AD_Name in Unsupervise_AD_Pool:
                # TSB-AD wrapper (unsupervised = no explicit train segment)
                output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)

            elif args.AD_Name == "StreamVAE":
                # Your method (semi-supervised in the same sense: fit on data_train)
                det = StreamVAE(
                    win_size=args.win_size,
                    feats=feats,
                    latent_dim=args.latent_dim,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    lr=args.lr,
                    validation_size=args.validation_size,
                    target_kl=args.target_kl,
                    event_l1_weight=args.event_l1_weight,
                )
                det.fit(data_train_n)
                output = det.decision_function(data_n)
                model_obj = getattr(det, "model", None)

            else:
                raise ValueError(f"{args.AD_Name} is not defined (not in wrapper pools and not StreamVAE).")

        except Exception as e:
            logging.error(f"Runtime error at {filename} using {args.AD_Name}: {e}")
            continue

        run_time = time.time() - t0

        # --- save scores ---
        if isinstance(output, np.ndarray):
            np.save(score_path, output)
            logging.info(f"Success at {filename} using {args.AD_Name} | Time: {run_time:.3f}s | len={n}")
        else:
            logging.error(f"Non-numpy output at {filename}: {output}")
            continue

        # --- save model weights (optional) ---
        if model_obj is not None and hasattr(model_obj, "state_dict"):
            try:
                torch.save(model_obj.state_dict(), models_dir / f"{stem}_{args.AD_Name}.pt")
            except Exception as e:
                logging.error(f"Failed to save model for {filename}: {e}")

        # --- metrics ---
        if args.save:
            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                row = {"file": filename, "Time": run_time, **{k: float(v) for k, v in evaluation_result.items()}}
            except Exception as e:
                logging.error(f"Metric error at {filename}: {e}")
                # keep row with zeros if metrics fail
                row = {"file": filename, "Time": run_time}
            write_rows.append(row)

            # temp save (append to existing)
            w_csv = pd.DataFrame(write_rows)
            if already_existing is not None:
                w_csv = pd.concat([already_existing, w_csv], ignore_index=True)
            w_csv.to_csv(save_path, index=False)

    total_time = time.time() - start_all
    print(f"\nDone. Total time: {total_time/60:.2f} min")
    if args.save and save_path.exists():
        print(f"Wrote metrics CSV: {save_path}")
