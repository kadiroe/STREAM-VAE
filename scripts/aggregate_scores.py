#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate aggregated metrics CSV from saved .npy score files.
"""
import argparse
import pathlib
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.model_wrapper import find_length_rank
except ImportError as e:
    raise ImportError("TSB-AD is required. Install with: pip install TSB-AD") from e

TR_RE = re.compile(r"_tr_(\d+)_", re.IGNORECASE)


def load_series(csv_path: str):
    """Load data and labels from CSV file."""
    df = pd.read_csv(csv_path).dropna()
    if "Label" not in df.columns:
        raise ValueError(f"Expected 'Label' column in {csv_path}")
    label = df["Label"].astype(int).to_numpy()
    data = df.iloc[:, 0:-1].values.astype(float)
    if data.ndim == 1:
        data = data[:, None]
    return data, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate saved .npy scores into metrics CSV")
    parser.add_argument("--dataset_dir", type=str, default="datasets/TSB-AD-M/",
                        help="Folder containing original CSV files with labels")
    parser.add_argument("--score_dir", type=str, default="eval/score/multi/StreamVAE/",
                        help="Folder containing saved .npy score files")
    parser.add_argument("--output_csv", type=str, default="eval/metrics/multi/StreamVAE.csv",
                        help="Output CSV file for aggregated metrics")
    parser.add_argument("--win_size", type=int, default=100,
                        help="Fallback window size if find_length_rank fails")
    
    args = parser.parse_args()
    
    score_dir = pathlib.Path(args.score_dir)
    dataset_dir = pathlib.Path(args.dataset_dir)
    output_path = pathlib.Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all .npy files
    npy_files = sorted(score_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {score_dir}")
        exit(1)
    
    print(f"Found {len(npy_files)} .npy score files")
    
    results = []
    
    for npy_path in tqdm(npy_files, desc="Computing metrics"):
        # Get corresponding CSV filename
        stem = npy_path.stem
        csv_filename = f"{stem}.csv"
        csv_path = dataset_dir / csv_filename
        
        if not csv_path.exists():
            print(f"Warning: CSV not found for {stem}, skipping...")
            continue
        
        try:
            # Load scores
            scores = np.load(npy_path)
            
            # Load labels and data
            data, label = load_series(str(csv_path))
            
            # Check length match
            if len(scores) != len(label):
                print(f"Warning: Length mismatch for {csv_filename} (scores={len(scores)}, labels={len(label)}), skipping...")
                continue
            
            # Compute sliding window using find_length_rank
            try:
                slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
            except Exception:
                slidingWindow = args.win_size
            
            # Compute metrics
            metrics = get_metrics(scores, label, slidingWindow=slidingWindow)
            
            # Store result
            row = {"file": csv_filename, **{k: float(v) for k, v in metrics.items()}}
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved aggregated metrics to: {output_path}")
        print(f"  Total files processed: {len(results)}")
        
        # Show summary statistics
        print("\nSummary statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numeric_cols].describe())
    else:
        print("No results to save.")
