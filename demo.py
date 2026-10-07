#!/usr/bin/env python3
"""
demo.py

Single-command live terminal demonstration for the Soft Computing project.
Executes live evaluations of the baseline and all four 500-feature models in ~6 seconds.

Usage:
    source .venv/bin/activate
    python demo.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score


def main():
    print("\n" + "=" * 84)
    print("      SOFT COMPUTING: MULTI-OBJECTIVE EVOLUTIONARY FEATURE SELECTION")
    print("                     LIVE MODEL DEMONSTRATION & EVALUATION")
    print("=" * 84)

    # 1. Load Data
    print("\n[1/5] Loading Processed Numerical Matrices & Vocabulary...")
    t0 = time.time()
    X_train = sparse.load_npz("X_train.npz").tocsc()
    X_val = sparse.load_npz("X_val.npz").tocsc()
    y_train = np.load("y_train.npy")
    y_val = np.load("y_val.npy")
    with open("feature_names.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
        
    print(f"  ✓ Loaded in {time.time() - t0:.2f}s | Train: {X_train.shape[0]:,} × {X_train.shape[1]:,} | Val: {X_val.shape[0]:,} × {X_val.shape[1]:,}")
    print(f"  ✓ Class balance: {np.mean(y_val)*100:.2f}% positive toxic comments")

    # 2. Inspect Sample CSV
    sample_csv = "processed_data/processed_sample.csv"
    if os.path.exists(sample_csv):
        df_sample = pd.read_csv(sample_csv)
        print(f"\n[2/5] Inspecting Numerical CSV Export ('{sample_csv}'):")
        print(f"  ✓ Verified shape: {df_sample.shape[0]} rows × {df_sample.shape[1]:,} columns")
        print(f"  ✓ Feature naming: '{df_sample.columns[0]}' ... '{df_sample.columns[-2]}'")
        print(f"  ✓ Label column:   '{df_sample.columns[-1]}' (unique values: {sorted(df_sample['target'].unique())})")

    # 3. Live Baseline Evaluation (10,000 features)
    print("\n[3/5] Live Execution of Baseline LinearSVC (All 10,000 Features)...")
    t0 = time.perf_counter()
    clf_base = LinearSVC(class_weight="balanced", C=1.0, random_state=42, dual="auto")
    clf_base.fit(X_train, y_train)
    t_fit_base = time.perf_counter() - t0
    
    _ = clf_base.predict(X_val) # warmup
    runs = [time.perf_counter() for _ in range(5)]
    durations = []
    for _ in range(5):
        t_p = time.perf_counter()
        y_pred_base = clf_base.predict(X_val)
        durations.append(time.perf_counter() - t_p)
    inf_base_ms = np.mean(durations) * 1000.0
    
    cm_base = confusion_matrix(y_val, y_pred_base)
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    f1_base = f1_score(y_val, y_pred_base)
    fpr_base = fp_b / (fp_b + tn_b)
    print(f"  ✓ Baseline Trained in {t_fit_base:.2f}s | F1 = {f1_base:.4f} | FPR = {fpr_base*100:.2f}% | Latency = {inf_base_ms:.3f} ms")

    # 4. Live Retraining of 4 Final 500-Feature Models
    print("\n[4/5] Live Retraining of 4 Evolutionary 500-Feature Models (95% Reduction)...")
    algos = ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]
    file_keys = {"NSGA-II": "nsga2", "SPEA2": "spea2", "MOEA/D": "moead", "MODE": "mode"}
    
    live_results = []
    t_eval_start = time.time()
    
    for algo in algos:
        key = file_keys[algo]
        indices = np.load(f"selected_features_{key}_balanced.npy")
        assert len(indices) == 500
        
        t0 = time.perf_counter()
        clf = LinearSVC(class_weight="balanced", C=1.0, random_state=42, dual="auto")
        clf.fit(X_train[:, indices], y_train)
        t_fit = time.perf_counter() - t0
        
        _ = clf.predict(X_val[:, indices]) # warmup
        runs = []
        for _ in range(5):
            t_p = time.perf_counter()
            y_pred = clf.predict(X_val[:, indices])
            runs.append(time.perf_counter() - t_p)
        inf_ms = np.mean(runs) * 1000.0
        
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        f1 = f1_score(y_val, y_pred)
        fpr = fp / (fp + tn)
        
        live_results.append({
            "Algorithm": algo,
            "Features": 500,
            "F1 Score": f1,
            "FPR": fpr,
            "Inference Time": inf_ms,
            "Training Time": t_fit
        })
        print(f"  ✓ {algo:7s} [500 Feat]: F1 = {f1:.4f} | FPR = {fpr*100:5.2f}% | Latency = {inf_ms:5.3f} ms | Fit = {t_fit:4.2f}s")
        
    print(f"  ✓ All 4 models trained and evaluated in {time.time() - t_eval_start:.2f}s total!")

    # 5. Display Authoritative Summary Table
    print("\n[5/5] Final Audited Comparison Table (From final_algorithm_comparison.csv):")
    if os.path.exists("final_algorithm_comparison.csv"):
        df_comp = pd.read_csv("final_algorithm_comparison.csv")
        # Format table print
        cols_to_show = ["Algorithm", "Features", "Balanced F1", "Balanced FPR", "Balanced Inference Time", "Final Pareto Solutions", "Hypervolume"]
        print(df_comp[cols_to_show].to_string(index=False))
    
    print("\n" + "=" * 84)
    print("CONCLUSION:")
    print("  • OVERALL WINNER: NSGA-II")
    print("    - Closest to ideal point (d* = 0.2096), highest hypervolume (0.00533716), and 14/26 global Pareto solutions.")
    print("  • SPECIALIST: SPEA2")
    print("    - Achieved FPR = 6.58%, OUTPERFORMING the 10,000-feature baseline (6.81%).")
    print("  • SPECIALIST: MOEA/D")
    print("    - Achieved peak F1 = 0.6446 (retaining 94.1% of baseline F1).")
    print("  • DIMENSIONALITY REDUCTION: 95.0% feature reduction cuts latency by up to 62.5%.")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
