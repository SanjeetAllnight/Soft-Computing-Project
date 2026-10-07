#!/usr/bin/env python3
"""
Baseline Classifier for Soft Computing Multi-Objective Feature Selection Project.

Model: LinearSVC (class_weight='balanced', random_state=42, C=1.0)
Feature Space: All 10,000 TF-IDF features (from X_train.npz, X_val.npz)
Target: Binary is_toxic (from y_train.npy, y_val.npy)
"""

import os
import sys
import time
import json
import joblib
import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def run_baseline():
    print("=" * 80)
    print("             BASELINE CLASSIFIER EVALUATION (LinearSVC)")
    print("=" * 80)
    
    # 1. Load artifacts
    print("\n[Step 1] Loading preprocessed artifacts...")
    t0 = time.time()
    
    x_train_path = "X_train.npz"
    x_val_path = "X_val.npz"
    y_train_path = "y_train.npy"
    y_val_path = "y_val.npy"
    
    for path in [x_train_path, x_val_path, y_train_path, y_val_path]:
        if not os.path.exists(path):
            print(f"Error: Required artifact '{path}' not found!", file=sys.stderr)
            sys.exit(1)
            
    X_train = sparse.load_npz(x_train_path)
    X_val = sparse.load_npz(x_val_path)
    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)
    
    n_train_samples, n_features = X_train.shape
    n_val_samples = X_val.shape[0]
    
    print(f"  • Training samples:   {n_train_samples:,}")
    print(f"  • Validation samples: {n_val_samples:,}")
    print(f"  • Features used:      {n_features:,}")
    print(f"  • Artifacts loaded in {time.time() - t0:.2f}s.")
    
    # 2. Train baseline model
    print("\n[Step 2] Training LinearSVC baseline...")
    print("  Model configuration:")
    print("  • class_weight = 'balanced'")
    print("  • random_state = 42")
    print("  • C            = 1.0")
    
    t_train_start = time.time()
    clf = LinearSVC(class_weight="balanced", random_state=42, C=1.0)
    clf.fit(X_train, y_train)
    train_duration = time.time() - t_train_start
    print(f"  Model trained in {train_duration:.2f}s.")
    
    # 3. Inference timing (5 runs)
    print("\n[Step 3] Measuring inference time on validation set...")
    # Warm-up run
    _ = clf.predict(X_val)
    
    n_timing_runs = 5
    run_durations = []
    y_pred = None
    
    for i in range(n_timing_runs):
        t_start = time.perf_counter()
        y_pred = clf.predict(X_val)
        t_elapsed = time.perf_counter() - t_start
        run_durations.append(t_elapsed)
        print(f"  • Timing Run {i+1}: {t_elapsed*1000:.3f} ms")
        
    avg_total_inference_time_sec = float(np.mean(run_durations))
    std_total_inference_time_sec = float(np.std(run_durations))
    avg_time_per_comment_us = (avg_total_inference_time_sec / n_val_samples) * 1e6
    avg_time_per_comment_ms = (avg_total_inference_time_sec / n_val_samples) * 1e3
    
    # 4. Compute metrics
    print("\n[Step 4] Computing project objective metrics...")
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    
    # Core project objectives:
    # 1. F1 Score
    f1 = float(f1_score(y_val, y_pred))
    # 2. False-Negative Rate: FNR = FN / (FN + TP)
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    # 3. Precision, Recall, Accuracy
    precision = float(precision_score(y_val, y_pred))
    recall = float(recall_score(y_val, y_pred))
    accuracy = float(accuracy_score(y_val, y_pred))
    
    # 5. Display detailed results
    print("\n" + "=" * 80)
    print("                     BASELINE CLASSIFIER RESULTS")
    print("=" * 80)
    print(f"• Dataset / Split:")
    print(f"  - Training samples:       {n_train_samples:,}")
    print(f"  - Validation samples:     {n_val_samples:,}")
    print(f"  - Number of features:     {n_features:,} (100% of TF-IDF features)")
    print(f"\n• Primary Project Objectives:")
    print(f"  1. F1 Score:              {f1:.4f} ({f1*100:.2f}%)")
    print(f"  2. False-Negative Rate:   {fnr:.4f} ({fnr*100:.2f}%)")
    print(f"  3. Inference Time (X_val):")
    print(f"     - Total Time (avg):    {avg_total_inference_time_sec*1000:.3f} ms (±{std_total_inference_time_sec*1000:.3f} ms)")
    print(f"     - Per Comment:         {avg_time_per_comment_us:.3f} µs ({avg_time_per_comment_ms:.6f} ms)")
    print(f"\n• Additional Performance Metrics:")
    print(f"  - Accuracy:               {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Precision:              {precision:.4f} ({precision*100:.2f}%)")
    print(f"  - Recall (Sensitivity):   {recall:.4f} ({recall*100:.2f}%)")
    print(f"\n• Confusion Matrix (Validation Set):")
    print(f"                     Predicted Non-Toxic (0)    Predicted Toxic (1)")
    print(f"  Actual Non-Toxic (0):    {tn:>12,} (TN)           {fp:>12,} (FP)")
    print(f"  Actual Toxic (1):        {fn:>12,} (FN)           {tp:>12,} (TP)")

    # 6. Save model and results
    print("\n" + "-" * 80)
    print("SAVING BASELINE ARTIFACTS")
    print("-" * 80)
    model_path = "baseline_model.joblib"
    results_path = "baseline_results.json"
    
    joblib.dump(clf, model_path, compress=3)
    print(f"  • Saved trained baseline model to: '{model_path}' ({os.path.getsize(model_path)/(1024**2):.2f} MB)")
    
    results_dict = {
        "model_type": "LinearSVC",
        "hyperparameters": {
            "C": 1.0,
            "class_weight": "balanced",
            "random_state": 42
        },
        "dataset": {
            "training_samples": n_train_samples,
            "validation_samples": n_val_samples,
            "features_used": n_features
        },
        "objectives": {
            "f1_score": f1,
            "false_negative_rate": fnr,
            "inference_time_total_seconds": avg_total_inference_time_sec,
            "inference_time_total_ms": avg_total_inference_time_sec * 1000.0,
            "inference_time_per_comment_us": avg_time_per_comment_us,
            "inference_time_per_comment_ms": avg_time_per_comment_ms
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "timing_runs_seconds": run_durations
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    print(f"  • Saved baseline results to: '{results_path}'")
    
    # 7. Print concise requested summary
    print("\n" + "=" * 80)
    print("CONCISE BASELINE SUMMARY")
    print("=" * 80)
    print("Baseline:")
    print(f"Features = {n_features:,}")
    print(f"F1 = {f1:.4f}")
    print(f"FNR = {fnr:.4f}")
    print(f"Inference time = {avg_total_inference_time_sec*1000:.3f} ms total ({avg_time_per_comment_us:.3f} µs/comment)")
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_baseline()
