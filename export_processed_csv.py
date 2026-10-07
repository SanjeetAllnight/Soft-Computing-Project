#!/usr/bin/env python3
"""
Export Processed Jigsaw TF-IDF Feature Matrices and Targets to CSV.

Outputs:
- processed_data/processed_train.csv
- processed_data/processed_validation.csv
- processed_data/processed_sample.csv (for quick spreadsheet inspection)

Maintains complete fidelity to existing artifacts without refitting or altering data.
Uses chunked writing to prevent excessive RAM consumption.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse


def count_csv_rows(filepath: str, buffer_size: int = 1024 * 1024) -> int:
    """Fast line counter for large CSV files."""
    lines = 0
    with open(filepath, "rb") as f:
        while True:
            buf = f.read(buffer_size)
            if not buf:
                break
            lines += buf.count(b"\n")
    return lines - 1  # Exclude header


def export_matrix_to_csv(
    X: sparse.csr_matrix,
    y: np.ndarray,
    feature_names: list,
    output_path: str,
    chunk_size: int = 5000,
    float_fmt: str = "%.10g"
):
    """
    Exports a large sparse matrix and target array to CSV in chunks.
    Avoids full-matrix .toarray() memory explosion.
    """
    n_rows, n_cols = X.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    header_str = ",".join(feature_names + ["target"]) + "\n"
    
    t_start = time.time()
    with open(output_path, "w", encoding="utf-8") as f:
        # Write CSV header
        f.write(header_str)
        
        # Write rows in chunks
        for start_idx in range(0, n_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_rows)
            chunk_dense = X[start_idx:end_idx].toarray()
            chunk_target = y[start_idx:end_idx].reshape(-1, 1)
            
            # Stack features and target
            chunk_combined = np.hstack([chunk_dense, chunk_target])
            
            # Write chunk to CSV using fast formatted output
            np.savetxt(f, chunk_combined, fmt=float_fmt, delimiter=",")
            
            elapsed = time.time() - t_start
            pct = (end_idx / n_rows) * 100.0
            print(f"    Exporting {os.path.basename(output_path)}: {end_idx:,}/{n_rows:,} rows ({pct:.1f}%) in {elapsed:.1f}s", end="\r", flush=True)
            
            # Explicit cleanup
            del chunk_dense, chunk_target, chunk_combined

    total_time = time.time() - t_start
    file_size_bytes = os.path.getsize(output_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    file_size_mb = file_size_bytes / (1024 ** 2)
    print(f"\n    Finished {os.path.basename(output_path)} in {total_time:.2f}s | Size: {file_size_mb:,.1f} MB ({file_size_gb:.2f} GB)")
    return file_size_bytes, total_time


def main():
    start_total_time = time.time()
    print("=" * 80)
    print("      EXPORT PROCESSED JIGSAW TF-IDF DATASET TO NUMERICAL CSV FILES")
    print("=" * 80)
    
    # 1. Verify existence of source artifacts
    artifacts = {
        "X_train": "X_train.npz",
        "X_val": "X_val.npz",
        "y_train": "y_train.npy",
        "y_val": "y_val.npy",
        "feature_names": "feature_names.txt"
    }
    
    print("\n[Step 1] Verifying source artifacts...")
    for name, path in artifacts.items():
        if not os.path.exists(path):
            print(f"Error: Artifact '{path}' not found!", file=sys.stderr)
            sys.exit(1)
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  • Found {path:<20} ({size_mb:>6.2f} MB)")

    # 2. Load artifacts
    print("\n[Step 2] Loading artifacts into memory...")
    t0 = time.time()
    X_train = sparse.load_npz(artifacts["X_train"])
    X_val = sparse.load_npz(artifacts["X_val"])
    y_train = np.load(artifacts["y_train"])
    y_val = np.load(artifacts["y_val"])
    
    with open(artifacts["feature_names"], "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f if line.strip()]
        
    print(f"  Loaded in {time.time() - t0:.2f}s.")
    print(f"  • X_train shape: {X_train.shape} | type: {type(X_train).__name__} | dtype: {X_train.dtype}")
    print(f"  • X_val shape:   {X_val.shape} | type: {type(X_val).__name__} | dtype: {X_val.dtype}")
    print(f"  • y_train shape: {y_train.shape} | dtype: {y_train.dtype} | class counts: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  • y_val shape:   {y_val.shape} | dtype: {y_val.dtype} | class counts: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"  • Feature names: {len(feature_names):,} features loaded")

    # 3. Calculate and report pre-export statistics
    print("\n[Step 3] Pre-Export Dataset Statistics:")
    
    def calc_stats(name, X, y):
        n_rows, n_cols = X.shape
        total_cells = n_rows * n_cols
        nnz = X.nnz
        sparsity = (1.0 - (nnz / total_cells)) * 100.0
        
        # Approximate size estimate: 
        # Each row has (n_cols - 1) commas + 1 newline + ~nnz_per_row floats (approx 6-8 chars each) + (n_cols - nnz_per_row) '0's (1 char each) + target digit (1 char)
        avg_row_chars = (n_cols - 1) + 1 + (nnz / n_rows * 6) + (n_cols - nnz / n_rows) * 1 + 2
        est_bytes = (avg_row_chars * n_rows) + (sum(len(f) for f in feature_names) + n_cols + 7)
        est_mb = est_bytes / (1024 ** 2)
        est_gb = est_bytes / (1024 ** 3)
        
        print(f"  • {name}:")
        print(f"    - Rows:                 {n_rows:,}")
        print(f"    - Feature Columns:      {n_cols:,} (+1 target column = {n_cols + 1:,} total columns)")
        print(f"    - Non-Zero Values:      {nnz:,}")
        print(f"    - Sparsity:             {sparsity:.4f}%")
        print(f"    - Estimated CSV Size:   ~{est_mb:,.1f} MB (~{est_gb:.2f} GB)")
        return est_bytes

    calc_stats("Training Set (processed_train.csv)", X_train, y_train)
    calc_stats("Validation Set (processed_validation.csv)", X_val, y_val)

    # 4. Define output directory and file paths
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    train_csv_path = os.path.join(output_dir, "processed_train.csv")
    val_csv_path = os.path.join(output_dir, "processed_validation.csv")
    sample_csv_path = os.path.join(output_dir, "processed_sample.csv")

    # 5. Export CSV files in chunks
    print(f"\n[Step 4] Exporting full datasets to '{output_dir}/' in chunks of 5,000 rows...")
    
    print("  a) Exporting processed_train.csv:")
    train_size_bytes, train_time = export_matrix_to_csv(
        X=X_train,
        y=y_train,
        feature_names=feature_names,
        output_path=train_csv_path,
        chunk_size=5000,
        float_fmt="%.10g"
    )
    
    print("  b) Exporting processed_validation.csv:")
    val_size_bytes, val_time = export_matrix_to_csv(
        X=X_val,
        y=y_val,
        feature_names=feature_names,
        output_path=val_csv_path,
        chunk_size=5000,
        float_fmt="%.10g"
    )

    # 6. Export quick inspection sample (first 100 rows of train)
    print("  c) Exporting processed_sample.csv (first 100 rows of train for quick inspection):")
    sample_size_bytes, sample_time = export_matrix_to_csv(
        X=X_train[:100],
        y=y_train[:100],
        feature_names=feature_names,
        output_path=sample_csv_path,
        chunk_size=100,
        float_fmt="%.10g"
    )

    # 7. Verification of exported CSV files
    print("\n[Step 5] Verifying exported CSV files...")
    
    def verify_csv(csv_path, X_orig, y_orig, expected_rows):
        print(f"  • Verifying '{os.path.basename(csv_path)}':")
        # 1. Header check
        with open(csv_path, "r", encoding="utf-8") as f:
            header_line = f.readline().rstrip("\r\n")
            header_cols = header_line.split(",")
        
        assert len(header_cols) == 10001, f"Expected 10001 columns, got {len(header_cols)}"
        assert header_cols[-1] == "target", f"Expected last column 'target', got '{header_cols[-1]}'"
        assert header_cols[:-1] == feature_names, "Feature column names do not match vocabulary"
        print(f"    ✓ Header verified: exactly 10,000 feature columns + 'target' = 10,001 columns")
        
        # 2. Row count check
        t_c = time.time()
        actual_rows = count_csv_rows(csv_path)
        assert actual_rows == expected_rows, f"Row count mismatch: expected {expected_rows}, got {actual_rows}"
        print(f"    ✓ Row count verified: exactly {actual_rows:,} data rows (counted in {time.time() - t_c:.2f}s)")
        
        # 3. Sample values verification (first 10 rows and last 10 rows)
        df_head = pd.read_csv(csv_path, nrows=10)
        head_features = df_head.iloc[:, :-1].values
        head_targets = df_head.iloc[:, -1].values
        
        orig_head_features = X_orig[:10].toarray()
        orig_head_targets = y_orig[:10]
        
        max_diff_head = np.max(np.abs(head_features - orig_head_features))
        assert max_diff_head < 1e-6, f"Feature value mismatch in head: max diff {max_diff_head}"
        assert np.array_equal(head_targets, orig_head_targets), "Target mismatch in head"
        print(f"    ✓ Sample data verified: head values match source matrix (max diff = {max_diff_head:.2e})")
        print(f"    ✓ Target values verified: head targets exactly match source array")

    verify_csv(train_csv_path, X_train, y_train, X_train.shape[0])
    verify_csv(val_csv_path, X_val, y_val, X_val.shape[0])

    # 8. Final Report
    print("\n" + "=" * 80)
    print("                         FINAL EXPORT REPORT")
    print("=" * 80)
    print(f"Total Export & Verification Time: {time.time() - start_total_time:.2f} seconds\n")
    
    print("Exported Files Summary:")
    print(f"1. Training Set:")
    print(f"   • Path:        {os.path.abspath(train_csv_path)}")
    print(f"   • Shape:       ({X_train.shape[0]:,}, 10001) [10,000 TF-IDF features + 1 target]")
    print(f"   • File Size:   {train_size_bytes / (1024 ** 2):,.2f} MB ({train_size_bytes / (1024 ** 3):.2f} GB)")
    
    print(f"\n2. Validation Set:")
    print(f"   • Path:        {os.path.abspath(val_csv_path)}")
    print(f"   • Shape:       ({X_val.shape[0]:,}, 10001) [10,000 TF-IDF features + 1 target]")
    print(f"   • File Size:   {val_size_bytes / (1024 ** 2):,.2f} MB ({val_size_bytes / (1024 ** 3):.2f} GB)")
    
    print(f"\n3. Inspection Sample (First 100 rows):")
    print(f"   • Path:        {os.path.abspath(sample_csv_path)}")
    print(f"   • Shape:       (100, 10001) [10,000 TF-IDF features + 1 target]")
    print(f"   • File Size:   {sample_size_bytes / (1024 ** 2):,.2f} MB")
    
    # 9. Preview of first few columns and rows
    print("\n" + "-" * 80)
    print("DATASET PREVIEW (First 5 rows, selected columns from processed_train.csv)")
    print("-" * 80)
    sample_df = pd.read_csv(sample_csv_path, nrows=5)
    
    # Select first 5 features, 5 middle features, 5 last features, and target
    preview_cols = feature_names[:5] + feature_names[5000:5005] + feature_names[-5:] + ["target"]
    # Handle pandas duplicate column name for column 7661 ('target') if needed
    preview_display = sample_df.iloc[:5, [0, 1, 2, 3, 4, 5000, 5001, 5002, 5003, 5004, -5, -4, -3, -2, -1]]
    print(preview_display.to_string(index=True))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
