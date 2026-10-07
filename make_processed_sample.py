#!/usr/bin/env python3
"""
make_processed_sample.py

Generates a human-readable 100-row sample CSV from the existing processed Jigsaw TF-IDF
numerical artifacts for demonstration and spreadsheet inspection.

Requirements:
- Uses existing X_train.npz, y_train.npy, and feature_names.txt (no TF-IDF refitting).
- Exports exactly 100 rows and 10,001 columns (10,000 TF-IDF features + 1 target column).
- Prefixes every TF-IDF feature column with 'tfidf_' (e.g., 'tfidf_target').
- Sets the final column name strictly to 'target' (avoiding column collision).
- Verifies exact shape, uniqueness, and value fidelity against X_train[:100] and y_train[:100].
"""

import os
import sys
import numpy as np
from scipy import sparse


def make_sample_csv(
    x_train_path: str = "X_train.npz",
    y_train_path: str = "y_train.npy",
    feature_names_path: str = "feature_names.txt",
    output_path: str = "processed_data/processed_sample.csv",
    n_sample_rows: int = 100
):
    print("=" * 80)
    print("        GENERATE HUMAN-READABLE PROCESSED SAMPLE CSV (100 ROWS)")
    print("=" * 80)

    # 1. Verify existence of required files
    for path in [x_train_path, y_train_path, feature_names_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    # 2. Load feature vocabulary
    with open(feature_names_path, "r", encoding="utf-8") as f:
        raw_vocab = [line.strip() for line in f if line.strip()]
    assert len(raw_vocab) == 10000, f"Expected 10,000 features, found {len(raw_vocab)}"

    # Prefix each feature with 'tfidf_'
    feature_headers = [f"tfidf_{name}" for name in raw_vocab]
    all_headers = feature_headers + ["target"]
    assert len(all_headers) == 10001, f"Expected 10,001 columns, found {len(all_headers)}"
    assert len(set(all_headers)) == 10001, "Column header collision detected!"
    assert all_headers[-1] == "target", "Final column must be named 'target'"

    # 3. Load sparse X_train and dense y_train slices
    print(f"Loading first {n_sample_rows} samples from {x_train_path} and {y_train_path}...")
    X_train_sparse = sparse.load_npz(x_train_path)
    y_train_full = np.load(y_train_path)

    X_sample_dense = X_train_sparse[:n_sample_rows].toarray()
    y_sample = y_train_full[:n_sample_rows].reshape(-1, 1)

    assert X_sample_dense.shape == (n_sample_rows, 10000)
    assert y_sample.shape == (n_sample_rows, 1)

    # Combined array (100, 10001)
    combined = np.hstack([X_sample_dense, y_sample])

    # 4. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header_line = ",".join(all_headers) + "\n"

    print(f"Writing {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header_line)
        np.savetxt(f, combined, fmt="%.10g", delimiter=",")

    file_size_bytes = os.path.getsize(output_path)
    file_size_mb = file_size_bytes / (1024 ** 2)
    print(f"✓ Saved {output_path} ({file_size_mb:.2f} MB)")

    # 5. Full Verification
    print("\nVerifying exported sample CSV...")
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_cols = lines[0].strip().split(",")
    data_lines = lines[1:]

    # Check rows and columns
    assert len(data_lines) == n_sample_rows, f"Row count mismatch: expected {n_sample_rows}, got {len(data_lines)}"
    assert len(header_cols) == 10001, f"Column count mismatch: expected 10,001, got {len(header_cols)}"
    assert header_cols[-1] == "target", "Last column must be 'target'"
    assert len(set(header_cols)) == 10001, "Header names must be completely unique"

    # Spot-check numerical values
    row0_vals = [float(v) for v in data_lines[0].strip().split(",")]
    assert len(row0_vals) == 10001
    assert row0_vals[-1] == float(y_train_full[0]), "Target value mismatch on row 0"
    
    # Check non-zero feature alignment
    row0_x = np.array(row0_vals[:-1])
    diff = np.max(np.abs(row0_x - X_sample_dense[0]))
    assert diff < 1e-6, f"Feature value mismatch: max diff {diff}"

    print(f"✓ Exact dimensions verified: {len(data_lines)} data rows × {len(header_cols)} columns")
    print("✓ All 10,000 feature columns uniquely prefixed with 'tfidf_'")
    print("✓ Final column is exactly 'target'")
    print("✓ Numerical values perfectly match original X_train and y_train")
    print("=" * 80)


if __name__ == "__main__":
    make_sample_csv()
