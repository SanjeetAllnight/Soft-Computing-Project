#!/usr/bin/env python3
"""
make_bundle.py

Creates a clean, portable, self-contained ZIP bundle:
Soft_Computing_Project_Demo_Bundle.zip

Contains everything required to run the live demonstration notebook on
Google Colab, Kaggle, or a local environment.
Excludes the multi-gigabyte dense CSV files and raw dataset files.
"""

import os
import zipfile
import shutil
import tempfile
import json
import numpy as np


BUNDLE_NAME = "Soft_Computing_Project_Demo_Bundle.zip"


README_CONTENT = """# Soft Computing Project Demo

This bundle contains the demonstration files, numerical matrices, and results for the **Multi-Objective Evolutionary Feature Selection** project on the Jigsaw Toxic Comment Classification dataset.

---

## Contents

- **`Soft_Computing_Project_Demo.ipynb`**: The primary demonstration notebook. Runs live model training and evaluation across the baseline and all four 500-feature models in ~6–8 seconds.
- **`demo.py`**: Standalone terminal demonstration script that runs all live evaluations in ~6 seconds (`python demo.py`).
- **`make_processed_sample.py`**: Utility script to regenerate the 100-row inspection sample from numerical matrices.
- **`X_train.npz`, `X_val.npz`**: Pre-extracted sparse TF-IDF feature matrices (10,000 features).
- **`y_train.npy`, `y_val.npy`**: Binary toxicity target labels ($0 = \\text{non-toxic}, 1 = \\text{toxic}$).
- **`feature_names.txt`**: The 10,000 TF-IDF vocabulary tokens.
- **`selected_features_*_balanced.{npy,txt}`**: The optimal 500 feature indices and tokens selected by **NSGA-II**, **SPEA2**, **MOEA/D**, and **MODE**.
- **`final_algorithm_comparison.csv`**: Audited comparison table including baseline, balanced solutions, extremal solutions, and hypervolumes.
- **`final_pareto_solutions.csv`**: The 53 true full-validation Pareto solutions.
- **`final_analysis.json`**: Machine-readable summary of normalization metrics and experimental results.
- **`baseline_results.json`, `experiment_config.json`**: Baseline classifier metrics and experiment configuration.
- **`processed_data/processed_sample.csv`**: Human-readable 100-row sample CSV (10,001 columns: `tfidf_*` and `target`) for spreadsheet inspection.
- **`plots/`**: Publication-quality comparison figures:
  - `pareto_f1_vs_fpr.png`: Pareto frontier (F1 vs. FPR).
  - `pareto_f1_vs_inference_time.png`: Pareto frontier (F1 vs. Inference Latency).
  - `baseline_vs_evolutionary_comparison.png`: 3-metric bar comparison (Baseline vs. 500-feature models).
  - `hypervolume_comparison.png`: Hypervolume and Pareto front size distribution.

---

## Google Colab Instructions

1. Upload `Soft_Computing_Project_Demo_Bundle.zip` to your Colab session storage (`/content/`).
2. Unzip the bundle by executing:
   ```python
   !unzip -o Soft_Computing_Project_Demo_Bundle.zip -d /content/
   ```
   *(Or simply upload the unzipped files directly to `/content/`)*.
3. Open `Soft_Computing_Project_Demo.ipynb` in Colab.
4. Execute the cells sequentially. The path autodetection automatically locates `/content/` or the current directory.
5. Normal live demo execution completes in **~6 to 8 seconds**.

---

## Kaggle Instructions

1. Create a new **Kaggle Dataset** and upload `Soft_Computing_Project_Demo_Bundle.zip` (or its contents).
2. Attach the dataset to your Kaggle Notebook.
3. Open `Soft_Computing_Project_Demo.ipynb`. The path autodetection automatically searches `/kaggle/input/...` directories.
4. Run all demo cells.

---

## Important Notes

1. **Lightweight & Fast**: The demonstration notebook evaluates the pre-computed 500-feature solutions live on the full 31,915-row validation partition in seconds using saved sparse matrices.
2. **Dense CSVs Excluded**: The full processed CSV files (`processed_train.csv` ~2.5 GB and `processed_validation.csv` ~0.6 GB) are intentionally excluded from this bundle to keep it lightweight (~85 MB) and fast to upload. The models train directly from the mathematically identical sparse `.npz` files.
3. **Inspection Sample**: `processed_data/processed_sample.csv` (100 rows × 10,001 columns) is provided specifically for human spreadsheet inspection.
4. **Optional Full Re-Run Disabled**: The optional from-scratch 16-minute evolutionary search cell at the end of the notebook is disabled by default (`RUN_FULL_EXPERIMENT = False`).
"""


def build_zip():
    print("=" * 80)
    print(f"      CREATING PORTABLE LIVE DEMO BUNDLE: {BUNDLE_NAME}")
    print("=" * 80)

    # 1. Write bundle README.md
    readme_path = "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README_CONTENT.strip() + "\n")
    print(f"✓ Created '{readme_path}'")

    # 2. Files to include
    bundle_files = [
        # Primary notebook
        "Soft_Computing_Project_Demo.ipynb",
        # Documentation & scripts
        "README.md",
        "demo.py",
        "make_processed_sample.py",
        # Core numerical data matrices
        "X_train.npz",
        "X_val.npz",
        "y_train.npy",
        "y_val.npy",
        "feature_names.txt",
        # 500-Feature selection sets (npy & txt)
        "selected_features_nsga2_balanced.npy",
        "selected_features_nsga2_balanced.txt",
        "selected_features_spea2_balanced.npy",
        "selected_features_spea2_balanced.txt",
        "selected_features_moead_balanced.npy",
        "selected_features_moead_balanced.txt",
        "selected_features_mode_balanced.npy",
        "selected_features_mode_balanced.txt",
        # Final result artifacts
        "final_algorithm_comparison.csv",
        "final_pareto_solutions.csv",
        "final_analysis.json",
        "baseline_results.json",
        "experiment_config.json",
        # Processed sample
        "processed_data/processed_sample.csv",
        # Visualizations
        "plots/pareto_f1_vs_fpr.png",
        "plots/pareto_f1_vs_inference_time.png",
        "plots/baseline_vs_evolutionary_comparison.png",
        "plots/hypervolume_comparison.png"
    ]

    # Verify all files exist
    missing = [f for f in bundle_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing bundle dependencies: {missing}")

    # Explicit exclusions check
    forbidden = [
        "processed_data/processed_train.csv",
        "processed_data/processed_validation.csv",
        "train.csv",
        "test.csv"
    ]
    for fb in forbidden:
        if fb in bundle_files:
            raise ValueError(f"Forbidden large file included in bundle: {fb}")

    # 3. Create ZIP archive
    print(f"\nCompressing {len(bundle_files)} items into '{BUNDLE_NAME}'...")
    zip_path = BUNDLE_NAME
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in bundle_files:
            zf.write(f, arcname=f)
            size_kb = os.path.getsize(f) / 1024
            print(f"  + Added {f:<45} ({size_kb:>8.1f} KB)")

    total_bytes = os.path.getsize(zip_path)
    total_mb = total_bytes / (1024 ** 2)
    print(f"\n✓ Successfully created '{zip_path}' | Size: {total_mb:.2f} MB ({total_bytes:,} bytes)")

    # 4. Verification & Clean Extraction Test
    print("\n" + "=" * 80)
    print("                    VERIFYING BUNDLE INTEGRITY")
    print("=" * 80)

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        print(f"ZIP Archive contains {len(namelist)} items:")
        for name in sorted(namelist):
            info = zf.getinfo(name)
            print(f"  • {name:<45} (uncompressed: {info.file_size / 1024:>8.1f} KB)")

        # Verify exclusion of huge files
        for fb in ["processed_train.csv", "processed_validation.csv", "train.csv", "test.csv"]:
            assert not any(fb in n for n in namelist), f"Error: {fb} found in ZIP!"
        print("✓ Confirmed: Huge processed CSVs and raw CSVs are strictly excluded.")

    # 5. Extraction test into temporary directory
    print("\n[Test] Extracting bundle into clean temporary sandbox...")
    temp_sandbox = tempfile.mkdtemp(prefix="demo_bundle_test_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_sandbox)
        print(f"  Extracted to temporary directory: {temp_sandbox}")

        # Check all files are present in extraction
        for f in bundle_files:
            extracted_f = os.path.join(temp_sandbox, f)
            assert os.path.exists(extracted_f), f"Extraction failed for {f}"
        print("  ✓ All files cleanly extracted into sandbox.")

        # Test notebook parsing and paths in extracted sandbox
        extracted_nb_path = os.path.join(temp_sandbox, "Soft_Computing_Project_Demo.ipynb")
        with open(extracted_nb_path, "r", encoding="utf-8") as nb_f:
            nb_data = json.load(nb_f)

        code_cells = [c for c in nb_data["cells"] if c["cell_type"] == "code"]
        print(f"  ✓ Notebook loaded from sandbox ({len(code_cells)} code cells).")

        # Test running demo cells with cwd = temp_sandbox
        orig_cwd = os.getcwd()
        try:
            os.chdir(temp_sandbox)
            test_context = {}
            # Run cell 1 (setup)
            exec("".join(code_cells[0]["source"]), test_context)
            assert os.path.exists(os.path.join(test_context.get("DATA_DIR", "."), "X_train.npz"))
            print(f"  ✓ Sandbox path autodetection confirmed (DATA_DIR='{test_context.get('DATA_DIR')}').")

            # Run cell 2 (sample CSV inspection)
            exec("".join(code_cells[1]["source"]), test_context)
            print("  ✓ Processed sample CSV loaded cleanly in sandbox.")

            # Run cell 7 (plots with find_plot)
            exec("".join(code_cells[6]["source"]), test_context)
            print("  ✓ All 4 plot files successfully located and loaded via find_plot() in sandbox.")
        finally:
            os.chdir(orig_cwd)

        print("✓ Sandbox verification PASSED completely.")
    finally:
        shutil.rmtree(temp_sandbox, ignore_errors=True)
        print("✓ Cleaned up sandbox.")

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    build_zip()
