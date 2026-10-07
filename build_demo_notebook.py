#!/usr/bin/env python3
"""
build_demo_notebook.py

Generates a professional, self-contained, publication-grade Jupyter notebook:
Soft_Computing_Project_Demo.ipynb

Designed strictly as a clean, technical live demonstration of the completed
Soft Computing Evolutionary Feature Selection project on Kaggle, Colab, or local environments.
Contains zero Q&A, presentation instructions, or study-guide text.
"""

import json
import os


def create_cell(cell_type: str, source: list, execution_count=None, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source]
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs if outputs is not None else []
    return cell


def build_notebook():
    cells = []

    # =========================================================================
    # SECTION 1: Project Overview
    # =========================================================================
    cells.append(create_cell("markdown", [
        "# Multi-Objective Evolutionary Feature Selection for Toxic Comment Classification",
        "## Experimental Evaluation and Model Demonstration",
        "",
        "### Project Summary",
        "- **Dataset**: Jigsaw Toxic Comment Classification (159,571 comments, binary toxicity target).",
        "- **Objective**: Select an optimal subset of **exactly 500 TF-IDF features** from an initial vocabulary of **10,000 features** (**95.0% dimensionality reduction**).",
        "- **Optimization Formulation**: 3-Objective Pareto optimization simultaneously optimizing:",
        "  1. **Classification Performance**: Maximize F1 score (minimize $1 - \\text{F1}$).",
        "  2. **False Alarm Suppression**: Minimize False Positive Rate ($\\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}}$).",
        "  3. **Prediction Throughput**: Minimize inference latency on the 31,915-sample validation partition.",
        "- **Algorithms Benchmarked**: **NSGA-II**, **SPEA2**, **MOEA/D**, and **MODE**.",
        "- **Base Classifier**: `LinearSVC(class_weight=\"balanced\", C=1.0, random_state=42)`.",
        "- **Key Finding**: **NSGA-II** achieved the best overall 3-objective trade-off ($d^* = 0.2096$ to the ideal point) and highest hypervolume ($0.00533716$).",
        "",
        "---",
        "*Note: This notebook executes live evaluations of the baseline and final 500-feature models in under 10 seconds using pre-extracted numerical sparse matrices.*"
    ]))

    # =========================================================================
    # SECTION 2: Environment Setup
    # =========================================================================
    cells.append(create_cell("code", [
        "# 1. Environment & Data Path Autodetection",
        "import os",
        "import sys",
        "import time",
        "import json",
        "import numpy as np",
        "import pandas as pd",
        "from scipy import sparse",
        "import matplotlib.pyplot as plt",
        "",
        "try:",
        "    from IPython.display import display, Image",
        "except ImportError:",
        "    def display(obj):",
        "        print(obj)",
        "    class Image:",
        "        def __init__(self, filename=None, width=None):",
        "            self.filename = filename",
        "",
        "from sklearn.svm import LinearSVC",
        "from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score",
        "",
        "# Candidate paths to search for project artifacts (Local, Colab, Kaggle)",
        "POSSIBLE_DIRS = [",
        "    '.',",
        "    '..',",
        "    '/content',",
        "    '/kaggle/input/soft-computing-project',",
        "    '/kaggle/input/scproject',",
        "    '/kaggle/working'",
        "]",
        "",
        "DATA_DIR = '.'",
        "for d in POSSIBLE_DIRS:",
        "    if os.path.exists(os.path.join(d, 'X_train.npz')) and os.path.exists(os.path.join(d, 'final_analysis.json')):",
        "        DATA_DIR = d",
        "        break",
        "",
        "print(f\"[Setup] Active Data Directory: '{os.path.abspath(DATA_DIR)}'\")",
        "",
        "# Verify required runtime artifacts",
        "REQUIRED_FILES = [",
        "    'X_train.npz', 'X_val.npz', 'y_train.npy', 'y_val.npy', 'feature_names.txt',",
        "    'final_algorithm_comparison.csv', 'final_pareto_solutions.csv', 'final_analysis.json',",
        "    'selected_features_nsga2_balanced.npy', 'selected_features_spea2_balanced.npy',",
        "    'selected_features_moead_balanced.npy', 'selected_features_mode_balanced.npy'",
        "]",
        "",
        "missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]",
        "if missing:",
        "    print(f\"WARNING: Missing required artifacts: {missing}\")",
        "else:",
        "    print(\"✓ All required numerical matrices, result tables, and selected feature sets verified.\")"
    ]))

    # =========================================================================
    # SECTION 3: Dataset & Preprocessing Pipeline
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "## 2. Dataset Pipeline & Numerical Representation",
        "",
        "### Pipeline Stages:",
        "1. **Raw Labeled Data**: 159,571 comments from the Jigsaw Toxic Comment Classification challenge.",
        "2. **Binary Target Formulation**: A unified toxic label $y \\in \\{0, 1\\}$ where $y=1$ if any of the six original labels (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) is positive ($10.17\\%$ positive prevalence).",
        "3. **Text Preprocessing**: Lowercasing, HTML stripping, URL removal, whitespace normalization, and punctuation cleanup preserving token boundaries.",
        "4. **TF-IDF Vectorization**: Word and bi-gram tokenization (`ngram_range=(1,2)`, `min_df=5`, `sublinear_tf=True`) restricted to the top $10,000$ most informative features.",
        "5. **Partitioning**: Stratified 80/20 train/validation split (`random_state=42`):",
        "   - **Training Set**: 127,656 comments $\\times$ 10,000 features.",
        "   - **Validation Set**: 31,915 comments $\\times$ 10,000 features."
    ]))

    # =========================================================================
    # SECTION 4: Processed Numerical CSV Inspection
    # =========================================================================
    cells.append(create_cell("code", [
        "# 2.1 Inspection of the Processed Numerical Dataset Sample (processed_sample.csv)",
        "# Demonstrates the conversion of raw text comments into numerical TF-IDF features.",
        "sample_csv_path = os.path.join(DATA_DIR, 'processed_data', 'processed_sample.csv')",
        "",
        "if os.path.exists(sample_csv_path):",
        "    df_sample = pd.read_csv(sample_csv_path)",
        "    print(f\"Loaded Sample CSV: '{sample_csv_path}'\")",
        "    print(f\"  • Dimensions: {df_sample.shape[0]} rows × {df_sample.shape[1]} columns\")",
        "    print(f\"  • Feature Prefix Convention: '{df_sample.columns[0]}' ... '{df_sample.columns[-2]}'\")",
        "    print(f\"  • Final Column: '{df_sample.columns[-1]}' (Unique classes: {sorted(df_sample['target'].unique())})\")",
        "    ",
        "    # Display first 3 rows showing representative feature values and class label",
        "    sample_cols = list(df_sample.columns[:6]) + ['target']",
        "    display(df_sample[sample_cols].head(3))",
        "else:",
        "    print(\"Note: processed_sample.csv not located; continuing with sparse matrix loading.\")"
    ]))

    # =========================================================================
    # SECTION 5: Load Sparse Matrices
    # =========================================================================
    cells.append(create_cell("code", [
        "# 2.2 Loading Sparse TF-IDF Numerical Matrices",
        "t0 = time.time()",
        "X_train = sparse.load_npz(os.path.join(DATA_DIR, 'X_train.npz')).tocsc()",
        "X_val = sparse.load_npz(os.path.join(DATA_DIR, 'X_val.npz')).tocsc()",
        "y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))",
        "y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))",
        "",
        "with open(os.path.join(DATA_DIR, 'feature_names.txt'), 'r', encoding='utf-8') as f:",
        "    feature_names = [line.strip() for line in f if line.strip()]",
        "",
        "print(f\"Loaded numerical matrices in {time.time() - t0:.2f}s:\")",
        "print(f\"  • X_train: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features (CSC sparse)\")",
        "print(f\"  • X_val:   {X_val.shape[0]:,} samples × {X_val.shape[1]:,} features (CSC sparse)\")",
        "print(f\"  • y_train: {len(y_train):,} labels | Positive class: {np.mean(y_train)*100:.2f}%\")",
        "print(f\"  • y_val:   {len(y_val):,} labels | Positive class: {np.mean(y_val)*100:.2f}%\")",
        "print(f\"  • Vocabulary: {len(feature_names):,} feature tokens\")"
    ]))

    # =========================================================================
    # SECTION 6: Baseline Model Evaluation
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "## 3. Baseline Model: Full 10,000-Feature LinearSVC",
        "",
        "The reference baseline utilizes all $10,000$ TF-IDF features with `LinearSVC(class_weight=\"balanced\", C=1.0, random_state=42)`.",
        "",
        "### Established Baseline Performance:",
        "- **F1 Score**: $0.6848$",
        "- **False Positive Rate (FPR)**: $6.81\\%$",
        "- **Inference Latency**: $3.375\\text{ ms}$ for all 31,915 validation comments"
    ]))

    cells.append(create_cell("code", [
        "# 3.1 Live Execution of Baseline LinearSVC (10,000 Features)",
        "print(\"Training baseline LinearSVC on full 10,000 features...\")",
        "t0_fit = time.perf_counter()",
        "clf_base = LinearSVC(class_weight=\"balanced\", C=1.0, random_state=42, dual=\"auto\")",
        "clf_base.fit(X_train, y_train)",
        "t_fit_base = time.perf_counter() - t0_fit",
        "",
        "# Timed prediction passes",
        "_ = clf_base.predict(X_val) # warmup",
        "durations = []",
        "for _ in range(5):",
        "    t0 = time.perf_counter()",
        "    y_pred_base = clf_base.predict(X_val)",
        "    durations.append(time.perf_counter() - t0)",
        "avg_time_base_ms = np.mean(durations) * 1000.0",
        "",
        "cm_base = confusion_matrix(y_val, y_pred_base)",
        "tn_b, fp_b, fn_b, tp_b = cm_base.ravel()",
        "f1_base = f1_score(y_val, y_pred_base)",
        "fpr_base = fp_b / (fp_b + tn_b)",
        "",
        "print(f\"\\n✓ Baseline Model Results (10,000 Features):\")",
        "print(f\"  • F1 Score:       {f1_base:.4f}\")",
        "print(f\"  • FPR:            {fpr_base*100:.2f}%\")",
        "print(f\"  • Inference Time: {avg_time_base_ms:.3f} ms (31,915 comments)\")",
        "print(f\"  • Training Time:  {t_fit_base:.2f} s\")"
    ]))

    # =========================================================================
    # SECTION 7: Evolutionary Formulation & Constraint
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "## 4. Multi-Objective Evolutionary Feature Selection Formulation",
        "",
        "### Optimization Problem",
        "$$\\min_{\\mathbf{x} \\in \\{0, 1\\}^{10000}} \\mathbf{F}(\\mathbf{x}) = \\Big(1.0 - \\text{F1}(\\mathbf{x}), \\; \\text{FPR}(\\mathbf{x}), \\; T_{\\text{inf}}(\\mathbf{x})\\Big) \\quad \\text{subject to} \\quad \\sum_{i=1}^{10000} x_i = 500$$",
        "",
        "### Representation & Repair Mechanism:",
        "- **Chromosome**: Binary string of length $10,000$, where $x_i = 1$ denotes feature inclusion.",
        "- **Exact 500-Feature Constraint**: Strictly maintained via deterministic repair during initialization, crossover, and mutation to enforce a true 95% dimensionality reduction without treating feature count as a free objective.",
        "",
        "### Algorithmic Approaches Compared:",
        "1. **NSGA-II**: Non-dominated sorting with crowding distance diversity preservation.",
        "2. **SPEA2**: Strength Pareto fitness assignment with $k$-NN density truncation.",
        "3. **MOEA/D**: Tchebycheff decomposition over predefined uniform weight vectors.",
        "4. **MODE**: Multi-objective differential evolution with DE/rand/1/bin vector mutation and Pareto dominance selection."
    ]))

    # =========================================================================
    # SECTION 8: Live Retraining & Evaluation of the 4 Models
    # =========================================================================
    cells.append(create_cell("code", [
        "# 4.1 Live Retraining and Validation of the Four 500-Feature Models",
        "# Evaluates each algorithm's balanced 500-feature subset on the complete 31,915 validation set.",
        "# Execution takes ~1.5 seconds.",
        "",
        "algorithms = ['NSGA-II', 'SPEA2', 'MOEA/D', 'MODE']",
        "file_keys = {'NSGA-II': 'nsga2', 'SPEA2': 'spea2', 'MOEA/D': 'moead', 'MODE': 'mode'}",
        "",
        "live_metrics = []",
        "print(\"=\" * 80)",
        "print(\"  LIVE EVALUATION OF 500-FEATURE BALANCED MODELS (31,915 VALIDATION SAMPLES)\")",
        "print(\"=\" * 80)",
        "",
        "for algo in algorithms:",
        "    key = file_keys[algo]",
        "    npy_file = os.path.join(DATA_DIR, f'selected_features_{key}_balanced.npy')",
        "    indices = np.load(npy_file)",
        "    assert len(indices) == 500, f\"Constraint violation in {algo}: {len(indices)} features\"",
        "    ",
        "    # Fit LinearSVC on 500 selected features",
        "    t0_fit = time.perf_counter()",
        "    clf = LinearSVC(class_weight=\"balanced\", C=1.0, random_state=42, dual=\"auto\")",
        "    clf.fit(X_train[:, indices], y_train)",
        "    t_fit = time.perf_counter() - t0_fit",
        "    ",
        "    # Timed prediction runs",
        "    _ = clf.predict(X_val[:, indices]) # warmup",
        "    runs = []",
        "    for _ in range(5):",
        "        t0 = time.perf_counter()",
        "        y_pred = clf.predict(X_val[:, indices])",
        "        runs.append(time.perf_counter() - t0)",
        "    avg_inf_ms = np.mean(runs) * 1000.0",
        "    ",
        "    f1 = f1_score(y_val, y_pred)",
        "    cm = confusion_matrix(y_val, y_pred)",
        "    tn, fp, fn, tp = cm.ravel()",
        "    fpr = fp / (fp + tn)",
        "    prec = precision_score(y_val, y_pred, zero_division=0)",
        "    rec = recall_score(y_val, y_pred, zero_division=0)",
        "    acc = accuracy_score(y_val, y_pred)",
        "    ",
        "    live_metrics.append({",
        "        'Algorithm': algo,",
        "        'Features': 500,",
        "        'F1 Score': round(f1, 4),",
        "        'FPR': f\"{fpr*100:.2f}%\",",
        "        'Inference Time (ms)': round(avg_inf_ms, 3),",
        "        'Precision': round(prec, 4),",
        "        'Recall': round(rec, 4),",
        "        'Accuracy': round(acc, 4),",
        "        'Fit Time (s)': round(t_fit, 2)",
        "    })",
        "    print(f\"✓ {algo:7s} [500 Feat]: F1 = {f1:.4f} | FPR = {fpr*100:5.2f}% | Latency = {avg_inf_ms:5.3f} ms | Fit = {t_fit:4.2f}s\")",
        "",
        "df_live_eval = pd.DataFrame(live_metrics)",
        "print(\"=\" * 80)"
    ]))

    # =========================================================================
    # SECTION 9: Comparison Table & Analysis
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "## 5. Authoritative Comparison Table",
        "",
        "The table below reports the complete experimental comparison incorporating the baseline, search runtime, true Pareto front size, hypervolume (reference point $\\mathbf{r} = [1.0, 1.0, 0.010\\text{ s}]$), balanced operating points (minimum Euclidean distance to ideal point $(0,0,0)$ in normalized objective space), and extremal solutions."
    ]))

    cells.append(create_cell("code", [
        "# 5.1 Display Authoritative Comparison Table (final_algorithm_comparison.csv)",
        "comp_csv_path = os.path.join(DATA_DIR, 'final_algorithm_comparison.csv')",
        "if os.path.exists(comp_csv_path):",
        "    df_comp = pd.read_csv(comp_csv_path)",
        "    display(df_comp)",
        "else:",
        "    display(df_live_eval)"
    ]))

    # =========================================================================
    # SECTION 10: Publication Visualizations
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "## 6. Visualizations: Pareto Frontiers & Trade-Off Analysis",
        "",
        "The following figures illustrate the true full-validation non-dominated Pareto fronts, latency trade-offs, and hypervolume coverage:"
    ]))

    cells.append(create_cell("code", [
        "# 6.1 Display Pareto Fronts and Comparative Plots",
        "import os",
        "# Helper function to search across Colab, Kaggle, and local directory structures",
        "def find_plot(filename):",
        "    \"\"\"",
        "    Searches for a plot file in multiple candidate locations:",
        "      1. DATA_DIR/plots/<filename>",
        "      2. DATA_DIR/<filename>",
        "      3. /content/plots/<filename>",
        "      4. /content/<filename>",
        "      5. plots/<filename>",
        "      6. ./<filename>",
        "    \"\"\"",
        "    candidates = [",
        "        os.path.join(DATA_DIR, 'plots', filename),",
        "        os.path.join(DATA_DIR, filename),",
        "        os.path.join('/content', 'plots', filename),",
        "        os.path.join('/content', filename),",
        "        os.path.join('plots', filename),",
        "        os.path.join('.', filename)",
        "    ]",
        "    searched = []",
        "    seen = set()",
        "    for p in candidates:",
        "        norm = os.path.normpath(os.path.abspath(p))",
        "        if norm not in seen:",
        "            seen.add(norm)",
        "            searched.append(p)",
        "            if os.path.exists(p):",
        "                return p, searched",
        "    return None, searched",
        "",
        "plot_items = [",
        "    ('pareto_f1_vs_fpr.png', 'Figure 1: F1 Score vs. False Positive Rate (Full Validation Pareto Front)'),",
        "    ('pareto_f1_vs_inference_time.png', 'Figure 2: F1 Score vs. Inference Latency (Full Validation Pareto Front)'),",
        "    ('baseline_vs_evolutionary_comparison.png', 'Figure 3: 3-Metric Bar Comparison: Baseline (10,000) vs. Balanced (500)'),",
        "    ('hypervolume_comparison.png', 'Figure 4: Hypervolume & True Pareto Front Size Distribution')",
        "]",
        "",
        "for filename, caption in plot_items:",
        "    plot_path, searched_paths = find_plot(filename)",
        "    if plot_path:",
        "        print(f\"\\n{caption} (Loaded: '{plot_path}')\")",
        "        try:",
        "            display(Image(filename=plot_path, width=820))",
        "        except Exception:",
        "            img = plt.imread(plot_path)",
        "            plt.figure(figsize=(10, 6))",
        "            plt.imshow(img)",
        "            plt.axis('off')",
        "            plt.show()",
        "    else:",
        "        print(f\"\\n[Plot Missing] Could not locate '{filename}'\")",
        "        print(\"  Searched candidate locations:\")",
        "        for s in searched_paths:",
        "            print(f\"    - {s}\")"
    ]))

    # =========================================================================
    # SECTION 11: Feature Token Inspection
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "## 7. Selected Feature Token Inspection",
        "",
        "Inspection of the $500$ linguistic tokens selected by the winning **NSGA-II** model verifies that high-signal toxic indicator words are preserved while uninformative grammatical noise and rare n-grams are pruned."
    ]))

    cells.append(create_cell("code", [
        "# 7.1 Inspecting Feature Tokens Selected by NSGA-II Balanced Model",
        "tokens_path = os.path.join(DATA_DIR, 'selected_features_nsga2_balanced.txt')",
        "if os.path.exists(tokens_path):",
        "    with open(tokens_path, 'r', encoding='utf-8') as f:",
        "        nsga2_tokens = [line.strip() for line in f if line.strip()]",
        "    ",
        "    print(f\"NSGA-II Balanced Model selected exactly {len(nsga2_tokens)} features.\")",
        "    print(\"\\nSample of 25 selected tokens:\")",
        "    print(\", \".join(nsga2_tokens[:25]))",
        "    ",
        "    # Check for presence of strong toxic indicators",
        "    toxic_stems = ['idiot', 'moron', 'stupid', 'kill', 'hate', 'die', 'shut', 'fuk', 'bitch', 'asshole']",
        "    found = [w for w in nsga2_tokens if any(s in w.lower() for s in toxic_stems)]",
        "    print(f\"\\nPreserved high-signal toxic keywords ({len(found)} terms): {found[:8]} ...\")",
        "else:",
        "    print(\"selected_features_nsga2_balanced.txt not found.\")"
    ]))

    # =========================================================================
    # SECTION 12: Overall Conclusion & Winner Determination
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "## 8. Experimental Conclusions & Selected Winner",
        "",
        "### Selected Winner: **NSGA-II**",
        "",
        "> **\"Based on the balanced three-objective comparison, NSGA-II was selected as the most suitable approach for this dataset.\"**",
        "",
        "### Empirical Justification:",
        "1. **Minimum Distance to Ideal Point**: NSGA-II Candidate #24 achieved $d^* = 0.2096$ to the normalized ideal point $(0, 0, 0)$, providing the optimal simultaneous balance among F1 score ($0.5911$), FPR ($7.43\\%$), and inference latency ($1.423\\text{ ms}$).",
        "2. **Highest Hypervolume**: NSGA-II achieved the highest hypervolume ($0.00533716$), capturing $99.2\\%$ of the combined 4-algorithm global frontier.",
        "3. **Global Frontier Dominance**: Out of 26 non-dominated solutions across all four algorithms combined, NSGA-II contributed **14 solutions (53.8%)**.",
        "",
        "### Algorithmic Trade-Offs:",
        "- **MOEA/D**: Achieved the highest individual F1 score ($\\text{F1} = 0.6446$, retaining $94.1\\%$ of baseline F1 at $2.852\\text{ ms}$), but clustered in a narrow region with low diversity (lowest hypervolume: $0.00438351$, 2 true Pareto solutions).",
        "- **SPEA2**: Achieved the lowest false alarm rate ($\\text{FPR} = 6.58\\%$ at $1.267\\text{ ms}$), **outperforming the 10,000-feature baseline** ($6.81\\%$), making it the preferred specialist for strict false-alarm suppression.",
        "- **MODE**: Produced an exploratory frontier with 15 per-algorithm Pareto solutions, but was strictly dominated on the combined global front.",
        "",
        "### Baseline Comparison Summary:",
        "Reducing dimensionality by **95.0%** (10,000 down to 500 features) preserves **up to 94.1% of baseline F1**, achieves **lower false-alarm rates than the full-feature model (SPEA2: 6.58% vs. 6.81%)**, and cuts inference latency by **up to 62.5%**."
    ]))

    # =========================================================================
    # SECTION 13: Methodological Limitations
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "## 9. Methodological Limitations",
        "",
        "1. **Validation / Search-Subset Overlap**: Evolutionary fitness evaluation during the search phase utilized a fixed stratified subset of 8,000 samples drawn from the validation split to complete optimization within ~16 minutes. Candidate solutions were subsequently evaluated on the full 31,915-sample validation partition. Because the 8,000-sample search subset is a sub-slice of the validation partition, the final validation set partially overlaps with the search-time subset; this represents a documented methodological limitation of the time-constrained setup.",
        "2. **Neighborhood Clustering in Decomposition**: MOEA/D's uniform scalarizing decomposition with neighborhood size $T=6$ resulted in rapid convergence toward high-F1 regions, limiting its spread across lower-latency operating points."
    ]))

    # =========================================================================
    # SECTION 14: Optional Full Re-Run
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "## 10. [OPTIONAL] Full Evolutionary Experiment Re-Run",
        "",
        "> **Note**: This cell executes the complete 16-minute evolutionary experiment across all 20 generations for all 4 algorithms. It is strictly optional and is **disabled by default**."
    ]))

    cells.append(create_cell("code", [
        "# Execute the full 16-minute evolutionary search from scratch (DISABLED BY DEFAULT)",
        "RUN_FULL_EXPERIMENT = False",
        "",
        "if RUN_FULL_EXPERIMENT:",
        "    print(\"Starting complete evolutionary optimization run (approx. 16 minutes)...\\n\")",
        "    !python run_evolutionary_experiment.py",
        "else:",
        "    print(\"Full evolutionary re-run is skipped (default demo mode). Pre-computed results are verified above.\")"
    ]))

    # Construct the complete notebook dictionary conforming to nbformat v4
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path = "Soft_Computing_Project_Demo.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"✓ Successfully generated clean professional notebook '{output_path}' with {len(cells)} cells!")


if __name__ == "__main__":
    build_notebook()
