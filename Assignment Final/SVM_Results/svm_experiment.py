#!/usr/bin/env python3
"""
MLRan SVM Classification Experiment
------------------------------------
Author: Machine Learning Engineer
Task: Complete, reproducible SVM classification benchmarking on MLRan RFE dataset.
Models: Linear SVM, RBF SVM, Polynomial SVM
Target: sample_type (0 = Benign, 1 = Malicious)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

# Set headless backend for matplotlib before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def resolve_paths():
    """Locate dataset files and output directory robustly."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    
    # If run from repository root
    if not os.path.exists(os.path.join(project_dir, "MLRan_X_train_RFE.csv")):
        # Check if current directory has the files
        if os.path.exists("Assignment Final/MLRan_X_train_RFE.csv"):
            project_dir = os.path.abspath("Assignment Final")
        elif os.path.exists("MLRan_X_train_RFE.csv"):
            project_dir = os.path.abspath(".")

    train_path = os.path.join(project_dir, "MLRan_X_train_RFE.csv")
    test_path = os.path.join(project_dir, "MLRan_X_test_RFE.csv")
    labels_path = os.path.join(project_dir, "MLRan_labels.csv")
    json_path = os.path.join(project_dir, "RFE_selected_feature_names_dic.json")
    output_dir = os.path.join(project_dir, "SVM_Results")

    os.makedirs(output_dir, exist_ok=True)
    return train_path, test_path, labels_path, json_path, output_dir


def load_and_validate_data(train_path, test_path, labels_path, json_path):
    """Load and rigorously validate data alignment and leakage safeguards."""
    print("=" * 80)
    print("STEP 1: DATA LOADING & INTEGRITY VERIFICATION")
    print("=" * 80)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    labels_df = pd.read_csv(labels_path)

    with open(json_path, "r", encoding="utf-8") as f:
        rfe_dict = json.load(f)

    print(f"Loaded Train: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Loaded Test:  {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    print(f"Loaded Labels: {labels_df.shape[0]} rows, {labels_df.shape[1]} columns")
    print(f"Loaded RFE JSON features: {len(rfe_dict)} keys")

    # Define metadata columns to explicitly exclude
    metadata_cols = ["sample_id", "sample_type", "family_label", "type_label"]
    for col in metadata_cols:
        assert col in train_df.columns, f"Missing {col} in train_df"
        assert col in test_df.columns, f"Missing {col} in test_df"

    # Extract feature columns
    train_feature_cols = [c for c in train_df.columns if c not in metadata_cols]
    test_feature_cols = [c for c in test_df.columns if c not in metadata_cols]

    # Verification 1: Exactly 483 feature columns
    assert len(train_feature_cols) == 483, f"Expected 483 train features, got {len(train_feature_cols)}"
    assert len(test_feature_cols) == 483, f"Expected 483 test features, got {len(test_feature_cols)}"
    assert train_feature_cols == test_feature_cols, "Train and test feature columns do not match in sequence!"
    assert train_feature_cols == list(rfe_dict.keys()), "Feature columns do not match JSON keys!"

    # Verification 2: Sample IDs disjointness and validity
    train_ids = set(train_df["sample_id"])
    test_ids = set(test_df["sample_id"])
    assert len(train_ids.intersection(test_ids)) == 0, "Train and test sample IDs overlap!"
    assert len(train_ids) == 3905, f"Expected 3905 train IDs, got {len(train_ids)}"
    assert len(test_ids) == 975, f"Expected 975 test IDs, got {len(test_ids)}"

    # Verification 3: No NaN / Inf in features
    X_train = train_df[train_feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[test_feature_cols].to_numpy(dtype=np.float64)
    assert not np.isnan(X_train).any(), "NaN found in X_train"
    assert not np.isnan(X_test).any(), "NaN found in X_test"
    assert not np.isinf(X_train).any(), "Inf found in X_train"
    assert not np.isinf(X_test).any(), "Inf found in X_test"

    # Targets: sample_type
    y_train = train_df["sample_type"].to_numpy(dtype=np.int64)
    y_test = test_df["sample_type"].to_numpy(dtype=np.int64)

    print(f"\n[Verification Passed] X_train shape: {X_train.shape}")
    print(f"[Verification Passed] X_test shape:  {X_test.shape}")
    print(f"[Verification Passed] y_train distribution: Benign (0)={np.sum(y_train==0)}, Malicious (1)={np.sum(y_train==1)}")
    print(f"[Verification Passed] y_test distribution:  Benign (0)={np.sum(y_test==0)}, Malicious (1)={np.sum(y_test==1)}")
    print(f"[Verification Passed] Metadata successfully excluded: {metadata_cols}")

    return X_train, y_train, X_test, y_test, train_feature_cols


def run_grid_search(model_name, pipeline, param_grid, cv, X_train, y_train):
    """Execute GridSearchCV using training data only."""
    print("\n" + "-" * 80)
    print(f"TUNING MODEL: {model_name}")
    print(f"Parameter Grid: {param_grid}")
    print("-" * 80)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    tuning_duration = time.time() - t0

    print(f"-> Best Parameters: {grid.best_params_}")
    print(f"-> Best 3-Fold Cross-Validation F1-score: {grid.best_score_:.4f}")
    print(f"-> Grid search wall-clock time: {tuning_duration:.2f} s")

    return grid.best_estimator_, grid.best_params_, grid.best_score_, grid.cv_results_


def evaluate_model_on_test(model_name, best_estimator, X_train, y_train, X_test, y_test):
    """Retrain best estimator on full training set and evaluate on test set."""
    print(f"\n[Evaluating {model_name} on untouched Test Set...]")

    # Measure exact training time on complete training set
    t0_fit = time.perf_counter()
    best_estimator.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t0_fit

    # Measure exact test inference time
    warmup = best_estimator.predict(X_test[:10])  # Warmup cache
    t0_inf = time.perf_counter()
    y_pred = best_estimator.predict(X_test)
    test_inf_time_sec = time.perf_counter() - t0_inf
    inf_time_per_sample_ms = (test_inf_time_sec / len(X_test)) * 1000.0
    inf_time_per_sample_us = inf_time_per_sample_ms * 1000.0

    # Decision function for ROC-AUC
    y_scores = best_estimator.decision_function(X_test)
    roc_auc = float(roc_auc_score(y_test, y_scores))
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_scores)

    # Core classification metrics (positive class = 1, Malicious)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1))
    rec = float(recall_score(y_test, y_pred, pos_label=1))
    f1 = float(f1_score(y_test, y_pred, pos_label=1))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    report = classification_report(
        y_test, y_pred,
        target_names=["Benign (0)", "Malicious (1)"],
        digits=4
    )

    print(f"  Test Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Test Precision:    {prec:.4f}")
    print(f"  Test Recall:       {rec:.4f}")
    print(f"  Test F1-score:     {f1:.4f}")
    print(f"  Test ROC-AUC:      {roc_auc:.4f}")
    print(f"  Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Training Time:     {train_time_sec:.3f} s")
    print(f"  Inference Time:    {test_inf_time_sec*1000:.2f} ms ({inf_time_per_sample_us:.2f} µs/sample)")

    results_dict = {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "specificity": spec,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "training_time_sec": float(train_time_sec),
        "test_inference_time_sec": float(test_inf_time_sec),
        "test_inference_time_ms": float(test_inf_time_sec * 1000.0),
        "inference_time_per_sample_us": float(inf_time_per_sample_us),
        "classification_report": report,
        "confusion_matrix": cm,
        "fpr_curve": fpr_curve,
        "tpr_curve": tpr_curve,
        "y_scores": y_scores,
        "y_pred": y_pred
    }
    return results_dict


def plot_confusion_matrices(eval_results, output_dir):
    """Plot and save individual and combined confusion matrices."""
    # 1. Combined 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, res in zip(axes, eval_results):
        cm = res["confusion_matrix"]
        name = res["model_name"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Benign (0)", "Malicious (1)"],
            yticklabels=["Benign (0)", "Malicious (1)"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"{name}\nAcc: {res['accuracy']*100:.2f}% | F1: {res['f1_score']:.4f}", fontsize=13, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
    
    plt.tight_layout()
    combined_path = os.path.join(output_dir, "confusion_matrices_all.png")
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"Saved: {combined_path}")

    # 2. Individual plots
    for res in eval_results:
        cm = res["confusion_matrix"]
        name = res["model_name"]
        safe_name = name.lower().replace(" ", "_")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax,
            xticklabels=["Benign (0)", "Malicious (1)"],
            yticklabels=["Benign (0)", "Malicious (1)"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {name}\n(Acc={res['accuracy']*100:.2f}%, F1={res['f1_score']:.4f})", fontsize=12, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.tight_layout()
        ind_path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
        plt.savefig(ind_path, dpi=300)
        plt.close()
        print(f"Saved: {ind_path}")


def plot_model_comparison(eval_results, output_dir):
    """Plot bar chart comparing core metrics across the models."""
    df_plot = []
    for res in eval_results:
        for metric, label in [("accuracy", "Accuracy"), ("precision", "Precision"), 
                             ("recall", "Recall"), ("f1_score", "F1-Score"), ("roc_auc", "ROC-AUC")]:
            df_plot.append({
                "Model": res["model_name"],
                "Metric": label,
                "Value": res[metric]
            })
    df_plot = pd.DataFrame(df_plot)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_plot,
        x="Metric",
        y="Value",
        hue="Model",
        palette=["#1f77b4", "#2ca02c", "#ff7f0e"]
    )
    plt.title("SVM Models Performance Comparison (MLRan RFE Dataset)", fontsize=14, weight="bold", pad=15)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.ylim(0.85, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, shadow=True, title="SVM Model")

    # Annotate bar values
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(
                f"{val:.3f}",
                (p.get_x() + p.get_width() / 2., val),
                ha='center', va='bottom',
                fontsize=9, weight="bold",
                xytext=(0, 3), textcoords='offset points'
            )

    plt.tight_layout()
    comp_path = os.path.join(output_dir, "model_metrics_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_path}")


def plot_roc_curves(eval_results, output_dir):
    """Plot overlayed ROC curves for all models."""
    plt.figure(figsize=(8, 7))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    for res, color in zip(eval_results, colors):
        plt.plot(
            res["fpr_curve"],
            res["tpr_curve"],
            color=color,
            lw=2.5,
            label=f"{res['model_name']} (AUC = {res['roc_auc']:.4f})"
        )

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, weight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    roc_path = os.path.join(output_dir, "roc_curves_comparison.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved: {roc_path}")


def main():
    train_path, test_path, labels_path, json_path, output_dir = resolve_paths()
    print(f"Working Directory: {os.getcwd()}")
    print(f"Output Directory:  {output_dir}")

    # Step 1: Load and Validate
    X_train, y_train, X_test, y_test, feature_names = load_and_validate_data(
        train_path, test_path, labels_path, json_path
    )

    # Step 2: Stratified Cross-Validation setup
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Step 3: Pipelines and Parameter Grids
    # 1. Linear SVM
    pipe_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", random_state=RANDOM_STATE))
    ])
    param_grid_linear = {
        "svm__C": [0.1, 1, 10]
    }

    # 2. RBF SVM
    pipe_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=RANDOM_STATE))
    ])
    param_grid_rbf = {
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", "auto"]
    }

    # 3. Polynomial SVM
    pipe_poly = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="poly", random_state=RANDOM_STATE))
    ])
    param_grid_poly = {
        "svm__C": [0.1, 1, 10],
        "svm__degree": [2, 3],
        "svm__gamma": ["scale"]
    }

    model_configs = [
        ("Linear SVM", pipe_linear, param_grid_linear, "svm_model_linear.joblib"),
        ("RBF SVM", pipe_rbf, param_grid_rbf, "svm_model_rbf.joblib"),
        ("Polynomial SVM", pipe_poly, param_grid_poly, "svm_model_poly.joblib")
    ]

    best_estimators = {}
    best_parameters_dict = {}
    eval_results = []

    print("\n" + "=" * 80)
    print("STEP 2: HYPERPARAMETER TUNING & MODEL TRAINING")
    print("=" * 80)

    for name, pipeline, param_grid, model_filename in model_configs:
        best_est, best_params, best_cv_f1, cv_res = run_grid_search(
            name, pipeline, param_grid, cv, X_train, y_train
        )
        best_estimators[name] = best_est
        best_parameters_dict[name] = {
            "best_params": best_params,
            "best_cv_f1_score": float(best_cv_f1)
        }

        # Final Evaluation on Untouched Test Set
        eval_res = evaluate_model_on_test(
            name, best_est, X_train, y_train, X_test, y_test
        )
        eval_results.append(eval_res)

        # Save Joblib model artifact
        model_save_path = os.path.join(output_dir, model_filename)
        joblib.dump(best_est, model_save_path)
        print(f"Saved Model Artifact: {model_save_path}")

    # Step 4: Save Best Parameters JSON
    json_out_path = os.path.join(output_dir, "svm_best_parameters.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(best_parameters_dict, f, indent=4)
    print(f"\nSaved Best Parameters JSON: {json_out_path}")

    # Step 5: Save Classification Reports TXT
    reports_txt_path = os.path.join(output_dir, "svm_classification_reports.txt")
    with open(reports_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MLRan SVM CLASSIFICATION EXPERIMENT - DETAILED TEST REPORTS\n")
        f.write("=" * 80 + "\n\n")
        for res in eval_results:
            f.write(f"MODEL: {res['model_name']}\n")
            f.write(f"Best Hyperparameters: {best_parameters_dict[res['model_name']]['best_params']}\n")
            f.write(f"Best CV F1: {best_parameters_dict[res['model_name']]['best_cv_f1_score']:.4f}\n")
            f.write(f"Test Accuracy:  {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)\n")
            f.write(f"Test F1-score:  {res['f1_score']:.4f}\n")
            f.write(f"Test Precision: {res['precision']:.4f}\n")
            f.write(f"Test Recall:    {res['recall']:.4f}\n")
            f.write(f"Test ROC-AUC:   {res['roc_auc']:.4f}\n")
            f.write(f"Training Time:  {res['training_time_sec']:.3f} s\n")
            f.write(f"Test Inference Time: {res['test_inference_time_ms']:.2f} ms ({res['inference_time_per_sample_us']:.2f} us/sample)\n")
            f.write(f"Confusion Matrix: TN={res['true_negatives']}, FP={res['false_positives']}, FN={res['false_negatives']}, TP={res['true_positives']}\n")
            f.write("\nClassification Report:\n")
            f.write(res["classification_report"] + "\n")
            f.write("-" * 80 + "\n\n")
    print(f"Saved Classification Reports: {reports_txt_path}")

    # Step 6: Save Results CSV
    csv_rows = []
    for res in eval_results:
        csv_rows.append({
            "Model": res["model_name"],
            "Best_Parameters": str(best_parameters_dict[res["model_name"]]["best_params"]),
            "CV_F1_Score": best_parameters_dict[res["model_name"]]["best_cv_f1_score"],
            "Test_Accuracy": res["accuracy"],
            "Test_Precision": res["precision"],
            "Test_Recall": res["recall"],
            "Test_F1_Score": res["f1_score"],
            "Test_ROC_AUC": res["roc_auc"],
            "Specificity": res["specificity"],
            "FPR": res["false_positive_rate"],
            "FNR": res["false_negative_rate"],
            "True_Negatives": res["true_negatives"],
            "False_Positives": res["false_positives"],
            "False_Negatives": res["false_negatives"],
            "True_Positives": res["true_positives"],
            "Training_Time_s": res["training_time_sec"],
            "Inference_Time_ms": res["test_inference_time_ms"],
            "Inference_Time_Per_Sample_us": res["inference_time_per_sample_us"]
        })
    results_df = pd.DataFrame(csv_rows)
    csv_out_path = os.path.join(output_dir, "svm_results.csv")
    results_df.to_csv(csv_out_path, index=False)
    print(f"Saved Comparison CSV: {csv_out_path}")

    # Step 7: Visualizations
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING HIGH-RESOLUTION PLOTS")
    print("=" * 80)
    plot_confusion_matrices(eval_results, output_dir)
    plot_model_comparison(eval_results, output_dir)
    plot_roc_curves(eval_results, output_dir)

    # Step 8: Terminal Summary
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (TEST SET)")
    print("=" * 80)
    summary_cols = ["Model", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1_Score", "Test_ROC_AUC", "Training_Time_s", "Inference_Time_ms"]
    display_df = results_df[summary_cols].copy()
    display_df["Test_Accuracy"] = display_df["Test_Accuracy"].map(lambda x: f"{x*100:.2f}%")
    display_df["Test_Precision"] = display_df["Test_Precision"].map(lambda x: f"{x:.4f}")
    display_df["Test_Recall"] = display_df["Test_Recall"].map(lambda x: f"{x:.4f}")
    display_df["Test_F1_Score"] = display_df["Test_F1_Score"].map(lambda x: f"{x:.4f}")
    display_df["Test_ROC_AUC"] = display_df["Test_ROC_AUC"].map(lambda x: f"{x:.4f}")
    display_df["Training_Time_s"] = display_df["Training_Time_s"].map(lambda x: f"{x:.2f} s")
    display_df["Inference_Time_ms"] = display_df["Inference_Time_ms"].map(lambda x: f"{x:.2f} ms")
    print(display_df.to_string(index=False))

    # Identify best model by test F1-score
    best_row = results_df.loc[results_df["Test_F1_Score"].idxmax()]
    print("\n" + "*" * 80)
    print(f"🏆 BEST-PERFORMING MODEL: {best_row['Model']}")
    print(f"   • Test F1-Score:    {best_row['Test_F1_Score']:.4f}")
    print(f"   • Test Accuracy:    {best_row['Test_Accuracy']*100:.2f}%")
    print(f"   • Test Precision:   {best_row['Test_Precision']:.4f}")
    print(f"   • Test Recall:      {best_row['Test_Recall']:.4f}")
    print(f"   • Test ROC-AUC:     {best_row['Test_ROC_AUC']:.4f}")
    print(f"   • Best Parameters:  {best_row['Best_Parameters']}")
    print(f"   • Training Latency: {best_row['Training_Time_s']:.2f} s")
    print(f"   • Infer. Latency:   {best_row['Inference_Time_ms']:.2f} ms")
    print("*" * 80 + "\n")

    return results_df, best_parameters_dict


if __name__ == "__main__":
    main()

