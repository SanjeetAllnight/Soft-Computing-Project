#!/usr/bin/env python3
"""
Generate publication-quality plots for Evolutionary Feature Selection experiment.
Backend: Agg (headless).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_all_plots():
    os.makedirs("plots", exist_ok=True)
    
    # 1. Load data
    df_pareto = pd.read_csv("pareto_solutions.csv")
    df_comp = pd.read_csv("algorithm_comparison.csv")
    
    results = {}
    for algo, key in [("NSGA-II", "nsgaii"), ("SPEA2", "spea2"), ("MOEA/D", "moead"), ("MODE", "mode")]:
        with open(f"results_{key}.json", "r", encoding="utf-8") as f:
            results[algo] = json.load(f)
            
    # Baseline specs
    baseline_f1 = 0.6848
    baseline_fpr = 6.81
    baseline_inf_time = 3.375
    
    colors = {
        "NSGA-II": "#1f77b4",
        "SPEA2": "#2ca02c",
        "MOEA/D": "#ff7f0e",
        "MODE": "#d62728"
    }
    markers = {
        "NSGA-II": "o",
        "SPEA2": "s",
        "MOEA/D": "^",
        "MODE": "D"
    }

    # -------------------------------------------------------------
    # Plot 1: Pareto Front (F1 vs FPR) on Full Validation Set
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    for algo in ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]:
        sub = df_pareto[df_pareto["Algorithm"] == algo]
        plt.scatter(
            sub["FPR"] * 100.0,
            sub["F1_Score"],
            label=f"{algo} (500 features)",
            color=colors[algo],
            marker=markers[algo],
            s=95,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=3
        )
        
    plt.scatter(
        [baseline_fpr],
        [baseline_f1],
        color="black",
        marker="X",
        s=200,
        label=f"Baseline (10,000 features | F1={baseline_f1:.4f}, FPR={baseline_fpr:.2f}%)",
        zorder=5
    )
    
    plt.xlabel("False Positive Rate (FPR %)", fontsize=12, fontweight="bold")
    plt.ylabel("F1 Score (Validation Set)", fontsize=12, fontweight="bold")
    plt.title("Pareto Front: F1 Score vs. False Positive Rate (Full 31,915 Validation Set)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5, zorder=1)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/pareto_f1_vs_fpr.png")
    plt.close()
    print("✓ Saved 'plots/pareto_f1_vs_fpr.png'")

    # -------------------------------------------------------------
    # Plot 2: F1 vs Inference Time
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    for algo in ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]:
        sub = df_pareto[df_pareto["Algorithm"] == algo]
        plt.scatter(
            sub["Inference_Time_ms"],
            sub["F1_Score"],
            label=f"{algo} (500 features)",
            color=colors[algo],
            marker=markers[algo],
            s=95,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=3
        )
        
    plt.scatter(
        [baseline_inf_time],
        [baseline_f1],
        color="black",
        marker="X",
        s=200,
        label=f"Baseline (10,000 features | Time={baseline_inf_time:.3f} ms)",
        zorder=5
    )
    
    plt.xlabel("Inference Time for 31,915 Comments (ms)", fontsize=12, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=12, fontweight="bold")
    plt.title("Inference Latency vs. F1 Score Trade-Off (Full Validation Set)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5, zorder=1)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/pareto_f1_vs_inference_time.png")
    plt.close()
    print("✓ Saved 'plots/pareto_f1_vs_inference_time.png'")

    # -------------------------------------------------------------
    # Plot 3: Convergence History across 20 Generations
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    for algo in ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]:
        hist = results[algo]["search_phase"]["history"]
        gens = [h["gen"] for h in hist]
        f1_prog = [h["best_f1"] for h in hist]
        plt.plot(
            gens,
            f1_prog,
            label=f"{algo}",
            color=colors[algo],
            linewidth=2.4,
            marker=markers[algo],
            markersize=5,
            alpha=0.9
        )
        
    plt.xlabel("Generation", fontsize=12, fontweight="bold")
    plt.ylabel("Best F1 Score (Search Subset)", fontsize=12, fontweight="bold")
    plt.title("Evolutionary Convergence Comparison across 20 Generations", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/algorithm_convergence.png")
    plt.close()
    print("✓ Saved 'plots/algorithm_convergence.png'")

    # -------------------------------------------------------------
    # Plot 4: Baseline vs 500-Feature Solution Bar Comparison
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Extract Knee solutions from df_comp
    algos = ["Baseline (10k)"] + list(df_comp["Algorithm"].iloc[1:])
    f1_vals = [baseline_f1] + [float(x) for x in df_comp["F1 Score"].iloc[1:]]
    fpr_vals = [baseline_fpr] + [float(str(x).replace("%", "")) for x in df_comp["False Positive Rate (FPR)"].iloc[1:]]
    
    x = np.arange(len(algos))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, f1_vals, width, label="F1 Score (Higher is better)", color="#1f4e79", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("F1 Score", fontsize=12, color="#1f4e79", fontweight="bold")
    ax1.set_ylim(0, 0.85)
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, fpr_vals, width, label="FPR % (Lower is better)", color="#c55a11", edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("False Positive Rate (%)", fontsize=12, color="#c55a11", fontweight="bold")
    ax2.set_ylim(0, 12.0)
    ax2.tick_params(axis="y", labelcolor="#c55a11")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, rotation=15, ha="right", fontsize=10, fontweight="bold")
    plt.title("Performance Comparison: 10,000-Feature Baseline vs. 500-Feature Solutions", fontsize=12, fontweight="bold", pad=12)
    
    # Values on bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1f4e79")
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#c55a11")
                     
    plt.tight_layout()
    plt.savefig("plots/baseline_vs_evolutionary_comparison.png")
    plt.close()
    print("✓ Saved 'plots/baseline_vs_evolutionary_comparison.png'")


if __name__ == "__main__":
    generate_all_plots()
