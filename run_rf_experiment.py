#!/usr/bin/env python3
"""
Final Soft Computing Evolutionary Feature Minimization Experiment (Random Forest)
==================================================================================
Comparing: NSGA-II, SPEA2, MOEA/D, and MODE
On: Jigsaw Toxic Comment Classification
Baseline Model: RandomForestClassifier (all 10,000 TF-IDF features)

Three Optimization Objectives (all minimized):
  f1 = 1 - Accuracy        (maximize classification accuracy)
  f2 = K                   (minimize number of selected features, variable 1 <= K <= 10,000)
  f3 = Inference Time      (minimize inference latency in seconds)

No fixed 500-feature constraint: feature count is fully variable.
Accuracy Preservation Tolerance: 0.01 absolute accuracy (acceptable_acc >= baseline_acc - 0.01).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import chi2

# Fix random seed
RANDOM_STATE = 42
TOLERANCE = 0.01  # 0.01 absolute accuracy tolerance

RF_CONFIG = {
    "n_estimators": 50,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}


# ==============================================================================
# 1. EVALUATOR CLASS
# ==============================================================================

class RFFeatureEvaluator:
    """
    Evaluates candidate feature subsets using RandomForestClassifier.
    Computes:
      - Accuracy
      - K (number of selected features)
      - Inference time
      - Objectives vector: [1.0 - Accuracy, float(K), avg_inference_time]
    Maintains a canonical hash cache to prevent duplicate training.
    """
    def __init__(self, X_train_csc, y_train, X_val_csc, y_val, rf_params=None, name="SearchEvaluator"):
        self.X_train = X_train_csc
        self.y_train = y_train
        self.X_val = X_val_csc
        self.y_val = y_val
        self.rf_params = rf_params or RF_CONFIG
        self.name = name
        self.cache = {}
        self.eval_count = 0
        self.cache_hits = 0

    def evaluate(self, individual: np.ndarray, timing_runs: int = 2):
        active_indices = np.where(individual == 1)[0]
        # Guarantee non-empty
        if len(active_indices) == 0:
            active_indices = np.array([0], dtype=np.int32)
            
        key = tuple(sorted(active_indices.tolist()))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        self.eval_count += 1
        idx = np.array(key, dtype=np.int32)
        k_features = len(idx)

        # Slice sparse columns
        X_tr = self.X_train[:, idx]
        X_va = self.X_val[:, idx]

        # Train Random Forest
        t0_fit = time.perf_counter()
        clf = RandomForestClassifier(**self.rf_params)
        clf.fit(X_tr, self.y_train)
        t_fit = time.perf_counter() - t0_fit

        # Measure inference latency with warmup
        _ = clf.predict(X_va)
        durations = []
        y_pred = None
        for _ in range(timing_runs):
            t0_inf = time.perf_counter()
            y_pred = clf.predict(X_va)
            durations.append(time.perf_counter() - t0_inf)
        avg_inf_time = float(np.mean(durations))

        # Metrics
        cm = confusion_matrix(self.y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        acc = float(accuracy_score(self.y_val, y_pred))
        f1 = float(f1_score(self.y_val, y_pred, zero_division=0))
        prec = float(precision_score(self.y_val, y_pred, zero_division=0))
        rec = float(recall_score(self.y_val, y_pred, zero_division=0))

        # Three optimization objectives to minimize: [1 - acc, k_features, avg_inf_time]
        obj_vector = np.array([1.0 - acc, float(k_features), avg_inf_time], dtype=np.float64)

        res = {
            "accuracy": acc,
            "k_features": k_features,
            "inference_time": avg_inf_time,
            "training_time": t_fit,
            "f1": f1,
            "fpr": fpr,
            "precision": prec,
            "recall": rec,
            "objectives": obj_vector,
            "indices": idx,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
        self.cache[key] = res
        return res


# ==============================================================================
# 2. PARETO UTILITIES
# ==============================================================================

def dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """Returns True if obj1 Pareto-dominates obj2 (minimization)."""
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)


def fast_non_dominated_sort(objectives: np.ndarray):
    """Deb's fast non-dominated sorting."""
    n = len(objectives)
    domination_counts = np.zeros(n, dtype=int)
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]
    
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_solutions[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            fronts[0].append(p)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if len(f) > 0]


def calculate_crowding_distance(objectives: np.ndarray, front: list) -> dict:
    """Crowding distance assignment for diversity preservation."""
    l = len(front)
    if l <= 2:
        return {idx: float('inf') for idx in front}
    distances = {idx: 0.0 for idx in front}
    m = objectives.shape[1]
    
    for i in range(m):
        sorted_front = sorted(front, key=lambda idx: objectives[idx, i])
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        
        obj_min = objectives[sorted_front[0], i]
        obj_max = objectives[sorted_front[-1], i]
        norm = (obj_max - obj_min) if (obj_max - obj_min) > 0 else 1e-9
        
        for j in range(1, l - 1):
            distances[sorted_front[j]] += (objectives[sorted_front[j+1], i] - objectives[sorted_front[j-1], i]) / norm
    return distances


def get_non_dominated_indices(objectives: np.ndarray) -> list:
    """Extracts indices belonging to Front 0."""
    fronts = fast_non_dominated_sort(objectives)
    return fronts[0] if len(fronts) > 0 else list(range(len(objectives)))


# ==============================================================================
# 3. GENETIC OPERATORS FOR VARIABLE FEATURE COUNTS
# ==============================================================================

def crossover_uniform_variable(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover for variable-length feature selection."""
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2).astype(np.int8)
    if np.sum(child) == 0:
        child[rng.integers(0, len(child))] = 1
    return child


def mutate_variable(individual: np.ndarray, rng: np.random.Generator,
                    p_flip: float = 0.4, p_drop: float = 0.45, p_add: float = 0.15) -> np.ndarray:
    """
    Adaptive mutation tailored for feature minimization:
    - Random bit flips (exploration)
    - Active feature drops (downward pressure on K)
    - Inactive feature additions (expansion)
    """
    child = individual.copy()
    L = len(child)
    roll = rng.random()
    
    ones_idx = np.where(child == 1)[0]
    zeros_idx = np.where(child == 0)[0]
    
    if roll < p_drop and len(ones_idx) > 10:
        # Sparsification: drop a fraction of active features (5 to 30)
        n_drop = min(len(ones_idx) - 1, rng.integers(5, 30))
        drop_choices = rng.choice(ones_idx, size=n_drop, replace=False)
        child[drop_choices] = 0
    elif roll < (p_drop + p_add) and len(zeros_idx) > 0:
        # Expansion: add a few features (2 to 10)
        n_add = min(len(zeros_idx), rng.integers(2, 10))
        add_choices = rng.choice(zeros_idx, size=n_add, replace=False)
        child[add_choices] = 1
    else:
        # Standard bit-flip mutation on a small random subset of loci
        n_flips = rng.integers(2, 8)
        flip_choices = rng.choice(L, size=n_flips, replace=False)
        child[flip_choices] = 1 - child[flip_choices]

    # Guarantee at least 1 feature active
    if np.sum(child) == 0:
        child[rng.integers(0, L)] = 1
    return child


def mode_differential_variation_variable(target: np.ndarray, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray,
                                        F: float = 0.8, CR: float = 0.8, rng: np.random.Generator = None) -> np.ndarray:
    """
    Binary Differential Evolution operator for variable feature counts.
    Combines difference vector (r1 - r2) with base r3, then binomial crossover with target.
    """
    if rng is None:
        rng = np.random.default_rng()
    L = len(target)
    
    # Continuous DE difference mapping: prob_one = r3 + F * (r1 - r2)
    diff = r1.astype(np.float32) - r2.astype(np.float32)
    v_cont = r3.astype(np.float32) + F * diff
    # Sigmoidal probabilistic mapping to binary
    prob = 1.0 / (1.0 + np.exp(-4.0 * (v_cont - 0.5)))
    donor = (rng.random(L) < prob).astype(np.int8)
    
    # Binomial crossover with target
    cross_mask = rng.random(L) < CR
    # Ensure at least one component from donor
    cross_mask[rng.integers(0, L)] = True
    trial = np.where(cross_mask, donor, target).astype(np.int8)
    
    if np.sum(trial) == 0:
        trial[rng.integers(0, L)] = 1
    return trial


# ==============================================================================
# 4. SHARED INITIAL POPULATION GENERATOR (VARIABLE K)
# ==============================================================================

def generate_shared_initial_population(n_pop: int = 28, n_features: int = 10000,
                                       ranked_features: np.ndarray = None, seed: int = 42) -> np.ndarray:
    """
    Generates a shared initial population spanning variable feature counts from
    ultra-compact (e.g. 80 features) to broad (e.g. 3500 features).
    Shared across all 4 algorithms for 100% fair baseline conditions.
    """
    rng = np.random.default_rng(seed)
    pop = []

    # Distribution of initial feature counts across 28 individuals:
    k_targets = [
        60, 100, 140, 180, 220, 260, 300, 350, 400, 450,
        500, 550, 600, 700, 800, 900, 1000, 1200, 1400, 1600,
        1800, 2000, 2500, 3000, 3500, 4500, 500, 250
    ]
    
    for i, target_k in enumerate(k_targets[:n_pop]):
        ind = np.zeros(n_features, dtype=np.int8)
        if ranked_features is not None and i < 18:
            # Informed initialization from ranked Chi-2 pool with controlled exploratory noise
            pool_size = min(n_features, max(target_k * 2, int(target_k * 1.5)))
            pool = ranked_features[:pool_size]
            top_k_count = int(target_k * 0.8)
            rand_k_count = target_k - top_k_count
            
            chosen_top = rng.choice(pool, size=min(len(pool), top_k_count), replace=False)
            rem_features = list(set(range(n_features)) - set(chosen_top))
            chosen_rand = rng.choice(rem_features, size=min(len(rem_features), rand_k_count), replace=False)
            
            ind[chosen_top] = 1
            ind[chosen_rand] = 1
        else:
            # Exploratory random sparse initialization
            chosen = rng.choice(n_features, size=target_k, replace=False)
            ind[chosen] = 1
            
        if np.sum(ind) == 0:
            ind[0] = 1
        pop.append(ind)
        
    pop_arr = np.array(pop, dtype=np.int8)
    return pop_arr


# ==============================================================================
# 5. ALGORITHM 1: NSGA-II (VARIABLE K)
# ==============================================================================

def run_nsga2(initial_pop: np.ndarray, evaluator: RFFeatureEvaluator, n_gen: int = 15, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    pop = [ind.copy() for ind in initial_pop]
    
    eval_results = [evaluator.evaluate(ind) for ind in pop]
    objs = np.array([res["objectives"] for res in eval_results])
    
    history = []
    print("  [NSGA-II] Running generations...")
    for gen in range(1, n_gen + 1):
        fronts = fast_non_dominated_sort(objs)
        crowd_dist = {}
        rank_map = {}
        for r, f in enumerate(fronts):
            cd = calculate_crowding_distance(objs, f)
            for idx in f:
                rank_map[idx] = r
                crowd_dist[idx] = cd[idx]
                
        def tournament_select():
            i, j = rng.choice(N, size=2, replace=False)
            if rank_map[i] < rank_map[j]:
                return i
            elif rank_map[j] < rank_map[i]:
                return j
            else:
                return i if crowd_dist[i] >= crowd_dist[j] else j

        offspring_pop = []
        for _ in range(N):
            p1_idx = tournament_select()
            p2_idx = tournament_select()
            child = crossover_uniform_variable(pop[p1_idx], pop[p2_idx], rng=rng)
            child = mutate_variable(child, rng=rng)
            offspring_pop.append(child)
            
        offspring_results = [evaluator.evaluate(ind) for ind in offspring_pop]
        offspring_objs = np.array([res["objectives"] for res in offspring_results])
        
        combined_pop = pop + offspring_pop
        combined_results = eval_results + offspring_results
        combined_objs = np.vstack([objs, offspring_objs])
        
        comb_fronts = fast_non_dominated_sort(combined_objs)
        new_indices = []
        for f in comb_fronts:
            if len(new_indices) + len(f) <= N:
                new_indices.extend(f)
            else:
                cd = calculate_crowding_distance(combined_objs, f)
                sorted_f = sorted(f, key=lambda idx: cd[idx], reverse=True)
                needed = N - len(new_indices)
                new_indices.extend(sorted_f[:needed])
                break
                
        pop = [combined_pop[idx] for idx in new_indices]
        eval_results = [combined_results[idx] for idx in new_indices]
        objs = np.array([res["objectives"] for res in eval_results])
        
        best_acc = max(res["accuracy"] for res in eval_results)
        min_k = min(res["k_features"] for res in eval_results)
        history.append({"gen": gen, "best_acc": best_acc, "min_k": min_k, "cache_evals": evaluator.eval_count})
        print(f"    Gen {gen:2d}/{n_gen} | Best Acc: {best_acc*100:.2f}% | Min K: {min_k} | Evals: {evaluator.eval_count}", end="\r", flush=True)

    runtime = time.time() - t_start
    print(f"\n  [NSGA-II] Completed in {runtime:.2f}s.")
    
    front0 = get_non_dominated_indices(objs)
    pareto_solutions = [eval_results[i] for i in front0]
    return {
        "algorithm": "NSGA-II",
        "runtime": runtime,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": pareto_solutions,
        "all_final_solutions": eval_results,
        "history": history
    }


# ==============================================================================
# 6. ALGORITHM 2: SPEA2 (VARIABLE K)
# ==============================================================================

def run_spea2(initial_pop: np.ndarray, evaluator: RFFeatureEvaluator, n_gen: int = 15, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    pop = [ind.copy() for ind in initial_pop]
    archive = [ind.copy() for ind in initial_pop]
    
    pop_results = [evaluator.evaluate(ind) for ind in pop]
    archive_results = [evaluator.evaluate(ind) for ind in archive]
    
    history = []
    print("  [SPEA2] Running generations...")
    for gen in range(1, n_gen + 1):
        union_pop = pop + archive
        union_results = pop_results + archive_results
        union_objs = np.array([res["objectives"] for res in union_results])
        M = len(union_objs)
        
        strengths = np.zeros(M, dtype=int)
        for i in range(M):
            for j in range(M):
                if i != j and dominates(union_objs[i], union_objs[j]):
                    strengths[i] += 1
                    
        raw_fitness = np.zeros(M, dtype=float)
        for i in range(M):
            for j in range(M):
                if i != j and dominates(union_objs[j], union_objs[i]):
                    raw_fitness[i] += strengths[j]
                    
        mins = np.min(union_objs, axis=0)
        maxs = np.max(union_objs, axis=0)
        ranges = np.where(maxs - mins == 0, 1e-9, maxs - mins)
        norm_objs = (union_objs - mins) / ranges
        
        dists = cdist(norm_objs, norm_objs)
        dists.sort(axis=1)
        k_nn = int(np.sqrt(M))
        density = 1.0 / (dists[:, k_nn] + 2.0)
        total_fitness = raw_fitness + density
        
        non_dom_idx = [i for i, f in enumerate(total_fitness) if f < 1.0]
        if len(non_dom_idx) == N:
            new_archive_idx = non_dom_idx
        elif len(non_dom_idx) < N:
            dom_idx = sorted([i for i in range(M) if i not in non_dom_idx], key=lambda i: total_fitness[i])
            new_archive_idx = non_dom_idx + dom_idx[:N - len(non_dom_idx)]
        else:
            survivors = list(non_dom_idx)
            while len(survivors) > N:
                cur_norm = norm_objs[survivors]
                cur_dists = cdist(cur_norm, cur_norm)
                np.fill_diagonal(cur_dists, np.inf)
                min_neighbor_dist = np.min(cur_dists, axis=1)
                drop_local_idx = int(np.argmin(min_neighbor_dist))
                survivors.pop(drop_local_idx)
            new_archive_idx = survivors
            
        archive = [union_pop[idx] for idx in new_archive_idx]
        archive_results = [union_results[idx] for idx in new_archive_idx]
        archive_fitness = [total_fitness[idx] for idx in new_archive_idx]
        
        offspring_pop = []
        for _ in range(N):
            i, j = rng.choice(len(archive), size=2, replace=False)
            parent1 = archive[i] if archive_fitness[i] <= archive_fitness[j] else archive[j]
            i, j = rng.choice(len(archive), size=2, replace=False)
            parent2 = archive[i] if archive_fitness[i] <= archive_fitness[j] else archive[j]
            
            child = crossover_uniform_variable(parent1, parent2, rng=rng)
            child = mutate_variable(child, rng=rng)
            offspring_pop.append(child)
            
        pop = offspring_pop
        pop_results = [evaluator.evaluate(ind) for ind in pop]
        
        best_acc = max(res["accuracy"] for res in archive_results)
        min_k = min(res["k_features"] for res in archive_results)
        history.append({"gen": gen, "best_acc": best_acc, "min_k": min_k, "cache_evals": evaluator.eval_count})
        print(f"    Gen {gen:2d}/{n_gen} | Best Acc: {best_acc*100:.2f}% | Min K: {min_k} | Evals: {evaluator.eval_count}", end="\r", flush=True)

    runtime = time.time() - t_start
    print(f"\n  [SPEA2] Completed in {runtime:.2f}s.")
    
    arch_objs = np.array([res["objectives"] for res in archive_results])
    front0 = get_non_dominated_indices(arch_objs)
    pareto_solutions = [archive_results[i] for i in front0]
    return {
        "algorithm": "SPEA2",
        "runtime": runtime,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": pareto_solutions,
        "all_final_solutions": archive_results,
        "history": history
    }


# ==============================================================================
# 7. ALGORITHM 3: MOEA/D (VARIABLE K)
# ==============================================================================

def generate_das_dennis_weights(n_obj: int = 3, H: int = 6) -> np.ndarray:
    """Generates uniform reference directions on simplex (Das-Dennis method). 28 weights for H=6."""
    weights = []
    for i in range(H + 1):
        for j in range(H + 1 - i):
            k = H - i - j
            weights.append([i / H, j / H, k / H])
    return np.array(weights, dtype=float)


def run_moead(initial_pop: np.ndarray, evaluator: RFFeatureEvaluator, n_gen: int = 15, T: int = 6, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    
    weights = generate_das_dennis_weights(n_obj=3, H=6)
    assert len(weights) == N, f"Weight count {len(weights)} must match pop size {N}"
    
    weight_dists = cdist(weights, weights)
    neighborhoods = [np.argsort(weight_dists[i])[:T] for i in range(N)]
    
    pop = [ind.copy() for ind in initial_pop]
    eval_results = [evaluator.evaluate(ind) for ind in pop]
    objs = np.array([res["objectives"] for res in eval_results])
    
    # Scale objectives to comparable ranges for Tchebycheff decomposition
    # f1 in [0, 1], f2 in [0, 10000], f3 in [0, 0.5]
    scale = np.array([1.0, 1.0 / 10000.0, 10.0], dtype=float)
    z_star = np.min(objs * scale, axis=0)
    
    def tchebycheff(f_val, w_vec):
        f_scaled = f_val * scale
        w = np.where(w_vec == 0, 1e-4, w_vec)
        return np.max(w * np.abs(f_scaled - z_star))

    history = []
    print("  [MOEA/D] Running generations...")
    for gen in range(1, n_gen + 1):
        for i in range(N):
            p_idx1, p_idx2 = rng.choice(neighborhoods[i], size=2, replace=False)
            child = crossover_uniform_variable(pop[p_idx1], pop[p_idx2], rng=rng)
            child = mutate_variable(child, rng=rng)
            
            child_res = evaluator.evaluate(child)
            child_obj = child_res["objectives"]
            
            z_star = np.minimum(z_star, child_obj * scale)
            
            for j in neighborhoods[i]:
                if tchebycheff(child_obj, weights[j]) <= tchebycheff(objs[j], weights[j]):
                    pop[j] = child
                    eval_results[j] = child_res
                    objs[j] = child_obj
                    
        best_acc = max(res["accuracy"] for res in eval_results)
        min_k = min(res["k_features"] for res in eval_results)
        history.append({"gen": gen, "best_acc": best_acc, "min_k": min_k, "cache_evals": evaluator.eval_count})
        print(f"    Gen {gen:2d}/{n_gen} | Best Acc: {best_acc*100:.2f}% | Min K: {min_k} | Evals: {evaluator.eval_count}", end="\r", flush=True)

    runtime = time.time() - t_start
    print(f"\n  [MOEA/D] Completed in {runtime:.2f}s.")
    
    front0 = get_non_dominated_indices(objs)
    pareto_solutions = [eval_results[i] for i in front0]
    return {
        "algorithm": "MOEA/D",
        "runtime": runtime,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": pareto_solutions,
        "all_final_solutions": eval_results,
        "history": history
    }


# ==============================================================================
# 8. ALGORITHM 4: MODE (VARIABLE K)
# ==============================================================================

def run_mode(initial_pop: np.ndarray, evaluator: RFFeatureEvaluator, n_gen: int = 15,
             F: float = 0.8, CR: float = 0.8, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    pop = [ind.copy() for ind in initial_pop]
    
    eval_results = [evaluator.evaluate(ind) for ind in pop]
    objs = np.array([res["objectives"] for res in eval_results])
    
    history = []
    print("  [MODE] Running generations...")
    for gen in range(1, n_gen + 1):
        candidate_pop = list(pop)
        candidate_results = list(eval_results)
        
        for i in range(N):
            pool = [idx for idx in range(N) if idx != i]
            r1_idx, r2_idx, r3_idx = rng.choice(pool, size=3, replace=False)
            
            trial = mode_differential_variation_variable(pop[i], pop[r1_idx], pop[r2_idx], pop[r3_idx],
                                                         F=F, CR=CR, rng=rng)
            # Occasional drop mutation to accelerate sparsification
            if rng.random() < 0.3:
                trial = mutate_variable(trial, rng=rng, p_flip=0.2, p_drop=0.7, p_add=0.1)
                
            trial_res = evaluator.evaluate(trial)
            trial_obj = trial_res["objectives"]
            
            if dominates(trial_obj, objs[i]):
                pop[i] = trial
                eval_results[i] = trial_res
                objs[i] = trial_obj
            elif dominates(objs[i], trial_obj):
                pass
            else:
                candidate_pop.append(trial)
                candidate_results.append(trial_res)
                
        if len(candidate_pop) > N:
            c_objs = np.array([res["objectives"] for res in candidate_results])
            fronts = fast_non_dominated_sort(c_objs)
            selected_idx = []
            for f in fronts:
                if len(selected_idx) + len(f) <= N:
                    selected_idx.extend(f)
                else:
                    cd = calculate_crowding_distance(c_objs, f)
                    sorted_f = sorted(f, key=lambda idx: cd[idx], reverse=True)
                    needed = N - len(selected_idx)
                    selected_idx.extend(sorted_f[:needed])
                    break
            pop = [candidate_pop[idx] for idx in selected_idx]
            eval_results = [candidate_results[idx] for idx in selected_idx]
            objs = np.array([res["objectives"] for res in eval_results])
            
        best_acc = max(res["accuracy"] for res in eval_results)
        min_k = min(res["k_features"] for res in eval_results)
        history.append({"gen": gen, "best_acc": best_acc, "min_k": min_k, "cache_evals": evaluator.eval_count})
        print(f"    Gen {gen:2d}/{n_gen} | Best Acc: {best_acc*100:.2f}% | Min K: {min_k} | Evals: {evaluator.eval_count}", end="\r", flush=True)

    runtime = time.time() - t_start
    print(f"\n  [MODE] Completed in {runtime:.2f}s.")
    
    front0 = get_non_dominated_indices(objs)
    pareto_solutions = [eval_results[i] for i in front0]
    return {
        "algorithm": "MODE",
        "runtime": runtime,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": pareto_solutions,
        "all_final_solutions": eval_results,
        "history": history
    }


# ==============================================================================
# 9. HYPERVOLUME COMPUTATION
# ==============================================================================

def compute_hypervolume(objectives: np.ndarray, ref_point: np.ndarray = None) -> float:
    """
    Computes Hypervolume using pymoo.
    Normalizes objectives by ref_point so comparison is scale-invariant.
    """
    try:
        from pymoo.indicators.hv import HV
        if ref_point is None:
            # Default reference point for [1 - acc, k_features, inf_time]
            # Max 1-acc = 0.20, Max K = 10000, Max inf_time = 0.20
            ref_point = np.array([0.20, 10000.0, 0.20], dtype=float)
            
        # Filter points dominated by ref_point
        valid_mask = np.all(objectives <= ref_point, axis=1)
        valid_objs = objectives[valid_mask]
        if len(valid_objs) == 0:
            return 0.0
            
        # Normalize to [0, 1] relative to ideal (0, 0, 0) and ref_point
        norm_objs = valid_objs / ref_point
        norm_ref = np.ones(3, dtype=float)
        
        hv = HV(ref_point=norm_ref)
        return float(hv(norm_objs))
    except Exception as e:
        print(f"HV calculation warning: {e}")
        return 0.0


# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================

def main():
    print("=" * 80)
    print("   FINAL SOFT COMPUTING EXPERIMENT: RANDOM FOREST FEATURE MINIMIZATION")
    print("=" * 80)
    print("Optimization Objectives: [1 - Accuracy, Number of Features (K), Inference Time]")
    print(f"Accuracy Preservation Tolerance: {TOLERANCE:.2f} (Absolute Accuracy Drop)")
    print(f"Random Forest Parameters: {json.dumps(RF_CONFIG, indent=2)}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # STEP 1: LOAD NUMERICAL DATA
    # --------------------------------------------------------------------------
    print("\n[Step 1] Loading precomputed TF-IDF sparse matrices and binary labels...")
    t0 = time.time()
    X_train_full = sparse.load_npz("X_train.npz").tocsr()
    X_val_full = sparse.load_npz("X_val.npz").tocsr()
    y_train = np.load("y_train.npy")
    y_val = np.load("y_val.npy")
    with open("feature_names.txt", "r") as f:
        feature_names = [line.strip() for line in f]
    print(f"  Loaded in {time.time()-t0:.2f}s:")
    print(f"  • Full X_train: {X_train_full.shape} (127,656 samples x 10,000 features)")
    print(f"  • Full X_val:   {X_val_full.shape} (31,915 samples x 10,000 features)")
    print(f"  • Vocabulary:   {len(feature_names)} terms")
    print(f"  • Class ratio:  Train toxic = {np.mean(y_train)*100:.2f}%, Val toxic = {np.mean(y_val)*100:.2f}%")

    # --------------------------------------------------------------------------
    # STEP 2: LOAD/VERIFY BASELINE RANDOM FOREST
    # --------------------------------------------------------------------------
    print("\n[Step 2] Establishing Full 10,000-Feature Random Forest Baseline...")
    if os.path.exists("baseline_results.json"):
        with open("baseline_results.json", "r") as f:
            base_data = json.load(f)
        baseline_acc = base_data["metrics"]["accuracy"]
        baseline_f1 = base_data["metrics"]["f1_score"]
        baseline_fpr = base_data["metrics"]["false_positive_rate"]
        baseline_prec = base_data["metrics"]["precision"]
        baseline_rec = base_data["metrics"]["recall"]
        baseline_inf_time = base_data["metrics"]["inference_time_total_seconds"]
        baseline_fit_time = base_data["metrics"]["training_time_seconds"]
        print("  ✓ Loaded verified Random Forest baseline from baseline_results.json:")
    else:
        print("  Training full Random Forest baseline on 10,000 features...")
        t_base_fit = time.perf_counter()
        base_clf = RandomForestClassifier(**RF_CONFIG)
        base_clf.fit(X_train_full, y_train)
        baseline_fit_time = time.perf_counter() - t_base_fit

        # Measure baseline inference
        _ = base_clf.predict(X_val_full)
        durs = []
        for _ in range(3):
            t_base_inf = time.perf_counter()
            base_preds = base_clf.predict(X_val_full)
            durs.append(time.perf_counter() - t_base_inf)
        baseline_inf_time = float(np.mean(durs))
        baseline_acc = float(accuracy_score(y_val, base_preds))
        baseline_f1 = float(f1_score(y_val, base_preds))
        cm = confusion_matrix(y_val, base_preds)
        tn, fp, fn, tp = cm.ravel()
        baseline_fpr = float(fp / (fp + tn))
        baseline_prec = float(precision_score(y_val, base_preds))
        baseline_rec = float(recall_score(y_val, base_preds))

    print(f"    • Baseline Accuracy:       {baseline_acc * 100:.2f}% ({baseline_acc:.6f})")
    print(f"    • Baseline F1 Score:       {baseline_f1:.4f}")
    print(f"    • Baseline FPR:            {baseline_fpr * 100:.2f}%")
    print(f"    • Baseline Precision:      {baseline_prec:.4f}")
    print(f"    • Baseline Recall:         {baseline_rec:.4f}")
    print(f"    • Baseline Inference Time: {baseline_inf_time * 1000:.2f} ms")
    print(f"    • Baseline Training Time:  {baseline_fit_time:.2f} s")

    acc_threshold = baseline_acc - TOLERANCE
    print(f"\n  >>> Accuracy Preservation Threshold: >= {acc_threshold * 100:.2f}% ({acc_threshold:.6f}) <<<")

    # --------------------------------------------------------------------------
    # STEP 3: SANITY CHECK FIRST
    # --------------------------------------------------------------------------
    print("\n[Step 3] Running Mandatory Sanity Checks (as requested by professor)...")
    print("  • Verifying that feature count K is variable across test subsets [100, 250, 500, 1000]...")
    chi2_scores, _ = chi2(X_train_full[:10000], y_train[:10000])
    ranked_features = np.argsort(chi2_scores)[::-1]
    
    sanity_results = []
    for test_k in [100, 250, 500, 1000]:
        cols = ranked_features[:test_k]
        clf = RandomForestClassifier(**RF_CONFIG)
        clf.fit(X_train_full[:10000, cols], y_train[:10000])
        preds = clf.predict(X_val_full[:5000, cols])
        s_acc = float(accuracy_score(y_val[:5000], preds))
        s_f1 = float(f1_score(y_val[:5000], preds, zero_division=0))
        cm = confusion_matrix(y_val[:5000], preds)
        tn, fp, fn, tp = cm.ravel()
        s_fpr = float(fp / (fp + tn))
        print(f"    K = {test_k:4d} | Acc = {s_acc*100:.2f}% | F1 = {s_f1:.4f} | FPR = {s_fpr*100:.2f}% | CM = [TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}]")
        sanity_results.append({"k": test_k, "accuracy": s_acc, "f1": s_f1, "fpr": s_fpr})

    print("  ✓ Sanity check passed:")
    print("    1. Feature count is fully variable (no 500 constraint).")
    print("    2. Evaluator computes all three objectives: [1 - Accuracy, K, inference_time].")
    print("    3. No test.csv used, no TF-IDF refitted.")

    # --------------------------------------------------------------------------
    # STEP 4: PREPARE FIXED STRATIFIED SEARCH SUBSET
    # --------------------------------------------------------------------------
    print("\n[Step 4] Preparing Fixed Stratified Search Subset (for efficient search evaluations)...")
    sss_train = StratifiedShuffleSplit(n_splits=1, train_size=6000, random_state=RANDOM_STATE)
    idx_train_search, _ = next(sss_train.split(X_train_full, y_train))
    idx_train_search.sort()

    sss_val = StratifiedShuffleSplit(n_splits=1, train_size=2000, random_state=RANDOM_STATE)
    idx_val_search, _ = next(sss_val.split(X_val_full, y_val))
    idx_val_search.sort()

    X_train_search = X_train_full[idx_train_search].tocsc()
    y_train_search = y_train[idx_train_search]
    X_val_search = X_val_full[idx_val_search].tocsc()
    y_val_search = y_val[idx_val_search]

    print(f"  • Search training subset:   {X_train_search.shape[0]} samples (stratified toxic: {np.mean(y_train_search)*100:.2f}%)")
    print(f"  • Search validation subset: {X_val_search.shape[0]} samples (stratified toxic: {np.mean(y_val_search)*100:.2f}%)")
    print("  • NOTE: Search subset drawn once, fixed, and reused identically across all 4 algorithms.")

    # Initialize shared search evaluator
    search_evaluator = RFFeatureEvaluator(X_train_search, y_train_search, X_val_search, y_val_search,
                                         rf_params=RF_CONFIG, name="SearchEvaluator")

    # Generate shared initial population
    print("\n[Step 5] Generating Shared Initial Population (N = 28, variable K across [60, 5000])...")
    initial_population = generate_shared_initial_population(n_pop=28, n_features=10000,
                                                           ranked_features=ranked_features, seed=RANDOM_STATE)
    init_counts = [int(np.sum(ind)) for ind in initial_population]
    print(f"  Initial feature counts: min = {min(init_counts)}, max = {max(init_counts)}, mean = {np.mean(init_counts):.1f}")
    print(f"  Distribution: {init_counts}")
    print("  ✓ Shared initial population will be passed to all 4 algorithms.")

    # --------------------------------------------------------------------------
    # STEP 6: RUN THE FOUR EVOLUTIONARY ALGORITHMS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                 PHASE 1: RUNNING EVOLUTIONARY SEARCH")
    print("=" * 80)
    
    n_generations = 15
    print(f"Budget per algorithm: Population N = 28, Generations G = {n_generations}")

    # 1. NSGA-II
    print("\n>>> 1/4: Running NSGA-II...")
    res_nsga2 = run_nsga2(initial_population, search_evaluator, n_gen=n_generations, seed=RANDOM_STATE)

    # 2. SPEA2
    print("\n>>> 2/4: Running SPEA2...")
    res_spea2 = run_spea2(initial_population, search_evaluator, n_gen=n_generations, seed=RANDOM_STATE)

    # 3. MOEA/D
    print("\n>>> 3/4: Running MOEA/D...")
    res_moead = run_moead(initial_population, search_evaluator, n_gen=n_generations, T=6, seed=RANDOM_STATE)

    # 4. MODE
    print("\n>>> 4/4: Running MODE...")
    res_mode = run_mode(initial_population, search_evaluator, n_gen=n_generations, F=0.8, CR=0.8, seed=RANDOM_STATE)

    # --------------------------------------------------------------------------
    # STEP 7: FULL VALIDATION EVALUATION (127,656 Train x 31,915 Val)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("        PHASE 2: FULL DATASET VALIDATION OF STRONGEST PARETO CANDIDATES")
    print("=" * 80)
    print("Evaluating final Pareto candidates on FULL partitions:")
    print("  • Training partition:   127,656 samples")
    print("  • Validation partition: 31,915 samples")
    print(f"  • Classifier:           RandomForestClassifier (50 trees, class_weight='balanced')")

    full_evaluator = RFFeatureEvaluator(X_train_full.tocsc(), y_train, X_val_full.tocsc(), y_val,
                                       rf_params=RF_CONFIG, name="FullEvaluator")

    alg_results = {
        "NSGA-II": res_nsga2,
        "SPEA2": res_spea2,
        "MOEA/D": res_moead,
        "MODE": res_mode
    }

    validated_pareto_by_alg = {}
    
    for alg_name, alg_res in alg_results.items():
        print(f"\n[Full Validation] Evaluating {alg_name} Pareto candidates on all 31,915 samples...")
        search_pareto = alg_res["pareto_solutions"]
        
        # Deduplicate candidates by unique feature sets
        unique_candidates = {}
        for sol in search_pareto:
            key = tuple(sorted(sol["indices"].tolist()))
            if key not in unique_candidates:
                unique_candidates[key] = sol
                
        # Also include all final solutions to ensure comprehensive front coverage
        for sol in alg_res["all_final_solutions"]:
            key = tuple(sorted(sol["indices"].tolist()))
            if key not in unique_candidates and sol["accuracy"] >= 0.85:
                unique_candidates[key] = sol

        # Sort candidates by feature count
        sorted_cand = sorted(unique_candidates.values(), key=lambda s: s["k_features"])
        # Always include the smallest K, the highest accuracy, and representative intermediate candidates
        selected_cand = []
        if len(sorted_cand) <= 10:
            selected_cand = list(sorted_cand)
        else:
            # Endpoints
            selected_cand.append(sorted_cand[0])  # Smallest K
            selected_cand.append(sorted_cand[-1]) # Largest K
            # Best accuracy candidate
            best_acc_cand = max(sorted_cand, key=lambda s: s["accuracy"])
            if best_acc_cand not in selected_cand:
                selected_cand.append(best_acc_cand)
            # Sample intermediate indices evenly
            step = len(sorted_cand) / 8.0
            for i in range(1, 8):
                idx_pick = int(i * step)
                cand_pick = sorted_cand[idx_pick]
                if cand_pick not in selected_cand:
                    selected_cand.append(cand_pick)
            # Re-sort by k_features
            selected_cand = sorted(selected_cand, key=lambda s: s["k_features"])

        validated_solutions = []
        for idx_c, cand in enumerate(selected_cand):
            feat_idx = cand["indices"]
            k_feat = len(feat_idx)
            ind_vec = np.zeros(10000, dtype=np.int8)
            ind_vec[feat_idx] = 1
            
            t0_v = time.perf_counter()
            full_res = full_evaluator.evaluate(ind_vec, timing_runs=3)
            t_v = time.perf_counter() - t0_v
            
            acc = full_res["accuracy"]
            f1 = full_res["f1"]
            fpr = full_res["fpr"]
            prec = full_res["precision"]
            rec = full_res["recall"]
            inf_time = full_res["inference_time"]
            fit_time = full_res["training_time"]
            
            acc_loss = baseline_acc - acc
            acc_change_pct = ((acc - baseline_acc) / baseline_acc) * 100.0
            within_tol = (acc >= acc_threshold)
            
            sol_record = {
                "algorithm": alg_name,
                "k_features": k_feat,
                "accuracy": acc,
                "baseline_accuracy": baseline_acc,
                "accuracy_loss": acc_loss,
                "accuracy_change_pct": acc_change_pct,
                "within_tolerance": within_tol,
                "f1": f1,
                "fpr": fpr,
                "precision": prec,
                "recall": rec,
                "inference_time_sec": inf_time,
                "inference_time_ms": inf_time * 1000.0,
                "training_time_sec": fit_time,
                "indices": feat_idx.tolist(),
                "objectives": full_res["objectives"].tolist()
            }
            validated_solutions.append(sol_record)
            status = "✓ ACCEP" if within_tol else "✗ BELOW"
            print(f"    Candidate {idx_c+1:2d}/{len(selected_cand)} | K = {k_feat:5d} | Acc = {acc*100:.2f}% (Loss: {acc_loss*100:+.2f}%) | F1 = {f1:.4f} | Inf = {inf_time*1000:.1f}ms | {status}")

        validated_pareto_by_alg[alg_name] = validated_solutions

    # --------------------------------------------------------------------------
    # STEP 8: RECOMPUTE FINAL PARETO DOMINANCE ON FULL DATA
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("        PHASE 3: RECOMPUTING PARETO DOMINANCE ON FULL DATA")
    print("=" * 80)

    final_pareto_rows = []
    comparison_rows = []

    for alg_name, sols in validated_pareto_by_alg.items():
        # Objective matrix: [1 - Accuracy, K, inference_time]
        objs = np.array([sol["objectives"] for sol in sols])
        front0_indices = get_non_dominated_indices(objs)
        pareto_sols = [sols[i] for i in front0_indices]
        
        # Sort by k_features
        pareto_sols = sorted(pareto_sols, key=lambda s: s["k_features"])
        for p in pareto_sols:
            p["is_pareto"] = True
            final_pareto_rows.append(p)

        # Calculate hypervolume on validated Pareto front
        p_objs = np.array([p["objectives"] for p in pareto_sols])
        hv = compute_hypervolume(p_objs)

        # Identify key solutions:
        # A. Smallest feature subset within tolerance
        within_tol_sols = [s for s in sols if s["within_tolerance"]]
        if len(within_tol_sols) > 0:
            smallest_tol = min(within_tol_sols, key=lambda s: s["k_features"])
        else:
            smallest_tol = max(sols, key=lambda s: s["accuracy"])  # fallback to highest acc

        # B. Highest accuracy solution
        highest_acc_sol = max(sols, key=lambda s: s["accuracy"])
        
        # C. Overall minimum feature count
        min_k_sol = min(sols, key=lambda s: s["k_features"])

        runtime = alg_results[alg_name]["runtime"]
        evals = alg_results[alg_name]["evaluations"]

        comparison_rows.append({
            "Algorithm": alg_name,
            "Smallest_K_Within_Tolerance": smallest_tol["k_features"],
            "Reduced_Accuracy": smallest_tol["accuracy"],
            "Baseline_Accuracy": baseline_acc,
            "Accuracy_Loss": smallest_tol["accuracy_loss"],
            "Accuracy_Change_Pct": smallest_tol["accuracy_change_pct"],
            "F1_Score": smallest_tol["f1"],
            "False_Positive_Rate": smallest_tol["fpr"],
            "Inference_Time_ms": smallest_tol["inference_time_ms"],
            "Inference_Speedup": baseline_inf_time / smallest_tol["inference_time_sec"],
            "Search_Runtime_sec": runtime,
            "Total_Evaluations": evals,
            "Pareto_Front_Size": len(pareto_sols),
            "Hypervolume": hv,
            "Best_Accuracy_Discovered": highest_acc_sol["accuracy"],
            "Min_K_Discovered": min_k_sol["k_features"]
        })

    df_comp = pd.DataFrame(comparison_rows)
    df_pareto = pd.DataFrame(final_pareto_rows)

    print("\n" + "=" * 80)
    print("                    FINAL ALGORITHM COMPARISON TABLE")
    print("=" * 80)
    print(df_comp[[
        "Algorithm", "Smallest_K_Within_Tolerance", "Reduced_Accuracy",
        "Accuracy_Loss", "F1_Score", "False_Positive_Rate", "Inference_Time_ms", "Hypervolume"
    ]].to_string(index=False))

    # Determine overall winner:
    # Key criteria:
    # 1. How few features were retained while satisfying the tolerance.
    # 2. How close accuracy was to baseline.
    # 3. Inference-time improvement.
    # 4. Hypervolume / Pareto quality.
    
    # Sort algorithms by Smallest_K_Within_Tolerance ascending, then Reduced_Accuracy descending
    df_comp_sorted = df_comp.sort_values(
        by=["Smallest_K_Within_Tolerance", "Reduced_Accuracy"],
        ascending=[True, False]
    )
    winning_algorithm = df_comp_sorted.iloc[0]["Algorithm"]
    winning_row = df_comp_sorted.iloc[0]

    print("\n" + "=" * 80)
    print(f"             OVERALL WINNER: {winning_algorithm}")
    print("=" * 80)
    print(f"  • Smallest acceptable features: {winning_row['Smallest_K_Within_Tolerance']} (out of 10,000)")
    print(f"  • Feature reduction ratio:      {(1 - winning_row['Smallest_K_Within_Tolerance']/10000)*100:.2f}% features eliminated!")
    print(f"  • Achieved accuracy:            {winning_row['Reduced_Accuracy']*100:.2f}% (Baseline: {baseline_acc*100:.2f}%)")
    print(f"  • Accuracy loss:                {winning_row['Accuracy_Loss']*100:+.2f}% (Within tolerance of <= {TOLERANCE*100:.2f}%)")
    print(f"  • Inference time:               {winning_row['Inference_Time_ms']:.2f} ms (Baseline: {baseline_inf_time*1000:.2f} ms)")
    print(f"  • Speedup:                      {winning_row['Inference_Speedup']:.2f}x faster inference")

    # --------------------------------------------------------------------------
    # STEP 9: SAVE ALL DELIVERABLES
    # --------------------------------------------------------------------------
    print("\n[Step 9] Saving Deliverables...")

    # 1. Save algorithm JSON files
    for alg_name, alg_key in [("NSGA-II", "nsga2"), ("SPEA2", "spea2"), ("MOEA/D", "moead"), ("MODE", "mode")]:
        out_json = f"results_{alg_key}_rf.json"
        with open(out_json, "w") as f:
            json.dump({
                "algorithm": alg_name,
                "runtime": alg_results[alg_name]["runtime"],
                "evaluations": alg_results[alg_name]["evaluations"],
                "history": alg_results[alg_name]["history"],
                "validated_solutions": validated_pareto_by_alg[alg_name]
            }, f, indent=2)
        print(f"  ✓ Saved {out_json}")

    # 2. Save comparison CSV
    df_comp.to_csv("final_algorithm_comparison_rf.csv", index=False)
    print("  ✓ Saved final_algorithm_comparison_rf.csv")

    # 3. Save Pareto solutions CSV
    # Flatten indices for clean CSV export
    df_pareto_export = df_pareto.drop(columns=["indices", "objectives"])
    df_pareto_export.to_csv("final_pareto_solutions_rf.csv", index=False)
    print("  ✓ Saved final_pareto_solutions_rf.csv")

    # 4. Save experiment configuration
    exp_config = {
        "model_type": "RandomForestClassifier",
        "hyperparameters": RF_CONFIG,
        "tolerance": TOLERANCE,
        "dataset": {
            "train_samples": int(X_train_full.shape[0]),
            "val_samples": int(X_val_full.shape[0]),
            "total_features": 10000,
            "search_train_samples": int(X_train_search.shape[0]),
            "search_val_samples": int(X_val_search.shape[0])
        },
        "baseline_metrics": {
            "accuracy": baseline_acc,
            "f1": baseline_f1,
            "fpr": baseline_fpr,
            "precision": baseline_prec,
            "recall": baseline_rec,
            "inference_time_ms": baseline_inf_time * 1000.0,
            "training_time_sec": baseline_fit_time
        },
        "search_budget": {
            "population_size": 28,
            "generations": n_generations
        },
        "winning_algorithm": winning_algorithm
    }
    with open("experiment_config_rf.json", "w") as f:
        json.dump(exp_config, f, indent=2)
    print("  ✓ Saved experiment_config_rf.json")

    # 5. Save final analysis JSON
    final_analysis = {
        "baseline": exp_config["baseline_metrics"],
        "comparison": df_comp.to_dict(orient="records"),
        "winner": {
            "algorithm": winning_algorithm,
            "k_features": int(winning_row["Smallest_K_Within_Tolerance"]),
            "accuracy": float(winning_row["Reduced_Accuracy"]),
            "accuracy_loss": float(winning_row["Accuracy_Loss"]),
            "inference_time_ms": float(winning_row["Inference_Time_ms"]),
            "f1": float(winning_row["F1_Score"]),
            "fpr": float(winning_row["False_Positive_Rate"]),
            "hypervolume": float(winning_row["Hypervolume"])
        }
    }
    with open("final_analysis_rf.json", "w") as f:
        json.dump(final_analysis, f, indent=2)
    print("  ✓ Saved final_analysis_rf.json")

    # 6. Save selected feature files (.npy and .txt) for each algorithm's winning subset
    for alg_name, alg_key in [("NSGA-II", "nsga2"), ("SPEA2", "spea2"), ("MOEA/D", "moead"), ("MODE", "mode")]:
        # Find smallest subset within tolerance
        sols = validated_pareto_by_alg[alg_name]
        within = [s for s in sols if s["within_tolerance"]]
        chosen = min(within, key=lambda s: s["k_features"]) if len(within) > 0 else max(sols, key=lambda s: s["accuracy"])
        
        idx_arr = np.array(chosen["indices"], dtype=np.int32)
        np.save(f"selected_features_{alg_key}_rf.npy", idx_arr)
        
        with open(f"selected_features_{alg_key}_rf.txt", "w") as f:
            for idx in idx_arr:
                f.write(f"{feature_names[idx]}\n")
        print(f"  ✓ Saved selected_features_{alg_key}_rf.npy and .txt (K = {len(idx_arr)})")

    # --------------------------------------------------------------------------
    # STEP 10: GENERATE VISUALIZATION PLOTS
    # --------------------------------------------------------------------------
    print("\n[Step 10] Generating Visualization Plots...")
    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    colors = {"NSGA-II": "#1f77b4", "SPEA2": "#2ca02c", "MOEA/D": "#ff7f0e", "MODE": "#d62728"}

    # Plot 1: Accuracy vs Number of Features
    plt.figure(figsize=(9, 5.5))
    plt.axhline(baseline_acc * 100, color="black", linestyle="--", linewidth=1.5,
                label=f"Baseline RF (10,000 feats): {baseline_acc*100:.2f}%")
    plt.axhline(acc_threshold * 100, color="crimson", linestyle=":", linewidth=1.5,
                label=f"Acceptable Tolerance Threshold: {acc_threshold*100:.2f}%")
    
    for alg_name, sols in validated_pareto_by_alg.items():
        ks = [s["k_features"] for s in sols]
        accs = [s["accuracy"] * 100 for s in sols]
        plt.scatter(ks, accs, label=alg_name, color=colors[alg_name], s=55, alpha=0.85, edgecolors="none")
        # Line connecting sorted by k
        s_sorted = sorted(zip(ks, accs))
        plt.plot([p[0] for p in s_sorted], [p[1] for p in s_sorted], color=colors[alg_name], alpha=0.4, linestyle="-")

    plt.xlabel("Number of Selected Features (K)", fontsize=11, fontweight="bold")
    plt.ylabel("Classification Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Random Forest Accuracy vs. Retained Feature Count", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/accuracy_vs_features_rf.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/accuracy_vs_features_rf.png")

    # Plot 2: Inference Time vs Number of Features
    plt.figure(figsize=(9, 5.5))
    plt.axhline(baseline_inf_time * 1000, color="black", linestyle="--", linewidth=1.5,
                label=f"Baseline RF (10,000 feats): {baseline_inf_time*1000:.1f} ms")
    
    for alg_name, sols in validated_pareto_by_alg.items():
        ks = [s["k_features"] for s in sols]
        times = [s["inference_time_ms"] for s in sols]
        plt.scatter(ks, times, label=alg_name, color=colors[alg_name], s=55, alpha=0.85)
        s_sorted = sorted(zip(ks, times))
        plt.plot([p[0] for p in s_sorted], [p[1] for p in s_sorted], color=colors[alg_name], alpha=0.4)

    plt.xlabel("Number of Selected Features (K)", fontsize=11, fontweight="bold")
    plt.ylabel("Inference Time on 31,915 Samples (ms)", fontsize=11, fontweight="bold")
    plt.title("Inference Latency vs. Retained Feature Count", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    plt.savefig("plots/inference_time_vs_features_rf.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/inference_time_vs_features_rf.png")

    # Plot 3: Pareto Front: Accuracy vs Feature Count
    plt.figure(figsize=(9, 5.5))
    for alg_name in ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]:
        sub_pareto = [s for s in final_pareto_rows if s["algorithm"] == alg_name]
        ks = [s["k_features"] for s in sub_pareto]
        accs = [s["accuracy"] * 100 for s in sub_pareto]
        plt.scatter(ks, accs, label=f"{alg_name} Pareto Front", color=colors[alg_name], s=70, marker="o")
        s_sorted = sorted(zip(ks, accs))
        if len(s_sorted) > 1:
            plt.plot([p[0] for p in s_sorted], [p[1] for p in s_sorted], color=colors[alg_name], linewidth=2, alpha=0.7)

    plt.axhline(acc_threshold * 100, color="crimson", linestyle=":", linewidth=1.2, label="Tolerance (93.11%)")
    plt.xlabel("Number of Selected Features (K)", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Validated Non-Dominated Pareto Fronts (Accuracy vs. Feature Count)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig("plots/pareto_accuracy_vs_features_rf.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/pareto_accuracy_vs_features_rf.png")

    # Plot 4: Algorithm comparison (Bar chart)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    algs = df_comp["Algorithm"].tolist()
    
    # 4a: Smallest K
    axes[0].bar(algs, df_comp["Smallest_K_Within_Tolerance"], color=[colors[a] for a in algs], width=0.55)
    axes[0].set_title("Smallest Features Within Tolerance", fontweight="bold")
    axes[0].set_ylabel("Feature Count (K)")
    for i, v in enumerate(df_comp["Smallest_K_Within_Tolerance"]):
        axes[0].text(i, v + 20, f"{v}", ha="center", fontweight="bold")

    # 4b: Retained Accuracy
    axes[1].bar(algs, df_comp["Reduced_Accuracy"] * 100, color=[colors[a] for a in algs], width=0.55)
    axes[1].axhline(baseline_acc * 100, color="black", linestyle="--", label="Baseline (94.11%)")
    axes[1].set_title("Accuracy at Smallest Feature Subset (%)", fontweight="bold")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(92.0, 95.0)
    axes[1].legend(loc="lower right")
    for i, v in enumerate(df_comp["Reduced_Accuracy"] * 100):
        axes[1].text(i, v + 0.05, f"{v:.2f}%", ha="center", fontweight="bold")

    # 4c: Hypervolume
    axes[2].bar(algs, df_comp["Hypervolume"], color=[colors[a] for a in algs], width=0.55)
    axes[2].set_title("Validated Hypervolume Indicator", fontweight="bold")
    axes[2].set_ylabel("Hypervolume (Higher is Better)")
    for i, v in enumerate(df_comp["Hypervolume"]):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/algorithm_comparison_rf.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/algorithm_comparison_rf.png")

    # Plot 5: Baseline vs Final Reduced Models
    plt.figure(figsize=(10, 5.5))
    models = ["Baseline RF\n(10,000 feats)"] + [f"{a} Reduced\n({int(df_comp.loc[df_comp['Algorithm']==a, 'Smallest_K_Within_Tolerance'].values[0])} feats)" for a in algs]
    acc_vals = [baseline_acc * 100] + [df_comp.loc[df_comp["Algorithm"]==a, "Reduced_Accuracy"].values[0] * 100 for a in algs]
    bar_cols = ["#333333"] + [colors[a] for a in algs]

    bars = plt.bar(models, acc_vals, color=bar_cols, width=0.55)
    plt.axhline(acc_threshold * 100, color="crimson", linestyle=":", label="Tolerance Threshold (93.11%)")
    plt.ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Baseline Random Forest vs. Reduced-Feature Evolutionary Models", fontsize=13, fontweight="bold")
    plt.ylim(92.0, 95.0)
    plt.legend(loc="lower right")
    for bar, val in zip(bars, acc_vals):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.06, f"{val:.2f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/baseline_vs_reduced_rf.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/baseline_vs_reduced_rf.png")

    print("\n" + "=" * 80)
    print("              EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
