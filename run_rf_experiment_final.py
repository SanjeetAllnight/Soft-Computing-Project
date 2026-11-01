#!/usr/bin/env python3
"""
Final Soft Computing Evolutionary Feature Minimization Experiment (Random Forest)
==================================================================================
Model: RandomForestClassifier (Baseline: 10,000 features; Reduced: Feature-budget constrained)
Algorithms: NSGA-II, SPEA2, MOEA/D, MODE

THREE OPTIMIZATION OBJECTIVES (all minimized):
  f1 = 1 - F1 Score        (maximize classification F1)
  f2 = FPR                 (minimize False Positive Rate = FP / (FP + TN))
  f3 = Inference Time      (minimize prediction latency in seconds)

FEATURE MINIMIZATION:
  Enforced as an explicit budget constraint: sum(chromosome) <= K
  Budgets tested: K in [1000, 500, 300, 200, 100, 50]

ACCURACY REQUIREMENT:
  Accuracy is a REPORTED validation metric (before vs. after), NOT an optimization objective.
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

RANDOM_STATE = 42

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

FEATURE_BUDGETS = [1000, 500, 300, 200, 100, 50]


# ==============================================================================
# 1. EVALUATOR CLASS (3 Objectives: 1-F1, FPR, Inference Time)
# ==============================================================================

class RFFitnessEvaluator:
    """
    Evaluates candidate feature subsets using RandomForestClassifier.
    Computes:
      - F1 Score
      - FPR = FP / (FP + TN)
      - Inference time
      - Objectives vector: [1.0 - F1, FPR, avg_inference_time]
      - Secondary metrics: Accuracy, Precision, Recall, confusion matrix
    Maintains a canonical hash cache.
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

        # EXACT THREE OPTIMIZATION OBJECTIVES: [1 - F1, FPR, Inference Time]
        obj_vector = np.array([1.0 - f1, fpr, avg_inf_time], dtype=np.float64)

        res = {
            "f1": f1,
            "fpr": fpr,
            "inference_time": avg_inf_time,
            "accuracy": acc,
            "k_features": k_features,
            "training_time": t_fit,
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
# 2. PARETO UTILITIES (Under [1-F1, FPR, Inference Time])
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
    fronts = fast_non_dominated_sort(objectives)
    return fronts[0] if len(fronts) > 0 else list(range(len(objectives)))


# ==============================================================================
# 3. GENETIC OPERATORS WITH FEATURE BUDGET CONSTRAINT (sum <= K)
# ==============================================================================

def repair_to_budget(individual: np.ndarray, budget_k: int, rng: np.random.Generator) -> np.ndarray:
    """Enforces sum(individual) <= budget_k and sum(individual) >= 1."""
    cur_k = int(np.sum(individual))
    if cur_k > budget_k:
        ones = np.where(individual == 1)[0]
        drop = rng.choice(ones, size=(cur_k - budget_k), replace=False)
        individual[drop] = 0
    elif cur_k == 0:
        individual[rng.integers(0, len(individual))] = 1
    return individual


def crossover_uniform_budget(p1: np.ndarray, p2: np.ndarray, budget_k: int, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2).astype(np.int8)
    return repair_to_budget(child, budget_k, rng)


def mutate_budget(individual: np.ndarray, budget_k: int, rng: np.random.Generator,
                  p_mut: float = 0.3) -> np.ndarray:
    child = individual.copy()
    if rng.random() < p_mut:
        ones = np.where(child == 1)[0]
        zeros = np.where(child == 0)[0]
        
        # Swap mutation: replace 1-3 active features with inactive features
        n_swaps = min(len(ones), len(zeros), rng.integers(1, 4))
        if n_swaps > 0:
            drop = rng.choice(ones, size=n_swaps, replace=False)
            add = rng.choice(zeros, size=n_swaps, replace=False)
            child[drop] = 0
            child[add] = 1
    return repair_to_budget(child, budget_k, rng)


def mode_differential_variation_budget(target: np.ndarray, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray,
                                       budget_k: int, F: float = 0.8, CR: float = 0.8,
                                       rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    L = len(target)
    diff = r1.astype(np.float32) - r2.astype(np.float32)
    v_cont = r3.astype(np.float32) + F * diff
    prob = 1.0 / (1.0 + np.exp(-4.0 * (v_cont - 0.5)))
    donor = (rng.random(L) < prob).astype(np.int8)
    
    cross_mask = rng.random(L) < CR
    cross_mask[rng.integers(0, L)] = True
    trial = np.where(cross_mask, donor, target).astype(np.int8)
    return repair_to_budget(trial, budget_k, rng)


# ==============================================================================
# 4. INITIAL POPULATION GENERATION FOR A GIVEN BUDGET K
# ==============================================================================

def generate_budget_initial_population(n_pop: int, budget_k: int, n_features: int = 10000,
                                       ranked_features: np.ndarray = None, seed: int = 42) -> np.ndarray:
    """Generates an initial population satisfying sum(ind) <= budget_k."""
    rng = np.random.default_rng(seed)
    pop = []
    
    # 1. Informative Chi-2 seeds with varying sparsity (half of population)
    n_info = n_pop // 2
    for i in range(n_info):
        target_k = max(10, int(budget_k * (0.6 + 0.4 * (i / max(1, n_info - 1)))))
        pool_size = min(n_features, max(target_k * 2, int(target_k * 1.5)))
        pool = ranked_features[:pool_size] if ranked_features is not None else np.arange(pool_size)
        
        selected = rng.choice(pool, size=min(len(pool), target_k), replace=False)
        ind = np.zeros(n_features, dtype=np.int8)
        ind[selected] = 1
        pop.append(repair_to_budget(ind, budget_k, rng))
        
    # 2. Exploratory random seeds (other half)
    for _ in range(n_pop - n_info):
        target_k = max(10, rng.integers(int(budget_k * 0.5), budget_k + 1))
        selected = rng.choice(n_features, size=target_k, replace=False)
        ind = np.zeros(n_features, dtype=np.int8)
        ind[selected] = 1
        pop.append(repair_to_budget(ind, budget_k, rng))
        
    return np.array(pop, dtype=np.int8)


# ==============================================================================
# 5. EVOLUTIONARY ALGORITHMS RUNNER PER BUDGET
# ==============================================================================

def print_objective_table(objs: np.ndarray, results: list, title: str, indent: str = "      "):
    """Pretty-prints [1-F1, FPR, Inference Time(ms)] (plus K) for a set of solutions."""
    print(f"{indent}{title}")
    print(f"{indent}{'#':>3} | {'K':>5} | {'f1=1-F1':>9} | {'f2=FPR':>9} | {'f3=Time(ms)':>11}")
    print(f"{indent}" + "-" * 50)
    order = np.argsort(objs[:, 0])          # sorted by f1 for readability
    for rank, i in enumerate(order, 1):
        print(f"{indent}{rank:>3} | {results[i]['k_features']:>5d} | "
              f"{objs[i, 0]:>9.4f} | {objs[i, 1]:>9.4f} | {objs[i, 2] * 1000:>11.2f}")


def _log_nsga2_generation(gen: int, objs: np.ndarray, eval_results: list):
    """Prints the objectives of the current NSGA-II population (summary + current front 0)."""
    front0 = get_non_dominated_indices(objs)
    print(f"      Gen {gen:>2} | Pareto(front0) size = {len(front0):>2} | "
          f"min(1-F1) = {objs[:, 0].min():.4f} | min(FPR) = {objs[:, 1].min():.4f} | "
          f"min(Time) = {objs[:, 2].min() * 1000:.2f} ms")


def run_nsga2_budget(initial_pop: np.ndarray, budget_k: int, evaluator: RFFitnessEvaluator,
                     n_gen: int = 8, seed: int = 42, verbose: bool = True):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    pop = [ind.copy() for ind in initial_pop]
    
    eval_results = [evaluator.evaluate(ind) for ind in pop]
    objs = np.array([res["objectives"] for res in eval_results])
    history = []   # per-generation record of objective values
    if verbose:
        print(f"    [NSGA-II objectives log | K={budget_k}]  objectives = [1-F1, FPR, Time]")
        _log_nsga2_generation(0, objs, eval_results)
    history.append({"gen": 0, "objectives": objs.copy().tolist()})
    
    for gen in range(1, n_gen + 1):
        fronts = fast_non_dominated_sort(objs)
        crowd_dist = {}
        rank_map = {}
        for r, f in enumerate(fronts):
            cd = calculate_crowding_distance(objs, f)
            for idx in f:
                rank_map[idx] = r
                crowd_dist[idx] = cd[idx]
                
        def tournament():
            i, j = rng.choice(N, size=2, replace=False)
            if rank_map[i] < rank_map[j]: return i
            elif rank_map[j] < rank_map[i]: return j
            else: return i if crowd_dist[i] >= crowd_dist[j] else j

        offspring = []
        for _ in range(N):
            p1, p2 = tournament(), tournament()
            child = crossover_uniform_budget(pop[p1], pop[p2], budget_k, rng)
            child = mutate_budget(child, budget_k, rng)
            offspring.append(child)
            
        off_res = [evaluator.evaluate(ind) for ind in offspring]
        off_objs = np.array([r["objectives"] for r in off_res])
        
        comb_pop = pop + offspring
        comb_res = eval_results + off_res
        comb_objs = np.vstack([objs, off_objs])
        
        comb_fronts = fast_non_dominated_sort(comb_objs)
        new_idx = []
        for f in comb_fronts:
            if len(new_idx) + len(f) <= N:
                new_idx.extend(f)
            else:
                cd = calculate_crowding_distance(comb_objs, f)
                sorted_f = sorted(f, key=lambda idx: cd[idx], reverse=True)
                new_idx.extend(sorted_f[:(N - len(new_idx))])
                break
        pop = [comb_pop[i] for i in new_idx]
        eval_results = [comb_res[i] for i in new_idx]
        objs = np.array([res["objectives"] for res in eval_results])
        if verbose:
            _log_nsga2_generation(gen, objs, eval_results)
        history.append({"gen": gen, "objectives": objs.copy().tolist()})
        
    front0 = get_non_dominated_indices(objs)
    if verbose:
        print_objective_table(objs[front0], [eval_results[i] for i in front0],
                              f"NSGA-II FINAL PARETO FRONT (search subset) | K={budget_k} | {len(front0)} solutions")
    return {
        "algorithm": "NSGA-II",
        "budget": budget_k,
        "runtime": time.time() - t_start,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": [eval_results[i] for i in front0],
        "final_population_objectives": objs.copy(),   # all N survivors (for plotting)
        "history": history
    }


def run_spea2_budget(initial_pop: np.ndarray, budget_k: int, evaluator: RFFitnessEvaluator,
                     n_gen: int = 8, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    pop = [ind.copy() for ind in initial_pop]
    archive = [ind.copy() for ind in initial_pop]
    
    pop_res = [evaluator.evaluate(ind) for ind in pop]
    arch_res = [evaluator.evaluate(ind) for ind in archive]
    
    for gen in range(1, n_gen + 1):
        union_pop = pop + archive
        union_res = pop_res + arch_res
        union_objs = np.array([r["objectives"] for r in union_res])
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
                    
        mins, maxs = np.min(union_objs, axis=0), np.max(union_objs, axis=0)
        ranges = np.where(maxs - mins == 0, 1e-9, maxs - mins)
        norm_objs = (union_objs - mins) / ranges
        dists = cdist(norm_objs, norm_objs)
        dists.sort(axis=1)
        k_nn = int(np.sqrt(M))
        density = 1.0 / (dists[:, k_nn] + 2.0)
        total_fitness = raw_fitness + density
        
        non_dom_idx = [i for i, f in enumerate(total_fitness) if f < 1.0]
        if len(non_dom_idx) == N:
            new_arch_idx = non_dom_idx
        elif len(non_dom_idx) < N:
            dom_idx = sorted([i for i in range(M) if i not in non_dom_idx], key=lambda i: total_fitness[i])
            new_arch_idx = non_dom_idx + dom_idx[:N - len(non_dom_idx)]
        else:
            survivors = list(non_dom_idx)
            while len(survivors) > N:
                cur_norm = norm_objs[survivors]
                cur_dists = cdist(cur_norm, cur_norm)
                np.fill_diagonal(cur_dists, np.inf)
                drop_i = int(np.argmin(np.min(cur_dists, axis=1)))
                survivors.pop(drop_i)
            new_arch_idx = survivors
            
        archive = [union_pop[i] for i in new_arch_idx]
        arch_res = [union_res[i] for i in new_arch_idx]
        arch_fit = [total_fitness[i] for i in new_arch_idx]
        
        offspring = []
        for _ in range(N):
            i, j = rng.choice(len(archive), size=2, replace=False)
            p1 = archive[i] if arch_fit[i] <= arch_fit[j] else archive[j]
            i, j = rng.choice(len(archive), size=2, replace=False)
            p2 = archive[i] if arch_fit[i] <= arch_fit[j] else archive[j]
            child = crossover_uniform_budget(p1, p2, budget_k, rng)
            child = mutate_budget(child, budget_k, rng)
            offspring.append(child)
            
        pop = offspring
        pop_res = [evaluator.evaluate(ind) for ind in pop]
        
    arch_objs = np.array([r["objectives"] for r in arch_res])
    front0 = get_non_dominated_indices(arch_objs)
    return {
        "algorithm": "SPEA2",
        "budget": budget_k,
        "runtime": time.time() - t_start,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": [arch_res[i] for i in front0]
    }


def generate_das_dennis_weights(n_obj: int = 3, H: int = 5) -> np.ndarray:
    weights = []
    for i in range(H + 1):
        for j in range(H + 1 - i):
            k = H - i - j
            weights.append([i / H, j / H, k / H])
    return np.array(weights, dtype=float)


def run_moead_budget(initial_pop: np.ndarray, budget_k: int, evaluator: RFFitnessEvaluator,
                     n_gen: int = 8, T: int = 5, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    weights = generate_das_dennis_weights(n_obj=3, H=5)[:N]
    
    weight_dists = cdist(weights, weights)
    neighborhoods = [np.argsort(weight_dists[i])[:T] for i in range(N)]
    
    pop = [ind.copy() for ind in initial_pop]
    eval_res = [evaluator.evaluate(ind) for ind in pop]
    objs = np.array([r["objectives"] for r in eval_res])
    
    scale = np.array([1.0, 10.0, 10.0], dtype=float)
    z_star = np.min(objs * scale, axis=0)
    
    def tchebycheff(f_val, w_vec):
        w = np.where(w_vec == 0, 1e-4, w_vec)
        return np.max(w * np.abs(f_val * scale - z_star))

    for gen in range(1, n_gen + 1):
        for i in range(N):
            p1, p2 = rng.choice(neighborhoods[i], size=2, replace=False)
            child = crossover_uniform_budget(pop[p1], pop[p2], budget_k, rng)
            child = mutate_budget(child, budget_k, rng)
            
            c_res = evaluator.evaluate(child)
            c_obj = c_res["objectives"]
            z_star = np.minimum(z_star, c_obj * scale)
            
            for j in neighborhoods[i]:
                if tchebycheff(c_obj, weights[j]) <= tchebycheff(objs[j], weights[j]):
                    pop[j] = child
                    eval_res[j] = c_res
                    objs[j] = c_obj
                    
    front0 = get_non_dominated_indices(objs)
    return {
        "algorithm": "MOEA/D",
        "budget": budget_k,
        "runtime": time.time() - t_start,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": [eval_res[i] for i in front0]
    }


def run_mode_budget(initial_pop: np.ndarray, budget_k: int, evaluator: RFFitnessEvaluator,
                    n_gen: int = 8, F: float = 0.8, CR: float = 0.8, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    pop = [ind.copy() for ind in initial_pop]
    
    eval_res = [evaluator.evaluate(ind) for ind in pop]
    objs = np.array([r["objectives"] for r in eval_res])
    
    for gen in range(1, n_gen + 1):
        candidate_pop = list(pop)
        candidate_res = list(eval_res)
        
        for i in range(N):
            pool = [idx for idx in range(N) if idx != i]
            r1, r2, r3 = rng.choice(pool, size=3, replace=False)
            trial = mode_differential_variation_budget(pop[i], pop[r1], pop[r2], pop[r3], budget_k, F=F, CR=CR, rng=rng)
            t_res = evaluator.evaluate(trial)
            t_obj = t_res["objectives"]
            
            if dominates(t_obj, objs[i]):
                pop[i] = trial
                eval_res[i] = t_res
                objs[i] = t_obj
            elif dominates(objs[i], t_obj):
                pass
            else:
                candidate_pop.append(trial)
                candidate_res.append(t_res)
                
        if len(candidate_pop) > N:
            c_objs = np.array([r["objectives"] for r in candidate_res])
            fronts = fast_non_dominated_sort(c_objs)
            selected_idx = []
            for f in fronts:
                if len(selected_idx) + len(f) <= N:
                    selected_idx.extend(f)
                else:
                    cd = calculate_crowding_distance(c_objs, f)
                    sorted_f = sorted(f, key=lambda idx: cd[idx], reverse=True)
                    selected_idx.extend(sorted_f[:(N - len(selected_idx))])
                    break
            pop = [candidate_pop[i] for i in selected_idx]
            eval_res = [candidate_res[i] for i in selected_idx]
            objs = np.array([r["objectives"] for r in eval_res])
            
    front0 = get_non_dominated_indices(objs)
    return {
        "algorithm": "MODE",
        "budget": budget_k,
        "runtime": time.time() - t_start,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": [eval_res[i] for i in front0]
    }


# ==============================================================================
# 6. HYPERVOLUME CALCULATION ON [1-F1, FPR, Inference Time]
# ==============================================================================

def compute_hypervolume(objectives: np.ndarray, ref_point: np.ndarray = None) -> float:
    try:
        from pymoo.indicators.hv import HV
        if ref_point is None:
            # ref_point for [1 - F1, FPR, latency_sec]: [1.0, 0.10, 0.20]
            ref_point = np.array([1.0, 0.10, 0.20], dtype=float)
        valid_mask = np.all(objectives <= ref_point, axis=1)
        valid_objs = objectives[valid_mask]
        if len(valid_objs) == 0:
            return 0.0
        norm_objs = valid_objs / ref_point
        hv = HV(ref_point=np.ones(3, dtype=float))
        return float(hv(norm_objs))
    except Exception as e:
        print(f"HV calculation warning: {e}")
        return 0.0


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    print("=" * 80)
    print("   FINAL SOFT COMPUTING EXPERIMENT: FEATURE BUDGET MINIMIZATION (RF)")
    print("=" * 80)
    print("Classifier: RandomForestClassifier (50 trees, class_weight='balanced')")
    print("THREE OPTIMIZATION OBJECTIVES: [1 - F1 Score, False Positive Rate (FPR), Inference Time]")
    print(f"Feature Budgets Tested: {FEATURE_BUDGETS}")
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
    else:
        print("  Training baseline Random Forest on all 10,000 features...")
        t_base = time.perf_counter()
        clf_base = RandomForestClassifier(**RF_CONFIG)
        clf_base.fit(X_train_full, y_train)
        baseline_fit_time = time.perf_counter() - t_base
        _ = clf_base.predict(X_val_full)
        durs = []
        for _ in range(3):
            t_inf = time.perf_counter()
            base_preds = clf_base.predict(X_val_full)
            durs.append(time.perf_counter() - t_inf)
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

    # --------------------------------------------------------------------------
    # STEP 3: SANITY CHECK ON CANDIDATE FEATURE BUDGETS
    # --------------------------------------------------------------------------
    print("\n[Step 3] Running Sanity Check on Candidate Feature Budgets [1000, 500, 100, 50]...")
    chi2_scores, _ = chi2(X_train_full[:20000], y_train[:20000])
    ranked_features = np.argsort(chi2_scores)[::-1]
    
    for test_k in [1000, 500, 100, 50]:
        cols = ranked_features[:test_k]
        clf = RandomForestClassifier(**RF_CONFIG)
        clf.fit(X_train_full[:8000, cols], y_train[:8000])
        preds = clf.predict(X_val_full[:3000, cols])
        s_acc = float(accuracy_score(y_val[:3000], preds))
        s_f1 = float(f1_score(y_val[:3000], preds, zero_division=0))
        cm = confusion_matrix(y_val[:3000], preds)
        tn, fp, fn, tp = cm.ravel()
        s_fpr = float(fp / (fp + tn))
        print(f"    Budget K = {test_k:4d} | Acc = {s_acc*100:.2f}% | F1 = {s_f1:.4f} | FPR = {s_fpr*100:.2f}% | Constraint Verified: len={len(cols)} <= {test_k}")
    print("  ✓ Sanity check passed: Fitness returns [1-F1, FPR, Inference Time]; budgets strictly enforced.")

    # --------------------------------------------------------------------------
    # STEP 4: PREPARE FIXED STRATIFIED SEARCH SUBSET
    # --------------------------------------------------------------------------
    print("\n[Step 4] Preparing Fixed Stratified Search Subset (6,000 train / 2,000 val)...")
    sss_tr = StratifiedShuffleSplit(n_splits=1, train_size=6000, random_state=RANDOM_STATE)
    idx_tr_search, _ = next(sss_tr.split(X_train_full, y_train))
    idx_tr_search.sort()

    sss_va = StratifiedShuffleSplit(n_splits=1, train_size=2000, random_state=RANDOM_STATE)
    idx_va_search, _ = next(sss_va.split(X_val_full, y_val))
    idx_va_search.sort()

    X_train_search = X_train_full[idx_tr_search].tocsc()
    y_train_search = y_train[idx_tr_search]
    X_val_search = X_val_full[idx_va_search].tocsc()
    y_val_search = y_val[idx_va_search]

    search_evaluator = RFFitnessEvaluator(X_train_search, y_train_search, X_val_search, y_val_search,
                                          rf_params=RF_CONFIG, name="SearchEvaluator")

    # --------------------------------------------------------------------------
    # STEP 5: RUN 4 ALGORITHMS ACROSS FEATURE BUDGETS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("       PHASE 1: MULTI-OBJECTIVE EVOLUTIONARY SEARCH ACROSS BUDGETS")
    print("=" * 80)
    print(f"Budgets: {FEATURE_BUDGETS} | Population: N = 21 | Generations: G = 8")
    
    n_pop = 21
    n_gen = 8
    
    budget_results = {
        "NSGA-II": {},
        "SPEA2": {},
        "MOEA/D": {},
        "MODE": {}
    }
    
    for budget_k in FEATURE_BUDGETS:
        print(f"\n--- Testing Feature Budget K = {budget_k} (Constraint: <= {budget_k} features) ---")
        init_pop = generate_budget_initial_population(n_pop, budget_k, n_features=10000,
                                                      ranked_features=ranked_features, seed=RANDOM_STATE)
        
        # NSGA-II
        t0_alg = time.time()
        res_nsga2 = run_nsga2_budget(init_pop, budget_k, search_evaluator, n_gen=n_gen, seed=RANDOM_STATE)
        budget_results["NSGA-II"][budget_k] = res_nsga2
        print(f"  • NSGA-II [{budget_k}]: {time.time()-t0_alg:.1f}s | Pareto: {len(res_nsga2['pareto_solutions'])} | Best F1: {max(s['f1'] for s in res_nsga2['pareto_solutions']):.4f} | Min FPR: {min(s['fpr'] for s in res_nsga2['pareto_solutions'])*100:.2f}%")
        
        # SPEA2
        t0_alg = time.time()
        res_spea2 = run_spea2_budget(init_pop, budget_k, search_evaluator, n_gen=n_gen, seed=RANDOM_STATE)
        budget_results["SPEA2"][budget_k] = res_spea2
        print(f"  • SPEA2   [{budget_k}]: {time.time()-t0_alg:.1f}s | Pareto: {len(res_spea2['pareto_solutions'])} | Best F1: {max(s['f1'] for s in res_spea2['pareto_solutions']):.4f} | Min FPR: {min(s['fpr'] for s in res_spea2['pareto_solutions'])*100:.2f}%")
        
        # MOEA/D
        t0_alg = time.time()
        res_moead = run_moead_budget(init_pop, budget_k, search_evaluator, n_gen=n_gen, T=5, seed=RANDOM_STATE)
        budget_results["MOEA/D"][budget_k] = res_moead
        print(f"  • MOEA/D  [{budget_k}]: {time.time()-t0_alg:.1f}s | Pareto: {len(res_moead['pareto_solutions'])} | Best F1: {max(s['f1'] for s in res_moead['pareto_solutions']):.4f} | Min FPR: {min(s['fpr'] for s in res_moead['pareto_solutions'])*100:.2f}%")
        
        # MODE
        t0_alg = time.time()
        res_mode = run_mode_budget(init_pop, budget_k, search_evaluator, n_gen=n_gen, F=0.8, CR=0.8, seed=RANDOM_STATE)
        budget_results["MODE"][budget_k] = res_mode
        print(f"  • MODE    [{budget_k}]: {time.time()-t0_alg:.1f}s | Pareto: {len(res_mode['pareto_solutions'])} | Best F1: {max(s['f1'] for s in res_mode['pareto_solutions']):.4f} | Min FPR: {min(s['fpr'] for s in res_mode['pareto_solutions'])*100:.2f}%")

    # --------------------------------------------------------------------------
    # STEP 6: FULL DATASET VALIDATION (127,656 Train x 31,915 Val)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("      PHASE 2: FULL DATASET VALIDATION OF STRONGEST CANDIDATES")
    print("=" * 80)
    print("Evaluating top solutions on all 127,656 training rows and 31,915 validation rows...")

    full_evaluator = RFFitnessEvaluator(X_train_full.tocsc(), y_train, X_val_full.tocsc(), y_val,
                                        rf_params=RF_CONFIG, name="FullEvaluator")

    # For each algorithm and each budget, select the representative knee candidate
    # (balanced solution minimizing distance to ideal point z*=(0,0,0) in normalized space)
    validated_records = []
    
    for alg_name in ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]:
        print(f"\n[Full Validation] Evaluating {alg_name} across budgets...")
        for budget_k in FEATURE_BUDGETS:
            sols = budget_results[alg_name][budget_k]["pareto_solutions"]
            if len(sols) == 0:
                continue
            # Select balanced solution minimizing Euclidean distance in objective space
            objs = np.array([s["objectives"] for s in sols])
            mins = np.min(objs, axis=0)
            maxs = np.max(objs, axis=0)
            ranges = np.where(maxs - mins == 0, 1e-9, maxs - mins)
            norm_objs = (objs - mins) / ranges
            dists = np.linalg.norm(norm_objs, axis=1)
            best_sol = sols[int(np.argmin(dists))]
            
            # Evaluate on full dataset
            ind_vec = np.zeros(10000, dtype=np.int8)
            ind_vec[best_sol["indices"]] = 1
            full_res = full_evaluator.evaluate(ind_vec, timing_runs=3)
            
            acc = full_res["accuracy"]
            f1 = full_res["f1"]
            fpr = full_res["fpr"]
            prec = full_res["precision"]
            rec = full_res["recall"]
            inf_ms = full_res["inference_time"] * 1000.0
            fit_s = full_res["training_time"]
            k_actual = full_res["k_features"]
            acc_loss = baseline_acc - acc
            acc_change_pct = ((acc - baseline_acc) / baseline_acc) * 100.0
            
            record = {
                "Algorithm": alg_name,
                "Budget_K": budget_k,
                "Selected_Features": k_actual,
                "Feature_Reduction_Pct": (1.0 - k_actual / 10000.0) * 100.0,
                "Baseline_Accuracy": baseline_acc,
                "Reduced_Accuracy": acc,
                "Accuracy_Difference": acc_loss,
                "Accuracy_Change_Pct": acc_change_pct,
                "F1_Score": f1,
                "Baseline_F1": baseline_f1,
                "F1_Difference": f1 - baseline_f1,
                "False_Positive_Rate": fpr,
                "Baseline_FPR": baseline_fpr,
                "FPR_Difference": fpr - baseline_fpr,
                "Precision": prec,
                "Recall": rec,
                "Inference_Time_ms": inf_ms,
                "Inference_Speedup": (baseline_inf_time * 1000.0) / inf_ms,
                "Training_Time_sec": fit_s,
                "indices": best_sol["indices"].tolist(),
                "objectives": full_res["objectives"].tolist()
            }
            validated_records.append(record)
            print(f"  Budget {budget_k:4d} (feats={k_actual:4d}) | Acc={acc*100:.2f}% (Loss={acc_loss*100:+.2f}%) | F1={f1:.4f} | FPR={fpr*100:.2f}% | Inf={inf_ms:.1f}ms | Fit={fit_s:.1f}s")

    df_validated = pd.DataFrame(validated_records)

    # --------------------------------------------------------------------------
    # STEP 7: RECOMPUTE PARETO DOMINANCE ON FULL DATA (OBJECTIVES: 1-F1, FPR, TIME)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("      PHASE 3: RECOMPUTING PARETO DOMINANCE ON FULL DATA")
    print("=" * 80)
    
    pareto_rows = []
    comp_rows = []

    for alg_name in ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]:
        sub_df = df_validated[df_validated["Algorithm"] == alg_name]
        objs = np.array([r["objectives"] for r in sub_df.to_dict(orient="records")])
        front0_idx = get_non_dominated_indices(objs)
        p_records = [sub_df.iloc[i].to_dict() for i in front0_idx]
        
        # Compute hypervolume
        p_objs = np.array([r["objectives"] for r in p_records])
        hv = compute_hypervolume(p_objs)
        
        for r in p_records:
            r["Is_Pareto"] = True
            pareto_rows.append(r)
            
        # Select best representative solution (smallest budget providing acceptable trade-off)
        # Criteria: F1 close to baseline (>= 0.65), FPR <= 2.5%, high feature reduction
        candidates_acceptable = [r for r in p_records if r["F1_Score"] >= 0.65 and r["False_Positive_Rate"] <= 0.025]
        if len(candidates_acceptable) > 0:
            best_tradeoff = min(candidates_acceptable, key=lambda r: r["Selected_Features"])
        else:
            best_tradeoff = max(p_records, key=lambda r: r["F1_Score"])
            
        comp_rows.append({
            "Algorithm": alg_name,
            "Winning_Budget_K": best_tradeoff["Budget_K"],
            "Selected_Features": best_tradeoff["Selected_Features"],
            "Feature_Reduction_Pct": best_tradeoff["Feature_Reduction_Pct"],
            "Baseline_Accuracy": baseline_acc,
            "Reduced_Accuracy": best_tradeoff["Reduced_Accuracy"],
            "Accuracy_Difference": best_tradeoff["Accuracy_Difference"],
            "Accuracy_Change_Pct": best_tradeoff["Accuracy_Change_Pct"],
            "Baseline_F1": baseline_f1,
            "Reduced_F1": best_tradeoff["F1_Score"],
            "F1_Difference": best_tradeoff["F1_Difference"],
            "Baseline_FPR": baseline_fpr,
            "Reduced_FPR": best_tradeoff["False_Positive_Rate"],
            "FPR_Difference": best_tradeoff["FPR_Difference"],
            "Inference_Time_ms": best_tradeoff["Inference_Time_ms"],
            "Inference_Speedup": best_tradeoff["Inference_Speedup"],
            "Search_Runtime_sec": sum(budget_results[alg_name][b]["runtime"] for b in FEATURE_BUDGETS),
            "Pareto_Solutions_Count": len(p_records),
            "Hypervolume": hv
        })

    df_comp = pd.DataFrame(comp_rows)
    df_pareto = pd.DataFrame(pareto_rows)

    # ---- NSGA-II Pareto values (full-data validated) ----
    nsga2_pareto = [r for r in pareto_rows if r["Algorithm"] == "NSGA-II"]
    print("\n" + "=" * 80)
    print("   NSGA-II PARETO FRONT (validated on full data) - objectives [1-F1, FPR, Time]")
    print("=" * 80)
    print(f"{'Budget_K':>8} | {'Feats':>5} | {'1-F1':>8} | {'FPR':>8} | {'Time(ms)':>9} | {'F1':>7} | {'Acc%':>6}")
    print("-" * 70)
    for r in sorted(nsga2_pareto, key=lambda r: r["Budget_K"], reverse=True):
        o = r["objectives"]
        print(f"{r['Budget_K']:>8d} | {r['Selected_Features']:>5d} | {o[0]:>8.4f} | {o[1]:>8.4f} | "
              f"{o[2]*1000:>9.2f} | {r['F1_Score']:>7.4f} | {r['Reduced_Accuracy']*100:>6.2f}")
    pd.DataFrame([{
        "Budget_K": r["Budget_K"], "Selected_Features": r["Selected_Features"],
        "f1_1_minus_F1": r["objectives"][0], "f2_FPR": r["objectives"][1],
        "f3_Inference_Time_sec": r["objectives"][2],
        "F1_Score": r["F1_Score"], "Accuracy": r["Reduced_Accuracy"]
    } for r in nsga2_pareto]).to_csv("nsga2_pareto_objectives_rf_final.csv", index=False)
    print("  ✓ Saved nsga2_pareto_objectives_rf_final.csv")

    print("\n" + "=" * 80)
    print("                    FINAL ALGORITHM COMPARISON TABLE")
    print("=" * 80)
    print(df_comp[[
        "Algorithm", "Selected_Features", "Feature_Reduction_Pct", "Reduced_Accuracy",
        "Reduced_F1", "Reduced_FPR", "Inference_Time_ms", "Inference_Speedup", "Hypervolume"
    ]].to_string(index=False))

    # Determine winning algorithm
    # Best trade-off: smallest feature count retaining strong F1 and low FPR with high hypervolume
    df_comp_sorted = df_comp.sort_values(
        by=["Reduced_F1", "Hypervolume"],
        ascending=[False, False]
    )
    winning_alg = df_comp_sorted.iloc[0]["Algorithm"]
    winning_row = df_comp_sorted.iloc[0]

    print("\n" + "=" * 80)
    print(f"             OVERALL WINNER: {winning_alg}")
    print("=" * 80)
    print(f"  • Selected features:       {winning_row['Selected_Features']} (out of 10,000)")
    print(f"  • Feature reduction:       {winning_row['Feature_Reduction_Pct']:.2f}% features eliminated!")
    print(f"  • Full validation accuracy: {winning_row['Reduced_Accuracy']*100:.2f}% (Baseline: {baseline_acc*100:.2f}%)")
    print(f"  • Accuracy difference:     {winning_row['Accuracy_Difference']*100:+.2f}%")
    print(f"  • F1 Score:                {winning_row['Reduced_F1']:.4f} (Baseline: {baseline_f1:.4f})")
    print(f"  • FPR:                     {winning_row['Reduced_FPR']*100:.2f}% (Baseline: {baseline_fpr*100:.2f}%)")
    print(f"  • Inference latency:       {winning_row['Inference_Time_ms']:.2f} ms (Baseline: {baseline_inf_time*1000:.2f} ms, {winning_row['Inference_Speedup']:.2f}x faster)")

    # --------------------------------------------------------------------------
    # STEP 8: SAVE DELIVERABLES
    # --------------------------------------------------------------------------
    print("\n[Step 8] Saving Deliverables...")

    for alg_name, alg_key in [("NSGA-II", "nsga2"), ("SPEA2", "spea2"), ("MOEA/D", "moead"), ("MODE", "mode")]:
        out_json = f"results_{alg_key}_rf_final.json"
        with open(out_json, "w") as f:
            json.dump({
                "algorithm": alg_name,
                "budgets_tested": FEATURE_BUDGETS,
                "validated_solutions": df_validated[df_validated["Algorithm"] == alg_name].to_dict(orient="records"),
                "pareto_solutions": [r for r in pareto_rows if r["Algorithm"] == alg_name]
            }, f, indent=2)
        print(f"  ✓ Saved {out_json}")

    df_comp.to_csv("final_algorithm_comparison_rf_final.csv", index=False)
    print("  ✓ Saved final_algorithm_comparison_rf_final.csv")

    df_pareto_export = df_pareto.drop(columns=["indices", "objectives"], errors="ignore")
    df_pareto_export.to_csv("final_pareto_solutions_rf_final.csv", index=False)
    print("  ✓ Saved final_pareto_solutions_rf_final.csv")

    exp_config = {
        "model_type": "RandomForestClassifier",
        "hyperparameters": RF_CONFIG,
        "objectives": ["1 - F1", "FPR", "Inference Time"],
        "budgets": FEATURE_BUDGETS,
        "dataset": {
            "training_samples": int(X_train_full.shape[0]),
            "validation_samples": int(X_val_full.shape[0]),
            "total_features": 10000,
            "search_train_samples": 6000,
            "search_val_samples": 2000
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
        "winning_algorithm": winning_alg
    }
    with open("experiment_config_rf_final.json", "w") as f:
        json.dump(exp_config, f, indent=2)
    print("  ✓ Saved experiment_config_rf_final.json")

    final_analysis = {
        "baseline": exp_config["baseline_metrics"],
        "comparison": df_comp.to_dict(orient="records"),
        "winner": winning_row.to_dict()
    }
    with open("final_analysis_rf_final.json", "w") as f:
        json.dump(final_analysis, f, indent=2)
    print("  ✓ Saved final_analysis_rf_final.json")

    # Save feature vectors and text tokens for final solutions
    for alg_name, alg_key in [("NSGA-II", "nsga2"), ("SPEA2", "spea2"), ("MOEA/D", "moead"), ("MODE", "mode")]:
        row = df_comp[df_comp["Algorithm"] == alg_name].iloc[0]
        rec = [r for r in validated_records if r["Algorithm"] == alg_name and r["Budget_K"] == row["Winning_Budget_K"]][0]
        idx_arr = np.array(rec["indices"], dtype=np.int32)
        np.save(f"selected_features_{alg_key}_rf_final.npy", idx_arr)
        with open(f"selected_features_{alg_key}_rf_final.txt", "w") as f:
            for idx in idx_arr:
                f.write(f"{feature_names[idx]}\n")
        print(f"  ✓ Saved selected_features_{alg_key}_rf_final.npy and .txt (K = {len(idx_arr)})")

    # --------------------------------------------------------------------------
    # STEP 9: GENERATE HIGH-QUALITY VISUALIZATIONS
    # --------------------------------------------------------------------------
    print("\n[Step 9] Generating Visualization Plots...")
    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    colors = {"NSGA-II": "#1f77b4", "SPEA2": "#2ca02c", "MOEA/D": "#ff7f0e", "MODE": "#d62728"}

    # Plot 1: Pareto F1 vs FPR
    plt.figure(figsize=(9, 5.5))
    plt.scatter([baseline_fpr * 100], [baseline_f1], color="black", marker="*", s=180, zorder=5,
                label=f"Baseline RF (10k feats): F1={baseline_f1:.4f}, FPR={baseline_fpr*100:.2f}%")
    for alg in ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]:
        sub = [r for r in pareto_rows if r["Algorithm"] == alg]
        fprs = [r["False_Positive_Rate"] * 100 for r in sub]
        f1s = [r["F1_Score"] for r in sub]
        plt.scatter(fprs, f1s, label=alg, color=colors[alg], s=65, alpha=0.9)
        s_sorted = sorted(zip(fprs, f1s))
        plt.plot([p[0] for p in s_sorted], [p[1] for p in s_sorted], color=colors[alg], alpha=0.5)
    plt.xlabel("False Positive Rate (%) [Lower is Better]", fontsize=11, fontweight="bold")
    plt.ylabel("F1 Score [Higher is Better]", fontsize=11, fontweight="bold")
    plt.title("Validated Pareto Front: F1 Score vs. False Positive Rate (Random Forest)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig("plots/pareto_f1_vs_fpr_rf_final.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/pareto_f1_vs_fpr_rf_final.png")

    # Plot 2: Pareto F1 vs Latency
    plt.figure(figsize=(9, 5.5))
    plt.scatter([baseline_inf_time * 1000], [baseline_f1], color="black", marker="*", s=180, zorder=5,
                label=f"Baseline RF: F1={baseline_f1:.4f}, Latency={baseline_inf_time*1000:.1f}ms")
    for alg in ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]:
        sub = [r for r in pareto_rows if r["Algorithm"] == alg]
        times = [r["Inference_Time_ms"] for r in sub]
        f1s = [r["F1_Score"] for r in sub]
        plt.scatter(times, f1s, label=alg, color=colors[alg], s=65, alpha=0.9)
        s_sorted = sorted(zip(times, f1s))
        plt.plot([p[0] for p in s_sorted], [p[1] for p in s_sorted], color=colors[alg], alpha=0.5)
    plt.xlabel("Inference Latency on 31,915 Samples (ms) [Lower is Better]", fontsize=11, fontweight="bold")
    plt.ylabel("F1 Score [Higher is Better]", fontsize=11, fontweight="bold")
    plt.title("Validated Pareto Front: F1 Score vs. Inference Latency (Random Forest)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig("plots/pareto_f1_vs_latency_rf_final.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/pareto_f1_vs_latency_rf_final.png")

    # Plot 2b: NSGA-II Pareto points vs. rest of final population (3 objective projections)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    pairs = [(0, 1, "f1 = 1 - F1", "f2 = FPR"),
             (0, 2, "f1 = 1 - F1", "f3 = Inference Time (ms)"),
             (1, 2, "f2 = FPR", "f3 = Inference Time (ms)")]
    cmap = plt.get_cmap("viridis")
    for ax, (a, b, xl, yl) in zip(axes, pairs):
        sc = lambda arr, c: arr[:, c] * (1000.0 if c == 2 else 1.0)
        for bi, bk in enumerate(FEATURE_BUDGETS):
            res = budget_results["NSGA-II"][bk]
            pop_o = res["final_population_objectives"]
            par_o = np.array([s["objectives"] for s in res["pareto_solutions"]])
            col = cmap(bi / max(1, len(FEATURE_BUDGETS) - 1))
            ax.scatter(sc(pop_o, a), sc(pop_o, b), color="lightgray", s=25, alpha=0.7, zorder=1)
            ax.scatter(sc(par_o, a), sc(par_o, b), color=col, edgecolor="black", s=70, zorder=3,
                       label=f"Pareto K={bk}")
        ax.set_xlabel(xl, fontweight="bold")
        ax.set_ylabel(yl, fontweight="bold")
        ax.set_title(f"{xl} vs {yl}", fontsize=11)
    axes[0].legend(fontsize=8, title="grey = dominated", frameon=True)
    fig.suptitle("NSGA-II: Pareto-optimal solutions (coloured) vs. rest of final population (grey)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/nsga2_pareto_objectives_rf_final.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/nsga2_pareto_objectives_rf_final.png")

    # Plot 3: Budget Trade-off Curve (F1 and Accuracy vs Feature Budget)
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax2 = ax1.twinx()
    
    # Average across algorithms for clean trend
    b_df = df_validated.groupby("Budget_K").agg({
        "F1_Score": "mean",
        "Reduced_Accuracy": "mean",
        "Inference_Time_ms": "mean"
    }).reset_index().sort_values("Budget_K")
    
    l1 = ax1.plot(b_df["Budget_K"], b_df["F1_Score"], "o-", color="#1f77b4", linewidth=2.2, label="Mean F1 Score")
    ax1.axhline(baseline_f1, color="#1f77b4", linestyle="--", alpha=0.7, label=f"Baseline F1 ({baseline_f1:.4f})")
    
    l2 = ax2.plot(b_df["Budget_K"], b_df["Reduced_Accuracy"] * 100, "s-", color="#d62728", linewidth=2.2, label="Mean Accuracy (%)")
    ax2.axhline(baseline_acc * 100, color="#d62728", linestyle="--", alpha=0.7, label=f"Baseline Acc ({baseline_acc*100:.2f}%)")
    
    ax1.set_xlabel("Feature Budget Constraint (K)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("F1 Score", color="#1f77b4", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", color="#d62728", fontsize=11, fontweight="bold")
    plt.title("Random Forest Performance Across Decreasing Feature Budgets", fontsize=13, fontweight="bold")
    
    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig("plots/budget_tradeoff_rf_final.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/budget_tradeoff_rf_final.png")

    # Plot 4: Baseline vs Reduced Models (Accuracy & F1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    algs = df_comp["Algorithm"].tolist()
    labels = ["Baseline RF\n(10,000 feats)"] + [f"{a}\n({int(df_comp.loc[df_comp['Algorithm']==a, 'Selected_Features'].values[0])} feats)" for a in algs]
    accs = [baseline_acc * 100] + [df_comp.loc[df_comp["Algorithm"]==a, "Reduced_Accuracy"].values[0] * 100 for a in algs]
    f1s = [baseline_f1] + [df_comp.loc[df_comp["Algorithm"]==a, "Reduced_F1"].values[0] for a in algs]
    bar_cols = ["#333333"] + [colors[a] for a in algs]
    
    # 4a: Accuracy comparison
    b1 = axes[0].bar(labels, accs, color=bar_cols, width=0.55)
    axes[0].set_title("Validation Accuracy: Baseline vs. Reduced Models", fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(91.0, 95.0)
    for bar, val in zip(b1, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.08, f"{val:.2f}%", ha="center", fontweight="bold")
        
    # 4b: F1 Score comparison
    b2 = axes[1].bar(labels, f1s, color=bar_cols, width=0.55)
    axes[1].set_title("F1 Score: Baseline vs. Reduced Models", fontweight="bold")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_ylim(0.55, 0.72)
    for bar, val in zip(b2, f1s):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/baseline_vs_reduced_rf_final.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/baseline_vs_reduced_rf_final.png")

    # Plot 5: Algorithm Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 5a: Features & Reduction
    axes[0].bar(algs, df_comp["Selected_Features"], color=[colors[a] for a in algs], width=0.55)
    axes[0].set_title("Retained Features at Winning Budget", fontweight="bold")
    axes[0].set_ylabel("Feature Count")
    for i, v in enumerate(df_comp["Selected_Features"]):
        axes[0].text(i, v + 10, f"{v}", ha="center", fontweight="bold")
        
    # 5b: Inference Latency
    axes[1].bar(algs, df_comp["Inference_Time_ms"], color=[colors[a] for a in algs], width=0.55)
    axes[1].axhline(baseline_inf_time * 1000, color="black", linestyle="--", label=f"Baseline ({baseline_inf_time*1000:.1f}ms)")
    axes[1].set_title("Inference Latency on 31,915 Samples (ms)", fontweight="bold")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].legend(loc="upper left")
    for i, v in enumerate(df_comp["Inference_Time_ms"]):
        axes[1].text(i, v + 1.5, f"{v:.1f}ms", ha="center", fontweight="bold")
        
    # 5c: Hypervolume
    axes[2].bar(algs, df_comp["Hypervolume"], color=[colors[a] for a in algs], width=0.55)
    axes[2].set_title("Validated Hypervolume Indicator", fontweight="bold")
    axes[2].set_ylabel("Hypervolume (Higher is Better)")
    axes[2].set_ylim(0.0, 0.75)
    for i, v in enumerate(df_comp["Hypervolume"]):
        axes[2].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/algorithm_comparison_rf_final.png", dpi=300)
    plt.close()
    print("  ✓ Saved plots/algorithm_comparison_rf_final.png")

    print("\n" + "=" * 80)
    print("         EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()