#!/usr/bin/env python3
"""
Multi-Objective Evolutionary Feature Selection Experiment
Comparing: NSGA-II, SPEA2, MOEA/D, and MODE
On: Jigsaw Toxic Comment Classification (10,000 TF-IDF features -> 500 features)

Objectives:
1. MAXIMIZE F1 Score (represented as minimizing 1 - F1)
2. MINIMIZE False Positive Rate (FPR = FP / (FP + TN))
3. MINIMIZE Inference Time (average prediction time in seconds)

Constraint: EXACTLY 500 selected features per candidate solution.
Classifier: LinearSVC(class_weight='balanced', C=1.0, random_state=42)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import chi2


# ==============================================================================
# 1. CORE OPERATORS & REPAIR
# ==============================================================================

def repair_to_k_features(binary_vector: np.ndarray, k: int = 500, rng: np.random.Generator = None) -> np.ndarray:
    """Guarantees exactly k active features (ones) in a binary vector."""
    if rng is None:
        rng = np.random.default_rng()
    cur_k = int(np.sum(binary_vector))
    if cur_k > k:
        ones_idx = np.where(binary_vector == 1)[0]
        drop_idx = rng.choice(ones_idx, size=(cur_k - k), replace=False)
        binary_vector[drop_idx] = 0
    elif cur_k < k:
        zeros_idx = np.where(binary_vector == 0)[0]
        add_idx = rng.choice(zeros_idx, size=(k - cur_k), replace=False)
        binary_vector[add_idx] = 1
    return binary_vector


def crossover_uniform(p1: np.ndarray, p2: np.ndarray, k: int = 500, rng: np.random.Generator = None) -> np.ndarray:
    """Uniform crossover with exact k-feature repair."""
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2).astype(np.int8)
    return repair_to_k_features(child, k=k, rng=rng)


def mutate_swap(ind: np.ndarray, p_mut: float = 0.3, n_swaps: int = 4, rng: np.random.Generator = None) -> np.ndarray:
    """Swap mutation: swaps n_swaps ones with zeros, preserving exact feature count."""
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p_mut:
        child = ind.copy()
        ones = np.where(child == 1)[0]
        zeros = np.where(child == 0)[0]
        s = min(n_swaps, len(ones), len(zeros))
        drop = rng.choice(ones, size=s, replace=False)
        add = rng.choice(zeros, size=s, replace=False)
        child[drop] = 0
        child[add] = 1
        return child
    return ind.copy()


def mode_differential_variation(target: np.ndarray, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray,
                              F: float = 0.8, CR: float = 0.8, k: int = 500, rng: np.random.Generator = None) -> np.ndarray:
    """
    Classical MODE variation (Xue et al., 2003 / Babu & Jehan, 2003) for binary feature selection.
    Combines donor base r1 with directional differential perturbation (r2 - r3),
    binomial crossover with target, and exact k-feature repair.
    """
    if rng is None:
        rng = np.random.default_rng()
    mutant = r1.copy()
    diff = r2.astype(int) - r3.astype(int)
    
    # Positive directional features: present in r2, absent in r3
    add_mask = (diff == 1) & (rng.random(len(mutant)) < F)
    mutant[add_mask] = 1
    
    # Negative directional features: present in r3, absent in r2
    drop_mask = (diff == -1) & (rng.random(len(mutant)) < F)
    mutant[drop_mask] = 0
    
    # Binomial crossover with target
    cross_mask = rng.random(len(target)) < CR
    j_rand = rng.integers(0, len(target))
    cross_mask[j_rand] = True
    
    trial = target.copy()
    trial[cross_mask] = mutant[cross_mask]
    return repair_to_k_features(trial, k=k, rng=rng)


# ==============================================================================
# 2. EVALUATOR WITH CANONICAL CACHING
# ==============================================================================

class Evaluator:
    """Evaluates candidate feature subsets on LinearSVC."""
    def __init__(self, X_train_csc, y_train, X_val_csc, y_val, name="evaluator"):
        self.X_train = X_train_csc
        self.y_train = y_train
        self.X_val = X_val_csc
        self.y_val = y_val
        self.name = name
        self.cache = {}
        self.eval_count = 0
        self.cache_hits = 0

    def evaluate(self, binary_vector: np.ndarray, timing_runs: int = 3):
        active_indices = np.where(binary_vector == 1)[0]
        assert len(active_indices) == 500, f"Expected exactly 500 features, got {len(active_indices)}"
        
        key = tuple(active_indices)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
            
        self.eval_count += 1
        idx = np.array(key, dtype=np.int32)
        
        # 1. Slice training columns
        X_tr = self.X_train[:, idx]
        
        # 2. Fit baseline LinearSVC configuration
        t_fit_start = time.perf_counter()
        clf = LinearSVC(class_weight="balanced", C=1.0, random_state=42, dual="auto")
        clf.fit(X_tr, self.y_train)
        t_fit = time.perf_counter() - t_fit_start
        
        # 3. Slice evaluation columns
        X_v = self.X_val[:, idx]
        
        # 4. Measure inference time on evaluation set (multiple runs)
        _ = clf.predict(X_v)  # Warmup
        durations = []
        y_pred = None
        for _ in range(timing_runs):
            t0 = time.perf_counter()
            y_pred = clf.predict(X_v)
            durations.append(time.perf_counter() - t0)
        avg_inf_time = float(np.mean(durations))
        
        # 5. Metrics
        cm = confusion_matrix(self.y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        f1 = float(f1_score(self.y_val, y_pred))
        acc = float(accuracy_score(self.y_val, y_pred))
        prec = float(precision_score(self.y_val, y_pred, zero_division=0))
        rec = float(recall_score(self.y_val, y_pred, zero_division=0))
        
        # Objectives to minimize: [1.0 - F1, FPR, inference_time]
        obj_vector = np.array([1.0 - f1, fpr, avg_inf_time], dtype=np.float64)
        
        res = {
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "inference_time": avg_inf_time,
            "training_time": t_fit,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "objectives": obj_vector,
            "indices": active_indices,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
        self.cache[key] = res
        return res


# ==============================================================================
# 3. PARETO UTILITIES
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
# 4. INITIAL POPULATION GENERATION
# ==============================================================================

def generate_initial_population(n_pop: int = 28, n_features: int = 10000, k: int = 500,
                                ranked_features: np.ndarray = None, seed: int = 42) -> np.ndarray:
    """
    Generates a balanced, diverse initial population of exactly k-feature individuals.
    Shared identically across all 4 algorithms to guarantee 100% fair baseline conditions.
    """
    rng = np.random.default_rng(seed)
    pop = []
    
    # 1. Informative candidate seeds (8 individuals): features sampled from top univariate ranks
    for i in range(8):
        pool_size = 700 + i * 150
        pool = ranked_features[:pool_size]
        selected = rng.choice(pool, size=k, replace=False)
        ind = np.zeros(n_features, dtype=np.int8)
        ind[selected] = 1
        pop.append(ind)
        
    # 2. Hybrid semi-directed candidates (6 individuals): 350 top + 150 exploratory
    for i in range(6):
        top_part = rng.choice(ranked_features[:2000], size=350, replace=False)
        rem = list(set(range(n_features)) - set(top_part))
        random_part = rng.choice(rem, size=150, replace=False)
        ind = np.zeros(n_features, dtype=np.int8)
        ind[top_part] = 1
        ind[random_part] = 1
        pop.append(ind)
        
    # 3. Pure random exploration candidates (14 individuals)
    for i in range(14):
        selected = rng.choice(n_features, size=k, replace=False)
        ind = np.zeros(n_features, dtype=np.int8)
        ind[selected] = 1
        pop.append(ind)
        
    pop_arr = np.array(pop, dtype=np.int8)
    for i, ind in enumerate(pop_arr):
        assert np.sum(ind) == k, f"Initial individual {i} does not have exactly {k} features"
    return pop_arr


# ==============================================================================
# 5. ALGORITHM 1: NSGA-II
# ==============================================================================

def run_nsga2(initial_pop: np.ndarray, evaluator: Evaluator, n_gen: int = 20, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    pop = [ind.copy() for ind in initial_pop]
    
    # Evaluate initial population
    eval_results = [evaluator.evaluate(ind) for ind in pop]
    objs = np.array([res["objectives"] for res in eval_results])
    
    history = []
    print("  [NSGA-II] Running generations...")
    for gen in range(1, n_gen + 1):
        # 1. Binary tournament selection using front rank & crowding distance
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

        # 2. Variation: Crossover + Swap Mutation
        offspring_pop = []
        for _ in range(N):
            p1_idx = tournament_select()
            p2_idx = tournament_select()
            child = crossover_uniform(pop[p1_idx], pop[p2_idx], k=500, rng=rng)
            child = mutate_swap(child, p_mut=0.3, n_swaps=4, rng=rng)
            offspring_pop.append(child)
            
        # 3. Evaluate offspring
        offspring_results = [evaluator.evaluate(ind) for ind in offspring_pop]
        offspring_objs = np.array([res["objectives"] for res in offspring_results])
        
        # 4. Elitist (mu + lambda) selection (2N -> N)
        combined_pop = pop + offspring_pop
        combined_results = eval_results + offspring_results
        combined_objs = np.vstack([objs, offspring_objs])
        
        comb_fronts = fast_non_dominated_sort(combined_objs)
        new_indices = []
        for f in comb_fronts:
            if len(new_indices) + len(f) <= N:
                new_indices.extend(f)
            else:
                # Truncate splitting front by crowding distance
                cd = calculate_crowding_distance(combined_objs, f)
                sorted_f = sorted(f, key=lambda idx: cd[idx], reverse=True)
                needed = N - len(new_indices)
                new_indices.extend(sorted_f[:needed])
                break
                
        pop = [combined_pop[idx] for idx in new_indices]
        eval_results = [combined_results[idx] for idx in new_indices]
        objs = np.array([res["objectives"] for res in eval_results])
        
        # Record history
        best_f1 = max(res["f1"] for res in eval_results)
        best_fpr = min(res["fpr"] for res in eval_results)
        history.append({"gen": gen, "best_f1": best_f1, "best_fpr": best_fpr, "cache_evals": evaluator.eval_count})
        print(f"    Gen {gen:2d}/{n_gen} | Best F1: {best_f1:.4f} | Best FPR: {best_fpr:.4f} | Total Evals: {evaluator.eval_count}", end="\r", flush=True)

    runtime = time.time() - t_start
    print(f"\n  [NSGA-II] Completed in {runtime:.2f}s.")
    
    # Extract final Pareto front
    front0 = get_non_dominated_indices(objs)
    pareto_solutions = [eval_results[i] for i in front0]
    return {
        "algorithm": "NSGA-II",
        "runtime": runtime,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": pareto_solutions,
        "history": history
    }


# ==============================================================================
# 6. ALGORITHM 2: SPEA2
# ==============================================================================

def run_spea2(initial_pop: np.ndarray, evaluator: Evaluator, n_gen: int = 20, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    pop = [ind.copy() for ind in initial_pop]
    archive = [ind.copy() for ind in initial_pop]
    
    # Evaluate initial
    pop_results = [evaluator.evaluate(ind) for ind in pop]
    archive_results = [evaluator.evaluate(ind) for ind in archive]
    
    history = []
    print("  [SPEA2] Running generations...")
    for gen in range(1, n_gen + 1):
        # Combine population and archive (2N)
        union_pop = pop + archive
        union_results = pop_results + archive_results
        union_objs = np.array([res["objectives"] for res in union_results])
        M = len(union_objs)
        
        # 1. Strength & Raw Fitness
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
                    
        # 2. Density estimation via k-NN in normalized objective space
        mins = np.min(union_objs, axis=0)
        maxs = np.max(union_objs, axis=0)
        ranges = np.where(maxs - mins == 0, 1e-9, maxs - mins)
        norm_objs = (union_objs - mins) / ranges
        
        dists = cdist(norm_objs, norm_objs)
        dists.sort(axis=1)
        k_nn = int(np.sqrt(M))
        density = 1.0 / (dists[:, k_nn] + 2.0)
        total_fitness = raw_fitness + density
        
        # 3. Environmental selection to update archive
        non_dom_idx = [i for i, f in enumerate(total_fitness) if f < 1.0]
        if len(non_dom_idx) == N:
            new_archive_idx = non_dom_idx
        elif len(non_dom_idx) < N:
            # Fill with best dominated
            dom_idx = sorted([i for i in range(M) if i not in non_dom_idx], key=lambda i: total_fitness[i])
            new_archive_idx = non_dom_idx + dom_idx[:N - len(non_dom_idx)]
        else:
            # Truncate non-dominated using iterative nearest-neighbor removal
            survivors = list(non_dom_idx)
            sub_dists = dists[survivors][:, :]
            while len(survivors) > N:
                # Find pair with minimum distance
                # Recompute pairwise among survivors
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
        
        # 4. Mating selection: Binary tournament on archive
        offspring_pop = []
        for _ in range(N):
            i, j = rng.choice(len(archive), size=2, replace=False)
            parent1 = archive[i] if archive_fitness[i] <= archive_fitness[j] else archive[j]
            i, j = rng.choice(len(archive), size=2, replace=False)
            parent2 = archive[i] if archive_fitness[i] <= archive_fitness[j] else archive[j]
            
            child = crossover_uniform(parent1, parent2, k=500, rng=rng)
            child = mutate_swap(child, p_mut=0.3, n_swaps=4, rng=rng)
            offspring_pop.append(child)
            
        pop = offspring_pop
        pop_results = [evaluator.evaluate(ind) for ind in pop]
        
        best_f1 = max(res["f1"] for res in archive_results)
        best_fpr = min(res["fpr"] for res in archive_results)
        history.append({"gen": gen, "best_f1": best_f1, "best_fpr": best_fpr, "cache_evals": evaluator.eval_count})
        print(f"    Gen {gen:2d}/{n_gen} | Best F1: {best_f1:.4f} | Best FPR: {best_fpr:.4f} | Total Evals: {evaluator.eval_count}", end="\r", flush=True)

    runtime = time.time() - t_start
    print(f"\n  [SPEA2] Completed in {runtime:.2f}s.")
    
    # Final archive Pareto front
    arch_objs = np.array([res["objectives"] for res in archive_results])
    front0 = get_non_dominated_indices(arch_objs)
    pareto_solutions = [archive_results[i] for i in front0]
    return {
        "algorithm": "SPEA2",
        "runtime": runtime,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": pareto_solutions,
        "history": history
    }


# ==============================================================================
# 7. ALGORITHM 3: MOEA/D
# ==============================================================================

def generate_das_dennis_weights(n_obj: int = 3, H: int = 6) -> np.ndarray:
    """Generates uniform reference directions on simplex (Das-Dennis method)."""
    weights = []
    for i in range(H + 1):
        for j in range(H + 1 - i):
            k = H - i - j
            weights.append([i / H, j / H, k / H])
    return np.array(weights, dtype=float)


def run_moead(initial_pop: np.ndarray, evaluator: Evaluator, n_gen: int = 20, T: int = 6, seed: int = 42):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    
    # 1. Reference weight vectors (28 weights for 3 objectives, H=6)
    weights = generate_das_dennis_weights(n_obj=3, H=6)
    assert len(weights) == N, f"Weight count {len(weights)} must match population size {N}"
    
    # 2. Neighborhood B(i): T closest weight vectors
    weight_dists = cdist(weights, weights)
    neighborhoods = [np.argsort(weight_dists[i])[:T] for i in range(N)]
    
    # 3. Initial population & evaluation
    pop = [ind.copy() for ind in initial_pop]
    eval_results = [evaluator.evaluate(ind) for ind in pop]
    objs = np.array([res["objectives"] for res in eval_results])
    
    # 4. Ideal reference point z*
    z_star = np.min(objs, axis=0)
    
    def tchebycheff(f_val, w_vec):
        w = np.where(w_vec == 0, 1e-4, w_vec)
        return np.max(w * np.abs(f_val - z_star))

    history = []
    print("  [MOEA/D] Running generations...")
    for gen in range(1, n_gen + 1):
        for i in range(N):
            # Select 2 distinct parents from neighborhood B(i)
            p_idx1, p_idx2 = rng.choice(neighborhoods[i], size=2, replace=False)
            
            # Genetic variation
            child = crossover_uniform(pop[p_idx1], pop[p_idx2], k=500, rng=rng)
            child = mutate_swap(child, p_mut=0.3, n_swaps=4, rng=rng)
            
            # Evaluate child
            child_res = evaluator.evaluate(child)
            child_obj = child_res["objectives"]
            
            # Update ideal point z*
            z_star = np.minimum(z_star, child_obj)
            
            # Update neighbors in B(i) using Tchebycheff scalarization
            for j in neighborhoods[i]:
                if tchebycheff(child_obj, weights[j]) <= tchebycheff(objs[j], weights[j]):
                    pop[j] = child
                    eval_results[j] = child_res
                    objs[j] = child_obj
                    
        best_f1 = max(res["f1"] for res in eval_results)
        best_fpr = min(res["fpr"] for res in eval_results)
        history.append({"gen": gen, "best_f1": best_f1, "best_fpr": best_fpr, "cache_evals": evaluator.eval_count})
        print(f"    Gen {gen:2d}/{n_gen} | Best F1: {best_f1:.4f} | Best FPR: {best_fpr:.4f} | Total Evals: {evaluator.eval_count}", end="\r", flush=True)

    runtime = time.time() - t_start
    print(f"\n  [MOEA/D] Completed in {runtime:.2f}s.")
    
    # Extract non-dominated solutions
    front0 = get_non_dominated_indices(objs)
    pareto_solutions = [eval_results[i] for i in front0]
    return {
        "algorithm": "MOEA/D",
        "runtime": runtime,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": pareto_solutions,
        "history": history
    }


# ==============================================================================
# 8. ALGORITHM 4: MODE (Multi-Objective Differential Evolution)
# ==============================================================================

def run_mode(initial_pop: np.ndarray, evaluator: Evaluator, n_gen: int = 20, F: float = 0.8, CR: float = 0.8, seed: int = 42):
    """
    Classical Multi-Objective Differential Evolution (MODE - Xue et al., 2003 / Babu & Jehan, 2003).
    DE/rand/1/bin variation + Pareto dominance replacement + non-dominated sorting crowding archive.
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)
    N = len(initial_pop)
    pop = [ind.copy() for ind in initial_pop]
    
    # Evaluate initial
    eval_results = [evaluator.evaluate(ind) for ind in pop]
    objs = np.array([res["objectives"] for res in eval_results])
    
    history = []
    print("  [MODE] Running generations...")
    for gen in range(1, n_gen + 1):
        candidate_pop = list(pop)
        candidate_results = list(eval_results)
        
        for i in range(N):
            # Pick 3 random distinct individuals r1, r2, r3 != i
            pool = [idx for idx in range(N) if idx != i]
            r1_idx, r2_idx, r3_idx = rng.choice(pool, size=3, replace=False)
            
            # Differential variation
            trial = mode_differential_variation(pop[i], pop[r1_idx], pop[r2_idx], pop[r3_idx],
                                               F=F, CR=CR, k=500, rng=rng)
            trial_res = evaluator.evaluate(trial)
            trial_obj = trial_res["objectives"]
            
            # Pareto dominance selection
            if dominates(trial_obj, objs[i]):
                # Trial strictly dominates target -> replaces target directly
                pop[i] = trial
                eval_results[i] = trial_res
                objs[i] = trial_obj
            elif dominates(objs[i], trial_obj):
                # Target dominates trial -> discard trial
                pass
            else:
                # Mutually non-dominating: retain both in candidate pool for generational truncation
                candidate_pop.append(trial)
                candidate_results.append(trial_res)
                
        # Generational environmental selection: truncate candidate_pop back to N
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
            
        best_f1 = max(res["f1"] for res in eval_results)
        best_fpr = min(res["fpr"] for res in eval_results)
        history.append({"gen": gen, "best_f1": best_f1, "best_fpr": best_fpr, "cache_evals": evaluator.eval_count})
        print(f"    Gen {gen:2d}/{n_gen} | Best F1: {best_f1:.4f} | Best FPR: {best_fpr:.4f} | Total Evals: {evaluator.eval_count}", end="\r", flush=True)

    runtime = time.time() - t_start
    print(f"\n  [MODE] Completed in {runtime:.2f}s.")
    
    front0 = get_non_dominated_indices(objs)
    pareto_solutions = [eval_results[i] for i in front0]
    return {
        "algorithm": "MODE",
        "runtime": runtime,
        "evaluations": evaluator.eval_count,
        "pareto_solutions": pareto_solutions,
        "history": history
    }


# ==============================================================================
# 9. FULL VALIDATION EVALUATION & SELECTION
# ==============================================================================

def evaluate_on_full_validation(pareto_solutions, X_train_csc, y_train, X_val_csc, y_val, feature_names):
    """
    Evaluates Pareto candidate subsets on ALL 31,915 validation samples.
    Definitive, unbiased full-set comparison.
    """
    full_eval_results = []
    
    for sol in pareto_solutions:
        idx = sol["indices"]
        assert len(idx) == 500
        
        # 1. Fit LinearSVC on full training set
        t0_fit = time.perf_counter()
        clf = LinearSVC(class_weight="balanced", C=1.0, random_state=42, dual="auto")
        clf.fit(X_train_csc[:, idx], y_train)
        t_fit = time.perf_counter() - t0_fit
        
        # 2. Predict on full validation set (5 timed runs)
        _ = clf.predict(X_val_csc[:, idx])  # Warmup
        durations = []
        y_pred = None
        for _ in range(5):
            t0 = time.perf_counter()
            y_pred = clf.predict(X_val_csc[:, idx])
            durations.append(time.perf_counter() - t0)
        avg_inf_time = float(np.mean(durations))
        
        # 3. Metrics on full validation set
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn))
        fnr = float(fn / (fn + tp))
        f1 = float(f1_score(y_val, y_pred))
        acc = float(accuracy_score(y_val, y_pred))
        prec = float(precision_score(y_val, y_pred, zero_division=0))
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        
        full_eval_results.append({
            "features_count": 500,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "inference_time_ms": avg_inf_time * 1000.0,
            "inference_time_per_comment_us": (avg_inf_time / len(y_val)) * 1e6,
            "training_time_s": t_fit,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "indices": idx.tolist(),
            "feature_tokens": [feature_names[i] for i in idx]
        })
        
    return full_eval_results


# ==============================================================================
# 10. MAIN EXPERIMENT ORCHESTRATION
# ==============================================================================

def main():
    total_experiment_start = time.time()
    print("=" * 80)
    print("      MAIN EVOLUTIONARY FEATURE-SELECTION BENCHMARK (NSGA-II, SPEA2, MOEA/D, MODE)")
    print("=" * 80)
    
    # 1. Load Preprocessed Artifacts
    print("\n[Phase 1] Loading preprocessed artifacts...")
    X_train_csr = sparse.load_npz("X_train.npz")
    X_val_csr = sparse.load_npz("X_val.npz")
    y_train = np.load("y_train.npy")
    y_val = np.load("y_val.npy")
    with open("feature_names.txt", "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f if line.strip()]
        
    # Convert to CSC format for fast column slicing
    print("  Converting matrices to CSC format for high-speed feature slicing...")
    X_train_csc = X_train_csr.tocsc()
    X_val_csc = X_val_csr.tocsc()
    print(f"  • X_train: {X_train_csc.shape} | X_val: {X_val_csc.shape} | Vocabulary: {len(feature_names):,} terms")

    # 2. Create Fixed Stratified Search Evaluation Subset (8,000 samples)
    print("\n[Phase 2] Creating fixed stratified search evaluation subset (8,000 samples)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=8000, random_state=42)
    _, search_idx = next(sss.split(X_val_csr, y_val))
    search_idx.sort()
    
    X_val_search_csc = X_val_csc[search_idx]
    y_val_search = y_val[search_idx]
    print(f"  • Search subset shape: {X_val_search_csc.shape}")
    print(f"  • Search class counts: Non-Toxic = {(y_val_search==0).sum():,}, Toxic = {(y_val_search==1).sum():,} ({np.mean(y_val_search)*100:.2f}% positive)")

    # 3. Compute Univariate Chi-Squared Rankings for Initial Genetic Seeds
    print("\n[Phase 3] Computing feature importance ranks for shared seed initialization...")
    chi2_scores, _ = chi2(X_train_csr, y_train)
    ranked_features = np.argsort(chi2_scores)[::-1]
    
    # 4. Generate Identical Initial Population (Shared across all 4 algorithms)
    N_POP = 28
    N_GEN = 20
    print(f"\n[Phase 4] Generating identical initial population (N = {N_POP}, K = 500 features)...")
    initial_population = generate_initial_population(n_pop=N_POP, n_features=10000, k=500,
                                                    ranked_features=ranked_features, seed=42)
    print(f"  • Initial population generated: shape = {initial_population.shape}")
    for ind in initial_population:
        assert np.sum(ind) == 500

    # 5. Initialize Evaluator
    evaluator = Evaluator(X_train_csc, y_train, X_val_search_csc, y_val_search, name="SearchEvaluator")

    # 6. Execute Four Evolutionary Algorithms
    results = {}
    
    print("\n" + "=" * 80)
    print("                     EXECUTING ALGORITHMS (20 GENERATIONS EACH)")
    print("=" * 80)
    
    # Algorithm 1: NSGA-II
    print("\n--- 1. Executing NSGA-II ---")
    results["NSGA-II"] = run_nsga2(initial_population, evaluator, n_gen=N_GEN, seed=42)
    
    # Algorithm 2: SPEA2
    print("\n--- 2. Executing SPEA2 ---")
    results["SPEA2"] = run_spea2(initial_population, evaluator, n_gen=N_GEN, seed=42)
    
    # Algorithm 3: MOEA/D
    print("\n--- 3. Executing MOEA/D ---")
    results["MOEA/D"] = run_moead(initial_population, evaluator, n_gen=N_GEN, T=6, seed=42)
    
    # Algorithm 4: MODE
    print("\n--- 4. Executing MODE ---")
    results["MODE"] = run_mode(initial_population, evaluator, n_gen=N_GEN, F=0.8, CR=0.8, seed=42)

    # 7. Final Full-Validation Evaluation
    print("\n" + "=" * 80)
    print("        PHASE 2: DEFINITIVE FULL VALIDATION EVALUATION (31,915 ROWS)")
    print("=" * 80)
    
    final_full_results = {}
    best_solutions_summary = []
    
    for algo_name, res in results.items():
        print(f"\nEvaluating Pareto solutions for {algo_name} on ALL 31,915 validation samples...")
        pareto_solutions = res["pareto_solutions"]
        print(f"  • {algo_name} produced {len(pareto_solutions)} Pareto solutions in search phase.")
        
        full_evals = evaluate_on_full_validation(pareto_solutions, X_train_csc, y_train, X_val_csc, y_val, feature_names)
        final_full_results[algo_name] = full_evals
        
        # Identify key representative solutions:
        # a) Best F1 solution
        best_f1_sol = max(full_evals, key=lambda s: s["f1"])
        # b) Best FPR solution
        best_fpr_sol = min(full_evals, key=lambda s: s["fpr"])
        # c) Balanced Knee solution: maximizes F1 / (1 + FPR)
        knee_sol = max(full_evals, key=lambda s: s["f1"] / (1.0 + s["fpr"]))
        
        best_solutions_summary.append({
            "Algorithm": algo_name,
            "Best_F1": best_f1_sol["f1"],
            "Best_F1_FPR": best_f1_sol["fpr"],
            "Best_F1_InfTime_ms": best_f1_sol["inference_time_ms"],
            "Best_FPR": best_fpr_sol["fpr"],
            "Best_FPR_F1": best_fpr_sol["f1"],
            "Knee_F1": knee_sol["f1"],
            "Knee_FPR": knee_sol["fpr"],
            "Knee_InfTime_ms": knee_sol["inference_time_ms"],
            "Knee_Precision": knee_sol["precision"],
            "Knee_Recall": knee_sol["recall"],
            "Search_Runtime_s": res["runtime"],
            "Pareto_Count": len(pareto_solutions)
        })
        
        print(f"  • {algo_name} Full Validation Summary:")
        print(f"    - Best F1:   {best_f1_sol['f1']:.4f} (FPR = {best_f1_sol['fpr']:.4f}, Inf Time = {best_f1_sol['inference_time_ms']:.3f} ms)")
        print(f"    - Best FPR:  {best_fpr_sol['fpr']:.4f} (F1 = {best_fpr_sol['f1']:.4f})")
        print(f"    - Knee Sol:  F1 = {knee_sol['f1']:.4f} | FPR = {knee_sol['fpr']:.4f} | Inf Time = {knee_sol['inference_time_ms']:.3f} ms")

    # 8. Baseline Reference
    baseline = {
        "Algorithm": "Baseline (LinearSVC)",
        "Features": 10000,
        "F1": 0.6848,
        "FPR": 0.0681,
        "Inference_Time_ms": 3.375,
        "Precision": 0.5809,
        "Recall": 0.8339,
        "Search_Runtime": "N/A",
        "Pareto_Solutions": "N/A"
    }

    # 9. Build Comparison Table DataFrame
    comp_rows = []
    # Add Baseline
    comp_rows.append({
        "Algorithm": "Baseline (Full Features)",
        "Features": 10000,
        "F1 Score": 0.6848,
        "False Positive Rate (FPR)": "6.81%",
        "Inference Time": "3.375 ms",
        "Search Runtime": "N/A",
        "Pareto Front Size": "N/A"
    })
    
    for s in best_solutions_summary:
        comp_rows.append({
            "Algorithm": s["Algorithm"],
            "Features": 500,
            "F1 Score": round(s["Knee_F1"], 4),
            "False Positive Rate (FPR)": f"{s['Knee_FPR']*100:.2f}%",
            "Inference Time": f"{s['Knee_InfTime_ms']:.3f} ms",
            "Search Runtime": f"{s['Search_Runtime_s']:.1f} s",
            "Pareto Front Size": s["Pareto_Count"]
        })
        
    df_comparison = pd.DataFrame(comp_rows)
    df_comparison.to_csv("algorithm_comparison.csv", index=False)
    print("\nSaved comparison table to 'algorithm_comparison.csv'")
    
    # Save all full-validation Pareto solutions to pareto_solutions.csv
    all_pareto_rows = []
    for algo_name, full_evals in final_full_results.items():
        for i, sol in enumerate(full_evals):
            all_pareto_rows.append({
                "Algorithm": algo_name,
                "Solution_ID": i + 1,
                "Features_Count": sol["features_count"],
                "F1_Score": sol["f1"],
                "FPR": sol["fpr"],
                "FNR": sol["fnr"],
                "Inference_Time_ms": sol["inference_time_ms"],
                "Inference_Time_per_comment_us": sol["inference_time_per_comment_us"],
                "Precision": sol["precision"],
                "Recall": sol["recall"],
                "Accuracy": sol["accuracy"],
                "Training_Time_s": sol["training_time_s"]
            })
    df_pareto = pd.DataFrame(all_pareto_rows)
    df_pareto.to_csv("pareto_solutions.csv", index=False)
    print("Saved all Pareto solutions to 'pareto_solutions.csv'")

    # 10. Save Detailed JSON Files & Feature Indices
    for algo_name in ["NSGA-II", "SPEA2", "MOEA/D", "MODE"]:
        clean_name = algo_name.lower().replace("-", "").replace("/", "")
        res_file = f"results_{clean_name}.json"
        
        # Save json with results and full evaluations
        algo_export = {
            "algorithm": algo_name,
            "search_phase": {
                "runtime_seconds": results[algo_name]["runtime"],
                "evaluations_count": results[algo_name]["evaluations"],
                "history": results[algo_name]["history"]
            },
            "full_validation_evaluations": final_full_results[algo_name]
        }
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(algo_export, f, indent=4)
        print(f"Saved {algo_name} results to '{res_file}'")
        
        # Save best knee solution feature indices & tokens
        best_sol = max(final_full_results[algo_name], key=lambda s: s["f1"] / (1.0 + s["fpr"]))
        np.save(f"selected_features_{clean_name}.npy", np.array(best_sol["indices"]))
        with open(f"selected_features_{clean_name}.txt", "w", encoding="utf-8") as f:
            for token in best_sol["feature_tokens"]:
                f.write(f"{token}\n")

    # Save experiment configuration
    exp_config = {
        "dataset": "Jigsaw Toxic Comment Classification",
        "raw_samples": 159571,
        "train_samples": 127656,
        "full_val_samples": 31915,
        "search_subset_samples": 8000,
        "feature_count_target": 500,
        "feature_reduction_percent": (10000 - 500) / 10000 * 100.0,
        "classifier": "LinearSVC(class_weight='balanced', C=1.0, random_state=42, dual='auto')",
        "algorithms": ["NSGA-II", "SPEA2", "MOEA/D", "MODE"],
        "population_size": N_POP,
        "generations": N_GEN,
        "total_unique_evaluations": evaluator.eval_count,
        "cache_hits": evaluator.cache_hits,
        "random_seed": 42
    }
    with open("experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=4)
    print("Saved experiment configuration to 'experiment_config.json'")

    # 11. Generate Visualizations
    print("\n[Phase 5] Generating visualization plots...")
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: Pareto Front (F1 vs FPR)
    plt.figure(figsize=(9, 6), dpi=300)
    colors = {"NSGA-II": "#1f77b4", "SPEA2": "#2ca02c", "MOEA/D": "#ff7f0e", "MODE": "#d62728"}
    markers = {"NSGA-II": "o", "SPEA2": "s", "MOEA/D": "^", "MODE": "D"}
    
    for algo_name, full_evals in final_full_results.items():
        f1_vals = [s["f1"] for s in full_evals]
        fpr_vals = [s["fpr"] * 100.0 for s in full_evals]
        plt.scatter(fpr_vals, f1_vals, label=f"{algo_name} (500 feats)",
                    color=colors[algo_name], marker=markers[algo_name], s=90, alpha=0.85, edgecolors="k", zorder=3)
        
    # Baseline point
    plt.scatter([baseline["FPR"] * 100.0], [baseline["F1"]], color="black", marker="X", s=180,
                label="Baseline (10,000 feats)", zorder=4)
    
    plt.xlabel("False Positive Rate (FPR %)", fontsize=12, fontweight="bold")
    plt.ylabel("F1 Score (Validation)", fontsize=12, fontweight="bold")
    plt.title("Pareto Front Comparison: F1 Score vs. False Positive Rate (Full Validation Set)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/pareto_f1_vs_fpr.png")
    plt.close()
    print("  • Saved 'plots/pareto_f1_vs_fpr.png'")

    # Plot 2: F1 vs Inference Time
    plt.figure(figsize=(9, 6), dpi=300)
    for algo_name, full_evals in final_full_results.items():
        f1_vals = [s["f1"] for s in full_evals]
        inf_vals = [s["inference_time_ms"] for s in full_evals]
        plt.scatter(inf_vals, f1_vals, label=f"{algo_name} (500 feats)",
                    color=colors[algo_name], marker=markers[algo_name], s=90, alpha=0.85, edgecolors="k", zorder=3)
        
    plt.scatter([baseline["Inference_Time_ms"]], [baseline["F1"]], color="black", marker="X", s=180,
                label="Baseline (10,000 feats)", zorder=4)
    
    plt.xlabel("Inference Time for 31,915 Samples (ms)", fontsize=12, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=12, fontweight="bold")
    plt.title("F1 Score vs. Inference Time Trade-Off", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/pareto_f1_vs_inference_time.png")
    plt.close()
    print("  • Saved 'plots/pareto_f1_vs_inference_time.png'")

    # Plot 3: Convergence History across Generations
    plt.figure(figsize=(9, 6), dpi=300)
    for algo_name, res in results.items():
        gens = [h["gen"] for h in res["history"]]
        f1_progress = [h["best_f1"] for h in res["history"]]
        plt.plot(gens, f1_progress, label=algo_name, color=colors[algo_name], linewidth=2.5, marker="o", markersize=4)
        
    plt.xlabel("Generation", fontsize=12, fontweight="bold")
    plt.ylabel("Best F1 Score (Search Subset)", fontsize=12, fontweight="bold")
    plt.title("Evolutionary Convergence Comparison across 20 Generations", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/algorithm_convergence.png")
    plt.close()
    print("  • Saved 'plots/algorithm_convergence.png'")

    # Plot 4: Baseline vs 500-Feature Solution Bar Comparison
    plt.figure(figsize=(10, 6), dpi=300)
    models = ["Baseline (10k)"] + [s["Algorithm"] + " (500)" for s in best_solutions_summary]
    f1_scores = [baseline["F1"]] + [s["Knee_F1"] for s in best_solutions_summary]
    fpr_scores = [baseline["FPR"] * 100.0] + [s["Knee_FPR"] * 100.0 for s in best_solutions_summary]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    rects1 = ax1.bar(x - width/2, f1_scores, width, label="F1 Score (Higher is better)", color="#2b5c8f")
    ax1.set_ylabel("F1 Score", fontsize=12, color="#2b5c8f", fontweight="bold")
    ax1.set_ylim(0, 0.85)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, fpr_scores, width, label="FPR % (Lower is better)", color="#d95f02")
    ax2.set_ylabel("False Positive Rate (%)", fontsize=12, color="#d95f02", fontweight="bold")
    ax2.set_ylim(0, 10.0)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax1.set_title("Performance Comparison: 10,000-Feature Baseline vs. 500-Feature Evolutionary Subsets", fontsize=12, fontweight="bold")
    
    # Values on bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    plt.tight_layout()
    plt.savefig("plots/baseline_vs_evolutionary_comparison.png")
    plt.close()
    print("  • Saved 'plots/baseline_vs_evolutionary_comparison.png'")

    # 12. Final Decision & Winner Determination
    print("\n" + "=" * 80)
    print("                       FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(df_comparison.to_string(index=False))
    
    total_experiment_time = time.time() - total_experiment_start
    print(f"\nTotal Experiment Execution Time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
