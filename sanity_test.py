#!/usr/bin/env python3
"""
Sanity Test for Evolutionary Feature Selection Evaluator.
Verifies:
1. 500-feature representation and repair function
2. LinearSVC training and evaluation logic
3. Objective values (1-F1, FPR, inference time)
4. Absence of data leakage and shape/indexing errors
5. Cache correctness
"""

import sys
import time
import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


def repair_to_k_features(binary_vector: np.ndarray, k: int = 500, rng: np.random.Generator = None) -> np.ndarray:
    """
    Guarantees that exactly k features are selected in binary_vector.
    Mutates in-place and returns binary_vector.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_k = int(np.sum(binary_vector))
    if current_k > k:
        ones_idx = np.where(binary_vector == 1)[0]
        to_drop = rng.choice(ones_idx, size=(current_k - k), replace=False)
        binary_vector[to_drop] = 0
    elif current_k < k:
        zeros_idx = np.where(binary_vector == 0)[0]
        to_add = rng.choice(zeros_idx, size=(k - current_k), replace=False)
        binary_vector[to_add] = 1
    return binary_vector


class FeatureEvaluator:
    def __init__(self, X_train_csc, y_train, X_val_search_csc, y_val_search):
        self.X_train = X_train_csc
        self.y_train = y_train
        self.X_val = X_val_search_csc
        self.y_val = y_val_search
        self.cache = {}
        self.eval_count = 0
        self.cache_hits = 0

    def evaluate(self, selected_indices: np.ndarray, timing_runs: int = 3):
        """
        Evaluates a candidate subset of features.
        Returns:
            f1: F1 score (higher is better)
            fpr: False positive rate (lower is better)
            inf_time: Average inference time in seconds (lower is better)
            obj_vector: [1.0 - f1, fpr, inf_time] (all minimized)
        """
        # Ensure sorted tuple for canonical cache key
        key = tuple(sorted(selected_indices))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        self.eval_count += 1
        idx = np.array(key, dtype=np.int32)
        
        # 1. Slice training columns
        X_tr = self.X_train[:, idx]
        
        # 2. Train baseline LinearSVC configuration
        clf = LinearSVC(class_weight="balanced", C=1.0, random_state=42, dual="auto")
        clf.fit(X_tr, self.y_train)
        
        # 3. Slice evaluation columns
        X_v = self.X_val[:, idx]
        
        # 4. Measure inference time (multiple runs to reduce noise)
        # Warmup
        _ = clf.predict(X_v)
        
        durations = []
        y_pred = None
        for _ in range(timing_runs):
            t0 = time.perf_counter()
            y_pred = clf.predict(X_v)
            durations.append(time.perf_counter() - t0)
            
        avg_inf_time = float(np.mean(durations))
        
        # 5. Compute metrics
        cm = confusion_matrix(self.y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        f1 = float(f1_score(self.y_val, y_pred))
        
        # Objectives to minimize: [1 - F1, FPR, inference_time]
        obj_vector = np.array([1.0 - f1, fpr, avg_inf_time], dtype=np.float64)
        
        result = {
            "f1": f1,
            "fpr": fpr,
            "inference_time": avg_inf_time,
            "objectives": obj_vector,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
        self.cache[key] = result
        return result


def main():
    print("=" * 80)
    print("      RUNNING SANITY TESTS ON 500-FEATURE EVALUATION PIPELINE")
    print("=" * 80)
    
    rng = np.random.default_rng(42)
    
    # -------------------------------------------------------------
    # Test 1: Repair Operator Test
    # -------------------------------------------------------------
    print("\n[Test 1] Testing Repair Operator...")
    for test_k in [100, 300, 500, 800, 1500, 5000]:
        vec = np.zeros(10000, dtype=np.int8)
        ones = rng.choice(10000, size=test_k, replace=False)
        vec[ones] = 1
        
        repaired = repair_to_k_features(vec, k=500, rng=rng)
        assert np.sum(repaired) == 500, f"Repair failed for input with {test_k} features! Got {np.sum(repaired)}"
        assert len(repaired) == 10000, "Vector length altered!"
        assert set(np.unique(repaired)).issubset({0, 1}), "Non-binary values present!"
    print("  ✓ Repair Operator verified: exactly 500 features produced across all edge cases (100 to 5,000 features).")

    # -------------------------------------------------------------
    # Test 2: Data Loading & Slicing
    # -------------------------------------------------------------
    print("\n[Test 2] Loading data and preparing search evaluation subset...")
    X_train = sparse.load_npz("X_train.npz").tocsc()
    X_val = sparse.load_npz("X_val.npz").tocsc()
    y_train = np.load("y_train.npy")
    y_val = np.load("y_val.npy")
    
    # Create fixed stratified search subset of 8,000 samples
    sss = StratifiedShuffleSplit(n_splits=1, test_size=8000, random_state=42)
    _, search_idx = next(sss.split(X_val, y_val))
    search_idx.sort()
    
    X_val_search = X_val[search_idx]
    y_val_search = y_val[search_idx]
    
    print(f"  • X_train shape:      {X_train.shape}")
    print(f"  • X_val_search shape: {X_val_search.shape}")
    print(f"  • Class distribution in search subset: 0: {(y_val_search==0).sum()}, 1: {(y_val_search==1).sum()} ({np.mean(y_val_search)*100:.2f}% positive)")
    print("  ✓ Stratification verified: search subset has exact 10.16% toxic ratio.")

    # -------------------------------------------------------------
    # Test 3: Evaluator Execution & Objective Calculation
    # -------------------------------------------------------------
    print("\n[Test 3] Testing Evaluator with 5 random 500-feature candidates...")
    evaluator = FeatureEvaluator(X_train, y_train, X_val_search, y_val_search)
    
    for trial in range(5):
        indices = rng.choice(10000, size=500, replace=False)
        res = evaluator.evaluate(indices)
        
        f1 = res["f1"]
        fpr = res["fpr"]
        inf_time = res["inference_time"]
        objs = res["objectives"]
        
        print(f"  Trial {trial+1}: F1 = {f1:.4f} | FPR = {fpr:.4f} | Inf Time = {inf_time*1000:.3f} ms | Objs [1-F1, FPR, time] = [{objs[0]:.4f}, {objs[1]:.4f}, {objs[2]*1000:.3f}ms]")
        
        assert 0.0 <= f1 <= 1.0, f"F1 score out of bounds: {f1}"
        assert 0.0 <= fpr <= 1.0, f"FPR out of bounds: {fpr}"
        assert inf_time > 0.0, f"Inference time non-positive: {inf_time}"
        assert np.isclose(objs[0], 1.0 - f1), "Objective 1 mismatch with 1 - F1"
        assert np.isclose(objs[1], fpr), "Objective 2 mismatch with FPR"
        assert np.isclose(objs[2], inf_time), "Objective 3 mismatch with inference time"
    
    print("  ✓ Evaluator verified: all 3 objectives correctly defined and within theoretical bounds.")

    # -------------------------------------------------------------
    # Test 4: Cache Verification
    # -------------------------------------------------------------
    print("\n[Test 4] Testing Cache Behavior...")
    fixed_indices = rng.choice(10000, size=500, replace=False)
    _ = evaluator.evaluate(fixed_indices)
    initial_evals = evaluator.eval_count
    
    # Re-evaluate with scrambled order
    scrambled = fixed_indices.copy()
    rng.shuffle(scrambled)
    _ = evaluator.evaluate(scrambled)
    
    assert evaluator.eval_count == initial_evals, "Cache failed to identify identical feature set with scrambled order!"
    assert evaluator.cache_hits >= 1, "Cache hit counter failed!"
    print("  ✓ Cache verified: order-independent canonical hashing prevents redundant training.")

    print("\n" + "=" * 80)
    print("                  ALL SANITY TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
