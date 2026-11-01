# MLRan SVM Classification Experiment Report

**Project Directory:** `Assignment Final/SVM_Results/`  
**Date:** November 2, 2026  
**Target:** Binary Classification on `sample_type` (Class 0: Benign, Class 1: Malicious)  
**Dataset:** MLRan Dynamic & Static Behavioral Telemetry with 483 RFE-Selected Features  
**Author:** Machine Learning Engineering Team  

---

## Executive Summary

This report documents the implementation, rigorous validation, and empirical benchmarking of Support Vector Machine (SVM) classifiers on the MLRan ransomware and malware behavioral telemetry dataset. Using the pre-partitioned 80/20 train/test split (3,905 training samples, 975 test samples) and 483 features selected by Recursive Feature Elimination (RFE), three distinct SVM architectures—**Linear SVM**, **Radial Basis Function (RBF) SVM**, and **Polynomial SVM**—were systematically optimized using 3-fold Stratified Cross-Validation on training data alone, and subsequently evaluated on the untouched test partition.

### Key Benchmark Results

| Model | Best Hyperparameters | 3-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Training Time (s) | Inference Time (ms) | Inference Time (µs/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | $C = 0.1$ | 0.9914 | 97.74% | 0.9763 | 0.9763 | 0.9763 | 0.9935 | **0.158 s** | **11.80 ms** | **12.10 µs** |
| **RBF SVM (Winner)** | $C = 10, \gamma = \text{'scale'}$ | **0.9928** | **98.15%** | **0.9745** | **0.9871** | **0.9808** | **0.9974** | 0.359 s | 98.84 ms | 101.38 µs |
| **Polynomial SVM** | $C = 10, d = 2, \gamma = \text{'scale'}$ | 0.9879 | 97.95% | 0.9744 | 0.9828 | 0.9786 | 0.9970 | 0.459 s | 60.98 ms | 62.54 µs |

* **Winning Classifier:** **RBF SVM** achieved the highest overall test performance with an **F1-score of 0.9808**, **Accuracy of 98.15%**, **Recall of 98.71%**, and **ROC-AUC of 0.9974**. It missed only 6 malicious samples out of 465 (False Negative Rate: 1.29%).
* **Operational Runner-Up:** **Linear SVM** delivered outstanding latency efficiency (**12.10 µs per sample**, 8.4x faster inference than RBF) with near-identical precision and an F1-score of **0.9763**.

---

## 1. Dataset Description & Partitioning

The MLRan dataset consists of behavioral telemetry captured from Windows PE executables running in an instrumented sandbox environment. 

### 1.1 Partition Dimensions

- **Total Samples:** 4,880 unique executable traces
- **Training Set (`MLRan_X_train_RFE.csv`):** 3,905 samples (80.02%)
- **Test Set (`MLRan_X_test_RFE.csv`):** 975 samples (19.98%)
- **Target Distribution:**
  - **Training:** 2,040 Benign (52.24%) vs. 1,865 Malicious (47.76%)
  - **Testing:** 510 Benign (52.31%) vs. 465 Malicious (47.69%)
  - **Balance Ratio:** ~1.09:1 (essentially balanced, no severe class imbalance)

```
                            Total Data: 4,880 Samples
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
         Train Split (80%)                          Test Split (20%)
           3,905 Samples                              975 Samples
     ┌────────────┴────────────┐                ┌───────────┴───────────┐
     ▼                         ▼                ▼                       ▼
Benign (0)              Malicious (1)      Benign (0)             Malicious (1)
 2,040 (52.2%)           1,865 (47.8%)       510 (52.3%)            465 (47.7%)
```

### 1.2 Integrity Verification

Before training, strict integrity assertions confirmed:
1. $\text{IDs}_{\text{train}} \cap \text{IDs}_{\text{test}} = \emptyset$ (zero sample overlap across splits).
2. All 4,880 samples have matching labels in `MLRan_labels.csv`.
3. Zero missing values (`NaN`), infinite values (`inf`), or duplicated samples.

---

## 2. Classification Objective

The primary operational objective is automated malware/ransomware triage:

$$\text{Input: } \mathbf{x} \in \{0, 1\}^{483} \quad \longrightarrow \quad \text{Output: } y \in \{0, 1\}$$

- **Class 0 (Negative / Benign):** Safe, legitimate system utilities and benign applications.
- **Class 1 (Positive / Malicious):** Ransomware, trojans, droppers, and backdoors.

### Objective Priority in Cyber Defense
In ransomware and malware triage:
1. **Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$) is critical:** A false negative (FN) allows undetected ransomware to execute, encrypt endpoints, and exfiltrate credentials.
2. **Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) must remain high:** Excessive false alarms overwhelm Security Operations Center (SOC) analysts and disrupt legitimate user tasks.
3. **F1-score:** Chosen as the primary optimization metric during cross-validation as it balances precision and recall harmonic mean.

---

## 3. Feature Preparation & Data Leakage Prevention

### 3.1 483 Selected RFE Features

The input matrices contain exactly 483 binary features selected via Recursive Feature Elimination (RFE) from an original pool of over 6.4 million sparse static and dynamic indicators. According to `RFE_selected_feature_names_dic.json`, these features span 8 telemetry domains:

```
Telemetry Domain Breakdown (483 Features):
  ├── STRING    : 223 features (embedded suspicious strings & API symbols)
  ├── REG       :  90 features (Windows Registry key reads, writes, and persistence)
  ├── API       :  63 features (Win32 / Native API invocations like NtProtectVirtualMemory)
  ├── SYSTEM    :  38 features (system queries: GetNativeSystemInfo, computer name)
  ├── DROP      :  23 features (dropped PE artifacts and script payloads)
  ├── SIGNATURE :  20 features (high-level behavioral detection signatures)
  ├── FILE      :  16 features (file creation, modification, and deletion calls)
  └── DIRECTORY :  10 features (directory queries and staging directories)
```

### 3.2 Explicit Exclusion of Metadata Columns

Both `MLRan_X_train_RFE.csv` and `MLRan_X_test_RFE.csv` contain four non-feature columns at indices 0–3: `sample_id`, `sample_type`, `family_label`, and `type_label`.

> [!CAUTION]
> **Data Leakage Safeguard:**
> If `sample_type`, `family_label`, or `type_label` were inadvertently included in the feature matrix $\mathbf{X}$, the model would learn the identity mapping $y = \text{sample\_type}$ and achieve trivial, artificial 100% test accuracy.
> 
> In `svm_experiment.py`, the feature extraction explicitly discards all metadata columns:
> ```python
> metadata_cols = ["sample_id", "sample_type", "family_label", "type_label"]
> feature_cols = [c for c in train_df.columns if c not in metadata_cols]
> X_train = train_df[feature_cols].to_numpy(dtype=np.float64)  # Shape: (3905, 483)
> X_test  = test_df[feature_cols].to_numpy(dtype=np.float64)   # Shape: (975, 483)
> ```

---

## 4. SVM Mathematical Formulation & Kernel Theory

Support Vector Machines construct an optimal separating hyperplane in a high-dimensional feature space that maximizes the geometric margin between classes.

### 4.1 Soft-Margin SVM Formulation

Given training pairs $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$, the primal optimization problem is:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$

$$\text{subject to } y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i = 1, \dots, n$$

where:
- $\mathbf{w}$ is the normal vector to the separating hyperplane.
- $b$ is the bias offset.
- $\xi_i$ are slack variables measuring margin violations for non-separable points.
- $C > 0$ is the regularization parameter governing the trade-off between margin width ($\frac{2}{\|\mathbf{w}\|}$) and classification errors.
- $\phi(\mathbf{x})$ is a non-linear mapping into a higher-dimensional Hilbert space.

The dual formulation expresses the decision function solely in terms of inner products:

$$f(\mathbf{x}) = \text{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)$$

where $\alpha_i$ are Lagrange multipliers and $K(\mathbf{x}_i, \mathbf{x}) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}) \rangle$ is the kernel function.

---

### 4.2 Evaluated Kernel Functions

#### 1. Linear Kernel
$$K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$$
- **Characteristics:** Directly evaluates the standard dot product in $\mathbb{R}^{483}$ without projection.
- **Strengths:** Maximum computational efficiency ($\mathcal{O}(d)$ per prediction), direct interpretability of primal feature weights $\mathbf{w}$, and strong robustness against overfitting in high-dimensional text/telemetry spaces.

#### 2. Radial Basis Function (RBF / Gaussian) Kernel
$$K(\mathbf{x}, \mathbf{z}) = \exp\left( -\gamma \|\mathbf{x} - \mathbf{z}\|^2 \right)$$
- **Characteristics:** Maps inputs into an infinite-dimensional reproducing kernel Hilbert space (RKHS).
- **Strengths:** Captures highly complex, non-linear multi-feature conjunctions (e.g., sample is malicious if API A is present AND registry key B is modified AND string C is absent).
- **Hyperparameter $\gamma$:**
  - $\text{"scale"}$: $\gamma = \frac{1}{d \cdot \text{Var}(X)} = \frac{1}{483 \cdot 1.0} \approx 0.00207$.
  - $\text{"auto"}$: $\gamma = \frac{1}{d} = \frac{1}{483} \approx 0.00207$.

#### 3. Polynomial Kernel
$$K(\mathbf{x}, \mathbf{z}) = \left( \gamma \mathbf{x}^T \mathbf{z} + r \right)^d$$
- **Characteristics:** Generates explicit combinations of all up to degree-$d$ interactions between features.
- **Strengths:** For degree $d=2$, it explicitly maps pairwise co-occurrences of behavioral calls ($x_i \cdot x_j$), reflecting how malware detection heuristics look for specific API combinations.

---

## 5. Scaling Methodology & Pipeline Architecture

Although the original features are binary indicators $\{0, 1\}$, feature scaling is essential for distance-based kernel methods:

### 5.1 Why Standardization Matters for Binary Telemetry
1. **Equivalence of Margin Penalties:** In unscaled binary data, features with high base occurrence (e.g., common APIs like `NtAllocateVirtualMemory` present in 70% of binaries) contribute the same variance as rare, highly specific ransomware APIs (e.g., `STRING:cryptdecrypt` present in only 2% of binaries). Standardizing each column to $\mu = 0, \sigma = 1$ ensures that rare, high-signal indicators receive appropriate weighting in the Euclidean distance calculation $\|\mathbf{x} - \mathbf{z}\|^2$.
2. **Numerical Stability:** Gradient descent and dual coordinate descent solvers converge significantly faster when feature dimensions share identical variance.

### 5.2 Preventing Preprocessing Leakage
To ensure strict separation between training and test distributions:
- `StandardScaler` is wrapped inside a `sklearn.pipeline.Pipeline`.
- During 3-fold cross-validation, the scaler is fit **only on the 2 training folds** ($\frac{2}{3}$ of data) and applied to the validation fold ($\frac{1}{3}$).
- For test evaluation, the scaler is fit on the full training set (3,905 samples) and applied to the test set (975 samples) via `.transform()` without recomputing $\mu$ and $\sigma$.

---

## 6. Hyperparameter Tuning Procedure

Hyperparameter optimization was executed strictly on the training partition using **3-Fold Stratified Cross-Validation** with `shuffle=True` and `random_state=42`. The optimization target was the **F1-score of the positive class (Malicious)**.

### 6.1 Search Grids & Tuning Progression

```
Grid Search Summary (3-Fold Stratified CV, Scoring: F1):
├── Linear SVM
│   └── C ∈ [0.1, 1, 10] (3 candidates × 3 folds = 9 fits)
│       └── Best: C = 0.1 | CV F1 = 0.9914 (Wall time: 2.36 s)
├── RBF SVM
│   └── C ∈ [0.1, 1, 10], γ ∈ ['scale', 'auto'] (6 candidates × 3 folds = 18 fits)
│       └── Best: C = 10, γ = 'scale' | CV F1 = 0.9928 (Wall time: 8.00 s)
└── Polynomial SVM
    └── C ∈ [0.1, 1, 10], degree ∈ [2, 3], γ = 'scale' (6 candidates × 3 folds = 18 fits)
        └── Best: C = 10, degree = 2, γ = 'scale' | CV F1 = 0.9879 (Wall time: 11.61 s)
```

### 6.2 Best Hyperparameters JSON Artifact

The exact parameters chosen by `GridSearchCV` (saved in `svm_best_parameters.json`):

```json
{
    "Linear SVM": {
        "best_params": {
            "svm__C": 0.1
        },
        "best_cv_f1_score": 0.9914172576141941
    },
    "RBF SVM": {
        "best_params": {
            "svm__C": 10,
            "svm__gamma": "scale"
        },
        "best_cv_f1_score": 0.9927512265079925
    },
    "Polynomial SVM": {
        "best_params": {
            "svm__C": 10,
            "svm__degree": 2,
            "svm__gamma": "scale"
        },
        "best_cv_f1_score": 0.98793669308397
    }
}
```

---

## 7. Comprehensive Test Evaluation & Comparative Results

Following hyperparameter selection, each optimal pipeline was retrained once on the full 3,905 training samples and evaluated on the 975 test samples.

### 7.1 Detailed Metrics Comparison Table

| Metric | Linear SVM | RBF SVM (Winner) | Polynomial SVM |
| :--- | :---: | :---: | :---: |
| **Kernel Type** | Linear | Gaussian / RBF | Polynomial (Degree 2) |
| **Optimal Hyperparameters** | $C = 0.1$ | $C = 10, \gamma = \text{'scale'}$ | $C = 10, d = 2, \gamma = \text{'scale'}$ |
| **3-Fold CV F1-Score** | 0.9914 | **0.9928** | 0.9879 |
| **Test Accuracy** | 97.74% | **98.15%** | 97.95% |
| **Test Precision (Malicious)** | **0.9763** | 0.9745 | 0.9744 |
| **Test Recall (Malicious)** | 0.9763 | **0.9871** | 0.9828 |
| **Test F1-Score** | 0.9763 | **0.9808** | 0.9786 |
| **Test ROC-AUC** | 0.9935 | **0.9974** | 0.9970 |
| **Specificity (Benign Recall)** | **0.9784** | 0.9765 | 0.9765 |
| **False Positive Rate (FPR)** | **2.16%** | 2.35% | 2.35% |
| **False Negative Rate (FNR)** | 2.37% | **1.29%** | 1.72% |
| **True Negatives (TN / 510)** | **499** | 498 | 498 |
| **False Positives (FP / 510)** | **11** | 12 | 12 |
| **False Negatives (FN / 465)** | 11 | **6** | 8 |
| **True Positives (TP / 465)** | 454 | **459** | 457 |
| **Full Training Time (s)** | **0.158 s** | 0.359 s | 0.459 s |
| **Total Test Inference Time (ms)** | **11.80 ms** | 98.84 ms | 60.98 ms |
| **Inference Time per Sample (µs)**| **12.10 µs** | 101.38 µs | 62.54 µs |

---

## 8. Confusion Matrix & Diagnostic Error Analysis

The confusion matrix breakdown on the 975 test samples highlights critical behavioral distinctions:

```
               Linear SVM                        RBF SVM (Winner)                     Polynomial SVM
       Predicted: 0   Predicted: 1        Predicted: 0   Predicted: 1        Predicted: 0   Predicted: 1
True 0:    499            11          True 0:    498            12          True 0:    498            12
True 1:     11           454          True 1:      6           459          True 1:      8           457
```

### 8.1 Detailed Analysis of Classification Errors

1. **False Negatives (Malware Missed):**
   - **RBF SVM:** Only **6 false negatives** out of 465 malicious samples ($\text{FNR} = 1.29\%$).
   - **Polynomial SVM:** **8 false negatives** ($\text{FNR} = 1.72\%$).
   - **Linear SVM:** **11 false negatives** ($\text{FNR} = 2.37\%$).
   - *Security Implication:* RBF reduces uncaught malware incidents by **45.5% relative to Linear SVM** (from 11 down to 6). In an enterprise environment, stopping those 5 additional ransomware variants prevents potential domain-wide encryption.

2. **False Positives (Benign Software Blocked):**
   - **Linear SVM:** **11 false positives** out of 510 benign applications ($\text{FPR} = 2.16\%$).
   - **RBF SVM:** **12 false positives** ($\text{FPR} = 2.35\%$).
   - **Polynomial SVM:** **12 false positives** ($\text{FPR} = 2.35\%$).
   - *Security Implication:* The difference is negligible (only 1 additional false alarm across 510 benign programs), proving that RBF achieves superior recall without sacrificing specificity.

---

## 9. Receiver Operating Characteristic (ROC) Analysis

The ROC curves plot True Positive Rate against False Positive Rate across all decision thresholds produced by `best_estimator.decision_function(X_test)`.

- **Linear SVM ROC-AUC:** `0.9935`
- **RBF SVM ROC-AUC:** `0.9974`
- **Polynomial SVM ROC-AUC:** `0.9970`

```
ROC-AUC Comparison:
  1.0 ┌───────────────────────────────────────────────┐  RBF SVM:        AUC = 0.9974
      │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  Polynomial SVM: AUC = 0.9970
      │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  Linear SVM:     AUC = 0.9935
  TPR │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  Random Chance:  AUC = 0.5000
      │                                               │
  0.0 └───────────────────────────────────────────────┘
     0.0                     FPR                     1.0
```

All three classifiers achieve near-perfect area under the curve (>0.993), confirming strong class separation. However, **RBF SVM achieves the highest separability (0.9974)**, remaining closest to the top-left ideal corner ($[0, 1]$) across operating thresholds.

---

## 10. Best-Performing Model Selection & Justification

### 🏆 Selected Champion: RBF SVM (`SVC(kernel="rbf", C=10, gamma="scale")`)

#### Justification:
1. **Top F1-Score (0.9808):** Outperformed Linear SVM (0.9763) and Polynomial SVM (0.9786) on the primary optimization metric.
2. **Superior Malware Recall (98.71%):** Successfully caught 459 out of 465 malicious samples. In malware detection, high sensitivity is paramount.
3. **Lowest Breach Risk:** Halved the number of missed ransomware attacks (only 6 misses vs. 11 for Linear).
4. **Highest Discrimination Capacity (ROC-AUC = 0.9974):** Highest margin separation across threshold shifts.
5. **Non-Linear Synergy:** RBF projection naturally captures complex non-linear combinations of API activations (such as memory injection accompanied by network socket setup), which linear hyperplanes cannot model natively.

---

## 11. Operational Trade-Offs & Deployment Scenarios

| Deployment Scenario | Recommended Model | Rationale |
| :--- | :---: | :--- |
| **High-Security Firewall / SOC Sandbox** | **RBF SVM** | Maximum recall (98.71%), lowest false negatives (6 missed), acceptable latency (98.8 ms per 975 samples). |
| **High-Throughput Endpoint Agent / Real-Time Gateway** | **Linear SVM** | Sub-millisecond latency (**12.1 µs per sample**), minimal memory footprint (594 KB vs 1.8 MB), and highly competitive 97.63% F1. |
| **Offline Forensic Triage** | **RBF SVM** | Near-zero false negative rate ensures thorough incident investigation. |

---

## 12. Artifacts & Deliverables Summary

All experiment deliverables have been generated and saved under `Assignment Final/SVM_Results/`:

| Artifact | File Type | Path / Description |
| :--- | :---: | :--- |
| Python Pipeline Script | `.py` | [`Assignment Final/SVM_Results/svm_experiment.py`](file:///home/sanjeet/Projects/SCProject/Assignment%20Final/SVM_Results/svm_experiment.py) |
| Benchmark Comparison Table | `.csv` | [`Assignment Final/SVM_Results/svm_results.csv`](file:///home/sanjeet/Projects/SCProject/Assignment%20Final/SVM_Results/svm_results.csv) |
| Classification Reports | `.txt` | [`Assignment Final/SVM_Results/svm_classification_reports.txt`](file:///home/sanjeet/Projects/SCProject/Assignment%20Final/SVM_Results/svm_classification_reports.txt) |
| Best Hyperparameters JSON | `.json` | [`Assignment Final/SVM_Results/svm_best_parameters.json`](file:///home/sanjeet/Projects/SCProject/Assignment%20Final/SVM_Results/svm_best_parameters.json) |
| Trained Linear Model | `.joblib` | `Assignment Final/SVM_Results/svm_model_linear.joblib` (594 KB) |
| Trained RBF Model | `.joblib` | `Assignment Final/SVM_Results/svm_model_rbf.joblib` (1.8 MB) |
| Trained Polynomial Model | `.joblib` | `Assignment Final/SVM_Results/svm_model_poly.joblib` (2.4 MB) |
| Combined Confusion Matrices | `.png` | [`Assignment Final/SVM_Results/confusion_matrices_all.png`](file:///home/sanjeet/Projects/SCProject/Assignment%20Final/SVM_Results/confusion_matrices_all.png) |
| Model Comparison Bar Chart | `.png` | [`Assignment Final/SVM_Results/model_metrics_comparison.png`](file:///home/sanjeet/Projects/SCProject/Assignment%20Final/SVM_Results/model_metrics_comparison.png) |
| ROC Overlay Curves | `.png` | [`Assignment Final/SVM_Results/roc_curves_comparison.png`](file:///home/sanjeet/Projects/SCProject/Assignment%20Final/SVM_Results/roc_curves_comparison.png) |
| Full Experiment Report | `.md` | [`Assignment Final/SVM_Results/svm_experiment_report.md`](file:///home/sanjeet/Projects/SCProject/Assignment%20Final/SVM_Results/svm_experiment_report.md) |

---

## 13. Methodological Limitations

1. **Pre-computed RFE Features:** The 483 features were pre-selected using an external feature selection routine prior to this experiment. If the RFE procedure had access to the full dataset, subtle indirect feature selection bias may exist.
2. **Temporal Distribution Shift:** Static and dynamic telemetry in malware evolves rapidly as authors change packing algorithms, obfuscation methods, and API hashing techniques. Longitudinal test sets over multiple quarters would further validate temporal decay resistance.
3. **Execution Time Measurement:** Latency was measured on CPU hardware (`x86_64`, Linux kernel). Hardware acceleration (e.g., Intel AVX-512 or cuML GPU acceleration) would yield even lower per-sample inference times.

