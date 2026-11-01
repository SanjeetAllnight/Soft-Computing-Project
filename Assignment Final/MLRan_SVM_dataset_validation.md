# MLRan RFE Dataset Validation Report for SVM Classification

**Date of Validation:** November 2, 2026  
**Project Folder:** `Assignment Final/`  
**Dataset Domain:** MLRan (Machine Learning Ransomware & Malware Behavioral Telemetry)  
**Status:** Validated — Ready for SVM Pipeline Development  

---

## Executive Summary

A thorough, lightweight validation was conducted on the four prepared MLRan RFE dataset files (`MLRan_labels.csv`, `MLRan_X_train_RFE.csv`, `MLRan_X_test_RFE.csv`, and `RFE_selected_feature_names_dic.json`). 

Key takeaways:
1. **Feature Alignment:** Exactly 483 numerical features are present in both train and test matrices, in identical column sequence, perfectly matching the 483 keys in `RFE_selected_feature_names_dic.json`. All features are strictly binary indicators ($x \in \{0, 1\}$).
2. **Data Integrity:** Zero missing values (`NaN` / `None`), zero infinite values (`inf`), zero duplicate sample IDs, and zero cross-split ID overlaps.
3. **Partitioning:** The dataset represents a strict 80/20 train/test split (3,905 train samples, 975 test samples; total 4,880 samples).
4. **Target Hierarchy:** Three targets are available:
   - `sample_type`: Binary classification (0 = Benign: 52.25%, 1 = Malicious: 47.75%). **Strongly recommended** as the primary SVM target.
   - `type_label`: 5-class multi-class classification (0 = Benign, 1–4 = Malware behavioral categories). Perfectly stratified 80/20. **Recommended** for multi-class SVM.
   - `family_label`: 65-class fine-grained family classification. **Not recommended** due to severe zero-shot representation mismatch (Family 56 appears 14 times in test but 0 times in train; 30 families have 0 test samples).
5. **Critical Leakage Alert for Pipeline Construction:** The files `MLRan_X_train_RFE.csv` and `MLRan_X_test_RFE.csv` contain the target columns (`sample_type`, `family_label`, `type_label`) alongside `sample_id` as the first four columns. These four metadata columns **must be explicitly dropped** from the feature matrix $\mathbf{X}$ prior to model training to prevent trivial 100% data leakage.

---

## 1. Dataset Structure

### 1.1 File Inventory and Dimensionality

| File Name | File Size on Disk | Row Count | Column Count | In-Memory Footprint | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `MLRan_labels.csv` | 60.2 KB | 4,880 | 4 | ~152.6 KB | Master label file for all samples |
| `MLRan_X_train_RFE.csv` | 3.82 MB | 3,905 | 487 | ~14.5 MB | Training set with 483 RFE features + 4 metadata/label cols |
| `MLRan_X_test_RFE.csv` | 957.4 KB | 975 | 487 | ~3.6 MB | Test set with 483 RFE features + 4 metadata/label cols |
| `RFE_selected_feature_names_dic.json` | 29.5 KB | — | 483 keys | ~42.3 KB | JSON dictionary mapping column names to API/telemetry names |

### 1.2 Column Breakdown and Data Types

- **Metadata / Identifier Columns (4 columns):**
  - `sample_id` (`int64`): Unique integer identifier for each sample execution.
  - `sample_type` (`int64`): Binary ground truth (0 or 1).
  - `family_label` (`int64`): Fine-grained malware family label (0 to 65).
  - `type_label` (`int64`): Categorical malware type label (0 to 4).
- **Feature Columns (483 columns):**
  - Columns named with original feature indices (e.g., `'5'`, `'8'`, `'18'`, ..., `'6468211'`).
  - Data type: Strictly `int64` with binary values $\in \{0, 1\}$.
  - Meaning: Indicator of whether a specific API call, registry modification, string, signature, or filesystem event was observed during behavioral execution.

### 1.3 Feature Dictionary Verification

The JSON dictionary (`RFE_selected_feature_names_dic.json`) maps the 483 feature column identifiers to semantic behavioral descriptors:
- **Total Keys in JSON:** 483
- **Order and Set Equality:** $\text{Keys}(\text{JSON}) \equiv \text{Columns}(\mathbf{X}_{\text{train}}) \equiv \text{Columns}(\mathbf{X}_{\text{test}})$ in exact matching sequence.
- **Categorization of Selected Features:**
  - `STRING`: 223 features (e.g., `STRING:cryptdecrypt`, `STRING:desktop.ini`, `STRING:wshom.ocx`)
  - `REG`: 90 features (e.g., `REG:RegSetValueExW`, `REG:RegCreateKeyExA`)
  - `API`: 63 features (e.g., `API:NtProtectVirtualMemory`, `API:CreateProcessInternalW`, `API:NtAllocateVirtualMemory`)
  - `SYSTEM`: 38 features (e.g., `SYSTEM:GetNativeSystemInfo`, `SYSTEM:GetComputerNameW`)
  - `DROP`: 23 features (dropped binary payloads / artifacts)
  - `SIGNATURE`: 20 features (malware behavioral heuristics / rule matches)
  - `FILE`: 16 features (e.g., `FILE:NtQueryInformationFile`, `FILE:NtWriteFile`)
  - `DIRECTORY`: 10 features (directory query / creation events)

### 1.4 Representative Data Samples

#### Training Matrix (`MLRan_X_train_RFE.csv`) — First 3 Rows (Metadata + First 6 Features):
```
   sample_id  sample_type  family_label  type_label  5  8  18  32  33  34
0      10001            1             3           2  1  1   1   1   1   1
1      10002            1            12           2  1  1   1   0   0   1
2      10003            1             3           2  0  0   1   0   0   1
```

#### Test Matrix (`MLRan_X_test_RFE.csv`) — First 3 Rows (Metadata + First 6 Features):
```
   sample_id  sample_type  family_label  type_label  5  8  18  32  33  34
0      11027            1             8           1  1  1   1   1   1   1
1      11030            1            31           2  1  0   1   0   1   1
2      11033            1             8           1  1  1   1   1   1   1
```

#### Master Label File (`MLRan_labels.csv`) — First 3 Rows:
```
   sample_id  sample_type  family_label  type_label
0      10001            1             3           2
1      10002            1            12           2
2      10003            1             3           2
```

---

## 2. Label Analysis

The dataset provides a three-tiered hierarchical labeling scheme:

```
                          Total Samples (4,880)
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
   sample_type = 0                                       sample_type = 1
   Benign / Goodware                                     Malicious / Ransomware
   [2,550 samples, 52.25%]                               [2,330 samples, 47.75%]
         │                                                     │
         ├───────────────────────────┐                         ├───────────────────────────┐
         ▼                           ▼                         ▼                           ▼
   type_label = 0              family_label = 0          type_label ∈ {1, 2, 3, 4}   family_label ∈ {1 .. 65}
   [2,550 samples]             [2,550 samples]           [4 Malware Categories]      [64 Malware Families]
```

### 2.1 Distribution of `sample_type` (Binary Target)

| Class | Semantic Meaning | Overall Count | Overall Pct (%) | Train Count | Train Pct (%) | Test Count | Test Pct (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | Benign / Clean / Goodware | 2,550 | 52.25% | 2,040 | 52.24% | 510 | 52.31% |
| `1` | Malicious / Ransomware | 2,330 | 47.75% | 1,865 | 47.76% | 465 | 47.69% |
| **Total** | — | **4,880** | **100.00%** | **3,905** | **100.00%** | **975** | **100.00%** |

- **Observation:** `sample_type` is exceptionally well-balanced (~52% vs ~48%) and stratified across train and test splits to within 0.07%.

### 2.2 Distribution of `type_label` (Multi-Class Category Target)

| Class | Semantic Category | Overall Count | Overall Pct (%) | Train Count | Train Pct (%) | Test Count | Test Pct (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | Benign / Non-malicious | 2,550 | 52.25% | 2,040 | 52.24% | 510 | 52.31% |
| `1` | Malware Type 1 | 273 | 5.60% | 218 | 5.58% | 55 | 5.64% |
| `2` | Malware Type 2 | 1,140 | 23.36% | 912 | 23.35% | 228 | 23.38% |
| `3` | Malware Type 3 | 468 | 9.59% | 376 | 9.63% | 92 | 9.44% |
| `4` | Malware Type 4 | 449 | 9.20% | 359 | 9.19% | 90 | 9.23% |
| **Total** | — | **4,880** | **100.00%** | **3,905** | **100.00%** | **975** | **100.00%** |

- **Observation:** `type_label` features 5 classes with zero missing categories between train and test. Every class is proportionally represented in the test set.

### 2.3 Distribution of `family_label` (Fine-Grained Family Target)

- **Total Unique Classes:** 65 (Class `0` = Benign: 2,550 samples; Classes `1` through `65` = 64 malware families).
- **Train Set Unique Families:** 64 classes.
- **Test Set Unique Families:** 35 classes.
- **Severe Discrepancy Found:**
  - 30 malware families present in the training set have **0 test samples**.
  - **Family 56** has **14 samples in the test set**, but **0 samples in the training set**!
- **Consequence:** Standard supervised SVM classifiers trained on `family_label` cannot classify Family 56 on the test set, leading to unavoidable misclassifications and evaluation distortion.

### 2.4 Classification Suitability Summary

| Target Column | Classification Type | Number of Classes | Supported by Dataset Split? | Recommendation |
| :--- | :---: | :---: | :---: | : |
| **`sample_type`** | Binary | 2 | **Yes (Perfect Stratification)** | **Primary Recommendation** for SVM assignment |
| **`type_label`** | Multi-class | 5 | **Yes (Perfect Stratification)** | **Secondary Recommendation** if multi-class is required |
| **`family_label`** | Multi-class | 65 | **No (Zero-shot / split mismatch)** | **Do Not Use** for supervised evaluation on this split |

---

## 3. Data Alignment and Integrity

### 3.1 ID Consistency and Partition Verification

| Verification Check | Result | Details |
| :--- | :---: | :--- |
| Train Sample IDs Count | 3,905 | Unique IDs in `MLRan_X_train_RFE.csv` |
| Test Sample IDs Count | 975 | Unique IDs in `MLRan_X_test_RFE.csv` |
| Master Labels IDs Count | 4,880 | Unique IDs in `MLRan_labels.csv` |
| Train & Test ID Overlap | **0** | Empty intersection ($\text{IDs}_{\text{train}} \cap \text{IDs}_{\text{test}} = \emptyset$) |
| Complete Partitioning | **100% Match** | $\text{IDs}_{\text{train}} \cup \text{IDs}_{\text{test}} \equiv \text{IDs}_{\text{labels}}$ |
| Duplicate IDs in Train | **0** | No duplicated `sample_id` values |
| Duplicate IDs in Test | **0** | No duplicated `sample_id` values |
| Duplicate IDs in Labels | **0** | No duplicated `sample_id` values |

### 3.2 Cross-File Label Verification

A join between `MLRan_X_train_RFE.csv`, `MLRan_X_test_RFE.csv`, and `MLRan_labels.csv` on `sample_id` revealed:
- **Train Label Mismatches:** 0 across `sample_type`, `family_label`, and `type_label`.
- **Test Label Mismatches:** 0 across `sample_type`, `family_label`, and `type_label`.
- **Row Ordering:** While row IDs in train and test do not strictly match the sequential ordering of `MLRan_labels.csv`, every sample ID possesses an exact 1-to-1 match. Because labels are already embedded inside `MLRan_X_train_RFE.csv` and `MLRan_X_test_RFE.csv`, row misalignment between the files cannot cause label shifting if loaded directly from the respective files.

---

## 4. Data Quality

### 4.1 Missing and Non-Finite Values

| Check | Train Set | Test Set | Master Labels | Status |
| :--- | :---: | :---: | :---: | :---: |
| Missing / Null Values (`NaN`, `None`) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **Passed** |
| Infinite Values (`+inf`, `-inf`) | 0 (0.0%) | 0 (0.0%) | N/A | **Passed** |
| Non-numerical Values in Features | 0 (0.0%) | 0 (0.0%) | N/A | **Passed** |

### 4.2 Feature Variability and Constant Features

- **Training Set Constant Features:** **0 features**. Every one of the 483 selected features exhibits variance ($\sigma > 0$) on the training set.
- **Test Set Constant Features:** **2 features** have all zeros in test:
  1. `4541826` (`STRING:cryptdecrypt`): 82 ones in train (2.1%), 0 in test.
  2. `4604965` (`STRING:desktop.ini`): 61 ones in train (1.6%), 0 in test.
- **Interpretation:** This is expected for sparse binary indicators of rare malware events when evaluated on a smaller sample (975 test rows). These features should **not** be dropped, as doing so would misalign the train and test dimensional spaces.

### 4.3 Feature Values and Sparsity

- **Value Range:** Minimum = $0$, Maximum = $1$ across all 483 features in both train and test.
- **Feature Nature:** Strictly binary indicator variables ($x_{ij} \in \{0, 1\}$).
- **Feature Sparsity (% of zero entries):**
  - Train matrix: **82.97%** zeros (17.03% active indicators).
  - Test matrix: **85.07%** zeros (14.93% active indicators).
- **Active Features Per Sample:**
  - Train: Minimum = 0, Median = 73, Mean = 82.3, Maximum = 220 active features.
  - Test: Minimum = 0, Median = 53, Mean = 72.1, Maximum = 223 active features.
  - Samples with zero active features: 120 in train (3.07%), 18 in test (1.85%).

### 4.4 Duplicate Feature Vectors and Label Consistency

In behavioral binary datasets, two different executables can invoke identical subsets of captured telemetry:
- **Unique Feature Vectors:** 3,004 in train (901 repeated instances), 689 in test (286 repeated instances).
- **Contradiction Analysis:**
  - `sample_type`: **0 contradictions**. When two samples share identical feature vectors, their binary label is 100% consistent (both benign or both malicious).
  - `type_label`: Only 10 instances in train and 11 in test where distinct malware sub-types triggered identical telemetry subsets.

---

## 5. SVM Readiness and Engineering Recommendations

### 5.1 Correct Dataset Construction ($\mathbf{X}$ and $\mathbf{y}$)

To build standard Scikit-Learn data arrays safely:

```python
import pandas as pd

# 1. Load data
train_df = pd.read_csv("Assignment Final/MLRan_X_train_RFE.csv")
test_df = pd.read_csv("Assignment Final/MLRan_X_test_RFE.csv")

# 2. Define metadata and feature columns
metadata_cols = ["sample_id", "sample_type", "family_label", "type_label"]
feature_cols = [col for col in train_df.columns if col not in metadata_cols]

# 3. Construct Feature Matrices (Shape: (3905, 483) and (975, 483))
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

# 4. Construct Target Vectors (Select primary target: sample_type)
y_train = train_df["sample_type"].values
y_test = test_df["sample_type"].values
```

### 5.2 Critical Data Leakage Warnings

> [!CAUTION]
> **Avoid Feature Matrix Pollution:**
> If `train_df.drop(columns=['sample_id'])` is executed without also dropping `sample_type`, `family_label`, and `type_label`, the target labels will remain inside $\mathbf{X}_{\text{train}}$ and $\mathbf{X}_{\text{test}}$. The SVM classifier will achieve trivial, meaningless 100% test accuracy. Ensure only the 483 RFE features are passed to the model.

> [!IMPORTANT]
> **Preprocessing / Scaler Leakage:**
> Any scaling object (e.g., `StandardScaler`, `MaxAbsScaler`) must be fitted **strictly on `X_train`** using `.fit_transform()` and applied to `X_test` using `.transform()`.

### 5.3 Target Recommendation

- **Recommended Primary Task:** **Binary Classification on `sample_type`** (Benign vs. Malicious).
  - Clean 52.2% / 47.8% balance.
  - Zero label contradictions.
  - Aligns with standard cybersecurity malware triage.
- **Recommended Secondary Task (if multi-class required):** **Multi-Class Classification on `type_label`** (5 classes).
  - 1 Benign class + 4 distinct malware categories.
  - Fully stratified 80/20 train/test representation across all 5 classes.
- **Discouraged Task:** `family_label` (65 classes) should be avoided due to the 30 empty test classes and the zero-shot Family 56 artifact in the test partition.

### 5.4 Suitable SVM Kernels

1. **Linear SVM (`SVC(kernel='linear')` or `LinearSVC`):**
   - **Why:** In high-dimensional binary spaces ($d = 483$), text and behavioral telemetry features are frequently linearly separable. Linear SVM trains substantially faster ($\mathcal{O}(n \cdot d)$ vs $\mathcal{O}(n^2 \cdot d)$) and provides interpretable primal feature weights (`coef_`).
2. **Radial Basis Function SVM (`SVC(kernel='rbf')`):**
   - **Why:** RBF is the gold standard for non-linear SVM classification. It captures complex non-linear combinations of API activations (e.g., presence of `NtAllocateVirtualMemory` AND `NtProtectVirtualMemory`).
3. **Polynomial Kernel (`SVC(kernel='poly', degree=2)`):**
   - **Why:** Second-degree polynomial kernels directly model pairwise logical conjunctions ($x_i \land x_j$) of malware behaviors, which aligns naturally with malware evasion rules.

### 5.5 Scaling Requirements for Binary Telemetry

Since all 483 features are binary $\{0, 1\}$:
1. **Unscaled ($\mathbf{X}$ as $\{0, 1\}$):**
   - Already bounded in $[0, 1]$.
   - Euclidean distances between sample vectors $\mathbf{u}$ and $\mathbf{v}$ are directly proportional to Hamming distances ($||\mathbf{u} - \mathbf{v}||_2^2 = \sum |u_i - v_i|$).
   - This works well for sparse binary data and preserves exact sparsity.
2. **StandardScaler ($\mu = 0, \sigma = 1$):**
   - Features with low occurrence frequency (e.g., rare API calls) will receive higher variance scaling, preventing high-frequency common APIs from dominating the decision boundary margin.
   - Note: Dense standardization (`with_mean=True`) converts zeros to non-zero values, increasing memory footprint, but at $3905 \times 483$ (~15 MB) this is completely negligible.
3. **MaxAbsScaler:**
   - Preserves sparsity and maps data to $[0, 1]$. Since features are already in $\{0, 1\}$, `MaxAbsScaler` leaves binary features unaltered.

---

## 6. Summary Checklist

- [x] All 4 files inspected and verified.
- [x] Row and column counts confirmed (Train: 3,905 × 487; Test: 975 × 487; Labels: 4,880 × 4; JSON: 483 keys).
- [x] 483 selected features confirmed against JSON dictionary.
- [x] Labels analyzed: `sample_type` (binary), `type_label` (5-class), and `family_label` (65-class).
- [x] Train and test sample IDs verified as mutually exclusive and completely covering master labels.
- [x] Data quality verified: zero nulls, zero infs, zero constant features in train, all numerical features.
- [x] Safe $\mathbf{X}, \mathbf{y}$ extraction defined to prevent target leakage.
- [x] Recommended classification targets and SVM kernels documented.

