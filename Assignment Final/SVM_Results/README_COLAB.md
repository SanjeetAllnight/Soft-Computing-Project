# Google Colab Execution Guide: MLRan SVM Classification

**Notebook File:** [`Assignment Final/SVM_Results/MLRan_SVM_Classification.ipynb`](file:///home/sanjeet/Projects/SCProject/Assignment%20Final/SVM_Results/MLRan_SVM_Classification.ipynb)  
**Experiment:** Benchmarking Linear, RBF, and Polynomial Support Vector Machines on 483 Behavioral Telemetry Features  
**Target:** Binary Classification on `sample_type` (0 = Benign, 1 = Malicious)  

---

## 1. Overview

This notebook is an end-to-end, self-contained implementation designed for academic demonstration to the professor. It is pre-rendered with genuine experimental outputs and figures, and is 100% reproducible when executed from scratch on **Google Colab (Free CPU Tier)**, **Kaggle**, or a **local Jupyter/VSCode environment**.

**Execution Time:** ~20 to 30 seconds total across all 3 models and 3-fold cross-validation.

---

## 2. Required Input Files

To execute the notebook, only four lightweight files (~4.7 MB total) are required from the `Assignment Final/` directory:

| Filename | Size | Purpose |
| :--- | :---: | :--- |
| `MLRan_X_train_RFE.csv` | ~3.7 MB | Training feature matrix (3,905 samples × 483 features + metadata) |
| `MLRan_X_test_RFE.csv` | ~936 KB | Test feature matrix (975 samples × 483 features + metadata) |
| `MLRan_labels.csv` | ~60 KB | Master label registry for integrity validation |
| `RFE_selected_feature_names_dic.json` | ~29 KB | Dictionary mapping 483 feature names to their telemetry domains |

---

## 3. How to Run in Google Colab (Step-by-Step)

### Option A: Direct Session Upload (Fastest — Under 1 Minute)

1. **Open Google Colab:**
   - Navigate to [https://colab.research.google.com/](https://colab.research.google.com/).
2. **Upload Notebook:**
   - Click **File** $\rightarrow$ **Upload notebook**.
   - Select `MLRan_SVM_Classification.ipynb` from your computer.
3. **Upload the 4 Dataset Files:**
   - On the left sidebar of Google Colab, click the **Folder icon** (`Files`).
   - Click the **Upload to session storage** button (or drag and drop):
     - `MLRan_X_train_RFE.csv`
     - `MLRan_X_test_RFE.csv`
     - `MLRan_labels.csv`
     - `RFE_selected_feature_names_dic.json`
   - The files will appear in the root `/content/` directory.
4. **Execute All Cells:**
   - Click **Runtime** $\rightarrow$ **Run all** (or press `Ctrl + F9` / `Cmd + F9`).
   - The notebook dynamically resolves `/content/`, verifies dataset integrity, executes grid search, trains all models, and renders all plots inline.

---

### Option B: Google Drive Mounting (Persistent Storage)

If you prefer to store the dataset in Google Drive so you don't need to re-upload files when the Colab session resets:

1. **Upload Dataset to Drive:**
   - In Google Drive, create a folder named `Assignment Final` (e.g., inside `My Drive`).
   - Upload the 4 required files into this folder:
     `My Drive/Assignment Final/`
2. **Open the Notebook in Colab.**
3. **Mount Google Drive:**
   - At the top of the notebook or in an empty cell, run:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
4. **Run All Cells:**
   - Click **Runtime** $\rightarrow$ **Run all**.
   - The path resolver will automatically detect `/content/drive/MyDrive/Assignment Final` and load the files seamlessly.

---

## 4. Expected Demonstration Results

When all cells finish executing, the notebook will display the following verified test benchmark metrics:

| Model Architecture | Optimal Hyperparameters | 3-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Inference Time (Total) | Inference Time (per Sample) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | $C = 0.1$ | 0.9914 | 97.74% | 0.9763 | 0.9763 | 0.9763 | 0.9935 | ~11.5 ms | **11.7 µs** |
| **RBF SVM (Champion)** | $C = 10, \gamma = \text{'scale'}$ | **0.9928** | **98.15%** | 0.9745 | **0.9871** | **0.9808** | **0.9974** | ~92.4 ms | 94.8 µs |
| **Polynomial SVM** | $C = 10, d = 2, \gamma = \text{'scale'}$ | 0.9879 | 97.95% | 0.9744 | 0.9828 | 0.9786 | 0.9970 | ~56.9 ms | 58.3 µs |

### Key Discussion Points for the Professor:
1. **Winning Model**: **RBF SVM** is the overall champion, attaining **98.15% accuracy**, **98.08% F1-score**, and **0.9974 ROC-AUC**.
2. **Critical Asymmetric Risk**: RBF SVM missed **only 6 malicious binaries out of 465** (False Negative Rate = 1.29%), compared to 11 missed by Linear SVM — representing a **45.5% reduction in uncaught malware attacks**.
3. **Efficiency Trade-off**: Linear SVM is **~8x faster per sample (11.7 µs vs. 94.8 µs)** with near-identical precision, making it an excellent choice for real-time edge firewalls where throughput is the limiting factor.
4. **Zero Data Leakage**: Standardized within `Pipeline`, metadata columns (`sample_id`, `sample_type`, `family_label`, `type_label`) explicitly stripped, and test set evaluated strictly once without hyperparameter exposure.

---

## 5. Visualizations Generated Inline

The notebook renders four publication-quality visualizations directly in the output cells:
1. **Feature Domain Distribution**: Horizontal bar chart breaking down the 483 features across 8 telemetry categories (`STRING`, `REG`, `API`, `SYSTEM`, `DROP`, `SIGNATURE`, `FILE`, `DIRECTORY`).
2. **Combined Confusion Matrices**: 3-panel side-by-side heatmaps displaying TN, FP, FN, and TP for Linear, RBF, and Polynomial SVMs.
3. **Model Performance Comparison**: Grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC.
4. **ROC Curves Overlay**: Multi-curve comparison showing sensitivity vs. specificity across thresholds, with exact AUC metrics in the legend.

