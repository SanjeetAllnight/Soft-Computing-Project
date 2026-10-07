#!/usr/bin/env python3
"""
Thorough Inspection Script for Jigsaw Toxic Comment Classification Dataset.

Inspects:
- train.csv
- test.csv
"""

import os
import sys
import pandas as pd
import numpy as np


def inspect_jigsaw():
    train_file = "train.csv"
    test_file = "test.csv"
    
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    print("=" * 80)
    print("      JIGSAW TOXIC COMMENT CLASSIFICATION - DATASET INSPECTION")
    print("=" * 80)
    
    # 1. File names and sizes
    files_info = []
    for f in [train_file, test_file]:
        size_bytes = os.path.getsize(f)
        size_mb = size_bytes / (1024 ** 2)
        files_info.append((f, size_bytes, size_mb))
        
    print("\n1 & 2. FILE NAMES AND SIZES:")
    for f, b, mb in files_info:
        print(f"   • {f:<12}: {b:>12,} bytes ({mb:>6.2f} MB)")
        
    # Load files
    print("\nLoading datasets into memory for analysis...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # 3 & 4. Rows and Columns
    print(f"\n3 & 4. DIMENSIONS:")
    print(f"   • {train_file:<12}: {df_train.shape[0]:>8,} rows, {df_train.shape[1]:>2} columns")
    print(f"   • {test_file:<12}: {df_test.shape[0]:>8,} rows, {df_test.shape[1]:>2} columns")
    
    # 5. Column names
    print(f"\n5. COLUMN NAMES:")
    print(f"   • {train_file}: {list(df_train.columns)}")
    print(f"   • {test_file}: {list(df_test.columns)}")
    
    # 6. Data types
    print(f"\n6. DATA TYPES:")
    print(f"   • {train_file}:")
    for col, dt in df_train.dtypes.items():
        print(f"     - {col:<15}: {str(dt)}")
    print(f"   • {test_file}:")
    for col, dt in df_test.dtypes.items():
        print(f"     - {col:<15}: {str(dt)}")
        
    # 7. Text column
    print(f"\n7. COMMENT TEXT COLUMN:")
    print(f"   • Column name: 'comment_text' (present in both train.csv and test.csv)")
    print(f"   • Contains free-form text comments extracted from Wikipedia talk page edits.")
    
    # 8. Toxicity label columns
    print(f"\n8. TOXICITY LABEL COLUMNS:")
    print(f"   • Exactly 6 binary label columns in train.csv: {label_cols}")
    print(f"   • (None of these labels are present in test.csv)")
    
    # 9. Target label distributions
    print(f"\n9. POSITIVE & NEGATIVE SAMPLES PER LABEL (train.csv):")
    total_train = len(df_train)
    print(f"   {'Label':<16} {'Positive (1)':<15} {'Pos %':<10} {'Negative (0)':<15} {'Neg %':<10}")
    print(f"   {'-'*65}")
    for col in label_cols:
        pos = int((df_train[col] == 1).sum())
        neg = int((df_train[col] == 0).sum())
        pos_pct = (pos / total_train) * 100.0
        neg_pct = (neg / total_train) * 100.0
        print(f"   {col:<16} {pos:>10,} ({pos_pct:5.2f}%) {neg:>12,} ({neg_pct:5.2f}%)")
        
    # 10. Missing values
    print(f"\n10. MISSING VALUES:")
    print(f"   • {train_file}:")
    for col in df_train.columns:
        n_miss = int(df_train[col].isna().sum())
        pct_miss = (n_miss / total_train) * 100.0
        print(f"     - {col:<15}: {n_miss:>6,} missing ({pct_miss:5.2f}%)")
    print(f"   • {test_file}:")
    total_test = len(df_test)
    for col in df_test.columns:
        n_miss = int(df_test[col].isna().sum())
        pct_miss = (n_miss / total_test) * 100.0
        print(f"     - {col:<15}: {n_miss:>6,} missing ({pct_miss:5.2f}%)")
        
    # 11. Duplicates
    print(f"\n11. DUPLICATE ANALYSIS:")
    dup_ids_train = df_train['id'].duplicated().sum()
    dup_text_train = df_train['comment_text'].duplicated().sum()
    dup_ids_test = df_test['id'].duplicated().sum()
    dup_text_test = df_test['comment_text'].duplicated().sum()
    
    print(f"   • train.csv Duplicate IDs:   {dup_ids_train}")
    print(f"   • train.csv Duplicate Texts: {dup_text_train:,} ({dup_text_train/total_train*100:.2f}%)")
    print(f"   • test.csv  Duplicate IDs:   {dup_ids_test}")
    print(f"   • test.csv  Duplicate Texts: {dup_text_test:,} ({dup_text_test/total_test*100:.2f}%)")
    
    # Check if duplicate texts in train have conflicting labels
    if dup_text_train > 0:
        dups_df = df_train[df_train['comment_text'].duplicated(keep=False)]
        nunique_labels_per_dup = dups_df.groupby('comment_text')[label_cols].nunique()
        has_conflicts = (nunique_labels_per_dup > 1).any().any()
        print(f"   • Conflicting labels among duplicate comments in train: {has_conflicts}")
        
    # Check text overlap between train and test
    train_texts = set(df_train['comment_text'])
    test_texts = set(df_test['comment_text'])
    overlap = len(train_texts.intersection(test_texts))
    print(f"   • Exact text overlap between train and test: {overlap:,} comments")

    # 12. Representative samples
    print(f"\n12. REPRESENTATIVE SAMPLE ROWS:")
    # Non-toxic sample
    clean_sample = df_train[df_train[label_cols].sum(axis=1) == 0].iloc[0]
    print(f"   [Sample A: Non-toxic / Clean]")
    print(f"   ID: {clean_sample['id']}")
    print(f"   Text: {clean_sample['comment_text'][:120]}...")
    print(f"   Labels: {clean_sample[label_cols].to_dict()}")
    
    # Mild/toxic sample
    mild_sample = df_train[(df_train['toxic'] == 1) & (df_train['severe_toxic'] == 0) & (df_train['threat'] == 0)].iloc[0]
    print(f"\n   [Sample B: Mild/Moderate Toxic]")
    print(f"   ID: {mild_sample['id']}")
    print(f"   Text: {mild_sample['comment_text'][:120]}...")
    print(f"   Labels: {mild_sample[label_cols].to_dict()}")
    
    # Multi-label toxic sample
    multi_sample = df_train[df_train[label_cols].sum(axis=1) >= 4].iloc[0]
    print(f"\n   [Sample C: High Severity Multi-Label Toxic]")
    print(f"   ID: {multi_sample['id']}")
    print(f"   Text: {multi_sample['comment_text'][:120]}...")
    print(f"   Labels: {multi_sample[label_cols].to_dict()}")

    # 13. Multi-label vs Binary Target Simplification Analysis
    print("\n" + "=" * 80)
    print("   MULTI-LABEL VS. BINARY TARGET SIMPLIFICATION ANALYSIS")
    print("=" * 80)
    
    label_sum = df_train[label_cols].sum(axis=1)
    df_train['is_toxic_binary'] = (label_sum > 0).astype(int)
    
    bin_counts = df_train['is_toxic_binary'].value_counts()
    n_non_toxic = int(bin_counts.get(0, 0))
    n_toxic = int(bin_counts.get(1, 0))
    pct_non_toxic = (n_non_toxic / total_train) * 100.0
    pct_toxic = (n_toxic / total_train) * 100.0
    imbalance_ratio = n_non_toxic / n_toxic
    
    print(f"\n1. Resulting Binary Target Distribution (is_toxic):")
    print(f"   • Non-Toxic (0, all labels = 0): {n_non_toxic:>8,} ({pct_non_toxic:5.2f}%)")
    print(f"   • Toxic     (1, >= 1 label = 1):  {n_toxic:>8,} ({pct_toxic:5.2f}%)")
    print(f"   • Class Imbalance Ratio:           {imbalance_ratio:.2f} : 1 (~8.8 : 1)")
    print(f"   • Total:                           {total_train:>8,} (100.00%)")
    
    print(f"\n2. Distribution of Toxicity Label Counts per Comment:")
    label_counts_dist = label_sum.value_counts().sort_index()
    for n_labels, count in label_counts_dist.items():
        pct = (count / total_train) * 100.0
        print(f"   • {n_labels} active labels: {count:>8,} ({pct:5.2f}%)")
        
    print(f"\n3. Label Overlap & Co-occurrence Matrix:")
    co_matrix = df_train[label_cols].T.dot(df_train[label_cols])
    print(co_matrix)
    
    print(f"\n4. Recommendation on Target Representation:")
    print("   • Recommendation: [SIMPLIFY TO BINARY TARGET]")
    print("   • Why:")
    print("     a) Primary Goal of Project: Multi-objective feature selection using evolutionary algorithms.")
    print("        In multi-objective feature selection, the objectives are:")
    print("        - Objective 1: Maximize predictive performance (e.g. F1-score / ROC-AUC / PR-AUC).")
    print("        - Objective 2: Minimize feature subset size (number of TF-IDF vocabulary tokens selected).")
    print("     b) Evolutionary Complexity: If keeping multi-label (6 separate binary classifiers or classifier chains),")
    print("        evaluating each individual solution during the evolutionary process requires evaluating 6 models,")
    print("        multiplying runtime by 6x. A single binary objective makes fitness evaluation fast, mathematically clean,")
    print("        and directly aligned with standard soft computing feature selection literature.")
    print("     c) Healthy Imbalance: At 89.83% clean vs. 10.17% toxic (~8.8 : 1), the binary target has a realistic,")
    print("        tractable imbalance (far better than Bosch's 171:1), where standard class weighting or PR-AUC works smoothly.")
    
    print("\n" + "=" * 80)
    print("                       RECOMMENDED NEXT PREPROCESSING STEPS")
    print("=" * 80)
    print("1. Text Cleaning & Normalization:")
    print("   - Lowercasing, removing URLs, IP addresses, HTML tags, punctuation, and extra whitespace.")
    print("   - Handling repeated characters / leetspeak (common in toxic comments).")
    print("2. Deduplication:")
    print("   - Remove duplicate identical comments in train.csv to prevent data leakage/overweighting.")
    print("3. Target Encoding:")
    print("   - Create the binary target column: `is_toxic = (df[label_cols].sum(axis=1) > 0).astype(int)`.")
    print("4. Train/Validation Split:")
    print("   - Stratified K-Fold or Stratified 80/20 split based on `is_toxic` before vectorization.")
    print("5. Feature Extraction (TF-IDF):")
    print("   - Generate unigram + bigram TF-IDF matrix (e.g. max_features=1,500 to 3,000).")
    print("   - This provides the authentic high-dimensional discrete feature pool from which the multi-objective")
    print("     evolutionary algorithm will select the optimal ~500 features.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    inspect_jigsaw()
