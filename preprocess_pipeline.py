#!/usr/bin/env python3
"""
Preprocessing and TF-IDF Feature Extraction Pipeline
for Jigsaw Toxic Comment Classification.

Multi-Objective Evolutionary Feature Selection Project.
"""

import os
import sys
import time
import html
import re
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """
    Simple, reproducible text cleaning:
    - Decodes HTML entities and strips HTML tags
    - Strips URLs
    - Lowercases text
    - Strips IP addresses
    - Removes punctuation while preserving alphanumeric words
    - Normalizes redundant whitespace
    """
    if not isinstance(text, str):
        return ""
    # 1. HTML entities & tags
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    # 2. URLs
    text = re.sub(r"https?://\S+|www\.\S+|ftp://\S+", " ", text)
    # 3. Lowercase
    text = text.lower()
    # 4. IP addresses
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " ", text)
    # 5. Obvious unnecessary punctuation/symbols
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_pipeline():
    start_total = time.time()
    raw_csv = "train.csv"
    
    print("=" * 80)
    print("      JIGSAW TOXIC COMMENT PREPROCESSING & TF-IDF PIPELINE")
    print("=" * 80)
    
    if not os.path.exists(raw_csv):
        print(f"Error: {raw_csv} not found!", file=sys.stderr)
        sys.exit(1)
        
    # 1. Load train.csv efficiently
    print(f"\n[Step 1] Loading {raw_csv}...")
    t0 = time.time()
    # Load only necessary columns
    cols_to_load = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df = pd.read_csv(raw_csv, usecols=cols_to_load)
    print(f"  Loaded {len(df):,} rows and {len(df.columns)} columns in {time.time() - t0:.2f}s.")
    
    # 2. Create binary target column `is_toxic`
    print("\n[Step 2] Creating binary target column 'is_toxic'...")
    toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['is_toxic'] = (df[toxicity_labels].sum(axis=1) > 0).astype(np.int8)
    
    total_samples = len(df)
    total_toxic = int(df['is_toxic'].sum())
    total_clean = total_samples - total_toxic
    print(f"  Target summary across entire dataset:")
    print(f"  • Total samples: {total_samples:,}")
    print(f"  • Non-Toxic (0): {total_clean:,} ({total_clean/total_samples*100:.2f}%)")
    print(f"  • Toxic     (1): {total_toxic:,} ({total_toxic/total_samples*100:.2f}%)")
    print(f"  • Imbalance ratio: {total_clean/total_toxic:.2f} : 1")
    
    # 3. Clean comment_text
    print("\n[Step 3] Applying text cleaning...")
    t0 = time.time()
    df['cleaned_text'] = df['comment_text'].apply(clean_text)
    print(f"  Cleaned {len(df):,} comments in {time.time() - t0:.2f}s.")
    
    # Free memory of raw text columns no longer needed
    del df['comment_text']
    for col in toxicity_labels:
        del df[col]
        
    # 4. Stratified Train / Validation Split (80% / 20%, random_state=42)
    print("\n[Step 4] Performing Stratified 80/20 Train/Validation Split...")
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        df['cleaned_text'],
        df['is_toxic'].values,
        test_size=0.20,
        random_state=42,
        stratify=df['is_toxic'].values
    )
    
    train_rows = len(X_train_text)
    val_rows = len(X_val_text)
    print(f"  • Training rows:   {train_rows:,} (80.0%)")
    print(f"  • Validation rows: {val_rows:,} (20.0%)")
    
    # Free dataframe to release memory before vectorization
    del df
    
    # 5 & 6. TF-IDF Vectorization
    print("\n[Step 5 & 6] Fitting TF-IDF Vectorizer on training text and transforming validation...")
    print("  Configuration:")
    print("  • max_features = 10,000")
    print("  • ngram_range  = (1, 2)")
    print("  • min_df       = 5")
    print("  • sublinear_tf = True")
    
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        sublinear_tf=True
    )
    
    X_train = vectorizer.fit_transform(X_train_text)
    print(f"  Fitted and transformed X_train in {time.time() - t0:.2f}s.")
    
    t1 = time.time()
    X_val = vectorizer.transform(X_val_text)
    print(f"  Transformed X_val in {time.time() - t1:.2f}s.")
    
    # Free text lists
    del X_train_text, X_val_text
    
    # 7. Metrics & structural analysis
    num_features = X_train.shape[1]
    
    # Sparsity calculation
    train_total_elements = X_train.shape[0] * X_train.shape[1]
    train_nnz = X_train.nnz
    train_sparsity = (1.0 - (train_nnz / train_total_elements)) * 100.0
    
    val_total_elements = X_val.shape[0] * X_val.shape[1]
    val_nnz = X_val.nnz
    val_sparsity = (1.0 - (val_nnz / val_total_elements)) * 100.0
    
    # Memory footprint in RAM
    train_mem_mb = (X_train.data.nbytes + X_train.indices.nbytes + X_train.indptr.nbytes) / (1024 ** 2)
    val_mem_mb = (X_val.data.nbytes + X_val.indices.nbytes + X_val.indptr.nbytes) / (1024 ** 2)
    
    # Class distribution in splits
    y_train_pos = int((y_train == 1).sum())
    y_train_neg = int((y_train == 0).sum())
    y_val_pos = int((y_val == 1).sum())
    y_val_neg = int((y_val == 0).sum())
    
    print("\n" + "=" * 80)
    print("                 VECTORIZATION RESULTS & STRUCTURAL REPORT")
    print("=" * 80)
    print(f"• Training rows:                  {train_rows:,}")
    print(f"• Validation rows:                {val_rows:,}")
    print(f"• Number of features produced:     {num_features:,}")
    print(f"• X_train Matrix Shape:           {X_train.shape}")
    print(f"• X_val Matrix Shape:             {X_val.shape}")
    print(f"• Sparse Matrix Representation:   {type(X_train).__name__} (scipy.sparse.csr_matrix)")
    print(f"• Memory Usage in RAM:")
    print(f"  - X_train:                      {train_mem_mb:.2f} MB ({train_nnz:,} non-zero entries)")
    print(f"  - X_val:                        {val_mem_mb:.2f} MB ({val_nnz:,} non-zero entries)")
    print(f"  - Combined in-memory footprint: {train_mem_mb + val_mem_mb:.2f} MB (Extremely memory-efficient)")
    print(f"• Sparsity Percentage:")
    print(f"  - Training Sparsity:            {train_sparsity:.4f}% ({train_nnz:,} / {train_total_elements:,} non-zero)")
    print(f"  - Validation Sparsity:          {val_sparsity:.4f}% ({val_nnz:,} / {val_total_elements:,} non-zero)")
    print(f"• Class Distribution:")
    print(f"  - Train Set: Non-Toxic = {y_train_neg:,} ({y_train_neg/train_rows*100:.2f}%), Toxic = {y_train_pos:,} ({y_train_pos/train_rows*100:.2f}%)")
    print(f"  - Val Set:   Non-Toxic = {y_val_neg:,} ({y_val_neg/val_rows*100:.2f}%), Toxic = {y_val_pos:,} ({y_val_pos/val_rows*100:.2f}%)")
    
    # 8. Display small sample of feature names & resulting matrix
    feature_names = vectorizer.get_feature_names_out()
    print("\n" + "-" * 80)
    print("SAMPLE FEATURE NAMES & TF-IDF VALUES")
    print("-" * 80)
    print(f"• First 20 vocabulary features (unigrams & bigrams):")
    print(f"  {list(feature_names[:20])}")
    print(f"\n• Middle 20 sample features:")
    print(f"  {list(feature_names[5000:5020])}")
    print(f"\n• Last 20 vocabulary features:")
    print(f"  {list(feature_names[-20:])}")
    
    print("\n• Sample non-zero TF-IDF entries from row 0 of X_train:")
    row_0 = X_train[0]
    row_0_indices = row_0.indices
    row_0_data = row_0.data
    for idx, val in zip(row_0_indices[:10], row_0_data[:10]):
        print(f"  Feature [{idx:>5}] '{feature_names[idx]}': {val:.4f}")
    if len(row_0_indices) > 10:
        print(f"  ... ({len(row_0_indices) - 10} more non-zero terms in this comment)")

    # 9. Save fitted TF-IDF vectorizer
    print("\n" + "-" * 80)
    print("SAVING PIPELINE ARTIFACTS")
    print("-" * 80)
    vec_path = "tfidf_vectorizer.joblib"
    t0 = time.time()
    joblib.dump(vectorizer, vec_path, compress=3)
    print(f"  • Saved fitted TF-IDF vectorizer to: '{vec_path}' ({os.path.getsize(vec_path)/(1024**2):.2f} MB in {time.time()-t0:.2f}s)")
    
    # 10. Save processed matrices & target arrays
    x_train_path = "X_train.npz"
    x_val_path = "X_val.npz"
    y_train_path = "y_train.npy"
    y_val_path = "y_val.npy"
    feat_names_path = "feature_names.txt"
    
    sparse.save_npz(x_train_path, X_train)
    print(f"  • Saved X_train to: '{x_train_path}' ({os.path.getsize(x_train_path)/(1024**2):.2f} MB)")
    
    sparse.save_npz(x_val_path, X_val)
    print(f"  • Saved X_val to:   '{x_val_path}' ({os.path.getsize(x_val_path)/(1024**2):.2f} MB)")
    
    np.save(y_train_path, y_train)
    print(f"  • Saved y_train to: '{y_train_path}' ({os.path.getsize(y_train_path)/(1024**2):.2f} MB)")
    
    np.save(y_val_path, y_val)
    print(f"  • Saved y_val to:   '{y_val_path}' ({os.path.getsize(y_val_path)/(1024**2):.2f} MB)")
    
    with open(feat_names_path, "w", encoding="utf-8") as f:
        for feat in feature_names:
            f.write(f"{feat}\n")
    print(f"  • Saved feature names vocabulary to: '{feat_names_path}' ({os.path.getsize(feat_names_path)/(1024**2):.2f} MB)")
    
    print("\n" + "=" * 80)
    print("                    PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total pipeline execution time: {time.time() - start_total:.2f} seconds.")
    print(f"Final Output Dimensions:")
    print(f"  • X_train:           {X_train.shape[0]:,} rows x {X_train.shape[1]:,} features")
    print(f"  • X_val:             {X_val.shape[0]:,} rows x {X_val.shape[1]:,} features")
    print(f"  • y_train:           {y_train.shape[0]:,} labels")
    print(f"  • y_val:             {y_val.shape[0]:,} labels")
    print(f"  • Total Features:    {num_features:,} TF-IDF n-grams")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_pipeline()
