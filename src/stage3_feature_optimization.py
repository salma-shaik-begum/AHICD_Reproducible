"""
Stage 3 — Adaptive Multi-Objective Feature Optimization (NSGA-II).
Selects a binary feature mask over the 1697-d concatenated feature space,
jointly optimizing validation macro-F1 (maximize) and feature-count (minimize).
Final selection uses a knee-point heuristic on the resulting Pareto front.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from config import CSV_DIR, CNN_FEAT_DIR, VIT_FEAT_DIR, HC_FEAT_DIR, RESULTS_DIR, MODELS_DIR, SEED

STAGE3_OUT = os.path.join(RESULTS_DIR, 'stage3_optimized')
os.makedirs(STAGE3_OUT, exist_ok=True)


def load_concatenated_features(df_subset):
    feats, labels = [], []
    for _, row in df_subset.iterrows():
        uid = row['unique_id']
        cnn_f = np.load(os.path.join(CNN_FEAT_DIR, f"{uid}.npy"))
        vit_f = np.load(os.path.join(VIT_FEAT_DIR, f"{uid}.npy"))
        hc_f =
