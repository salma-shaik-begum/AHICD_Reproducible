"""
Stage 4 — Adaptive Intelligent Clinical Decision Engine.
Trains 5 heterogeneous classifiers on the Stage 3 optimized feature set and
combines them via confidence-weighted soft voting (each model's ensemble
weight is proportional to its own validation macro-F1).
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report
import xgboost as xgb
from config import RESULTS_DIR, MODELS_DIR, CSV_DIR, SEED

STAGE3_OUT = os.path.join(RESULTS_DIR, 'stage3_optimized')
STAGE4_OUT = os.path.join(RESULTS_DIR, 'stage4_decision_engine')
os.makedirs(STAGE4_OUT, exist_ok=True)


def build_model_specs():
    return {
        "xgboost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                       objective='multi:softprob', num_class=5, eval_metric='mlogloss',
                                       random_state=SEED, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                                  random_state=SEED, n_jobs=-1),
        "svm": SVC(kernel='rbf', C=2.0, class_weight='balanced',
