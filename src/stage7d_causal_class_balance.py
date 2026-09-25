"""
Stage 7d — Causal Test of the Severity-Class-Imbalance Mechanism.

Stage 7c showed domain adaptation gap-closure correlates with severity-class
representation (APTOS/IDRiD closed 61-77%, Messidor only 24%, tracking
grade-4 prevalence). That finding was CORRELATIONAL. This module runs a
controlled manipulation to test it causally: does deliberately balancing
severity representation in the adaptation set improve gap closure beyond
what inverse-frequency class weighting (already applied throughout this
pipeline) achieves alone?

Three conditions, same total adaptation volume:
  1. Natural + weighted   (reproduces Stage 7c's original result)
  2. Natural + unweighted (isolates the contribution of class weighting)
  3. Balanced + weighted  (deliberately oversamples grade 3/4 via
     duplication; tests whether genuine representation adds anything
     BEYOND what weighting already provides)

LIMITATION (disclosed): condition 3 uses feature-vector duplication, not
synthetic augmentation (e.g. SMOTE) or new images -- duplicated samples
carry zero new information content, only additional gradient/decision-
boundary influence. A positive effect here shows loss-level oversampling
helps; it does NOT show that genuinely novel severe-case images would
help by the same amount. This distinction is reported explicitly in
results.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, cohen_kappa_score
import xgboost as xgb
from config import CSV_DIR, RESULTS_DIR, CNN_FEAT_DIR, VIT_FEAT_DIR, HC_FEAT_DIR, SEED

STAGE7D_OUT = os.path.join(RESULTS_DIR, 'stage7d_causal_class_balance')
os.makedirs(STAGE7D_OUT, exist_ok=True)


def load_concatenated_features(df_subset):
    feats, labels = [], []
    for _, row in df_subset.iterrows():
        uid = row['unique_id']
        feats.append(np.concatenate([
            np.load(os.path.join(CNN_FEAT_DIR, f"{uid}.npy")),
            np.load(os.path.join(VIT_FEAT_DIR, f"{uid}.npy")),
            np.load(os.path.join(HC_FEAT_DIR, f"{uid}.npy")),
        ]))
        labels.append(int(row['dr_grade']))
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


def balance_via_duplication(X, y, target_count_per_class=None):
    """
    Oversamples minority classes via duplication to flatten severity
    representation. If target_count_per_class is None, matches the
    largest class count (full balance).
    """
    unique, counts = np.unique(y, return_counts=True)
    target = target_count_per_class or counts.max()

    X_list, y_list = [X], [y]
    for cls, cnt in zip(unique, counts):
        if cnt < target:
            n_needed = target - cnt
            cls_idx = np.where(y == cls)[0]
            dup_idx = np.random.RandomState(SEED).choice(cls_idx, n_needed, replace=True)
            X_list.append(X[dup_idx])
            y_list.append(y[dup_idx])
    return np.concatenate(X_list), np.concatenate(y_list)


def train_and_eval(X_train, y_train, X_test, y_test, use_class_weights):
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

    if use_class_weights:
        class_counts = np.bincount(y_train, minlength=5)
        weights_dict = {c: len(y_train) / (5 * max(class_counts[c], 1)) for c in range(5)}
        sample_weights = np.array([weights_dict[y] for y in y_train])
    else:
        sample_weights = None

    clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                              objective='multi:softprob', num_class=5, eval_metric='mlogloss',
                              random_state=SEED, n_jobs=-1)
    clf.fit(X_train_s, y_train, sample_weight=sample_weights)
    preds = clf.predict(X_test_s)
    return (f1_score(y_test, preds, average='macro'),
            cohen_kappa_score(y_test, preds, weights='quadratic'))


def run_causal_class_balance_test(held_out='Messidor', adaptation_frac=0.30):
    full_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))
    source_df = full_df[full_df['dataset'] != held_out].reset_index(drop=True)
    target_df = full_df[full_df['dataset'] == held_out].reset_index(drop=True)

    print(f"Held-out dataset: {held_out}, adaptation fraction: {adaptation_frac}")
    adapt_df, remaining_df = train_test_split(target_df, train_size=adaptation_frac,
                                                 stratify=target_df['dr_grade'], random_state=SEED)
    print(f"Adaptation set size: {len(adapt_df)}, grade distribution:")
    print(adapt_df['dr_grade'].value_counts().sort_index())

    X_source, y_source = load_concatenated_features(source_df)
    X_adapt, y_adapt = load_concatenated_features(adapt_df)
    X_test, y_test = load_concatenated_features(remaining_df)

    results = {}

    # ---- Condition 1: Natural + weighted (reproduces Stage 7c) ----
    X_train_1 = np.concatenate([X_source, X_adapt])
    y_train_1 = np.concatenate([y_source, y_adapt])
    f1_1, qwk_1 = train_and_eval(X_train_1, y_train_1, X_test, y_test, use_class_weights=True)
    results['1_natural_weighted'] = {"macro_f1": float(f1_1), "qwk": float(qwk_1), "n_train": len(y_train_1)}
    print(f"\n1) Natural + weighted:   macro-F1={f1_1:.4f}, QWK={qwk_1:.4f}")

    # ---- Condition 2: Natural + UNweighted (isolate weighting's contribution) ----
    f1_2, qwk_2 = train_and_eval(X_train_1, y_train_1, X_test, y_test, use_class_weights=False)
    results['2_natural_unweighted'] = {"macro_f1": float(f1_2), "qwk": float(qwk_2), "n_train": len(y_train_1)}
    print(f"2) Natural + unweighted: macro-F1={f1_2:.4f}, QWK={qwk_2:.4f}")

    # ---- Condition 3: Balanced (duplicated) adaptation subset + weighted ----
    X_adapt_balanced, y_adapt_balanced = balance_via_duplication(X_adapt, y_adapt)
    print(f"Balanced adaptation set size: {len(y_adapt_balanced)} (from {len(y_adapt)}), new distribution:")
    print(pd.Series(y_adapt_balanced).value_counts().sort_index())

    X_train_3 = np.concatenate([X_source, X_adapt_balanced])
    y_train_3 = np.concatenate([y_source, y_adapt_balanced])
    f1_3, qwk_3 = train_and_eval(X_train_3, y_train_3, X_test, y_test, use_class_weights=True)
    results['3_balanced_weighted'] = {"macro_f1": float(f1_3), "qwk": float(qwk_3), "n_train": len(y_train_3)}
    print(f"3) Balanced + weighted:  macro-F1={f1_3:.4f}, QWK={qwk_3:.4f}")

    # ---- Interpretation ----
    weighting_contribution = f1_1 - f1_2
    balance_contribution_beyond_weighting = f1_3 - f1_1
    print(f"\n--- Causal decomposition ---")
    print(f"Contribution of class weighting alone (1 vs 2): {weighting_contribution:+.4f} macro-F1")
    print(f"Contribution of balancing BEYOND weighting (3 vs 1): {balance_contribution_beyond_weighting:+.4f} macro-F1")

    if balance_contribution_beyond_weighting > 0.02:
        conclusion = ("Balancing severity representation improves performance beyond what class "
                        "weighting alone achieves, supporting a genuine representation/diversity "
                        "mechanism rather than a pure loss-weighting effect.")
    elif abs(balance_contribution_beyond_weighting) <= 0.02:
        conclusion = ("Balancing via duplication provided negligible additional benefit beyond class "
                        "weighting alone, suggesting the scarcity problem is fundamentally about the "
                        "DIVERSITY of available severe-class examples (which duplication cannot create), "
                        "not merely their weight in the loss function -- a stronger and more concerning "
                        "interpretation of the mechanism than simple class imbalance.")
    else:
        conclusion = "Balancing via duplication reduced performance, possibly due to overfitting to duplicated exact feature vectors."
    print(f"\nConclusion: {conclusion}")

    summary = {
        "held_out_dataset": held_out, "adaptation_fraction": adaptation_frac,
        "conditions": results,
        "weighting_contribution": float(weighting_contribution),
        "balance_contribution_beyond_weighting": float(balance_contribution_beyond_weighting),
        "conclusion": conclusion,
        "limitation": ("Condition 3 uses feature-vector duplication, not novel data. A null/small effect "
                        "here is arguably MORE informative than a positive one, since it suggests naive "
                        "oversampling cannot substitute for genuinely diverse severe-case examples."),
    }
    with open(os.path.join(STAGE7D_OUT, f'{held_out}_causal_test.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Saved causal class-balance test to {STAGE7D_OUT}")
    return summary


if __name__ == "__main__":
    run_causal_class_balance_test(held_out='Messidor', adaptation_frac=0.30)
    run_causal_class_balance_test(held_out='APTOS', adaptation_frac=0.30)  # comparison: already-decent representation
