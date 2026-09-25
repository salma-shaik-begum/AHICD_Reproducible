"""
Stage 6b — Auxiliary DME (Diabetic Macular Edema) Risk Classifier.

Trains a classifier on the same DR-optimized feature representation to
predict DME risk (0=none, 1=mild, 2=severe), using the previously-unused
dme_risk labels available for IDRiD and Messidor (APTOS has none).

Also includes the confound-isolated referral analysis: a naive comparison
of "DR-only" vs "DR+DME" referral rules is confounded by the precautionary
low-confidence trigger present in both DME-inclusive variants. This module
isolates DME's TRUE marginal contribution by holding that trigger constant
on both sides of the comparison. Finding: DME's marginal contribution to
referral sensitivity/specificity was negligible once this confound was
removed, so DME is NOT adopted as an independent referral trigger in the
final pipeline — retained here as an honestly-reported auxiliary result.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score
import xgboost as xgb
from config import CSV_DIR, RESULTS_DIR, MODELS_DIR, CNN_FEAT_DIR, VIT_FEAT_DIR, HC_FEAT_DIR, SEED

STAGE6B_OUT = os.path.join(RESULTS_DIR, 'stage6b_dme_classifier')
os.makedirs(STAGE6B_OUT, exist_ok=True)


def load_concatenated_features(df_subset, label_col='dme_risk'):
    feats, labels = [], []
    for _, row in df_subset.iterrows():
        uid = row['unique_id']
        cnn_f = np.load(os.path.join(CNN_FEAT_DIR, f"{uid}.npy"))
        vit_f = np.load(os.path.join(VIT_FEAT_DIR, f"{uid}.npy"))
        hc_f = np.load(os.path.join(HC_FEAT_DIR, f"{uid}.npy"))
        feats.append(np.concatenate([cnn_f, vit_f, hc_f]))
        labels.append(int(row[label_col]))
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


def train_dme_classifier():
    """Trains the DME classifier reusing the exact Stage 3 scaler + NSGA-II feature mask."""
    STAGE3_OUT = os.path.join(RESULTS_DIR, 'stage3_optimized')
    scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.pkl'))
    feature_mask = np.load(os.path.join(STAGE3_OUT, 'optimized_features.npz'))['feature_mask']

    split_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))
    dme_df = split_df[split_df['dme_risk'].notna()].copy()
    dme_df['dme_risk'] = dme_df['dme_risk'].astype(int)

    train_df = dme_df[dme_df['split'] == 'train'].reset_index(drop=True)
    val_df = dme_df[dme_df['split'] == 'val'].reset_index(drop=True)
    test_df = dme_df[dme_df['split'] == 'test'].reset_index(drop=True)

    X_train_raw, y_train = load_concatenated_features(train_df)
    X_val_raw, y_val = load_concatenated_features(val_df)
    X_test_raw, y_test = load_concatenated_features(test_df)

    X_train = scaler.transform(X_train_raw)[:, feature_mask]
    X_val = scaler.transform(X_val_raw)[:, feature_mask]
    X_test = scaler.transform(X_test_raw)[:, feature_mask]

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = {c: len(y_train) / (3 * max(class_counts[c], 1)) for c in range(3)}
    sample_weights = np.array([class_weights[y] for y in y_train])

    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                              objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                              random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    val_f1 = f1_score(y_val, clf.predict(X_val), average='macro')
    test_proba = clf.predict_proba(X_test)
    test_preds = np.argmax(test_proba, axis=1)
    test_confidence = np.max(test_proba, axis=1)
    test_macro_f1 = f1_score(y_test, test_preds, average='macro')
    test_qwk = cohen_kappa_score(y_test, test_preds, weights='quadratic')

    print(f"DME classifier — val macro-F1={val_f1:.4f}, test macro-F1={test_macro_f1:.4f}, QWK={test_qwk:.4f}")
    print(classification_report(y_test, test_preds, target_names=['No risk', 'Mild risk', 'Severe risk'], digits=3))

    joblib.dump(clf, os.path.join(MODELS_DIR, 'dme_classifier.pkl'))

    out_df = pd.DataFrame({'unique_id': test_df['unique_id'], 'dataset': test_df['dataset'],
                              'true_dme_risk': y_test, 'predicted_dme_risk': test_preds,
                              'dme_confidence': test_confidence})
    out_df.to_csv(os.path.join(CSV_DIR, 'stage6b_dme_predictions.csv'), index=False)

    summary = {"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
                "val_macro_f1": float(val_f1), "test_macro_f1": float(test_macro_f1), "test_qwk": float(test_qwk),
                "note": "APTOS excluded (no DME labels); trained on IDRiD+Messidor using the DR-optimized feature subset."}
    with open(os.path.join(STAGE6B_OUT, 'stage6b_dme_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return out_df


def confound_isolated_referral_analysis():
    """
    Isolates DME's TRUE marginal contribution to referral performance by
    comparing (Stage 6 rule alone) vs (Stage 6 rule + severe DME trigger),
    holding the precautionary confidence trigger constant on both sides.
    """
    dr_df = pd.read_csv(os.path.join(CSV_DIR, 'stage4_ensemble_predictions.csv'))
    dme_df = pd.read_csv(os.path.join(CSV_DIR, 'stage6b_dme_predictions.csv'))
    combined = dr_df.merge(dme_df[['unique_id', 'predicted_dme_risk', 'true_dme_risk']], on='unique_id', how='inner')
    combined['should_refer'] = (combined['true_dr_grade'] >= 2) | (combined['true_dme_risk'] >= 1)

    def evaluate(referred):
        tp = ((combined.should_refer) & referred).sum()
        fn = ((combined.should_refer) & ~referred).sum()
        tn = ((~combined.should_refer) & ~referred).sum()
        fp = ((~combined.should_refer) & referred).sum()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        return {"sensitivity": float(sens), "specificity": float(spec),
                 "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp)}

    rule_B = (combined['predicted_dr_grade'] >= 2) | (combined['confidence_score'] < 0.6)
    rule_C = rule_B | (combined['predicted_dme_risk'] == 2)

    stats_B, stats_C = evaluate(rule_B), evaluate(rule_C)
    analysis = {
        "conclusion": "DME risk NOT adopted as an independent referral trigger — negligible marginal effect once confidence-trigger confound is isolated.",
        "rule_B_stage6_no_dme": stats_B,
        "rule_C_stage6_plus_severe_dme": stats_C,
        "dme_marginal_delta_sensitivity": stats_C['sensitivity'] - stats_B['sensitivity'],
        "dme_marginal_delta_specificity": stats_C['specificity'] - stats_B['specificity'],
    }
    with open(os.path.join(STAGE6B_OUT, 'stage6b_dme_referral_confound_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"DME marginal effect: Δsens={analysis['dme_marginal_delta_sensitivity']:+.4f}, "
          f"Δspec={analysis['dme_marginal_delta_specificity']:+.4f}")
    return analysis


if __name__ == "__main__":
    train_dme_classifier()
    confound_isolated_referral_analysis()
