"""
Stage 7 — Cross-Dataset Validation. (Path convention fixed to match the
flat-file layout actually used across the project, consistent with how
Stage 0-6/8 and the auxiliary modules save their outputs.)
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
import xgboost as xgb
from config import CSV_DIR, RESULTS_DIR, CNN_FEAT_DIR, VIT_FEAT_DIR, HC_FEAT_DIR, SEED


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


def get_fast_model_specs():
    return {
        "xgboost": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                                       objective='multi:softprob', num_class=5, eval_metric='mlogloss',
                                       random_state=SEED, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=SEED, n_jobs=-1),
        "svm": SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=SEED),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, early_stopping=True, random_state=SEED),
        "logistic_regression": LogisticRegression(max_iter=300, class_weight='balanced', n_jobs=-1, random_state=SEED),
    }


def train_weighted_ensemble(X_train, y_train, X_val, y_val):
    class_counts = np.bincount(y_train, minlength=5)
    weights_dict = {c: len(y_train) / (5 * max(class_counts[c], 1)) for c in range(5)}
    sample_weights = np.array([weights_dict[y] for y in y_train])
    models, val_f1 = {}, {}
    for name, clf in get_fast_model_specs().items():
        if name == "xgboost":
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train, y_train)
        val_f1[name] = f1_score(y_val, clf.predict(X_val), average='macro')
        models[name] = clf
    total = sum(val_f1.values())
    return models, {n: f / total for n, f in val_f1.items()}, val_f1


def ensemble_predict_proba(X, models, weights):
    proba = np.zeros((X.shape[0], 5))
    for name, clf in models.items():
        proba += weights[name] * clf.predict_proba(X)
    return proba


def run_leave_one_dataset_out():
    full_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))
    for held_out in ['APTOS', 'IDRiD', 'Messidor']:
        result_path = os.path.join(RESULTS_DIR, f'stage7_holdout_{held_out}.json')  # flat, matches migration
        if os.path.exists(result_path):
            print(f"{held_out}: cached, skipping")
            continue
        train_pool = full_df[full_df['dataset'] != held_out].copy()
        test_df = full_df[full_df['dataset'] == held_out].reset_index(drop=True)
        train_pool['strata'] = train_pool['dataset'] + '_' + train_pool['dr_grade'].astype(int).astype(str)
        safe = train_pool['strata'].value_counts()
        safe_idx = train_pool['strata'].isin(safe[safe >= 2].index)
        train_df, val_df = train_test_split(train_pool[safe_idx], train_size=0.85,
                                              stratify=train_pool[safe_idx]['strata'], random_state=SEED)
        train_df = pd.concat([train_df, train_pool[~safe_idx]], ignore_index=True)

        X_train, y_train = load_concatenated_features(train_df)
        X_val, y_val = load_concatenated_features(val_df)
        X_test, y_test = load_concatenated_features(test_df)
        scaler = StandardScaler()
        X_train_s, X_val_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

        models, weights, _ = train_weighted_ensemble(X_train_s, y_train, X_val_s, y_val)
        test_proba = ensemble_predict_proba(X_test_s, models, weights)
        test_preds = np.argmax(test_proba, axis=1)

        result = {"held_out_dataset": held_out, "test_macro_f1": float(f1_score(y_test, test_preds, average='macro')),
                   "test_qwk": float(cohen_kappa_score(y_test, test_preds, weights='quadratic')),
                   "test_accuracy": float((test_preds == y_test).mean()),
                   "confusion_matrix": confusion_matrix(y_test, test_preds).tolist()}
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"{held_out}: macro-F1={result['test_macro_f1']:.4f}")


def run_kfold_cv(n_folds=5):
    result_path = os.path.join(RESULTS_DIR, 'stage7b_kfold_cv_summary.json')  # flat, matches migration
    if os.path.exists(result_path):
        print("5-fold CV: cached, skipping")
        return
    full_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))
    X_all, y_all = load_concatenated_features(full_df)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = {}
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr_full, y_tr_full = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        X_train, X_val, y_train, y_val = train_test_split(X_tr_full, y_tr_full, train_size=0.85,
                                                              stratify=y_tr_full, random_state=SEED)
        scaler = StandardScaler()
        X_train_s, X_val_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
        models, weights, _ = train_weighted_ensemble(X_train_s, y_train, X_val_s, y_val)
        test_proba = ensemble_predict_proba(X_test_s, models, weights)
        test_preds = np.argmax(test_proba, axis=1)
        fold_results[fold_idx] = {"fold": fold_idx, "test_macro_f1": float(f1_score(y_test, test_preds, average='macro')),
                                     "test_qwk": float(cohen_kappa_score(y_test, test_preds, weights='quadratic'))}
        print(f"Fold {fold_idx}: macro-F1={fold_results[fold_idx]['test_macro_f1']:.4f}")

    f1s = [r['test_macro_f1'] for r in fold_results.values()]
    qwks = [r['test_qwk'] for r in fold_results.values()]
    summary = {"mean_macro_f1": float(np.mean(f1s)), "std_macro_f1": float(np.std(f1s)),
                "mean_qwk": float(np.mean(qwks)), "std_qwk": float(np.std(qwks)),
                "per_fold": {str(k): v for k, v in fold_results.items()}}
    with open(result_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"5-fold CV: mean macro-F1={summary['mean_macro_f1']:.4f}")


def run_domain_adaptation_sweep(fractions=(0.0, 0.05, 0.10, 0.20, 0.30)):
    full_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))
    for held_out in ['APTOS', 'IDRiD', 'Messidor']:
        source_df = full_df[full_df['dataset'] != held_out].reset_index(drop=True)
        target_df = full_df[full_df['dataset'] == held_out].reset_index(drop=True)
        X_source, y_source = load_concatenated_features(source_df)
        for frac in fractions:
            result_path = os.path.join(RESULTS_DIR, f'stage7c_{held_out}_frac{int(frac*100)}.json')
            if os.path.exists(result_path):
                continue
            if frac == 0.0:
                adapt_df, remaining_df = pd.DataFrame(), target_df
            else:
                adapt_df, remaining_df = train_test_split(target_df, train_size=frac,
                                                             stratify=target_df['dr_grade'], random_state=SEED)
            X_test, y_test = load_concatenated_features(remaining_df)
            if len(adapt_df):
                X_adapt, y_adapt = load_concatenated_features(adapt_df)
                X_train_c, y_train_c = np.concatenate([X_source, X_adapt]), np.concatenate([y_source, y_adapt])
            else:
                X_train_c, y_train_c = X_source, y_source
            scaler = StandardScaler()
            X_train_s, X_test_s = scaler.fit_transform(X_train_c), scaler.transform(X_test)
            class_counts = np.bincount(y_train_c, minlength=5)
            weights_dict = {c: len(y_train_c) / (5 * max(class_counts[c], 1)) for c in range(5)}
            sample_weights = np.array([weights_dict[y] for y in y_train_c])
            clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                                      objective='multi:softprob', num_class=5, eval_metric='mlogloss',
                                      random_state=SEED, n_jobs=-1)
            clf.fit(X_train_s, y_train_c, sample_weight=sample_weights)
            preds = clf.predict(X_test_s)
            result = {"held_out": held_out, "target_fraction": frac, "n_adapt_samples": len(adapt_df),
                       "n_test_samples": len(remaining_df),
                       "test_macro_f1": float(f1_score(y_test, preds, average='macro')),
                       "test_qwk": float(cohen_kappa_score(y_test, preds, weights='quadratic'))}
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"{held_out} frac={frac}: macro-F1={result['test_macro_f1']:.4f}")


if __name__ == "__main__":
    run_leave_one_dataset_out()
    run_kfold_cv()
    run_domain_adaptation_sweep()
