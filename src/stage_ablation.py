"""
Ablation Study — isolates the contribution of each feature source and the
NSGA-II optimization step by evaluating: CNN only, ViT only, CNN+ViT,
CNN+ViT+Handcrafted (pre-optimization), and the Full AHICF pipeline
(NSGA-II optimized, from Stage 4).

Finding (see paper Section 4/Discussion): ViT alone matched or exceeded
every hybrid configuration on both macro-F1 and QWK, indicating the
hybrid fusion + optimization's primary value in this study is
dimensionality reduction (55% fewer dims, no significant accuracy cost)
and interpretability rather than raw accuracy gain.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score
import xgboost as xgb
from config import CSV_DIR, RESULTS_DIR, TABLES_DIR, CNN_FEAT_DIR, VIT_FEAT_DIR, HC_FEAT_DIR, SEED

ABLATION_OUT = os.path.join(RESULTS_DIR, 'ablation_study')
os.makedirs(ABLATION_OUT, exist_ok=True)


def load_all_feature_types(df_subset):
    cnn_list, vit_list, hc_list, labels = [], [], [], []
    for _, row in df_subset.iterrows():
        uid = row['unique_id']
        cnn_list.append(np.load(os.path.join(CNN_FEAT_DIR, f"{uid}.npy")))
        vit_list.append(np.load(os.path.join(VIT_FEAT_DIR, f"{uid}.npy")))
        hc_list.append(np.load(os.path.join(HC_FEAT_DIR, f"{uid}.npy")))
        labels.append(int(row['dr_grade']))
    return (np.array(cnn_list, dtype=np.float32), np.array(vit_list, dtype=np.float32),
             np.array(hc_list, dtype=np.float32), np.array(labels, dtype=np.int64))


def get_fast_model_specs():
    """Lighter configs (linear SVM, fewer trees) tuned for CPU feasibility, consistent with Stage 7."""
    return {
        "xgboost": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                                       objective='multi:softprob', num_class=5, eval_metric='mlogloss',
                                       random_state=SEED, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=SEED, n_jobs=-1),
        "svm": SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=SEED),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, early_stopping=True, random_state=SEED),
        "logistic_regression": LogisticRegression(max_iter=300, class_weight='balanced', n_jobs=-1, random_state=SEED),
    }


def train_and_eval_ensemble(X_train, X_val, X_test, y_train, y_val, y_test):
    scaler = StandardScaler()
    X_train_s, X_val_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    class_counts = np.bincount(y_train, minlength=5)
    weights_dict = {c: len(y_train) / (5 * max(class_counts[c], 1)) for c in range(5)}
    sample_weights = np.array([weights_dict[y] for y in y_train])

    models, val_f1 = {}, {}
    for name, clf in get_fast_model_specs().items():
        if name == "xgboost":
            clf.fit(X_train_s, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train_s, y_train)
        val_f1[name] = f1_score(y_val, clf.predict(X_val_s), average='macro')
        models[name] = clf
    total = sum(val_f1.values())
    weights = {n: f / total for n, f in val_f1.items()}

    test_proba = np.zeros((X_test_s.shape[0], 5))
    for name, clf in models.items():
        test_proba += weights[name] * clf.predict_proba(X_test_s)
    test_preds = np.argmax(test_proba, axis=1)
    return (f1_score(y_test, test_preds, average='macro'),
            cohen_kappa_score(y_test, test_preds, weights='quadratic'))


def run_ablation_study():
    split_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))
    train_df = split_df[split_df['split'] == 'train'].reset_index(drop=True)
    val_df = split_df[split_df['split'] == 'val'].reset_index(drop=True)
    test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)

    cnn_tr, vit_tr, hc_tr, y_train = load_all_feature_types(train_df)
    cnn_v, vit_v, hc_v, y_val = load_all_feature_types(val_df)
    cnn_te, vit_te, hc_te, y_test = load_all_feature_types(test_df)

    configs = {
        "CNN only": (cnn_tr, cnn_v, cnn_te),
        "ViT only": (vit_tr, vit_v, vit_te),
        "CNN + ViT": (np.concatenate([cnn_tr, vit_tr], 1), np.concatenate([cnn_v, vit_v], 1), np.concatenate([cnn_te, vit_te], 1)),
        "CNN + ViT + Handcrafted": (np.concatenate([cnn_tr, vit_tr, hc_tr], 1), np.concatenate([cnn_v, vit_v, hc_v], 1), np.concatenate([cnn_te, vit_te, hc_te], 1)),
    }

    results = {}
    for name, (Xtr, Xv, Xte) in configs.items():
        result_path = os.path.join(ABLATION_OUT, f"{name.replace(' ', '_').replace('+', 'and')}.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                results[name] = json.load(f)
            continue
        f1, qwk = train_and_eval_ensemble(Xtr, Xv, Xte, y_train, y_val, y_test)
        results[name] = {"macro_f1": float(f1), "qwk": float(qwk), "feature_dim": int(Xtr.shape[1])}
        with open(result_path, 'w') as f_out:
            json.dump(results[name], f_out, indent=2)
        print(f"{name}: macro-F1={f1:.4f}, QWK={qwk:.4f}")

    # Pull in the Full AHICF (Stage 4, NSGA-II optimized) result for the complete comparison
    with open(os.path.join(RESULTS_DIR, 'stage4_ensemble_summary.json')) as f:
        stage4_summary = json.load(f)
    results["Full AHICF (NSGA-II optimized)"] = {
        "macro_f1": stage4_summary.get('ensemble_test_macro_f1', stage4_summary.get('test_macro_f1')),
        "qwk": stage4_summary.get('test_qwk', stage4_summary.get('ensemble_test_qwk')),
        "feature_dim": 768,
    }

    ablation_df = pd.DataFrame([
        {"Configuration": k, "Feature_dim": v["feature_dim"], "Macro_F1": v["macro_f1"], "QWK": v.get("qwk")}
        for k, v in results.items()
    ])
    ablation_df.to_csv(os.path.join(TABLES_DIR, 'table5_ablation_study.csv'), index=False)
    print("\n" + ablation_df.to_string(index=False))
    return ablation_df


if __name__ == "__main__":
    run_ablation_study()
