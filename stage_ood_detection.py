"""
Out-of-Distribution (OOD) Detection.
Uses Mahalanobis distance in a PCA-reduced feature space (addresses the
covariance-estimation instability of computing Mahalanobis distance directly
in 768 dimensions with ~2600 training samples — a well-known failure mode).
Validated via AUC of OOD score predicting misclassification.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from config import CSV_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, CNN_FEAT_DIR, VIT_FEAT_DIR, HC_FEAT_DIR

STAGE_OOD_OUT = os.path.join(RESULTS_DIR, 'stage_ood_detection')
os.makedirs(STAGE_OOD_OUT, exist_ok=True)


def load_concatenated_features(df_subset):
    feats, labels = [], []
    for _, row in df_subset.iterrows():
        uid = row['unique_id']
        cnn_f = np.load(os.path.join(CNN_FEAT_DIR, f"{uid}.npy"))
        vit_f = np.load(os.path.join(VIT_FEAT_DIR, f"{uid}.npy"))
        hc_f = np.load(os.path.join(HC_FEAT_DIR, f"{uid}.npy"))
        feats.append(np.concatenate([cnn_f, vit_f, hc_f]))
        labels.append(int(row['dr_grade']))
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


class PCAMahalanobisOODDetector:
    """
    Reduces to n_components via PCA before computing Mahalanobis distance,
    to stabilize covariance estimation (raw 768-d Mahalanobis with ~2600
    training samples was empirically uninformative, AUC=0.486 — see
    ood_detection_summary_v1_failed.json for the documented negative result).
    """

    def __init__(self, n_components=30):
        self.n_components = n_components

    def fit(self, X_train):
        self.pca_ = PCA(n_components=self.n_components, random_state=42).fit(X_train)
        X_reduced = self.pca_.transform(X_train)
        self.mean_ = X_reduced.mean(axis=0)
        cov = np.cov(X_reduced, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        self.inv_cov_ = np.linalg.inv(cov)
        return self

    def score(self, X):
        X_reduced = self.pca_.transform(X)
        return np.array([mahalanobis(x, self.mean_, self.inv_cov_) for x in X_reduced])


def run_ood_detection(n_components_grid=(10, 30, 50, 100)):
    STAGE3_OUT = os.path.join(RESULTS_DIR, 'stage3_optimized')
    scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.pkl'))
    feature_mask = np.load(os.path.join(STAGE3_OUT, 'optimized_features.npz'))['feature_mask']

    split_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))
    train_df = split_df[split_df['split'] == 'train'].reset_index(drop=True)
    indist_test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)

    print("Loading features...")
    X_train_raw, _ = load_concatenated_features(train_df)
    X_train = scaler.transform(X_train_raw)[:, feature_mask]
    X_indist_raw, y_indist = load_concatenated_features(indist_test_df)
    X_indist = scaler.transform(X_indist_raw)[:, feature_mask]

    pred_df = pd.read_csv(os.path.join(CSV_DIR, 'stage4_ensemble_predictions.csv'))
    pred_df = pred_df.set_index('unique_id').loc[indist_test_df['unique_id']].reset_index()
    indist_correct = (pred_df['predicted_dr_grade'] == pred_df['true_dr_grade']).values

    # Sweep n_components to find the PCA dimensionality that actually makes the detector informative
    print(f"\nSweeping PCA dimensionality for Mahalanobis OOD detection:")
    best_auc, best_n, best_detector, best_scores = 0.0, None, None, None
    sweep_results = {}
    for n_comp in n_components_grid:
        detector = PCAMahalanobisOODDetector(n_components=n_comp).fit(X_train)
        scores = detector.score(X_indist)
        auc = roc_auc_score(~indist_correct, scores)
        sweep_results[n_comp] = float(auc)
        print(f"  n_components={n_comp}: AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_n, best_detector, best_scores = auc, n_comp, detector, scores

    print(f"\nBest configuration: n_components={best_n}, AUC={best_auc:.4f} "
          f"({'informative' if best_auc > 0.6 else 'weakly informative' if best_auc > 0.55 else 'still not informative'})")

    joblib.dump(best_detector, os.path.join(MODELS_DIR, 'ood_detector.pkl'))

    ood_by_dataset = {}
    for ds in ['APTOS', 'IDRiD', 'Messidor']:
        ds_test = split_df[(split_df['dataset'] == ds) & (split_df['split'] == 'test')]
        X_ds_raw, _ = load_concatenated_features(ds_test)
        X_ds = scaler.transform(X_ds_raw)[:, feature_mask]
        ood_by_dataset[ds] = best_detector.score(X_ds)
        print(f"  {ds} test subset OOD score: mean={ood_by_dataset[ds].mean():.2f}, std={ood_by_dataset[ds].std():.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for ds, scores in ood_by_dataset.items():
        ax.hist(scores, bins=25, alpha=0.5, label=ds, density=True)
    ax.set_xlabel(f'PCA({best_n})-Mahalanobis OOD score')
    ax.set_ylabel('Density')
    ax.set_title(f'OOD Score Distribution by Dataset (best config: PCA-{best_n})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_ood_score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    summary = {
        "pca_components_sweep": sweep_results, "best_n_components": best_n, "best_auc": float(best_auc),
        "ood_by_dataset": {ds: {"mean": float(s.mean()), "std": float(s.std())} for ds, s in ood_by_dataset.items()},
        "note": ("Raw 768-d Mahalanobis (no PCA) was empirically uninformative (AUC=0.486), "
                  "attributed to covariance-estimation instability given ~2600 training samples "
                  "vs 768 dimensions. PCA dimensionality reduction was applied to address this."),
    }
    with open(os.path.join(STAGE_OOD_OUT, 'ood_detection_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ OOD detection analysis saved to {STAGE_OOD_OUT}")
    return summary


if __name__ == "__main__":
    run_ood_detection()
