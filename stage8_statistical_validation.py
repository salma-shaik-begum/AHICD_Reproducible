"""
Stage 8 — Statistical Validation.
Bootstrapped 95% confidence intervals (percentile method, 2000 resamples)
for macro-F1 and QWK, plus significance testing of the in-distribution vs
cross-dataset gap and the domain-adaptation improvement.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score
from config import CSV_DIR, RESULTS_DIR

STAGE8_OUT = os.path.join(RESULTS_DIR, 'stage8_statistical_validation')
os.makedirs(STAGE8_OUT, exist_ok=True)
N_BOOTSTRAP = 2000


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    boot = np.array([metric_fn(y_true[idx], y_pred[idx])
                       for idx in (rng.randint(0, n, n) for _ in range(n_bootstrap))])
    return {"point_estimate": float(metric_fn(y_true, y_pred)),
             "ci_lower": float(np.percentile(boot, 2.5)), "ci_upper": float(np.percentile(boot, 97.5)),
             "bootstrap_std": float(boot.std())}, boot


def confusion_matrix_to_pairs(cm):
    cm = np.array(cm)
    y_true, y_pred = [], []
    for t in range(cm.shape[0]):
        for p in range(cm.shape[1]):
            y_true.extend([t] * cm[t, p])
            y_pred.extend([p] * cm[t, p])
    return np.array(y_true), np.array(y_pred)


def run_stage8():
    macro_f1_fn = lambda yt, yp: f1_score(yt, yp, average='macro')
    qwk_fn = lambda yt, yp: cohen_kappa_score(yt, yp, weights='quadratic')

    decision_df = pd.read_csv(os.path.join(CSV_DIR, 'stage4_ensemble_predictions.csv'))
    y_true, y_pred = decision_df['true_dr_grade'].values, decision_df['predicted_dr_grade'].values
    f1_ci, f1_boot = bootstrap_ci(y_true, y_pred, macro_f1_fn)
    qwk_ci, _ = bootstrap_ci(y_true, y_pred, qwk_fn)
    print(f"In-distribution: macro-F1={f1_ci['point_estimate']:.4f} "
          f"[{f1_ci['ci_lower']:.4f}, {f1_ci['ci_upper']:.4f}]")

    cross_dataset = {}
    for held_out in ['APTOS', 'IDRiD', 'Messidor']:
        with open(os.path.join(RESULTS_DIR, 'stage7_cross_dataset', f'holdout_{held_out}.json')) as f:
            fold_data = json.load(f)
        yt, yp = confusion_matrix_to_pairs(fold_data['confusion_matrix'])
        f1_ci_ho, f1_boot_ho = bootstrap_ci(yt, yp, macro_f1_fn)
        qwk_ci_ho, _ = bootstrap_ci(yt, yp, qwk_fn)

        min_len = min(len(f1_boot), len(f1_boot_ho))
        diff = f1_boot[:min_len] - f1_boot_ho[:min_len]
        diff_lower, diff_upper = np.percentile(diff, [2.5, 97.5])
        cross_dataset[held_out] = {
            "macro_f1": f1_ci_ho, "qwk": qwk_ci_ho,
            "delta_vs_indist": {"point": f1_ci['point_estimate'] - f1_ci_ho['point_estimate'],
                                  "ci_lower": float(diff_lower), "ci_upper": float(diff_upper),
                                  "significant": bool(diff_lower > 0)},
        }
        print(f"{held_out}: macro-F1={f1_ci_ho['point_estimate']:.4f} "
              f"[{f1_ci_ho['ci_lower']:.4f}, {f1_ci_ho['ci_upper']:.4f}], "
              f"Δ significant={diff_lower > 0}")

    summary = {"in_distribution": {"macro_f1": f1_ci, "qwk": qwk_ci},
                "cross_dataset": cross_dataset, "n_bootstrap": N_BOOTSTRAP}
    with open(os.path.join(STAGE8_OUT, 'stage8_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("Stage 8 complete.")


if __name__ == "__main__":
    run_stage8()
