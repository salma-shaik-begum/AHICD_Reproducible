"""
Additional Statistical Tests: McNemar's test (paired classifier comparison
on the same test set) and Wilcoxon signed-rank test (paired comparison
across folds). Complements Stage 8's bootstrap CIs.

NOTE: paths below point to the flat-file locations produced by the original
project migration (Stage 0-8 ad-hoc runs), NOT the nested-folder convention
used by stage7_validation.py's save logic — those differ because
stage7_validation.py has not yet been re-executed as a standalone module
since migration. If re-run, update these paths to match, or re-migrate.
"""
import os
import json
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
from config import CSV_DIR, RESULTS_DIR

STAGE_STATS_OUT = os.path.join(RESULTS_DIR, 'stage_statistical_tests')
os.makedirs(STAGE_STATS_OUT, exist_ok=True)


def mcnemar_test(y_true, preds_a, preds_b, label_a="Model A", label_b="Model B"):
    correct_a = (np.array(preds_a) == np.array(y_true))
    correct_b = (np.array(preds_b) == np.array(y_true))
    both_correct = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))
    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=(a_only + b_only < 25))
    print(f"McNemar's test — {label_a} vs {label_b}:")
    print(f"  Both correct: {both_correct}, {label_a} only: {a_only}, {label_b} only: {b_only}, Both wrong: {both_wrong}")
    print(f"  statistic={result.statistic:.4f}, p-value={result.pvalue:.4f}")
    return {"label_a": label_a, "label_b": label_b, "both_correct": both_correct,
             "a_only": a_only, "b_only": b_only, "both_wrong": both_wrong,
             "statistic": float(result.statistic), "p_value": float(result.pvalue),
             "significant": bool(result.pvalue < 0.05)}


def wilcoxon_test(scores_a, scores_b, label_a="Config A", label_b="Config B"):
    stat, p_value = wilcoxon(scores_a, scores_b)
    print(f"\nWilcoxon signed-rank test — {label_a} vs {label_b}:")
    print(f"  statistic={stat:.4f}, p-value={p_value:.4f}")
    return {"label_a": label_a, "label_b": label_b, "statistic": float(stat), "p_value": float(p_value),
             "significant": bool(p_value < 0.05)}


def run_statistical_tests():
    results = {}

    # ---- Wilcoxon: 5-fold CV macro-F1 scores vs leave-one-dataset-out macro-F1 scores ----
    # NOTE: n=3 for the cross-dataset side is genuinely thin for Wilcoxon; treat as
    # illustrative/exploratory, not a primary statistical claim (Stage 8's bootstrap
    # CIs remain the primary rigorous evidence for the generalization gap).
    with open(os.path.join(RESULTS_DIR, 'stage7b_kfold_cv_summary.json')) as f:
        kfold_summary = json.load(f)
    fold_f1s = [v['test_macro_f1'] for v in kfold_summary['per_fold'].values()]

    cross_dataset_f1s = []
    for name in ['APTOS', 'IDRiD', 'Messidor']:
        with open(os.path.join(RESULTS_DIR, f'stage7_holdout_{name}.json')) as f:
            cross_dataset_f1s.append(json.load(f)['test_macro_f1'])

    sorted_fold_f1s = sorted(fold_f1s)[:3]
    wilcoxon_result = wilcoxon_test(sorted_fold_f1s, cross_dataset_f1s,
                                      "5-fold CV (within-distribution)", "Leave-one-dataset-out (cross-distribution)")
    wilcoxon_result["caveat"] = "n=3 paired samples — illustrative only, not a primary statistical claim"
    results['wilcoxon_kfold_vs_cross_dataset'] = wilcoxon_result

    with open(os.path.join(STAGE_STATS_OUT, 'statistical_tests_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Statistical tests saved to {STAGE_STATS_OUT}")
    return results


if __name__ == "__main__":
    run_statistical_tests()
