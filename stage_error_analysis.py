"""
Error Analysis — consolidates and formalizes failure patterns identified
throughout the pipeline: per-class error breakdown, "confidently wrong"
cases (high confidence + incorrect — the most clinically concerning failure
mode), cross-referencing errors against dataset source and image quality,
and identification of the hardest individual cases for qualitative review.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CSV_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, DR_GRADE_NAMES

STAGE_ERROR_OUT = os.path.join(RESULTS_DIR, 'stage_error_analysis')
os.makedirs(STAGE_ERROR_OUT, exist_ok=True)


def run_error_analysis():
    pred_df = pd.read_csv(os.path.join(CSV_DIR, 'stage4_ensemble_predictions.csv'))
    pred_df['correct'] = pred_df['predicted_dr_grade'] == pred_df['true_dr_grade']
    pred_df['error_magnitude'] = (pred_df['predicted_dr_grade'] - pred_df['true_dr_grade']).abs()

    # ---- 1. Per-class error rate + typical confusion direction ----
    print("=" * 60)
    print("PER-CLASS ERROR BREAKDOWN")
    print("=" * 60)
    per_class = []
    for grade in range(5):
        subset = pred_df[pred_df['true_dr_grade'] == grade]
        error_rate = 1 - subset['correct'].mean()
        avg_error_mag = subset[~subset['correct']]['error_magnitude'].mean() if (~subset['correct']).any() else 0
        most_common_wrong = subset[~subset['correct']]['predicted_dr_grade'].mode()
        most_common_wrong = int(most_common_wrong.iloc[0]) if len(most_common_wrong) else None
        print(f"Grade {grade} ({DR_GRADE_NAMES[grade]}): n={len(subset)}, error_rate={error_rate:.3f}, "
              f"avg_error_magnitude={avg_error_mag:.2f}, most_common_misprediction={most_common_wrong}")
        per_class.append({"true_grade": grade, "n": len(subset), "error_rate": float(error_rate),
                            "avg_error_magnitude": float(avg_error_mag), "most_common_wrong_prediction": most_common_wrong})

    # ---- 2. "Confidently wrong" cases — high confidence + incorrect (most clinically concerning) ----
    wrong = pred_df[~pred_df['correct']].copy()
    confidently_wrong = wrong[wrong['confidence_score'] >= 0.7].sort_values('confidence_score', ascending=False)
    print(f"\n--- Confidently wrong cases (confidence >= 0.7, n={len(confidently_wrong)}/{len(wrong)} total errors) ---")
    print(confidently_wrong[['unique_id', 'dataset', 'true_dr_grade', 'predicted_dr_grade', 'confidence_score']].head(15).to_string(index=False))

    # ---- 3. Error rate by dataset source ----
    print(f"\n--- Error rate by dataset ---")
    dataset_errors = {}
    for ds in pred_df['dataset'].unique():
        subset = pred_df[pred_df['dataset'] == ds]
        err_rate = 1 - subset['correct'].mean()
        print(f"  {ds}: n={len(subset)}, error_rate={err_rate:.3f}")
        dataset_errors[ds] = {"n": len(subset), "error_rate": float(err_rate)}

    # ---- 4. Cross-reference with image quality (if available) ----
    quality_path = os.path.join(CSV_DIR, 'stage_image_quality_metrics.csv')
    quality_correlation = None
    if os.path.exists(quality_path):
        quality_df = pd.read_csv(quality_path)
        merged = pred_df.merge(quality_df[['unique_id', 'blur_score', 'brightness', 'contrast']], on='unique_id', how='left')
        confidently_wrong_ids = confidently_wrong['unique_id'].tolist()
        cw_quality = merged[merged['unique_id'].isin(confidently_wrong_ids)]
        rest_quality = merged[~merged['unique_id'].isin(confidently_wrong_ids)]
        print(f"\n--- Image quality: confidently-wrong cases vs. rest ---")
        for metric in ['blur_score', 'brightness', 'contrast']:
            print(f"  {metric}: confidently-wrong mean={cw_quality[metric].mean():.2f}, "
                  f"rest mean={rest_quality[metric].mean():.2f}")
        quality_correlation = {
            m: {"confidently_wrong_mean": float(cw_quality[m].mean()), "rest_mean": float(rest_quality[m].mean())}
            for m in ['blur_score', 'brightness', 'contrast']
        }

    # ---- 5. Figure: error magnitude distribution ----
    fig, ax = plt.subplots(figsize=(7, 5))
    error_mags = pred_df[~pred_df['correct']]['error_magnitude']
    ax.hist(error_mags, bins=range(1, 6), align='left', rwidth=0.7, color='#C00000', alpha=0.8)
    ax.set_xlabel('Error magnitude (|predicted grade - true grade|)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Misclassification Severity (n={len(error_mags)} errors)')
    ax.set_xticks(range(1, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_error_magnitude_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    adjacent_error_pct = (error_mags == 1).mean() * 100
    print(f"\n{adjacent_error_pct:.1f}% of errors are adjacent-class (magnitude=1), "
          f"consistent with the ordinal QWK finding (Section 4.1)")

    pd.DataFrame(per_class).to_csv(os.path.join(TABLES_DIR, 'table7_per_class_error_breakdown.csv'), index=False)
    confidently_wrong.to_csv(os.path.join(CSV_DIR, 'stage_confidently_wrong_cases.csv'), index=False)

    summary = {
        "per_class_errors": per_class,
        "n_confidently_wrong": len(confidently_wrong),
        "n_total_errors": len(wrong),
        "pct_adjacent_class_errors": float(adjacent_error_pct),
        "error_rate_by_dataset": dataset_errors,
        "quality_vs_confidently_wrong": quality_correlation,
    }
    with open(os.path.join(STAGE_ERROR_OUT, 'error_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Error analysis saved to {STAGE_ERROR_OUT}")
    return summary


if __name__ == "__main__":
    run_error_analysis()
