"""
Model Calibration — Expected Calibration Error (ECE) and Reliability Diagram.
Validates whether the ensemble's confidence scores (relied upon by the Stage 6
referral logic) are actually trustworthy: does 80% confidence really mean
the prediction is correct 80% of the time?
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CSV_DIR, RESULTS_DIR, FIGURES_DIR

STAGE_CAL_DIR = os.path.join(RESULTS_DIR, 'stage_calibration')
os.makedirs(STAGE_CAL_DIR, exist_ok=True)


def compute_ece(confidences, correct, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            bin_stats.append({'bin_range': f'{lo:.1f}-{hi:.1f}', 'n': 0, 'avg_confidence': None, 'accuracy': None})
            continue
        avg_conf, avg_acc = confidences[in_bin].mean(), correct[in_bin].mean()
        ece += (n_in_bin / len(confidences)) * abs(avg_conf - avg_acc)
        bin_stats.append({'bin_range': f'{lo:.1f}-{hi:.1f}', 'n': int(n_in_bin),
                            'avg_confidence': float(avg_conf), 'accuracy': float(avg_acc)})
    return ece, bin_stats


def plot_reliability_diagram(bin_stats, ece, save_path):
    valid = [b for b in bin_stats if b['n'] > 0]
    bin_centers = [(float(b['bin_range'].split('-')[0]) + float(b['bin_range'].split('-')[1])) / 2 for b in valid]
    accuracies = [b['accuracy'] for b in valid]
    confs = [b['avg_confidence'] for b in valid]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    ax.bar(bin_centers, accuracies, width=0.08, alpha=0.7, edgecolor='black',
           label='Observed accuracy', color='#5B9BD5')
    ax.scatter(confs, accuracies, color='#C00000', zorder=5, label='Bin avg. confidence vs accuracy')
    ax.set_xlabel('Predicted confidence')
    ax.set_ylabel('Observed accuracy')
    ax.set_title(f'Reliability Diagram — Ensemble Calibration (ECE = {ece:.4f})')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_calibration_analysis():
    decision_df = pd.read_csv(os.path.join(CSV_DIR, 'stage4_ensemble_predictions.csv'))
    confidences = decision_df['confidence_score'].values
    correct = (decision_df['predicted_dr_grade'] == decision_df['true_dr_grade']).values.astype(int)

    ece, bin_stats = compute_ece(confidences, correct, n_bins=10)
    print(f"Expected Calibration Error (ECE): {ece:.4f} "
          f"({'well-calibrated' if ece < 0.05 else 'needs recalibration'} for clinical use)")

    plot_reliability_diagram(bin_stats, ece, os.path.join(FIGURES_DIR, 'fig_reliability_diagram.png'))

    with open(os.path.join(STAGE_CAL_DIR, 'calibration_summary.json'), 'w') as f:
        json.dump({"ece": float(ece), "n_bins": 10, "bin_stats": bin_stats}, f, indent=2)
    print(f"Calibration analysis complete, saved to {STAGE_CAL_DIR}")
    return ece, bin_stats


if __name__ == "__main__":
    run_calibration_analysis()
