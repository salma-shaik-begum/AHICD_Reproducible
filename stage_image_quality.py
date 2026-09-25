"""
Image Quality Assessment.
Computes blur (Laplacian variance), brightness, and contrast metrics per
enhanced fundus image. Validates whether: (1) low-quality images correlate
with higher misclassification rates, and (2) datasets differ systematically
in quality, as a candidate additional explanation for the cross-dataset
generalization gap alongside the severity-class-imbalance mechanism.
"""
import os
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr, f_oneway
from config import CSV_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR

STAGE_QUALITY_OUT = os.path.join(RESULTS_DIR, 'stage_image_quality')
os.makedirs(STAGE_QUALITY_OUT, exist_ok=True)


def compute_quality_metrics(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blur: variance of the Laplacian — lower = blurrier (standard, widely-used metric)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness: mean pixel intensity — flags over/under-exposed images
    brightness = gray.mean()

    # Contrast: std of pixel intensity — low contrast can obscure lesion visibility
    contrast = gray.std()

    return {"blur_score": float(blur_score), "brightness": float(brightness), "contrast": float(contrast)}


def run_image_quality_assessment():
    split_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))
    test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)

    print(f"Computing quality metrics for {len(split_df)} images (full manifest)...")
    quality_records = []
    for _, row in split_df.iterrows():
        img = cv2.imread(row['enhanced_path'])
        if img is None:
            continue
        metrics = compute_quality_metrics(img)
        metrics['unique_id'] = row['unique_id']
        metrics['dataset'] = row['dataset']
        metrics['split'] = row['split']
        quality_records.append(metrics)

    quality_df = pd.DataFrame(quality_records)
    quality_df.to_csv(os.path.join(CSV_DIR, 'stage_image_quality_metrics.csv'), index=False)
    print(f"Computed quality metrics for {len(quality_df)} images")

    # ---- 1. Does quality differ systematically by dataset? (candidate 2nd generalization-gap mechanism) ----
    print("\nQuality metrics by dataset (mean ± std):")
    dataset_quality_summary = {}
    for metric in ['blur_score', 'brightness', 'contrast']:
        print(f"\n{metric}:")
        groups = []
        for ds in ['APTOS', 'IDRiD', 'Messidor']:
            vals = quality_df[quality_df['dataset'] == ds][metric]
            print(f"  {ds}: {vals.mean():.2f} ± {vals.std():.2f}")
            groups.append(vals.values)
        f_stat, p_val = f_oneway(*groups)
        print(f"  ANOVA: F={f_stat:.2f}, p={p_val:.6f} ({'significant difference across datasets' if p_val < 0.05 else 'no significant difference'})")
        dataset_quality_summary[metric] = {
            "by_dataset": {ds: {"mean": float(quality_df[quality_df['dataset']==ds][metric].mean()),
                                   "std": float(quality_df[quality_df['dataset']==ds][metric].std())}
                            for ds in ['APTOS', 'IDRiD', 'Messidor']},
            "anova_f": float(f_stat), "anova_p": float(p_val), "significant": bool(p_val < 0.05),
        }

    # ---- 2. Does quality correlate with prediction correctness on the in-distribution test set? ----
    pred_df = pd.read_csv(os.path.join(CSV_DIR, 'stage4_ensemble_predictions.csv'))
    test_quality = quality_df[quality_df['split'] == 'test'].merge(
        pred_df[['unique_id', 'predicted_dr_grade', 'true_dr_grade']], on='unique_id', how='inner')
    test_quality['correct'] = (test_quality['predicted_dr_grade'] == test_quality['true_dr_grade']).astype(int)

    print(f"\n--- Quality vs. prediction correctness (n={len(test_quality)}) ---")
    correlation_results = {}
    for metric in ['blur_score', 'brightness', 'contrast']:
        corr, p_val = pointbiserialr(test_quality['correct'], test_quality[metric])
        auc = roc_auc_score(1 - test_quality['correct'], -test_quality[metric])  # low quality -> predict error
        print(f"  {metric}: point-biserial r={corr:.4f} (p={p_val:.4f}), AUC(predicts error)={auc:.4f}")
        correlation_results[metric] = {"point_biserial_r": float(corr), "p_value": float(p_val), "auc": float(auc)}

    # ---- Figure: quality distributions by dataset ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ['blur_score', 'brightness', 'contrast']):
        for ds in ['APTOS', 'IDRiD', 'Messidor']:
            vals = quality_df[quality_df['dataset'] == ds][metric]
            ax.hist(vals, bins=30, alpha=0.5, label=ds, density=True)
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_image_quality_by_dataset.png'), dpi=150, bbox_inches='tight')
    plt.close()

    summary = {"dataset_quality_differences": dataset_quality_summary,
                "quality_vs_correctness": correlation_results, "n_images_processed": len(quality_df)}
    with open(os.path.join(STAGE_QUALITY_OUT, 'image_quality_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    quality_table = pd.DataFrame([
        {"Metric": m, **{f"{ds}_mean": dataset_quality_summary[m]['by_dataset'][ds]['mean'] for ds in ['APTOS','IDRiD','Messidor']},
          "ANOVA_p": dataset_quality_summary[m]['anova_p']}
        for m in ['blur_score', 'brightness', 'contrast']
    ])
    quality_table.to_csv(os.path.join(TABLES_DIR, 'table6_image_quality_by_dataset.csv'), index=False)

    print(f"\n✅ Image quality assessment saved to {STAGE_QUALITY_OUT}")
    return summary


if __name__ == "__main__":
    run_image_quality_assessment()
