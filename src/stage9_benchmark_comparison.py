"""
Stage 9 — Benchmark Comparison with Published State-of-the-Art Methods.

IMPORTANT METHODOLOGICAL NOTE: the entries below are literature-cited
(numbers as reported in the original papers), NOT independently reproduced
under our own pipeline/splits. This is disclosed explicitly per our paper's
own thesis: standard-split reported numbers can overstate real-world
performance, so these comparisons should be read as "what is typically
reported in this literature" rather than "verified equivalent performance."
Where a paper additionally reports cross-dataset results, this is flagged
distinctly, as it is directly comparable in spirit (though not in exact
setup) to our own Stage 7 findings.

Only entries where the metric was unambiguously 5-class DR severity grading
(via quadratic weighted kappa, the standard metric for this exact task) are
included in the primary comparison, to avoid conflating with the
substantially easier binary (referable/non-referable) DR detection task
common elsewhere in the literature.
"""
import os
import json
import pandas as pd
from config import RESULTS_DIR, TABLES_DIR

STAGE9_OUT = os.path.join(RESULTS_DIR, 'stage9_benchmark_comparison')
os.makedirs(STAGE9_OUT, exist_ok=True)


# ============================================================
# Literature-cited comparison entries (5-class DR grading, QWK metric)
# ============================================================
LITERATURE_BENCHMARKS = [
    {
        "method": "ResNet50 (baseline)", "citation": "Reported in EfficientNetB0+MSAG study (Springer, 2025)",
        "dataset": "APTOS 2019", "evaluation_type": "Single-dataset, standard split (in-distribution)",
        "qwk": 0.901, "notes": "Baseline comparison within a 2025 paper; not independently reproduced here.",
    },
    {
        "method": "DenseNet121 (baseline)", "citation": "Reported in EfficientNetB0+MSAG study (Springer, 2025)",
        "dataset": "APTOS 2019", "evaluation_type": "Single-dataset, standard split (in-distribution)",
        "qwk": 0.908, "notes": "Baseline comparison within a 2025 paper; not independently reproduced here.",
    },
    {
        "method": "EfficientNetB0 + Multi-Scale Attention Gates",
        "citation": "Discover Applied Sciences (Springer Nature), 2025",
        "dataset": "APTOS 2019", "evaluation_type": "Single-dataset, standard split (in-distribution)",
        "qwk": 0.923, "notes": "QWK reported as 0.923 ± 0.008; macro-AUC 0.976. Single-dataset only, not cross-dataset validated.",
    },
    {
        "method": "Ensemble Transfer Learning (EfficientNet-based)",
        "citation": "Chilukoti, Shan, Tida, Maida, Hei — BMC Medical Informatics and Decision Making, 2024",
        "dataset": "APTOS 2019", "evaluation_type": "Single-dataset, standard split (in-distribution)",
        "qwk": 0.967, "notes": "Same study reports Messidor QWK=0.944, EyePACS QWK=0.901 (dataset-specific pretraining per target).",
    },
    {
        "method": "Ensemble Transfer Learning (EfficientNet-based)",
        "citation": "Chilukoti, Shan, Tida, Maida, Hei — BMC Medical Informatics and Decision Making, 2024",
        "dataset": "Messidor", "evaluation_type": "Single-dataset, standard split (in-distribution)",
        "qwk": 0.944, "notes": "Dataset-specific pretraining/fine-tuning per target dataset, not a unified multi-dataset model.",
    },
    {
        "method": "Dual-Branch Deep Learning Network",
        "citation": "arXiv:2308.09945 / ScienceDirect, 2024",
        "dataset": "Merged: APTOS 2019 + Messidor-2 + IDRiD", "evaluation_type": "Merged-pool, standard split (in-distribution)",
        "qwk": 0.930, "notes": ("Most methodologically comparable prior work: also trains on a merged "
                                   "APTOS+Messidor-2+IDRiD pool, as AHICF does. Accuracy also reported: 89.6%. "
                                   "Does not report cross-dataset (leave-one-out) generalization."),
    },
    {
        "method": "Dual-Resolution Attention with Ordinal Regression",
        "citation": "arXiv:2604.17341, 2026",
        "dataset": "APTOS (train) -> Messidor-2 (cross-dataset test)",
        "evaluation_type": "CROSS-DATASET (train on APTOS, test on unseen Messidor-2)",
        "qwk_indist": 0.88, "qwk_cross": 0.68,
        "notes": ("Directly relevant independent corroboration of our Stage 7 finding: this study "
                   "also reports a substantial QWK drop under cross-dataset evaluation (0.88 -> 0.68, "
                   "a 22.7% relative decrease), from a different research group and architecture, "
                   "supporting that the standard-split overstatement phenomenon documented in this "
                   "paper is not specific to our own pipeline."),
    },
]


# ============================================================
# Our own results, for direct placement alongside the above
# ============================================================
def get_ahicf_results():
    return [
        {"method": "AHICF (ours)", "citation": "This work", "dataset": "Pooled (APTOS+IDRiD+Messidor)",
          "evaluation_type": "In-distribution, single stratified split", "qwk": 0.762,
          "notes": "Macro-F1=0.524. Bootstrapped 95% CI: [0.709, 0.807]."},
        {"method": "AHICF (ours)", "citation": "This work", "dataset": "Pooled (APTOS+IDRiD+Messidor)",
          "evaluation_type": "In-distribution, 5-fold stratified CV", "qwk": 0.770,
          "notes": "Mean across 5 folds, std=0.007. Macro-F1=0.524±0.016."},
        {"method": "AHICF (ours)", "citation": "This work", "dataset": "Held-out APTOS",
          "evaluation_type": "CROSS-DATASET (leave-one-dataset-out)", "qwk": 0.658,
          "notes": "Macro-F1=0.412 [0.372, 0.447]. Significant drop vs in-distribution (p<0.05, bootstrap)."},
        {"method": "AHICF (ours)", "citation": "This work", "dataset": "Held-out IDRiD",
          "evaluation_type": "CROSS-DATASET (leave-one-dataset-out)", "qwk": 0.707,
          "notes": "Macro-F1=0.391 [0.352, 0.429]. Significant drop vs in-distribution (p<0.05, bootstrap)."},
        {"method": "AHICF (ours)", "citation": "This work", "dataset": "Held-out Messidor",
          "evaluation_type": "CROSS-DATASET (leave-one-dataset-out)", "qwk": 0.372,
          "notes": "Macro-F1=0.297 [0.255, 0.337]. Largest drop; attributed to grade-4 scarcity (2.0% prevalence)."},
    ]


def run_stage9_comparison():
    ahicf_results = get_ahicf_results()

    all_rows = []
    for entry in LITERATURE_BENCHMARKS:
        row = {
            "Method": entry["method"], "Citation": entry["citation"], "Dataset": entry["dataset"],
            "Evaluation_Type": entry["evaluation_type"],
            "QWK": entry.get("qwk", entry.get("qwk_indist")),
            "QWK_cross_dataset": entry.get("qwk_cross", None),
            "Source": "Literature-cited (not independently reproduced)", "Notes": entry["notes"],
        }
        all_rows.append(row)
    for entry in ahicf_results:
        all_rows.append({
            "Method": entry["method"], "Citation": entry["citation"], "Dataset": entry["dataset"],
            "Evaluation_Type": entry["evaluation_type"], "QWK": entry["qwk"], "QWK_cross_dataset": None,
            "Source": "This work (independently computed, bootstrapped CIs available)", "Notes": entry["notes"],
        })

    comparison_df = pd.DataFrame(all_rows)
    comparison_df.to_csv(os.path.join(TABLES_DIR, 'table8_benchmark_comparison.csv'), index=False)

    print("=" * 100)
    print("STAGE 9: BENCHMARK COMPARISON")
    print("=" * 100)
    print(comparison_df.to_string(index=False))

    # ---- Honest summary statistics ----
    indist_literature_qwks = [e["qwk"] for e in LITERATURE_BENCHMARKS if "qwk" in e]
    print(f"\n--- Honest positioning summary ---")
    print(f"Published in-distribution QWK range (literature): {min(indist_literature_qwks):.3f} - {max(indist_literature_qwks):.3f}")
    print(f"AHICF in-distribution QWK: 0.762 (below this range)")
    print(f"\nMost methodologically comparable prior work (merged 3-dataset pool, Dual-Branch Network 2024): QWK=0.930")
    print(f"AHICF equivalent setting: QWK=0.762 (0.168 below)")
    print(f"\nKey corroborating evidence: independent cross-dataset study (arXiv:2604.17341) shows the SAME")
    print(f"phenomenon we document — QWK drop from 0.88 (in-dist) to 0.68 (cross-dataset), a 22.7% relative decrease.")
    print(f"AHICF's Messidor cross-dataset drop: 0.762 -> 0.372, a 51.2% relative decrease (more severe, "
          f"consistent with our identified grade-4-scarcity mechanism being particularly acute for Messidor).")

    summary = {
        "literature_indist_qwk_range": [min(indist_literature_qwks), max(indist_literature_qwks)],
        "ahicf_indist_qwk": 0.762,
        "most_comparable_prior_work": {"method": "Dual-Branch Network (merged 3-dataset pool)", "qwk": 0.930},
        "ahicf_equivalent_qwk": 0.762,
        "independent_cross_dataset_corroboration": {
            "source": "arXiv:2604.17341 (2026)", "qwk_indist": 0.88, "qwk_cross": 0.68,
            "relative_decrease_pct": round((0.88 - 0.68) / 0.88 * 100, 1),
        },
        "ahicf_messidor_cross_dataset": {
            "qwk_indist": 0.762, "qwk_cross": 0.372,
            "relative_decrease_pct": round((0.762 - 0.372) / 0.762 * 100, 1),
        },
        "honest_conclusion": (
            "AHICF's absolute in-distribution performance (QWK=0.762) is below the typical range reported "
            "in published single-dataset DR grading literature (0.90-0.97), reflecting this study's "
            "prioritization of computational tractability, multi-objective feature optimization, and "
            "rigorous generalization testing over accuracy-maximizing architecture search. The primary "
            "contribution of this work is not state-of-the-art classification accuracy, but rather the "
            "rigorous quantification of the cross-dataset generalization gap and its mechanistic "
            "explanation (severity-class imbalance) -- a phenomenon independently corroborated by at "
            "least one other 2026 study using a different architecture and dataset pairing."
        ),
    }
    with open(os.path.join(STAGE9_OUT, 'stage9_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Stage 9 benchmark comparison saved to {STAGE9_OUT}")
    print(f"✅ Table saved to /tables/table8_benchmark_comparison.csv")
    return comparison_df, summary


if __name__ == "__main__":
    run_stage9_comparison()
