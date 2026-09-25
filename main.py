"""
AHICF full pipeline orchestrator. Run stages sequentially; each stage is
independently checkpointed and safe to re-run without recomputation.

Usage:
  python main.py --stages 0 1 2 3 4 5 6 7 8
  python main.py --stages 6b cal ablation      (auxiliary/robustness stages)
"""
import argparse
import stage0_data_collection as s0
import stage1_image_enhancement as s1
import stage2_feature_extraction as s2
import stage3_feature_optimization as s3
import stage4_clinical_decision_engine as s4
import stage5_explainability as s5
import stage6_clinical_decision_support as s6
import stage6b_dme_classifier as s6b
import stage7_validation as s7
import stage8_statistical_validation as s8
import stage_calibration as scal
import stage_ablation as sabl

STAGE_FUNCS = {
    0: s0.build_unified_manifest,
    1: s1.run_stage1,
    2: s2.run_stage2,
    3: s3.run_stage3,
    4: s4.run_stage4,
    5: s5.finetune_cnn_for_gradcam,
    6: s6.run_stage6,
    "6b": lambda: (s6b.train_dme_classifier(), s6b.confound_isolated_referral_analysis()),
    7: lambda: (s7.run_leave_one_dataset_out(), s7.run_kfold_cv(), s7.run_domain_adaptation_sweep()),
    8: s8.run_stage8,
    "cal": scal.run_calibration_analysis,
    "ablation": sabl.run_ablation_study,
}

STAGE_DESCRIPTIONS = {
    0: "Multi-Dataset Collection", 1: "Intelligent Image Enhancement",
    2: "Adaptive Hybrid Feature Learning", 3: "Adaptive Multi-Objective Feature Optimization",
    4: "Adaptive Intelligent Clinical Decision Engine", 5: "Explainable AI (Grad-CAM fine-tuning)",
    6: "Clinical Decision Support (DR referral)", "6b": "Auxiliary DME Risk Classifier + Confound Analysis",
    7: "Cross-Dataset Validation (leave-one-out + k-fold CV + domain adaptation)",
    8: "Statistical Validation (bootstrapped CIs + significance tests)",
    "cal": "Model Calibration (ECE + Reliability Diagram)",
    "ablation": "Ablation Study (feature-source contribution)",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', nargs='+', default=[str(i) for i in range(9)])
    args = parser.parse_args()

    for raw_stage in args.stages:
        # Allow both int-like stages ("0","1"...) and named stages ("6b","cal","ablation")
        stage_key = int(raw_stage) if raw_stage.isdigit() else raw_stage
        print(f"\n{'='*70}\nSTAGE {stage_key}: {STAGE_DESCRIPTIONS.get(stage_key, '')}\n{'='*70}")
        STAGE_FUNCS[stage_key]()
