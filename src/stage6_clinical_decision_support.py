"""
Stage 6 — Clinical Decision Support.
Converts ensemble predictions into referral recommendations: urgent (grade>=3),
routine (grade==2), or precautionary (confidence < threshold), validated
against ground-truth "should refer" (true grade>=2) sensitivity/specificity.
"""
import os
import json
import pandas as pd
from config import CSV_DIR, RESULTS_DIR

STAGE6_OUT = os.path.join(RESULTS_DIR, 'stage6_clinical_decision_support')
os.makedirs(STAGE6_OUT, exist_ok=True)
CONFIDENCE_THRESHOLD = 0.6


def determine_referral(row, threshold=CONFIDENCE_THRESHOLD):
    grade, confidence = row['predicted_dr_grade'], row['confidence_score']
    if grade >= 3:
        return 'REFER', 'URGENT', f'Predicted grade {grade} — sight-threatening, refer promptly'
    elif grade == 2:
        return 'REFER', 'ROUTINE', f'Predicted grade {grade} — refer for ophthalmologist review'
    elif confidence < threshold:
        return 'REFER', 'PRECAUTIONARY', f'Predicted grade {grade}, low confidence ({confidence:.2f})'
    return 'NO_REFERRAL', 'ROUTINE_RESCREEN', f'Predicted grade {grade}, high confidence ({confidence:.2f})'


def validate_referral_logic(df):
    df['should_refer'] = df['true_dr_grade'] >= 2
    df['model_referred'] = df['referral_decision'] == 'REFER'
    tp = ((df.should_refer) & (df.model_referred)).sum()
    fn = ((df.should_refer) & (~df.model_referred)).sum()
    tn = ((~df.should_refer) & (~df.model_referred)).sum()
    fp = ((~df.should_refer) & (df.model_referred)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    return sensitivity, specificity, int(tp), int(fn), int(tn), int(fp)


def run_stage6():
    df = pd.read_csv(os.path.join(CSV_DIR, 'stage4_ensemble_predictions.csv'))
    results = df.apply(determine_referral, axis=1, result_type='expand')
    results.columns = ['referral_decision', 'urgency', 'reason']
    df = pd.concat([df, results], axis=1)

    sensitivity, specificity, tp, fn, tn, fp = validate_referral_logic(df)
    print(f"Referral sensitivity={sensitivity:.4f}, specificity={specificity:.4f} "
          f"(TP={tp}, FN={fn}, TN={tn}, FP={fp})")

    df.to_csv(os.path.join(CSV_DIR, 'stage6_referral_decisions.csv'), index=False)
    summary = {"confidence_threshold": CONFIDENCE_THRESHOLD, "sensitivity": sensitivity,
               "specificity": specificity, "true_positives": tp, "false_negatives": fn,
               "true_negatives": tn, "false_positives": fp}
    with open(os.path.join(STAGE6_OUT, 'stage6_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("Stage 6 complete.")


if __name__ == "__main__":
    run_stage6()
