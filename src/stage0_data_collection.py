"""
Stage 0 — Multi-Dataset Collection (APTOS + IDRiD + Messidor).

Builds a unified, collision-safe manifest across all three datasets. IDRiD's
train/test image folders share filenames (e.g. IDRiD_001.jpg exists in both
with different patients/labels); a unique_id column disambiguates them and
is verified to be globally unique before saving.
"""
import os
import pandas as pd
from config import DATASET_DIR, CSV_DIR


def load_aptos():
    root = os.path.join(DATASET_DIR, 'APTOS', 'aptos2019-blindness-detection')
    df = pd.read_csv(os.path.join(root, 'train.csv'))
    df['image_path'] = df['id_code'].apply(lambda x: os.path.join(root, 'train_images', f'{x}.png'))
    df = df.rename(columns={'diagnosis': 'dr_grade'})
    df['unique_id'] = df['id_code']
    df['dataset'] = 'APTOS'
    df['dme_risk'] = None
    df = df[df['image_path'].apply(os.path.exists)]  # keep only locally available files
    return df[['image_path', 'unique_id', 'dr_grade', 'dme_risk', 'dataset']]


def load_idrid():
    root = os.path.join(DATASET_DIR, 'IDRiD')
    frames = []
    splits = [
        ('a. IDRiD_Disease Grading_Training Labels.csv', 'a. Training Set', 'train'),
        ('b. IDRiD_Disease Grading_Testing Labels.csv', 'b. Testing Set', 'test'),
    ]
    for csv_name, img_folder, split_tag in splits:
        df = pd.read_csv(os.path.join(root, '2. Groundtruths', csv_name))
        df = df[['Image name', 'Retinopathy grade', 'Risk of macular edema ']].copy()
        df.columns = ['image_id', 'dr_grade', 'dme_risk']
        df['image_path'] = df['image_id'].apply(
            lambda x: os.path.join(root, '1. Original Images', img_folder, f'{x}.jpg'))
        df['unique_id'] = df['image_id'] + f'_{split_tag}'  # disambiguates train/test collision
        df['dataset'] = 'IDRiD'
        frames.append(df[['image_path', 'unique_id', 'dr_grade', 'dme_risk', 'dataset']])
    return pd.concat(frames, ignore_index=True)


def load_messidor():
    root = os.path.join(DATASET_DIR, 'Messidor')
    img_dir = os.path.join(root, 'IMAGES')
    lower_to_actual = {f.lower(): f for f in os.listdir(img_dir)}  # fixes .jpg/.JPG case mismatch

    df = pd.read_csv(os.path.join(root, 'messidor_data.csv'))
    df = df.dropna(subset=['adjudicated_dr_grade']).copy()

    def resolve_path(image_id):
        actual = lower_to_actual.get(image_id.lower())
        return os.path.join(img_dir, actual) if actual else None

    df['image_path'] = df['image_id'].apply(resolve_path)
    df = df.dropna(subset=['image_path'])
    df = df.rename(columns={'adjudicated_dr_grade': 'dr_grade', 'adjudicated_dme': 'dme_risk'})
    df['unique_id'] = df['image_id'].apply(lambda x: os.path.splitext(x)[0])
    df['dataset'] = 'Messidor'
    return df[['image_path', 'unique_id', 'dr_grade', 'dme_risk', 'dataset']]


def build_unified_manifest():
    aptos_df, idrid_df, messidor_df = load_aptos(), load_idrid(), load_messidor()
    unified = pd.concat([aptos_df, idrid_df, messidor_df], ignore_index=True)

    assert len(unified) == unified['unique_id'].nunique(), "Duplicate unique_id detected — collision not resolved!"
    unified['exists'] = unified['image_path'].apply(os.path.exists)
    assert unified['exists'].all(), "Some referenced image files are missing on disk!"
    unified = unified.drop(columns=['exists'])

    out_path = os.path.join(CSV_DIR, 'stage0_data_manifest.csv')
    unified.to_csv(out_path, index=False)
    print(f"Stage 0 complete: {len(unified)} images "
          f"(APTOS={len(aptos_df)}, IDRiD={len(idrid_df)}, Messidor={len(messidor_df)})")
    print(f"Saved to {out_path}")
    return unified


if __name__ == "__main__":
    build_unified_manifest()
