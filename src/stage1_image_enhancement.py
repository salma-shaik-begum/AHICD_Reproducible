"""
Stage 1 — Intelligent Image Enhancement.
Crops the circular fundus region, applies CLAHE contrast enhancement, and
resizes to a uniform resolution. Checkpointed so re-runs skip completed images.
"""
import os
import cv2
import pandas as pd
from tqdm import tqdm
from config import CSV_DIR, ENHANCED_DIR, IMAGE_SIZE

CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID = (8, 8)
CROP_MIN_FRACTION = 0.3


def crop_to_fundus_circle(img, min_fraction=CROP_MIN_FRACTION):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < img.shape[1] * min_fraction or h < img.shape[0] * min_fraction:
        return img
    return img[y:y+h, x:x+w]


def apply_clahe(img, clip_limit=CLAHE_CLIP_LIMIT, tile_grid=CLAHE_TILE_GRID):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)


def enhance_image(img_path, target_size=IMAGE_SIZE):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = crop_to_fundus_circle(img)
    img = apply_clahe(img)
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)


def run_stage1():
    df = pd.read_csv(os.path.join(CSV_DIR, 'stage0_data_manifest.csv'))
    for ds in df['dataset'].unique():
        os.makedirs(os.path.join(ENHANCED_DIR, ds), exist_ok=True)

    enhanced_paths, failed = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Enhancing images"):
        dst = os.path.join(ENHANCED_DIR, row['dataset'], f"{row['unique_id']}.png")
        if os.path.exists(dst):
            enhanced_paths.append(dst)
            continue
        enhanced = enhance_image(row['image_path'])
        if enhanced is None:
            failed.append(row['image_path'])
            enhanced_paths.append(None)
            continue
        cv2.imwrite(dst, enhanced)
        enhanced_paths.append(dst)

    df['enhanced_path'] = enhanced_paths
    out_df = df.dropna(subset=['enhanced_path']).reset_index(drop=True)
    out_path = os.path.join(CSV_DIR, 'stage1_enhanced_manifest.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Stage 1 complete: {len(out_df)}/{len(df)} images enhanced ({len(failed)} failed)")
    print(f"Saved to {out_path}")
    return out_df


if __name__ == "__main__":
    run_stage1()
