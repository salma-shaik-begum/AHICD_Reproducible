"""
Stage 2 — Adaptive Hybrid Feature Learning.
Extracts CNN (EfficientNet-B0, 1280-d), ViT (DeiT-Small, 384-d), and 33
handcrafted clinical features (GLCM texture, vessel density, optic disc
localization, color/intensity statistics) per image.
"""
import os
import numpy as np
import pandas as pd
import cv2
import torch
import timm
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
from config import CSV_DIR, CNN_FEAT_DIR, VIT_FEAT_DIR, HC_FEAT_DIR, IMAGE_SIZE, CNN_BACKBONE, VIT_BACKBONE

for d in [CNN_FEAT_DIR, VIT_FEAT_DIR, HC_FEAT_DIR]:
    os.makedirs(d, exist_ok=True)

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_handcrafted_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    glcm_feats = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        glcm_feats.extend(graycoprops(glcm, prop).flatten())

    green_blur = cv2.GaussianBlur(img_bgr[:, :, 1], (5, 5), 0)
    vessel_thresh = cv2.adaptiveThreshold(green_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 3)
    vessel_density = np.sum(vessel_thresh > 0) / vessel_thresh.size

    _, bright = cv2.threshold(gray, np.percentile(gray, 95), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        disc_area = cv2.contourArea(largest) / (gray.shape[0] * gray.shape[1])
        M = cv2.moments(largest)
        disc_cx = M['m10']/M['m00']/gray.shape[1] if M['m00'] else 0.5
        disc_cy = M['m01']/M['m00']/gray.shape[0] if M['m00'] else 0.5
    else:
        disc_area, disc_cx, disc_cy = 0.0, 0.5, 0.5

    mean_r, mean_g, mean_b = [img_bgr[:, :, c].mean()/255.0 for c in [2, 1, 0]]
    return np.array(glcm_feats + [vessel_density, disc_area, disc_cx, disc_cy,
                                    gray.mean()/255.0, gray.std()/255.0, mean_r, mean_g, mean_b],
                     dtype=np.float32)


def run_stage2(batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(os.path.join(CSV_DIR, 'stage1_enhanced_manifest.csv'))

    cnn_model = timm.create_model(CNN_BACKBONE, pretrained=True, num_classes=0).eval().to(device)
    vit_model = timm.create_model(VIT_BACKBONE, pretrained=True, num_classes=0).eval().to(device)

    remaining = df[~df['unique_id'].apply(
        lambda uid: os.path.exists(os.path.join(CNN_FEAT_DIR, f"{uid}.npy")))]

    with torch.no_grad():
        for start in tqdm(range(0, len(remaining), batch_size), desc="Extracting features"):
            batch = remaining.iloc[start:start+batch_size]
            tensors, ids, hc_feats = [], [], []
            for _, row in batch.iterrows():
                img = cv2.imread(row['enhanced_path'])
                if img is None:
                    continue
                tensors.append(PREPROCESS(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                ids.append(row['unique_id'])
                hc_feats.append(extract_handcrafted_features(img))
            if not tensors:
                continue
            batch_tensor = torch.stack(tensors).to(device)
            cnn_out = cnn_model(batch_tensor).cpu().numpy()
            vit_out = vit_model(batch_tensor).cpu().numpy()
            for i, uid in enumerate(ids):
                np.save(os.path.join(CNN_FEAT_DIR, f"{uid}.npy"), cnn_out[i])
                np.save(os.path.join(VIT_FEAT_DIR, f"{uid}.npy"), vit_out[i])
                np.save(os.path.join(HC_FEAT_DIR, f"{uid}.npy"), hc_feats[i])

    out_path = os.path.join(CSV_DIR, 'stage2_features_manifest.csv')
    df.to_csv(out_path, index=False)
    n_cnn, n_vit, n_hc = len(os.listdir(CNN_FEAT_DIR)), len(os.listdir(VIT_FEAT_DIR)), len(os.listdir(HC_FEAT_DIR))
    print(f"Stage 2 complete. Feature counts — CNN: {n_cnn}, ViT: {n_vit}, Handcrafted: {n_hc}")
    print(f"Saved manifest to {out_path}")


if __name__ == "__main__":
    run_stage2()
