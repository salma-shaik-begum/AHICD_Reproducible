"""
Stage 5 — Explainable AI (SHAP + Grad-CAM).
SHAP explains the Stage 4 XGBoost component on the optimized feature set.
Grad-CAM requires a faithfully end-to-end classifier, so a separate
EfficientNet-B0 is fine-tuned on raw enhanced images specifically for this stage.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, cohen_kappa_score
from tqdm import tqdm
from config import CSV_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR, IMAGE_SIZE, CNN_BACKBONE, SEED

STAGE5_OUT = os.path.join(RESULTS_DIR, 'stage5_explainability')
os.makedirs(STAGE5_OUT, exist_ok=True)


# ---------------- Grad-CAM: fine-tune EfficientNet end-to-end ----------------
class DRDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df.reset_index(drop=True), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row['enhanced_path']), cv2.COLOR_BGR2RGB)
        return self.transform(img), int(row['dr_grade']), row['unique_id']


def finetune_cnn_for_gradcam(epochs=15, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))

    train_tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15),
        transforms.ColorJitter(0.15, 0.15), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(DRDataset(split_df[split_df.split == 'train'], train_tf), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DRDataset(split_df[split_df.split == 'val'], eval_tf), batch_size=batch_size)

    model = timm.create_model(CNN_BACKBONE, pretrained=True, num_classes=5).to(device)
    class_counts = split_df[split_df.split == 'train']['dr_grade'].value_counts().sort_index().values
    class_weights = torch.tensor([len(split_df[split_df.split == 'train']) / (5 * c) for c in class_counts],
                                   dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_f1, best_path = 0.0, os.path.join(MODELS_DIR, 'finetuned_efficientnet_b0.pt')

    for epoch in range(epochs):
        model.train()
        for imgs, labels, _ in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
                trues.extend(labels.numpy())
        val_f1 = f1_score(trues, preds, average='macro')
        print(f"Epoch {epoch+1}: val macro-F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
    print(f"Best val macro-F1: {best_val_f1:.4f}, saved to {best_path}")
    return best_path


def generate_gradcam(model, device, img_tensor, target_class, target_layer, img_bgr_for_mask):
    activations, gradients = {}, {}
    fh = target_layer.register_forward_hook(lambda m, i, o: activations.update(value=o))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.update(value=go[0]))

    img_tensor = img_tensor.unsqueeze(0).to(device)
    img_tensor.requires_grad_()
    output = model(img_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    acts, grads = activations['value'][0], gradients['value'][0]
    weights = grads.mean(dim=(1, 2))
    cam = sum(w * acts[i] for i, w in enumerate(weights))
    cam = F.relu(cam).detach().cpu().numpy()
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    gray = cv2.cvtColor(img_bgr_for_mask, cv2.COLOR_BGR2GRAY)
    _, fundus_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    fundus_mask = cv2.erode(fundus_mask, np.ones((9, 9), np.uint8)).astype(np.float32) / 255.0
    cam = cam * fundus_mask
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    fh.remove(); bh.remove()
    return cam


def run_shap_analysis():
    """SHAP feature-importance analysis on the Stage 4 XGBoost ensemble member."""
    import shap
    STAGE3_OUT = os.path.join(RESULTS_DIR, 'stage3_optimized')
    data = np.load(os.path.join(STAGE3_OUT, 'optimized_features.npz'))
    X_test = data['X_test']
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))

    explainer = shap.TreeExplainer(xgb_model)
    n_sample = min(200, X_test.shape[0])
    idx = np.random.RandomState(SEED).choice(X_test.shape[0], n_sample, replace=False)
    shap_raw = explainer.shap_values(X_test[idx])

    if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        shap_values = [shap_raw[:, :, c] for c in range(shap_raw.shape[2])]
    elif isinstance(shap_raw, list):
        shap_values = shap_raw
    else:
        shap_values = [shap_raw]

    overall = np.stack([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0).mean(axis=0)
    return overall, shap_values


if __name__ == "__main__":
    finetune_cnn_for_gradcam()
    run_shap_analysis()
