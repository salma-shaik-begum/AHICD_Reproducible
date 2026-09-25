"""
Stage 2b — RETFound Feature Extraction (retina-specific foundation model).
Adds a fourth feature branch alongside CNN/ViT/Handcrafted, using RETFound
(ViT-Large, MAE-pretrained specifically on retinal fundus images) rather
than generic ImageNet-pretrained backbones — hypothesis: domain-specific
pretraining may improve cross-dataset generalization (Stage 7) relative to
ImageNet-pretrained CNN/ViT.

IMPORTANT: requires internet access to download RETFound weights (~1.2GB)
and is computationally heavy on CPU (ViT-Large). Checkpointed per-image
batch; safe to interrupt and resume.
"""
import os
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms
import cv2
from tqdm import tqdm
from config import CSV_DIR, PROC_DIR, IMAGE_SIZE

RETFOUND_FEAT_DIR = os.path.join(PROC_DIR, 'stage2_features', 'retfound')
os.makedirs(RETFOUND_FEAT_DIR, exist_ok=True)

RETFOUND_INPUT_SIZE = 224  # RETFound's native input resolution


def load_retfound_model(device):
    """
    Attempts to load RETFound weights via Hugging Face Hub. If this fails
    (repo path may have changed, or requires authentication), falls back
    to a plain ViT-Large architecture with ImageNet weights and prints a
    clear warning — results in that case should NOT be reported as RETFound.
    """
    try:
        from huggingface_hub import hf_hub_download
        # NOTE: verify this repo/filename is current before relying on results —
        # RETFound weight distribution may have moved since this code was written.
        weight_path = hf_hub_download(repo_id="YukunZhou/RETFound_mae_natureCFP", filename="RETFound_mae_natureCFP.pth")
        model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0)
        state_dict = torch.load(weight_path, map_location=device)
        state_dict = state_dict.get('model', state_dict)  # some checkpoints nest under 'model'
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded RETFound weights (missing keys: {len(missing)}, unexpected: {len(unexpected)})")
        is_real_retfound = True
    except Exception as e:
        print(f"⚠️ Could not load RETFound weights ({e})")
        print("⚠️ Falling back to ImageNet-pretrained ViT-Large — NOT true RETFound. "
              "Manually verify the weight source and retry before trusting downstream results.")
        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=0)
        is_real_retfound = False

    return model.eval().to(device), is_real_retfound


def run_retfound_extraction(batch_size=8):
    """Smaller batch size than Stage 2's CNN/ViT extraction — ViT-Large is far heavier per-image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("⚠️ CPU detected. ViT-Large feature extraction for 3740 images will be slow — "
              "budget significant time and rely on checkpointing to resume across sessions.")

    model, is_real_retfound = load_retfound_model(device)
    if not is_real_retfound:
        print("Stopping — fix weight loading before proceeding, to avoid mislabeling ImageNet-ViT-Large as RETFound.")
        return None

    preprocess = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((RETFOUND_INPUT_SIZE, RETFOUND_INPUT_SIZE)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))
    remaining = df[~df['unique_id'].apply(lambda uid: os.path.exists(os.path.join(RETFOUND_FEAT_DIR, f"{uid}.npy")))]
    print(f"Extracting RETFound features for {len(remaining)} remaining images...")

    with torch.no_grad():
        for start in tqdm(range(0, len(remaining), batch_size), desc="RETFound extraction"):
            batch = remaining.iloc[start:start + batch_size]
            tensors, ids = [], []
            for _, row in batch.iterrows():
                img = cv2.imread(row['enhanced_path'])
                if img is None:
                    continue
                tensors.append(preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                ids.append(row['unique_id'])
            if not tensors:
                continue
            batch_tensor = torch.stack(tensors).to(device)
            features = model(batch_tensor).cpu().numpy()
            for i, uid in enumerate(ids):
                np.save(os.path.join(RETFOUND_FEAT_DIR, f"{uid}.npy"), features[i])

    n_done = len(os.listdir(RETFOUND_FEAT_DIR))
    print(f"✅ RETFound extraction complete: {n_done}/{len(df)} images processed")
    print(f"Feature dimension: {features.shape[1] if 'features' in dir() else 1024} (ViT-Large)")


if __name__ == "__main__":
    run_retfound_extraction()
