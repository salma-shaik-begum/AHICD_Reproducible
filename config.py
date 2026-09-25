"""
AHICF — Shared configuration

Central configuration for paths, constants, model settings, and
reproducibility parameters used across the AHICF pipeline.

The project root and dataset directory can be configured using:

```
AHICF_BASE_DIR
AHICF_DATASET_DIR
```

If these environment variables are not supplied, the configuration
falls back to the historical Google Colab project location used during
development.
"""

import os

# ---------------------------------------------------------------------

# Base paths

# ---------------------------------------------------------------------

# Historical development location used in Google Colab.

DEFAULT_BASE_DIR = "/content/drive/MyDrive/AHICD"

# Allow Docker/local execution to override the historical Colab path.

BASE_DIR = os.environ.get("AHICF_BASE_DIR", DEFAULT_BASE_DIR)

# Dataset location can be specified independently.

DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

DATASET_DIR = os.environ.get(
"AHICF_DATASET_DIR",
DEFAULT_DATASET_DIR
)

# ---------------------------------------------------------------------

# Project output directories

# ---------------------------------------------------------------------

PROJECT_DIR = os.path.join(BASE_DIR, "AHICF_Project")

CSV_DIR = os.path.join(PROJECT_DIR, "csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TABLES_DIR = os.path.join(PROJECT_DIR, "tables")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

# ---------------------------------------------------------------------

# Intermediate processing directories

# ---------------------------------------------------------------------

PROC_DIR = os.path.join(BASE_DIR, "Processed")

ENHANCED_DIR = os.path.join(
PROC_DIR,
"stage1_enhanced_fixed"
)

CNN_FEAT_DIR = os.path.join(
PROC_DIR,
"stage2_features",
"cnn_fixed"
)

VIT_FEAT_DIR = os.path.join(
PROC_DIR,
"stage2_features",
"vit_fixed"
)

HC_FEAT_DIR = os.path.join(
PROC_DIR,
"stage2_features",
"handcrafted_fixed"
)

# ---------------------------------------------------------------------

# Reproducibility

# ---------------------------------------------------------------------

SEED = 42

# ---------------------------------------------------------------------

# Image and dataset configuration

# ---------------------------------------------------------------------

IMAGE_SIZE = 224

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ---------------------------------------------------------------------

# Feature extraction backbones

# ---------------------------------------------------------------------

CNN_BACKBONE = "efficientnet_b0"

VIT_BACKBONE = "deit_small_patch16_224"

# ---------------------------------------------------------------------

# Diabetic retinopathy grade labels

# ---------------------------------------------------------------------

DR_GRADE_NAMES = {
0: "No DR",
1: "Mild NPDR",
2: "Moderate NPDR",
3: "Severe NPDR",
4: "Proliferative DR",
}

# ---------------------------------------------------------------------

# Create required output directories

# ---------------------------------------------------------------------

for directory in [
CSV_DIR,
RESULTS_DIR,
TABLES_DIR,
FIGURES_DIR,
MODELS_DIR,
OUTPUTS_DIR,
]:
os.makedirs(directory, exist_ok=True)

# ---------------------------------------------------------------------

# Configuration summary

# ---------------------------------------------------------------------

print("AHICF configuration loaded.")
print(f"Base directory:    {BASE_DIR}")
print(f"Dataset directory: {DATASET_DIR}")
print(f"Project directory: {PROJECT_DIR}")
print(f"Random seed:       {SEED}")
