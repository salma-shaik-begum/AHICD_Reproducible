# AHICF — A Generalizable Hybrid Deep Learning Framework for Automated Diabetic Retinopathy Grading and Clinical Decision Support

**AHICF (Adaptive Hybrid Intelligent Clinical Framework)** is a reproducibility package accompanying the manuscript **“A Generalizable Hybrid Deep Learning Framework for Automated Diabetic Retinopathy Grading and Clinical Decision Support”**, currently under journal review.

This repository contains the source code, precomputed outputs, figures, tables, CSV manifests, JSON result summaries, and environment definitions required to inspect and reproduce the reported analyses.

> **Reproducibility note:** This repository contains the reconstructed and verified reproducibility package corresponding to the manuscript. Environment and configuration files have been aligned with the documented execution workflow. See the **Known Issues and Reproducibility Notes** section before attempting an exact numerical reproduction.

---

## 1. Overview

Diabetic retinopathy (DR) is a major cause of preventable vision loss worldwide. Automated DR grading systems can experience reduced generalization across heterogeneous clinical datasets because of domain shift, imaging variability, and class imbalance.

AHICF is a hybrid deep learning framework designed for automated DR grading and clinical decision support. The framework combines:

* CNN-based retinal feature extraction
* Vision Transformer (ViT) representations
* Clinical and handcrafted image features
* Adaptive multi-objective feature optimization using **NSGA-II**
* Confidence-weighted ensemble classification
* Supervised domain adaptation using limited target-domain samples
* Explainable AI using Grad-CAM and SHAP
* Clinical referral decision support
* Cross-dataset and out-of-distribution evaluation
* Statistical and causal class-balance analyses

The framework is evaluated using three publicly available retinal image datasets:

* **APTOS 2019 Blindness Detection**
* **IDRiD**
* **Messidor**

The evaluation includes in-distribution testing, cross-dataset evaluation, leave-one-dataset-out validation, domain adaptation, ablation analysis, statistical validation, image-quality assessment, calibration, and class-balance analysis.

---

## 2. Repository Contents

```text
AHICD_Reproducible/
│
├── src/                    Pipeline source code (Stages 0–9 + auxiliary analyses)
│
├── notebooks/              Original Colab development notebook (provenance only)
│
├── csv/                    Manifests and per-stage tabular outputs
│
├── figures/                Generated figures used in the manuscript
│
├── results/                Per-stage JSON/NPZ result summaries and metrics
│
├── requirements.txt        Python dependencies
│
├── Dockerfile              CPU-oriented container environment
│
├── .dockerignore           Docker build exclusions
│
├── .gitignore              Git exclusions
│
├── LICENSE                 Source-code license
│
├── CITATION.cff            Citation metadata
│
└── README.md               This file
```

The `csv/`, `figures/`, and `results/` directories contain **precomputed outputs generated during the manuscript analysis**. These files allow users to inspect the reported results without rerunning the complete computational pipeline.

The `src/` directory contains the corresponding source code required to regenerate the analyses.

> **Important:** Raw retinal image datasets are not redistributed in this repository because they are subject to their respective dataset licenses and distribution conditions.

---

# 3. AHICF Pipeline

The main pipeline is orchestrated by:

```text
src/main.py
```

The pipeline is organized into independently checkpointed stages. Each stage reads its required inputs from disk and writes outputs that can be consumed by subsequent stages.

| Stage        | Description                                                                                           | Main pipeline |
| ------------ | ----------------------------------------------------------------------------------------------------- | ------------- |
| **0**        | Multi-dataset collection and collision-safe manifest construction                                     | Yes           |
| **1**        | Intelligent image enhancement                                                                         | Yes           |
| **2**        | Adaptive hybrid feature learning using CNN + ViT + handcrafted features                               | Yes           |
| **2b**       | Optional RETFound retina-foundation-model feature branch                                              | Manual        |
| **3**        | Adaptive multi-objective feature optimization using NSGA-II                                           | Yes           |
| **4**        | Adaptive intelligent clinical decision engine using a 5-model ensemble                                | Yes           |
| **5**        | Explainable AI using Grad-CAM and SHAP                                                                | Yes           |
| **6**        | Clinical decision support and DR referral logic                                                       | Yes           |
| **6b**       | Auxiliary DME risk classifier and confound-isolated referral analysis                                 | Auxiliary     |
| **7**        | Cross-dataset validation, leave-one-dataset-out evaluation, k-fold CV, and domain-adaptation analysis | Yes           |
| **7d**       | Causal class-balance analysis                                                                         | Manual        |
| **8**        | Statistical validation, bootstrap confidence intervals, and significance testing                      | Yes           |
| **9**        | Benchmark comparison against published baselines                                                      | Manual        |
| **cal**      | Model calibration, including ECE and reliability analysis                                             | Auxiliary     |
| **ablation** | Feature-source contribution and ablation analysis                                                     | Auxiliary     |

Additional standalone analyses are available in `src/` but are not called automatically by `main.py`:

```text
stage_error_analysis.py
stage_image_quality.py
stage_ood_detection.py
stage_statistical_tests.py
stage_tsne_visualization.py
```

These scripts can be executed independently when their corresponding analyses need to be regenerated.

---

# 4. Running the Pipeline

## Run the core pipeline

From the repository root:

```bash
cd src

python main.py --stages 0 1 2 3 4 5 6 7 8
```

## Run auxiliary analyses

```bash
python main.py --stages 6b cal ablation
```

## Run individual standalone analyses

For example:

```bash
python stage_error_analysis.py
python stage_image_quality.py
python stage_ood_detection.py
python stage_statistical_tests.py
python stage_tsne_visualization.py
```

The exact output locations are controlled through the project configuration.

---

# 5. Environment Setup

## Option A — Docker

Docker provides a reproducible CPU-oriented environment.

Build the image:

```bash
docker build -t ahicf:latest .
```

Run the core pipeline:

```bash
docker run --rm -it \
  -e AHICF_BASE_DIR=/app/data \
  -v "$(pwd)/Dataset:/app/data/Dataset" \
  -v "$(pwd)/outputs:/app/data/AHICF_Project" \
  ahicf:latest --stages 0 1 2 3 4 5 6 7 8
```

The container uses CPU-only PyTorch for portability.

If a CUDA-enabled GPU is available, install a compatible CUDA build of PyTorch and torchvision and run the container with the appropriate GPU configuration, for example:

```bash
docker run --gpus all ...
```

The appropriate PyTorch installation should be selected according to the CUDA version supported by the target system.

---

## Option B — Local Virtual Environment

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it.

### Linux/macOS

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
cd src

python main.py --stages 0 1 2 3 4 5 6 7 8
```

The development workflow was tested with Python **3.10–3.12**.

---

# 6. Dataset Availability and Organization

The raw retinal image datasets used in the study are **not redistributed in this repository**. They must be obtained separately from their original sources and placed under a local `Dataset/` directory.

The expected directory organization is:

```text
Dataset/
│
├── APTOS/
│   └── aptos2019-blindness-detection/
│       ├── train.csv
│       └── train_images/
│
├── IDRiD/
│   ├── 2. Groundtruths/
│   └── 1. Original Images/
│
└── Messidor/
    ├── messidor_data.csv
    └── IMAGES/
```

### APTOS 2019 Blindness Detection

The APTOS dataset was obtained from the Kaggle APTOS 2019 Blindness Detection competition.

### IDRiD

The **Indian Diabetic Retinopathy Image Dataset (IDRiD)** is distributed through IEEE DataPort.

### Messidor

The Messidor dataset is distributed through the Messidor/Messidor-2 consortium and associated research-access mechanisms.

Users should obtain each dataset from its original source and comply with the corresponding dataset terms and conditions.

---

# 7. Data and Output Configuration

By default, the pipeline expects the dataset under:

```text
<repo_root>/Dataset
```

and writes analysis outputs under the repository/project output directories, including:

```text
csv/
figures/
results/
tables/
models/
outputs/
```

The base directories can be configured using:

```text
AHICF_BASE_DIR
AHICF_DATASET_DIR
```

For example:

```bash
export AHICF_BASE_DIR=/path/to/AHICF
export AHICF_DATASET_DIR=/path/to/Dataset
```

On Windows PowerShell:

```powershell
$env:AHICF_BASE_DIR="C:\path\to\AHICF"
$env:AHICF_DATASET_DIR="C:\path\to\Dataset"
```

The corresponding configuration is implemented in:

```text
src/config.py
```

---

# 8. RETFound Feature Branch

Stage **2b** provides an optional feature-extraction branch using **RETFound**, a retina-specific foundation model.

This stage:

* is not currently called automatically by `main.py`;
* requires access to the pretrained model weights;
* may download approximately 1.2 GB of pretrained weights at runtime;
* requires internet access when the weights are not already available locally.

If the RETFound weights cannot be obtained, the implementation may fall back to an ImageNet-pretrained ViT-Large model with a console warning.

> **Important:** Results generated using the fallback model must **not** be reported as RETFound results. Users reproducing the RETFound experiments should verify that the intended pretrained weights were successfully loaded before interpreting or reporting the results.

---

# 9. Key Results

The following values correspond to the manuscript analysis and are also available in the precomputed files under `results/`.

## In-distribution ensemble performance

The reported ensemble performance is:

* **Macro-F1:** 0.524
  95% CI: 0.454–0.584

* **Quadratic-weighted kappa (QWK):** 0.762
  95% CI: 0.709–0.807

Five-fold cross-validation produced:

* **Macro-F1:** 0.524 ± 0.016
* **QWK:** 0.770 ± 0.007

Relevant result files include:

```text
results/stage8_statistical_validation_summary.json
results/stage7b_kfold_cv_summary.json
```

---

## Cross-dataset generalization

Leave-one-dataset-out holdout evaluation produced the following macro-F1 values:

| Held-out dataset | Macro-F1 |
| ---------------- | -------: |
| APTOS            |    0.412 |
| IDRiD            |    0.391 |
| Messidor         |    0.297 |

The corresponding result files are available under:

```text
results/stage7_holdout_*.json
```

---

## Clinical referral decision support

The reported referral decision-support configuration produced:

* **Sensitivity:** 87.4%
* **Specificity:** 83.3%

The corresponding summary is available at:

```text
results/stage6_referral_summary.json
```

---

## Model calibration

The reported Expected Calibration Error (ECE) is:

```text
ECE = 0.019
```

The corresponding calibration summary is available at:

```text
results/stage_calibration/calibration_summary.json
```

---

# 10. Precomputed Results

The repository includes precomputed outputs generated during the manuscript analysis.

These include:

```text
csv/
    Dataset manifests
    Stage-wise tabular outputs
    Analysis CSV files

figures/
    Manuscript figures
    Evaluation plots
    Calibration plots
    Explainability visualizations

results/
    JSON summaries
    NPZ feature/results files
    Evaluation metrics
    Statistical analysis outputs
    Cross-dataset results
```

Therefore, users do **not** need to rerun the complete pipeline simply to inspect the reported results.

---

# 11. Reproducibility and Random Seed

The primary stochastic operations use:

```text
SEED = 42
```

The seed is applied to relevant stochastic operations, including:

* dataset splitting;
* model initialization where applicable;
* stochastic training procedures;
* bootstrap resampling;
* other randomized analysis components.

Exact numerical reproduction may still vary across hardware, operating-system, Python, PyTorch, torchvision, timm, and other dependency versions.

---

# 12. Feature Optimization

Stage 3 performs adaptive multi-objective feature optimization using the **NSGA-II** algorithm.

The optimized feature representation is saved as:

```text
results/stage3_optimized/optimized_features.npz
```

This file is consumed by downstream stages, including the clinical decision engine.

---

# 13. Ensemble Clinical Decision Engine

Stage 4 uses a confidence-weighted ensemble consisting of five classifiers:

1. XGBoost
2. Random Forest
3. Support Vector Machine (SVM)
4. Multi-Layer Perceptron (MLP)
5. Logistic Regression

The ensemble is used as part of the adaptive intelligent clinical decision pipeline for DR grading.

---

# 14. Explainable AI

Stage 5 provides model interpretability through:

* **Grad-CAM** for visual localization of image regions contributing to predictions;
* **SHAP** for feature-level attribution.

These analyses are intended to support inspection of model behavior rather than to constitute independent clinical validation.

---

# 15. Domain Adaptation and Cross-Dataset Evaluation

Stage 7 evaluates generalization under dataset shift using:

* leave-one-dataset-out evaluation;
* k-fold cross-validation;
* cross-dataset testing;
* supervised domain adaptation using limited target-domain samples.

The purpose is to examine how model performance changes when the target retinal-image distribution differs from the source training distribution.

The reported cross-dataset results should therefore be interpreted in the context of the datasets, preprocessing procedures, target-sample availability, and evaluation protocol used in the study.

---

# 16. Clinical Decision Support

Stage 6 implements DR referral decision logic based on model outputs.

An auxiliary Stage 6b analysis additionally evaluates:

* DME risk classification;
* confound-isolated referral behavior.

The clinical decision-support outputs are intended as computational research results and **not as a standalone clinical diagnostic system**.

Further validation using large, independent clinical cohorts is required before clinical deployment.

---

# 17. Limitations and Reproducibility Scope

Several limitations should be considered when reproducing or extending this work:

* The evaluation uses publicly available retinal image datasets rather than a newly collected multi-center clinical cohort.
* Dataset heterogeneity introduces domain shift across imaging sources.
* Class imbalance remains an important factor in DR grading performance.
* Cross-dataset performance can be substantially lower than in-distribution performance.
* Domain adaptation improves target-domain behavior under the evaluated setting but does not establish universal domain robustness.
* The available target-domain adaptation samples are limited.
* The clinical referral analysis is computational and does not constitute prospective clinical validation.
* Further validation on larger and independently collected clinical cohorts is required.

Accordingly, the results demonstrate **measurable domain-adaptation behavior and cross-dataset generalization under the evaluated experimental conditions**, rather than complete robustness across all clinical settings.

---

# 18. Known Issues and Reproducibility Notes

### 18.1 Configuration paths

Earlier development versions of the project used a hard-coded Google Drive path:

```text
/content/drive/MyDrive/AHICD
```

The reproducibility package uses configurable base and dataset directories through:

```text
AHICF_BASE_DIR
AHICF_DATASET_DIR
```

Users should verify the values in:

```text
src/config.py
```

before running the pipeline.

---

### 18.2 Dependency versions

The supplied `requirements.txt` describes the dependencies required by the reconstructed pipeline.

It is **not a pip-freeze of the exact environment used to generate every manuscript result**.

For exact numerical reproduction, differences in versions of packages such as:

```text
torch
torchvision
timm
scikit-learn
xgboost
```

may produce small numerical differences.

Where exact historical reproduction is required, the original training environment should be reconstructed as closely as possible.

---

### 18.3 Pipeline coverage

Not every analysis script in `src/` is automatically invoked by `main.py`.

The following analyses require manual execution:

```text
Stage 2b — RETFound feature branch
Stage 7d — Causal class-balance analysis
Stage 9  — Benchmark comparison
stage_error_analysis.py
stage_image_quality.py
stage_ood_detection.py
stage_statistical_tests.py
stage_tsne_visualization.py
```

This is intentional: these analyses are maintained as auxiliary or standalone components rather than being forced into the default core execution path.

---

### 18.4 RETFound fallback

The Stage 2b implementation may fall back to an ImageNet-pretrained ViT-Large model if RETFound weights are unavailable.

Researchers reproducing RETFound experiments must verify the loaded architecture and pretrained weights before interpreting the results.

Fallback outputs should not be labeled or reported as RETFound results.

---

# 19. Provenance Notebook

The repository contains the original Google Colab development notebook for provenance:

```text
notebooks/Untitled0 (1).ipynb
```

The notebook contains development references to Google Drive paths and is **not required for the primary reproduction workflow**.

For reproducibility, use:

```text
src/
requirements.txt
Dockerfile
```

rather than relying on the original notebook environment.

---

# 20. Recommended Reproduction Workflow

A recommended reproduction sequence is:

```text
1. Obtain the APTOS, IDRiD, and Messidor datasets
                ↓
2. Place datasets under Dataset/
                ↓
3. Configure AHICF_BASE_DIR and AHICF_DATASET_DIR
                ↓
4. Install dependencies or build the Docker image
                ↓
5. Run Stage 0
                ↓
6. Run Stages 1–4
                ↓
7. Run Stage 5 explainability analyses
                ↓
8. Run Stage 6 clinical decision-support analysis
                ↓
9. Run Stage 7 cross-dataset/domain-adaptation evaluation
                ↓
10. Run Stage 8 statistical validation
                ↓
11. Run auxiliary analyses as required
                ↓
12. Compare regenerated outputs with results/
```

For users interested only in inspecting the manuscript results, the precomputed files in:

```text
csv/
figures/
results/
```

can be used without rerunning the complete pipeline.

---

# 21. Citation

If you use the AHICF source code, analysis pipeline, or reproducibility package, please cite the associated manuscript.

Citation metadata are provided in:

```text
CITATION.cff
```

The citation metadata should be updated with the final author list, journal information, DOI, and publication details once the manuscript is formally published.

---

# 22. License

The source code in this repository is released under the **MIT License**.

See:

```text
LICENSE
```

for the complete license text.

The raw retinal datasets are **not covered by the repository's MIT License**. Each dataset remains subject to its own license, access conditions, and terms of use.

---

# 23. Disclaimer

This repository is provided for **research and reproducibility purposes**.

The AHICF models and clinical decision-support components are not intended to replace professional ophthalmic examination, diagnosis, or clinical judgment.

The reported experiments demonstrate performance under the datasets and experimental protocols described in the associated manuscript. Additional validation on independent, larger, and clinically representative cohorts is required before considering deployment in clinical practice.

---

## 24. Repository Summary

**Framework:** AHICF — Adaptive Hybrid Intelligent Clinical Framework
**Task:** Automated diabetic retinopathy grading and clinical decision support
**Datasets:** APTOS, IDRiD, Messidor
**Feature learning:** CNN + Vision Transformer + handcrafted features
**Feature optimization:** NSGA-II multi-objective optimization
**Classifier:** Confidence-weighted ensemble
**Domain adaptation:** Supervised target-domain adaptation
**Explainability:** Grad-CAM + SHAP
**Evaluation:** In-distribution, cross-dataset, leave-one-dataset-out, ablation, calibration, statistical, OOD, and class-balance analyses
**Primary seed:** 42
**Environment:** Python 3.10–3.12; Docker-supported
**License:** MIT for source code
**Raw datasets:** Not redistributed
