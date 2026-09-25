"""
Cross-Dataset Feature Visualization (t-SNE).
Visualizes whether APTOS/IDRiD/Messidor form separable clusters in the
optimized feature space — a visual, intuitive companion to the numeric
cross-dataset generalization gap reported in Stage 7/8.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config import CSV_DIR, FIGURES_DIR, RESULTS_DIR, CNN_FEAT_DIR, VIT_FEAT_DIR, HC_FEAT_DIR, SEED
import json

TSNE_OUT = os.path.join(RESULTS_DIR, 'stage_tsne')
os.makedirs(TSNE_OUT, exist_ok=True)


def load_concatenated_features(df_subset):
    feats, labels, datasets = [], [], []
    for _, row in df_subset.iterrows():
        uid = row['unique_id']
        cnn_f = np.load(os.path.join(CNN_FEAT_DIR, f"{uid}.npy"))
        vit_f = np.load(os.path.join(VIT_FEAT_DIR, f"{uid}.npy"))
        hc_f = np.load(os.path.join(HC_FEAT_DIR, f"{uid}.npy"))
        feats.append(np.concatenate([cnn_f, vit_f, hc_f]))
        labels.append(int(row['dr_grade']))
        datasets.append(row['dataset'])
    return np.array(feats, dtype=np.float32), np.array(labels), np.array(datasets)


def run_tsne_visualization(n_samples_per_dataset=400):
    """Subsamples per dataset for tractable t-SNE runtime, stratified by DR grade where possible."""
    split_df = pd.read_csv(os.path.join(CSV_DIR, 'stage_split_manifest.csv'))

    sampled = []
    for ds in ['APTOS', 'IDRiD', 'Messidor']:
        subset = split_df[split_df['dataset'] == ds]
        n = min(n_samples_per_dataset, len(subset))
        sampled.append(subset.sample(n, random_state=SEED))
    sample_df = pd.concat(sampled, ignore_index=True)

    print(f"Running t-SNE on {len(sample_df)} sampled images "
          f"({sample_df['dataset'].value_counts().to_dict()})...")
    X, y_grade, y_dataset = load_concatenated_features(sample_df)

    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca', learning_rate='auto')
    embedding = tsne.fit_transform(X)

    # ---- Plot 1: colored by dataset (shows domain shift) ----
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {'APTOS': '#2E5395', 'IDRiD': '#C00000', 'Messidor': '#548235'}
    for ds in ['APTOS', 'IDRiD', 'Messidor']:
        mask = y_dataset == ds
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=ds, alpha=0.6, s=15, color=colors[ds])
    ax.set_title('t-SNE Projection of Fused Feature Space — Colored by Dataset Source')
    ax.legend()
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_tsne_by_dataset.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Plot 2: colored by DR grade (shows whether severity is well-separated) ----
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y_grade, cmap='RdYlGn_r', alpha=0.6, s=15)
    plt.colorbar(scatter, label='DR Grade')
    ax.set_title('t-SNE Projection of Fused Feature Space — Colored by DR Severity')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_tsne_by_dr_grade.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Quantify visual separation with a simple silhouette-style proxy: mean inter-dataset centroid distance
    from scipy.spatial.distance import pdist, squareform
    centroids = {ds: embedding[y_dataset == ds].mean(axis=0) for ds in ['APTOS', 'IDRiD', 'Messidor']}
    centroid_matrix = np.array(list(centroids.values()))
    dist_matrix = squareform(pdist(centroid_matrix))
    dataset_names = list(centroids.keys())

    print(f"\nInter-dataset centroid distances in t-SNE space:")
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"  {dataset_names[i]} <-> {dataset_names[j]}: {dist_matrix[i, j]:.2f}")

    summary = {
        "n_samples": len(sample_df),
        "n_samples_per_dataset": sample_df['dataset'].value_counts().to_dict(),
        "inter_dataset_centroid_distances": {
            f"{dataset_names[i]}_vs_{dataset_names[j]}": float(dist_matrix[i, j])
            for i in range(3) for j in range(i + 1, 3)
        },
    }
    with open(os.path.join(TSNE_OUT, 'tsne_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Saved t-SNE figures and summary")
    return embedding, y_grade, y_dataset


if __name__ == "__main__":
    run_tsne_visualization()
