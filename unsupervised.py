"""
Unsupervised learning module for SolarSense.

Techniques:
- K-Means clustering
- Hierarchical (Agglomerative) clustering
- DBSCAN
- PCA
- t-SNE
- UMAP (optional dependency: umap-learn)
- Isolation Forest anomaly detection

The models intentionally exclude FAULT_FLAG, PLANT_ID and SOURCE_KEY from
the feature matrix. These are labels/identifiers, not unsupervised features.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "static" / "unsupervised_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Physical/operational variables available in SolarSense.
PREFERRED_FEATURES = [
    "DC_POWER",
    "AC_POWER",
    "DAILY_YIELD",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "AC_DC_EFFICIENCY",
]

CACHE = {}


def _prepare(df):
    """Clean and standardize unsupervised-learning features."""
    features = [c for c in PREFERRED_FEATURES if c in df.columns]
    if len(features) < 3:
        raise ValueError(
            "At least three numeric solar/weather features are required."
        )

    x = df[features].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0)

    # Remove constant columns.
    features = [c for c in features if x[c].nunique() > 1]
    x = x[features]

    scaler = StandardScaler()
    xs = scaler.fit_transform(x)

    return x, xs, features


def _sample_indices(n, max_n=4000, seed=42):
    rng = np.random.default_rng(seed)
    if n <= max_n:
        return np.arange(n)
    return rng.choice(n, size=max_n, replace=False)


def _save(fig, name):
    path = PLOT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"unsupervised_plots/{name}"


def _safe_silhouette(x, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    valid = labels[mask]
    if len(np.unique(valid)) < 2 or len(valid) < 3:
        return None
    # Keep scoring fast on very large datasets.
    xx = x[mask]
    if len(xx) > 5000:
        idx = _sample_indices(len(xx), 5000)
        xx, valid = xx[idx], valid[idx]
    return round(float(silhouette_score(xx, valid)), 4)


def run_kmeans(x_scaled, seed=42):
    """Find four broad operating-condition clusters."""
    model = KMeans(n_clusters=4, random_state=seed, n_init=10)
    labels = model.fit_predict(x_scaled)
    return labels, model, _safe_silhouette(x_scaled, labels)


def run_hierarchical(x_scaled):
    """Agglomerative clustering on a representative sample."""
    idx = _sample_indices(len(x_scaled), 1800)
    model = AgglomerativeClustering(n_clusters=4, linkage="ward")
    labels = model.fit_predict(x_scaled[idx])
    return idx, labels, _safe_silhouette(x_scaled[idx], labels)


def run_dbscan(x_scaled):
    """Density-based clustering; -1 means noise."""
    idx = _sample_indices(len(x_scaled), 5000)
    model = DBSCAN(eps=0.8, min_samples=12, n_jobs=-1)
    labels = model.fit_predict(x_scaled[idx])
    return idx, labels


def run_pca(x_scaled):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)
    variance = [round(float(v * 100), 2) for v in pca.explained_variance_ratio_]
    return coords, variance


def run_tsne(x_scaled):
    idx = _sample_indices(len(x_scaled), 2500)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
        max_iter=750,
    )
    coords = tsne.fit_transform(x_scaled[idx])
    return idx, coords


def run_umap(x_scaled):
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP is not installed. Run: pip install umap-learn"
        ) from exc

    idx = _sample_indices(len(x_scaled), 4000)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    coords = reducer.fit_transform(x_scaled[idx])
    return idx, coords


def run_anomaly_detection(x_scaled):
    """Isolation Forest over the full dataset."""
    model = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    labels = model.fit_predict(x_scaled)
    scores = -model.score_samples(x_scaled)
    return labels, scores


def run_all(df, force=False):
    """Run all unsupervised techniques and create dashboard plots."""
    cache_key = (len(df), tuple(df.columns))
    if cache_key in CACHE and not force:
        return CACHE[cache_key]

    raw, x_scaled, features = _prepare(df)

    # ---------------- K-MEANS ----------------
    k_labels, k_model, k_silhouette = run_kmeans(x_scaled)

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(
        raw["IRRADIATION"] if "IRRADIATION" in raw else raw.iloc[:, 0],
        raw["AC_POWER"] if "AC_POWER" in raw else raw.iloc[:, 1],
        c=k_labels,
        s=5,
        alpha=0.35,
    )
    plt.xlabel("Irradiation")
    plt.ylabel("AC Power")
    plt.title("K-Means: Solar Operating Conditions")
    k_plot = _save(fig, "kmeans_clusters.png")

    # ---------------- HIERARCHICAL ----------------
    h_idx, h_labels, h_silhouette = run_hierarchical(x_scaled)

    pca_for_h = PCA(n_components=2, random_state=42)
    h_coords = pca_for_h.fit_transform(x_scaled[h_idx])

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(h_coords[:, 0], h_coords[:, 1], c=h_labels, s=9, alpha=0.55)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Hierarchical Clustering")
    h_plot = _save(fig, "hierarchical_clusters.png")

    # ---------------- DBSCAN ----------------
    d_idx, d_labels = run_dbscan(x_scaled)
    d_coords = PCA(n_components=2, random_state=42).fit_transform(x_scaled[d_idx])

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(d_coords[:, 0], d_coords[:, 1], c=d_labels, s=9, alpha=0.55)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("DBSCAN: Dense Regions and Noise")
    d_plot = _save(fig, "dbscan_clusters.png")

    d_cluster_count = len(set(d_labels)) - (1 if -1 in d_labels else 0)
    d_noise_count = int(np.sum(d_labels == -1))

    # ---------------- PCA ----------------
    pca_coords, pca_variance = run_pca(x_scaled)
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(
        pca_coords[:, 0],
        pca_coords[:, 1],
        c=k_labels,
        s=5,
        alpha=0.35,
    )
    plt.xlabel(f"PC1 ({pca_variance[0]}%)")
    plt.ylabel(f"PC2 ({pca_variance[1]}%)")
    plt.title("PCA: 2D Solar Data Representation")
    pca_plot = _save(fig, "pca_2d.png")

    # ---------------- t-SNE ----------------
    t_idx, t_coords = run_tsne(x_scaled)
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(t_coords[:, 0], t_coords[:, 1], c=k_labels[t_idx], s=8, alpha=0.55)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE: Nonlinear Structure")
    tsne_plot = _save(fig, "tsne_2d.png")

    # ---------------- UMAP ----------------
    umap_error = None
    umap_plot = None
    try:
        u_idx, u_coords = run_umap(x_scaled)
        fig = plt.figure(figsize=(8, 5))
        plt.scatter(u_coords[:, 0], u_coords[:, 1], c=k_labels[u_idx], s=8, alpha=0.55)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title("UMAP: Nonlinear Structure")
        umap_plot = _save(fig, "umap_2d.png")
    except ImportError as exc:
        umap_error = str(exc)

    # ---------------- ANOMALY DETECTION ----------------
    anomaly_labels, anomaly_scores = run_anomaly_detection(x_scaled)
    anomaly_mask = anomaly_labels == -1
    anomaly_count = int(anomaly_mask.sum())
    anomaly_pct = round(anomaly_count / len(anomaly_labels) * 100, 2)

    # Plot a physically meaningful view of anomalies.
    fig = plt.figure(figsize=(8, 5))
    normal = ~anomaly_mask
    if "IRRADIATION" in raw.columns and "AC_POWER" in raw.columns:
        plt.scatter(
            raw.loc[normal, "IRRADIATION"],
            raw.loc[normal, "AC_POWER"],
            s=4,
            alpha=0.18,
            label="Normal",
        )
        plt.scatter(
            raw.loc[anomaly_mask, "IRRADIATION"],
            raw.loc[anomaly_mask, "AC_POWER"],
            s=10,
            alpha=0.7,
            label="Anomaly",
        )
        plt.xlabel("Irradiation")
        plt.ylabel("AC Power")
    else:
        plt.scatter(np.arange(len(raw)), anomaly_scores, s=4)
        plt.xlabel("Observation")
        plt.ylabel("Anomaly score")
    plt.title("Isolation Forest: Anomalous Solar Observations")
    plt.legend()
    anomaly_plot = _save(fig, "anomaly_detection.png")

    # Cluster summaries for dashboard.
    cluster_summary = []
    for cluster in sorted(np.unique(k_labels)):
        mask = k_labels == cluster
        cluster_summary.append({
            "cluster": int(cluster),
            "records": int(mask.sum()),
            "percentage": round(float(mask.mean() * 100), 2),
            "avg_irradiation": round(float(raw.loc[mask, "IRRADIATION"].mean()), 4)
                if "IRRADIATION" in raw else None,
            "avg_ac_power": round(float(raw.loc[mask, "AC_POWER"].mean()), 2)
                if "AC_POWER" in raw else None,
        })

    result = {
        "features": features,
        "rows": int(len(raw)),
        "kmeans": {
            "clusters": 4,
            "silhouette": k_silhouette,
            "plot": k_plot,
            "summary": cluster_summary,
        },
        "hierarchical": {
            "clusters": 4,
            "sample_size": int(len(h_idx)),
            "silhouette": h_silhouette,
            "plot": h_plot,
        },
        "dbscan": {
            "clusters": int(d_cluster_count),
            "sample_size": int(len(d_idx)),
            "noise": d_noise_count,
            "noise_percent": round(d_noise_count / len(d_idx) * 100, 2),
            "plot": d_plot,
        },
        "pca": {
            "variance": pca_variance,
            "total_variance": round(sum(pca_variance), 2),
            "plot": pca_plot,
        },
        "tsne": {
            "sample_size": int(len(t_idx)),
            "plot": tsne_plot,
        },
        "umap": {
            "sample_size": int(len(u_idx)) if umap_plot else 0,
            "plot": umap_plot,
            "error": umap_error,
        },
        "anomaly": {
            "count": anomaly_count,
            "percentage": anomaly_pct,
            "plot": anomaly_plot,
        },
    }

    CACHE[cache_key] = result
    return result
