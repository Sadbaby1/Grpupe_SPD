"""
Assignment 6 — Unsupervised Learning Using K-Means Clustering on California Housing Data
- Features: Longitude, Latitude, MedInc
- KMeans: silhouette optimization over k grid
- Visuals: Silhouette vs k, geospatial cluster map (+ centroids), boxplot of MedInc by cluster
- Compare vs DBSCAN (several eps values) and report which performs better
- Outputs saved to ./assignment6_outputs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# ------------------ Config ------------------
OUTDIR = "assignment6_outputs"
RANDOM_STATE = 42
K_GRID = [2, 3, 4, 5, 6, 8, 10]
DBSCAN_EPS = [0.15, 0.2, 0.25, 0.3]     # tuned for standardized features
DBSCAN_MIN_SAMPLES = 50
POINT_SIZE = 6
# --------------------------------------------

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def scatter_clusters(df, labels, centers=None, title="", fname=None):
    plt.figure(figsize=(8,6))
    sc = plt.scatter(df["Longitude"], df["Latitude"], c=labels, s=POINT_SIZE, cmap="tab10", alpha=0.8)
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title(title)
    plt.colorbar(sc, label="Cluster")
    if centers is not None:
        # centers are in standardized space; we pass the geospatial coords separately
        c_long, c_lat = centers[:, 0], centers[:, 1]
        plt.scatter(c_long, c_lat, s=120, c="black", marker="X", label="Centroids")
        plt.legend()
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=150)
    plt.close()

def boxplot_income_by_cluster(df, labels, title, fname=None):
    tmp = df.copy()
    tmp["cluster"] = labels
    order = sorted(tmp["cluster"].unique())
    data = [tmp.loc[tmp["cluster"]==c, "MedInc"] for c in order]
    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=[str(x) for x in order], showmeans=True)
    plt.title(title); plt.xlabel("Cluster"); plt.ylabel("Median Income")
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=150)
    plt.close()

def main():
    ensure_dir(OUTDIR)

    # 1) Load dataset and select features
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df.rename(columns={"Longitude":"Longitude", "Latitude":"Latitude", "MedInc":"MedInc"}, inplace=True)
    df = df[["Longitude", "Latitude", "MedInc"]].dropna().reset_index(drop=True)

    # 2) Standardize (important for clustering)
    X = df.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 3) KMeans: sweep k, compute silhouette
    km_results = []
    best_k = None
    best_sil = -1.0
    for k in K_GRID:
        km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        km_results.append({"k":k, "silhouette":sil})
        if sil > best_sil:
            best_sil = sil
            best_k = k

    km_df = pd.DataFrame(km_results).sort_values("k")
    print("Silhouette scores (KMeans):")
    print(km_df.to_string(index=False))

    # Plot silhouette vs k
    plt.figure(figsize=(7,4))
    plt.plot(km_df["k"], km_df["silhouette"], marker="o")
    plt.title("KMeans: Silhouette Score vs k")
    plt.xlabel("k"); plt.ylabel("Silhouette score")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "kmeans_silhouette_vs_k.png"), dpi=150)
    plt.close()

    # 4) Fit final KMeans with best k
    final_km = KMeans(n_clusters=best_k, n_init="auto", random_state=RANDOM_STATE)
    final_labels = final_km.fit_predict(Xs)
    final_sil = silhouette_score(Xs, final_labels)

    # Convert centroids back to original feature space for plotting
    centers_std = final_km.cluster_centers_                  # standardized centers
    centers_orig = scaler.inverse_transform(centers_std)     # [Longitude, Latitude, MedInc]
    centers_geo = centers_orig[:, :2]

    # Plot: geospatial clusters with centroids
    scatter_clusters(
        df=df, labels=final_labels,
        centers=centers_geo,
        title=f"KMeans Clusters (k={best_k}) — Silhouette={final_sil:.3f}",
        fname=os.path.join(OUTDIR, f"kmeans_clusters_k{best_k}.png")
    )

    # Boxplot: median income by cluster
    boxplot_income_by_cluster(
        df=df, labels=final_labels,
        title=f"Median Income by Cluster (KMeans, k={best_k})",
        fname=os.path.join(OUTDIR, f"kmeans_income_boxplot_k{best_k}.png")
    )

    # Save cluster summary
    df_km = df.copy()
    df_km["cluster"] = final_labels
    cluster_summary = df_km.groupby("cluster").agg(
        count=("MedInc", "size"),
        medinc_mean=("MedInc", "mean"),
        medinc_median=("MedInc", "median"),
        lon_mean=("Longitude", "mean"),
        lat_mean=("Latitude", "mean"),
    ).reset_index().sort_values("cluster")
    cluster_summary.to_csv(os.path.join(OUTDIR, f"kmeans_cluster_summary_k{best_k}.csv"), index=False)

    # 5) Compare with DBSCAN (basic sweep over eps)
    dbscan_rows = []
    best_db = None
    best_db_sil = -1.0
    for eps in DBSCAN_EPS:
        db = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
        labels = db.fit_predict(Xs)  # -1 = noise
        n_noise = int(np.sum(labels == -1))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Silhouette only valid if there are at least 2 clusters and not all are noise
        sil = np.nan
        valid = (n_clusters >= 2)
        if valid:
            try:
                sil = silhouette_score(Xs[labels!=-1], labels[labels!=-1])
            except Exception:
                sil = np.nan

        dbscan_rows.append({
            "eps": eps,
            "min_samples": DBSCAN_MIN_SAMPLES,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": sil,
        })

        if valid and (not np.isnan(sil)) and sil > best_db_sil:
            best_db_sil = sil
            best_db = (eps, labels.copy())

    db_df = pd.DataFrame(dbscan_rows)
    print("\nDBSCAN sweep:")
    print(db_df.to_string(index=False))

    db_df.to_csv(os.path.join(OUTDIR, "dbscan_results.csv"), index=False)

    # Visualize the best DBSCAN (if any valid clustering)
    if best_db is not None:
        best_eps, best_labels = best_db
        # plot all points; show noise as a separate color (-1)
        plt.figure(figsize=(8,6))
        sc = plt.scatter(df["Longitude"], df["Latitude"], c=best_labels, s=POINT_SIZE, cmap="tab10", alpha=0.8)
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.title(f"DBSCAN Clusters (eps={best_eps}) — noise={np.sum(best_labels==-1)}")
        plt.colorbar(sc, label="Cluster (-1 = noise)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"dbscan_clusters_eps{best_eps}.png"), dpi=150)
        plt.close()
    else:
        print("\nDBSCAN did not produce a valid multi-cluster partition with the tested eps values.")

    # 6) Final comparison summary
    summary = {
        "kmeans_best_k": best_k,
        "kmeans_silhouette": round(float(final_sil), 4),
        "dbscan_best_silhouette": round(float(best_db_sil), 4) if best_db_sil > 0 else np.nan,
        "dbscan_any_valid": bool(best_db is not None),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)

    print("\n=== Summary ===")
    print(summary)
    print(f"\nArtifacts saved to: {OUTDIR}")
    print("Saved files:")
    for fn in sorted(os.listdir(OUTDIR)):
        print(" -", fn)

if __name__ == "__main__":
    main()

