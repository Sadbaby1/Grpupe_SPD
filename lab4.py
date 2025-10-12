import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state

RANDOM_STATE = 42

def load_mnist_robust(random_state=RANDOM_STATE):
    rs = check_random_state(random_state)
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist['data'].astype(np.float32)
        y = mnist['target'].astype(np.int64)
        return X, y, {'name':'MNIST (OpenML)', 'shape':X.shape}
    except Exception:
        pass
    local = 'mnist.npz'
    if os.path.exists(local):
        with np.load(local) as f:
            Xtr, ytr = f['x_train'], f['y_train']
            Xte, yte = f['x_test'],  f['y_test']
        X = np.concatenate([Xtr.reshape(len(Xtr), -1),
                            Xte.reshape(len(Xte), -1)], axis=0).astype(np.float32)
        y = np.concatenate([ytr, yte]).astype(np.int64)
        return X, y, {'name':'MNIST (local npz)', 'shape':X.shape}
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    return X, y, {'name':'Digits 8x8 (fallback)', 'shape':X.shape}

def save_and_show(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight")
    print("Saved:", path)
    plt.show()

def plot_cm(cm, title, outpath, normalized=False):
    import matplotlib as mpl
    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay(cm).plot(
        ax=ax,
        include_values=True,
        cmap=("Blues" if normalized else mpl.cm.viridis),
        colorbar=True,
        xticks_rotation="vertical",
        values_format=(".2f" if normalized else "d")
    )
    ax.set_title(title)
    fig.tight_layout()
    save_and_show(fig, outpath)

def main(random_state=RANDOM_STATE):
    outdir = Path("assignment4_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load & split
    X, y, meta = load_mnist_robust(random_state)
    print(f"Dataset: {meta['name']} | shape={meta['shape']}")
    test_size = 10000 if meta['shape'][0] >= 60000 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 2) One-shot strong pipeline: StandardScaler -> PCA(60) -> KNN(3, distance, p=2)
    knn_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("pca", PCA(n_components=60, random_state=random_state, whiten=False)),
        ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance", p=2, algorithm="brute"))
    ])
    knn_pipe.fit(X_train, y_train)
    y_pred = knn_pipe.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred)
    print("KNN (PCA60, k=3, distance, euclidean) — Test accuracy:", f"{knn_acc:.4f}")

    # Save summary + classification report
    pd.DataFrame({
        "dataset":[meta["name"]],
        "pipeline":["Scaler->PCA(60)->KNN(k=3, distance, p=2)"],
        "test_accuracy":[knn_acc]
    }).to_csv(outdir / "knn_ultrafast_summary.csv", index=False)

    report = classification_report(y_test, y_pred, digits=4)
    (outdir / "knn_ultrafast_classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

    # Confusion matrices
    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, "KNN (Ultrafast) — Test Confusion Matrix",
            outdir / "knn_ultrafast_confusion_matrix.png", normalized=False)
    cm_norm = (cm.T / cm.sum(axis=1)).T
    plot_cm(cm_norm, "KNN (Ultrafast) — Test Confusion Matrix (Normalized)",
            outdir / "knn_ultrafast_confusion_matrix_normalized.png", normalized=True)

    # 3) Quick baselines (kept light)
    sgd = make_pipeline(StandardScaler(with_mean=True),
                        SGDClassifier(max_iter=1000, tol=1e-3, random_state=random_state))
    sgd.fit(X_train, y_train)
    sgd_acc = accuracy_score(y_test, sgd.predict(X_test))

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    comp = pd.DataFrame({
        "model": ["KNN (PCA60,k=3,dist)", "SGD", "RandomForest(100)"],
        "test_accuracy": [knn_acc, sgd_acc, rf_acc]
    })
    comp.to_csv(outdir / "baselines_vs_knn_ultrafast.csv", index=False)
    print("\nComparison (test accuracy):\n", comp)

    # Bar chart
    fig, ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(comp["model"], comp["test_accuracy"],
                  color=["#1f77b4", "#f39c12", "#2ca02c"])
    ax.set_ylim(0.80, 1.00)
    ax.set_title("Model Comparison — Test Accuracy")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.25)
    for r in bars:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.004, f"{h:.3f}",
                ha='center', va='bottom')
    fig.tight_layout()
    save_and_show(fig, outdir / "model_comparison_bar_ultrafast.png")

    with open(outdir / "README_ultrafast.txt", "w", encoding="utf-8") as f:
        f.write("Assignment 4 (ULTRAFAST) artifacts:\n")
        for p in sorted(outdir.iterdir()):
            f.write(f"- {p.name}\n")
    print("Artifacts saved to:", outdir)

if _name_ == "_main_":
    main()