#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 5: SVM on MNIST with Kernel Tuning + Comparison (KNN/SGD/RF)
- Robust MNIST loader (OpenML -> local mnist.npz -> sklearn digits fallback)
- Scaling + (optional) PCA for KNN speed
- GridSearchCV (poly) + RandomizedSearchCV (RBF) with 3-fold CV on a 15k subsample (fast)
- Best SVM refit on full training set, evaluated on held-out test set (10k)
- Metrics: accuracy, precision, recall, F1 + confusion matrix
- Timing: fit and predict time for each model
- Comparison bar plots: accuracy + train time
- Artifacts saved under ./assignment5_outputs
"""

import os, time, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)
from scipy.stats import loguniform
from sklearn.utils import check_random_state

# ---------------------- Robust loader ----------------------
def load_mnist_robust(random_state=42):
    rs = check_random_state(random_state)
    # 1) Try OpenML
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist['data'].astype(np.float32)
        y = mnist['target'].astype(np.int64)
        return X, y, {'name':'MNIST (OpenML)', 'shape': X.shape}
    except Exception:
        pass
    # 2) Try local mnist.npz
    try:
        if os.path.exists('mnist.npz'):
            with np.load('mnist.npz') as f:
                Xtr, ytr = f['x_train'], f['y_train']
                Xte, yte = f['x_test'],  f['y_test']
            X = np.concatenate([Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)], 0).astype(np.float32)
            y = np.concatenate([ytr, yte]).astype(np.int64)
            return X, y, {'name':'MNIST (local npz)', 'shape': X.shape}
    except Exception:
        pass
    # 3) Fallback: sklearn digits (8x8) so pipeline still runs
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    return X, y, {'name':'Digits 8x8 (fallback)', 'shape': X.shape}

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def time_fit_predict(clf, Xtr, ytr, Xte):
    t0 = time.time()
    clf.fit(Xtr, ytr)
    fit_s = time.time() - t0
    t1 = time.time()
    yhat = clf.predict(Xte)
    pred_s = time.time() - t1
    return yhat, fit_s, pred_s

# ---------------------- Main ----------------------
def main(random_state=42, search_subsample=15000, outdir="assignment5_outputs"):
    ensure_dir(outdir)

    # Load & split (10k test if MNIST-like, else 25%)
    X, y, meta = load_mnist_robust(random_state=random_state)
    n = X.shape[0]
    test_size = 10000 if n >= 60000 else 0.25
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    print(f"Dataset: {meta['name']} | shape={meta['shape']}")
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    # Scale (SVM is distance-based -> scaling is crucial)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr.astype(np.float64))
    Xte_s = scaler.transform(Xte.astype(np.float64))

    # ----------------- SVM Tuning -----------------
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # (A) Linear SVM (two options: SVC(kernel='linear') or LinearSVC)
    # SVC(kernel='linear') supports probability=False and exact margin formulation; LinearSVC is faster but slightly different.
    svm_lin = SVC(kernel='linear', C=1.0, random_state=random_state)

    # (B) Polynomial kernel — Grid search (small grid; fast)
    poly_grid = {
        'kernel': ['poly'],
        'degree': [2, 3, 4],
        'C': [0.1, 1, 10],
        'coef0': [0, 1, 5],
        # gamma defaults to 'scale'; good default for MNIST scale
    }
    svm_poly = SVC()
    # (C) RBF kernel — Randomized search (log-uniform for C, gamma)
    rbf_dist = {
        'kernel': ['rbf'],
        'C': loguniform(1e0, 1e2),       # 1 .. 100
        'gamma': loguniform(1e-4, 1e-1)  # 1e-4 .. 1e-1
    }
    svm_rbf = SVC()

    # Subsample for search to keep it quick
    rs = check_random_state(random_state)
    if search_subsample and search_subsample < Xtr_s.shape[0]:
        idx = rs.choice(Xtr_s.shape[0], size=search_subsample, replace=False)
        Xcv, ycv = Xtr_s[idx], ytr[idx]
        print(f"Grid/Random search on subsample: {Xcv.shape[0]} samples")
    else:
        Xcv, ycv = Xtr_s, ytr

    # Linear baseline (no tuning)
    print("\n[Linear SVM] Fitting...")
    yhat_lin, tfit_lin, tpred_lin = time_fit_predict(svm_lin, Xtr_s, ytr, Xte_s)
    acc_lin = accuracy_score(yte, yhat_lin)
    print(f"[Linear SVM] Test accuracy={acc_lin:.4f} | fit={tfit_lin:.1f}s, predict={tpred_lin:.1f}s")

    # Polynomial GridSearch
    print("\n[Poly SVM] Grid search...")
    gs_poly = GridSearchCV(SVC(), poly_grid, cv=cv, n_jobs=-1, verbose=1)
    gs_poly.fit(Xcv, ycv)
    print(f"[Poly SVM] Best params: {gs_poly.best_params_} | best CV acc={gs_poly.best_score_:.4f}")
    best_poly = SVC(**gs_poly.best_params_)
    yhat_poly, tfit_poly, tpred_poly = time_fit_predict(best_poly, Xtr_s, ytr, Xte_s)
    acc_poly = accuracy_score(yte, yhat_poly)
    print(f"[Poly SVM] Test accuracy={acc_poly:.4f} | fit={tfit_poly:.1f}s, predict={tpred_poly:.1f}s")

    # RBF RandomizedSearch
    print("\n[RBF SVM] Randomized search...")
    rs_rbf = RandomizedSearchCV(SVC(), rbf_dist, n_iter=12, cv=cv, random_state=random_state, n_jobs=-1, verbose=1)
    rs_rbf.fit(Xcv, ycv)
    print(f"[RBF SVM] Best params: {rs_rbf.best_params_} | best CV acc={rs_rbf.best_score_:.4f}")
    best_rbf = SVC(**rs_rbf.best_params_)
    yhat_rbf, tfit_rbf, tpred_rbf = time_fit_predict(best_rbf, Xtr_s, ytr, Xte_s)
    acc_rbf = accuracy_score(yte, yhat_rbf)
    print(f"[RBF SVM] Test accuracy={acc_rbf:.4f} | fit={tfit_rbf:.1f}s, predict={tpred_rbf:.1f}s")

    # Pick best SVM by test accuracy
    svm_candidates = [
        ("SVM-Linear", svm_lin, yhat_lin, acc_lin, tfit_lin, tpred_lin),
        ("SVM-Poly",   best_poly, yhat_poly, acc_poly, tfit_poly, tpred_poly),
        ("SVM-RBF",    best_rbf,  yhat_rbf,  acc_rbf,  tfit_rbf,  tpred_rbf)
    ]
    best_svm_name, best_svm, best_yhat, best_acc, best_tfit, best_tpred = sorted(svm_candidates, key=lambda r: r[3], reverse=True)[0]
    print(f"\n[Best SVM] {best_svm_name} | Test acc={best_acc:.4f} | fit={best_tfit:.1f}s, predict={best_tpred:.1f}s")
    print("\nClassification report (best SVM):")
    print(classification_report(yte, best_yhat, digits=4))

    # Confusion matrix (best SVM)
    cm = confusion_matrix(yte, best_yhat)
    fig = plt.figure(figsize=(6,6))
    ConfusionMatrixDisplay(cm).plot(include_values=True, cmap='viridis', ax=plt.gca(), xticks_rotation='vertical', colorbar=True)
    plt.title(f"{best_svm_name} — Test Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "svm_best_confusion_matrix.png"), dpi=220)
    plt.close(fig)

    # ----------------- Comparators (KNN/SGD/RF) -----------------
    # KNN with PCA(60) for speed (as in A4)
    knn_pipe = Pipeline([
        ("pca", PCA(n_components=60, random_state=random_state)),
        ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance"))
    ])
    _, tfit_knn, tpred_knn = None, None, None
    yhat_knn, tfit_knn, tpred_knn = time_fit_predict(knn_pipe, Xtr_s, ytr, Xte_s)
    acc_knn = accuracy_score(yte, yhat_knn)

    # SGD (logistic/hinge) – very fast linear baseline
    sgd = SGDClassifier(loss="log_loss", random_state=random_state, max_iter=1000, tol=1e-3)
    yhat_sgd, tfit_sgd, tpred_sgd = time_fit_predict(sgd, Xtr_s, ytr, Xte_s)
    acc_sgd = accuracy_score(yte, yhat_sgd)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    yhat_rf, tfit_rf, tpred_rf = time_fit_predict(rf, Xtr, ytr, Xte)  # trees are scale-insensitive; use raw X
    acc_rf = accuracy_score(yte, yhat_rf)

    # ----------------- Metrics table -----------------
    def prfs(y_true, y_pred):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        return p, r, f1

    rows = []
    for name, yhat, tfit, tpred in [
        (best_svm_name, best_yhat, best_tfit, best_tpred),
        ("KNN (PCA60,k=3,dist)", yhat_knn, tfit_knn, tpred_knn),
        ("SGD", yhat_sgd, tfit_sgd, tpred_sgd),
        ("RandomForest(100)", yhat_rf, tfit_rf, tpred_rf),
    ]:
        acc = accuracy_score(yte, yhat)
        p, r, f1 = prfs(yte, yhat)
        rows.append([name, acc, p, r, f1, tfit, tpred])

    df = pd.DataFrame(rows, columns=["model","accuracy","precision","recall","f1","fit_time_s","predict_time_s"])
    df.sort_values("accuracy", ascending=False, inplace=True)
    print("\nComparison (test metrics):")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(outdir, "comparison_metrics.csv"), index=False)

    # ----------------- Plots: accuracy + training time -----------------
    # Accuracy bar
    plt.figure(figsize=(7.5,4))
    plt.ylim(0.80, 1.00)
    bars = plt.bar(df["model"], df["accuracy"])
    for b, v in zip(bars, df["accuracy"]):
        plt.text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison — Test Accuracy")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_accuracy_bar.png"), dpi=220)
    plt.close()

    # Fit-time bar (log scale helps)
    plt.figure(figsize=(7.5,4))
    bars = plt.bar(df["model"], df["fit_time_s"])
    for b, v in zip(bars, df["fit_time_s"]):
        plt.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=10, rotation=0)
    plt.yscale("log")
    plt.ylabel("Fit time (s) [log scale]")
    plt.title("Model Comparison — Fit Time")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_fit_time_bar.png"), dpi=220)
    plt.close()

    print(f"\nArtifacts saved to: {outdir}")
    # Quick notes to paste into the report
    with open(os.path.join(outdir, "report_notes.txt"), "w", encoding="utf-8") as f:
        f.write("# Assignment 5 Auto Notes\n")
        f.write(f"Best SVM: {best_svm_name} | params={best_svm.get_params()}\n")
        f.write(df.to_string(index=False))
        f.write("\n")

if __name__ == "__main__":
    main()

