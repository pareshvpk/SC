"""Train the hybrid ML candidate ranker (small MLP) and export numpy weights.

Loads ranker_data.npz (from make_ranker_data.py), trains a scikit-learn
MLPClassifier to predict P(candidate is the true site) from the classical
candidate features, evaluates PAIR-LEVEL selection accuracy on a held-out split,
and exports the learned weights to ml_ranker.npz for the pure-numpy inference
path in localize.py.

The notebook train_ranker.ipynb mirrors this file cell-for-cell (deliverable).
"""
from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from localize import FEATURE_ORDER


def pair_selection_accuracy(probs, y, g, tol_label=1):
    """Fraction of pairs whose highest-probability candidate is a true site.
    Reported over all pairs and over 'solvable' pairs (>=1 true candidate)."""
    total = solvable = hit = hit_solvable = 0
    for gid in np.unique(g):
        m = g == gid
        pg, yg = probs[m], y[m]
        total += 1
        picked_true = yg[np.argmax(pg)] >= tol_label
        hit += picked_true
        if yg.max() >= 1:
            solvable += 1
            hit_solvable += picked_true
    return hit / total, (hit_solvable / solvable if solvable else float("nan")), solvable, total


def main():
    d = np.load("ranker_data.npz")
    Xtr, ytr, gtr = d["X_train"], d["y_train"], d["g_train"]
    Xte, yte, gte = d["X_test"], d["y_test"], d["g_test"]

    scaler = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                        alpha=1e-3, max_iter=3000, early_stopping=True,
                        n_iter_no_change=30, random_state=0)
    clf.fit(scaler.transform(Xtr), ytr)

    ptr = clf.predict_proba(scaler.transform(Xtr))[:, 1]
    pte = clf.predict_proba(scaler.transform(Xte))[:, 1]
    print("per-candidate ROC-AUC:  train %.3f   test %.3f"
          % (roc_auc_score(ytr, ptr), roc_auc_score(yte, pte)))

    acc, acc_s, solv, tot = pair_selection_accuracy(pte, yte, gte)
    print("pair-level selection (test): %.1f%% of all %d pairs, %.1f%% of %d solvable pairs"
          % (100 * acc, tot, 100 * acc_s, solv))

    # feature importance proxy: |input-layer weight| aggregated
    w0 = np.abs(clf.coefs_[0]).sum(axis=1)
    order = np.argsort(-w0)
    print("\ntop features by |input weight|:")
    for i in order[:6]:
        print(f"  {FEATURE_ORDER[i]:<18} {w0[i]:.2f}")

    np.savez("ml_ranker.npz",
             mean=scaler.mean_, scale=scaler.scale_,
             n_layers=len(clf.coefs_),
             **{f"coef_{i}": c for i, c in enumerate(clf.coefs_)},
             **{f"intercept_{i}": b for i, b in enumerate(clf.intercepts_)})
    print("\nexported ml_ranker.npz")


if __name__ == "__main__":
    main()
