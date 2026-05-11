"""
BBM409 Assignment 4 - GAK + SVM Classifier
Global Alignment Kernel with SVM for lip movement sequence classification.

NOTE: GAK kernel matrix computation is O(N^2 * T^2) — it can be slow
on large datasets. If it's too slow, use the subsample option below.
"""

import numpy as np
from tslearn.metrics import cdist_gak, cdist_soft_dtw
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time


# ── Helpers ────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def print_ablation_table(results):
    print(f"\n{'='*60}")
    print(f"{'sigma':>10} {'C':>10} {'test_acc':>12} {'time(s)':>10}")
    print(f"{'-'*60}")
    for r in sorted(results, key=lambda x: -x["test_acc"]):
        print(f"{r['sigma']:>10.3f} {r['C']:>10.3f} "
              f"{r['test_acc']:>12.4f} {r['fit_time']:>10.1f}")


# ── Subsample ──────────────────────────────────────────────────────────────────
def subsample(X, y, max_per_class, random_state=42):
    """
    Limit to max_per_class samples per class to keep GAK tractable.
    GAK kernel matrix grows as O(N^2) so this matters a lot.
    """
    rng = np.random.RandomState(random_state)
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min(max_per_class, len(cls_idx)), replace=False)
        indices.extend(chosen)
    indices = np.array(indices)
    return X[indices], y[indices]


# ── GAK kernel matrix ──────────────────────────────────────────────────────────
def compute_gak_kernel(X_train, X_test, sigma):
    """
    Compute the GAK kernel matrices needed for SVM.

    cdist_gak returns a SIMILARITY matrix (higher = more similar).
    SVC with kernel="precomputed" expects a kernel (similarity) matrix,
    NOT a distance matrix — so we pass it directly.

    Args:
        X_train : (N_train, T, D)
        X_test  : (N_test,  T, D)
        sigma   : bandwidth parameter for GAK

    Returns:
        K_train : (N_train, N_train) — training kernel matrix
        K_test  : (N_test,  N_train) — test kernel matrix (test vs train)
    """
    print(f"  Computing train kernel ({X_train.shape[0]}x{X_train.shape[0]})...")
    K_train = cdist_gak(X_train, X_train, sigma=sigma)

    print(f"  Computing test  kernel ({X_test.shape[0]}x{X_train.shape[0]})...")
    K_test = cdist_gak(X_test, X_train, sigma=sigma)

    return K_train, K_test


# ── Single run ─────────────────────────────────────────────────────────────────
def run_gak_svm(X_train, X_test, y_train, y_test, sigma, C, class_names):
    """
    Fit a GAK+SVM model with given sigma and C, return test accuracy.
    """
    t0 = time.time()

    K_train, K_test = compute_gak_kernel(X_train, X_test, sigma)

    print(f"  Fitting SVM (C={C})...")
    svm = SVC(kernel="precomputed", C=C, random_state=42)
    svm.fit(K_train, y_train)

    y_pred = svm.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - t0

    print(f"  Test accuracy: {acc:.4f} | Time: {elapsed:.1f}s")

    plot_confusion_matrix(
        y_test, y_pred, class_names,
        title=f"GAK+SVM  sigma={sigma}  C={C}  acc={acc:.3f}"
    )

    return acc, y_pred, elapsed


# ── Ablation study ─────────────────────────────────────────────────────────────
def run_ablation(X_train, X_test, y_train, y_test, class_names, configs,
                 max_per_class=None):
    """
    Run GAK+SVM for multiple (sigma, C) combinations.

    Args:
        configs      : list of dicts with keys: sigma, C
        max_per_class: if set, subsample to this many samples per class
                       before computing the kernel (speeds things up a lot)
    """
    if max_per_class is not None:
        print(f"Subsampling to {max_per_class} samples per class...")
        X_train, y_train = subsample(X_train, y_train, max_per_class)
        X_test,  y_test  = subsample(X_test,  y_test,  max_per_class)
        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    results = []

    for i, cfg in enumerate(configs):
        sigma, C = cfg["sigma"], cfg["C"]
        print(f"\n{'='*60}")
        print(f"Config {i+1}/{len(configs)}: sigma={sigma}, C={C}")
        print(f"{'='*60}")

        acc, y_pred, elapsed = run_gak_svm(
            X_train, X_test, y_train, y_test,
            sigma=sigma, C=C, class_names=class_names
        )
        results.append({"sigma": sigma, "C": C, "test_acc": acc, "fit_time": elapsed})

    return results


# ── Sigma heuristic ────────────────────────────────────────────────────────────
def estimate_sigma(X, n_samples=50, random_state=42):
    """
    Estimate a reasonable starting sigma using the median heuristic:
    sigma = median of pairwise L2 norms between random sequence pairs.

    This is a common heuristic for kernel bandwidth selection.
    tslearn's cdist_gak also accepts sigma="auto" internally,
    but this gives you a concrete starting value to ablate around.
    """
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    subset = X[idx]

    norms = []
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            norms.append(np.linalg.norm(subset[i] - subset[j]))

    sigma = np.median(norms)
    print(f"Median heuristic sigma estimate: {sigma:.4f}")
    return sigma


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load prepared data
    data = np.load("prepared_data.npz", allow_pickle=True)
    X_train = data["X_train"]      # (N, T, 80)
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    class_names = list(data["classes"])

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Classes: {class_names}\n")

    # Estimate a good starting sigma from the data
    sigma_est = estimate_sigma(X_train)

    # ── Ablation configurations ────────────────────────────────────────────────
    # We vary sigma (kernel bandwidth) and C (SVM regularization).
    # sigma controls how strictly sequences must align — smaller sigma
    # is stricter, larger is more permissive.
    # C controls the SVM margin — larger C = less regularization.
    configs = [
        # Around the median heuristic estimate
        {"sigma": sigma_est * 0.5, "C": 1.0},
        {"sigma": sigma_est,       "C": 1.0},   # baseline
        {"sigma": sigma_est * 2.0, "C": 1.0},
        # Vary C with best sigma
        {"sigma": sigma_est,       "C": 0.1},
        {"sigma": sigma_est,       "C": 10.0},
        {"sigma": sigma_est,       "C": 100.0},
    ]

    # ── IMPORTANT: GAK is slow on large datasets ───────────────────────────────
    # If you have many samples, set max_per_class to limit computation time.
    # Start with 50 to verify everything works, then increase.
    # Set to None to use all data (may take very long).
    MAX_PER_CLASS = 50

    results = run_ablation(
        X_train, X_test, y_train, y_test,
        class_names=class_names,
        configs=configs,
        max_per_class=MAX_PER_CLASS,
    )

    print_ablation_table(results)
