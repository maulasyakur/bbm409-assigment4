"""
BBM409 Assignment 4 - Shapelets + MLP Classifier
Extracts shapelet-based features from lip movement sequences,
then classifies with a PyTorch MLP.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tslearn.shapelets import ShapeletModel
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time


# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


def plot_history(history, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title(f"Loss — {title}"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title(f"Accuracy — {title}"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout(); plt.show()


def print_ablation_table(results):
    print(f"\n{'='*90}")
    print(f"{'n_shapelets':>12} {'shapelet_len':>13} {'hidden':>8} "
          f"{'dropout':>8} {'lr':>8} {'test_acc':>10} {'time(s)':>10}")
    print(f"{'-'*90}")
    for r in sorted(results, key=lambda x: -x["test_acc"]):
        print(f"{r['n_shapelets']:>12} {r['shapelet_len_frac']:>13.2f} "
              f"{r['hidden_sizes']:>8} {r['dropout']:>8.2f} "
              f"{r['lr']:>8.5f} {r['test_acc']:>10.4f} {r['time']:>10.1f}")


# ── Step 1: Shapelet transform ─────────────────────────────────────────────────
def fit_shapelet_transform(X_train, y_train, n_shapelets_per_size,
                           shapelet_len_frac, max_iter=50, random_state=42):
    """
    Fit a ShapeletModel and transform sequences into fixed-length distance vectors.

    Each shapelet is a short subsequence. For each input sequence, the transform
    computes the minimum distance to each shapelet → fixed-length feature vector
    of size (n_shapelets_per_size * num_sizes).

    Args:
        X_train             : (N, T, D)
        y_train             : (N,)
        n_shapelets_per_size: number of shapelets to learn per length
        shapelet_len_frac   : shapelet length as fraction of sequence length
                              e.g. 0.1 means 10% of T
        max_iter            : training iterations for shapelet learning

    Returns:
        model     : fitted ShapeletModel
        X_feat    : (N, n_shapelets) transformed feature matrix
    """
    T = X_train.shape[1]

    # Shapelet lengths to try — use one size for simplicity,
    # centered around shapelet_len_frac of T
    shapelet_len = max(2, int(T * shapelet_len_frac))
    shapelet_sizes = {shapelet_len: n_shapelets_per_size}

    print(f"  Fitting shapelets: {n_shapelets_per_size} shapelets "
          f"of length {shapelet_len} (={shapelet_len_frac:.0%} of T={T})")

    model = ShapeletModel(
        n_shapelets_per_size=shapelet_sizes,
        optimizer="sgd",
        weight_regularizer=0.01,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )
    model.fit(X_train, y_train)

    X_feat = model.transform(X_train)
    print(f"  Shapelet feature shape: {X_feat.shape}")  # (N, n_shapelets)
    return model, X_feat


# ── Step 2: MLP ────────────────────────────────────────────────────────────────
class ShapeletMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout):
        """
        Args:
            input_size   : number of shapelet features
            hidden_sizes : list of hidden layer sizes, e.g. [256, 128]
            num_classes  : number of word classes
            dropout      : dropout probability
        """
        super().__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def train_mlp(model, train_loader, val_loader, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y_b)
            t_correct += (logits.argmax(1) == y_b).sum().item()
            t_total += len(y_b)

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                loss = criterion(logits, y_b)
                v_loss += loss.item() * len(y_b)
                preds = logits.argmax(1)
                v_correct += (preds == y_b).sum().item()
                v_total += len(y_b)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_b.cpu().numpy())

        scheduler.step(v_loss / v_total)

        history["train_loss"].append(t_loss / t_total)
        history["train_acc"].append(t_correct / t_total)
        history["val_loss"].append(v_loss / v_total)
        history["val_acc"].append(v_correct / v_total)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Train loss: {t_loss/t_total:.4f}, acc: {t_correct/t_total:.3f} | "
                  f"Val   loss: {v_loss/v_total:.4f}, acc: {v_correct/v_total:.3f}")

    return history, np.array(all_preds), np.array(all_labels)


# ── Full pipeline ──────────────────────────────────────────────────────────────
def run_shapelets_mlp(X_train, X_test, y_train, y_test, cfg, class_names):
    """
    Full pipeline: shapelet transform → normalize → MLP.
    """
    t0 = time.time()
    num_classes = len(class_names)

    # 1. Fit shapelets on training data and transform both sets
    shapelet_model, X_train_feat = fit_shapelet_transform(
        X_train, y_train,
        n_shapelets_per_size=cfg["n_shapelets"],
        shapelet_len_frac=cfg["shapelet_len_frac"],
        max_iter=cfg["shapelet_iter"],
    )
    X_test_feat = shapelet_model.transform(X_test)

    # 2. Normalize features (important: fit scaler on train only)
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat  = scaler.transform(X_test_feat)

    # 3. Build DataLoaders
    train_ds = FeatureDataset(X_train_feat, y_train)
    test_ds  = FeatureDataset(X_test_feat,  y_test)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"])

    # 4. Build and train MLP
    input_size = X_train_feat.shape[1]
    model = ShapeletMLP(
        input_size=input_size,
        hidden_sizes=cfg["hidden_sizes"],
        num_classes=num_classes,
        dropout=cfg["dropout"],
    ).to(device)

    print(f"  MLP input size: {input_size} | Architecture: "
          f"{[input_size] + cfg['hidden_sizes'] + [num_classes]}")

    history, y_pred, y_true = train_mlp(
        model, train_loader, test_loader,
        num_epochs=cfg["num_epochs"],
        lr=cfg["lr"],
    )

    acc = accuracy_score(y_true, y_pred)
    elapsed = time.time() - t0
    print(f"\n  Final test accuracy: {acc:.4f} | Time: {elapsed:.1f}s")

    return acc, history, y_pred, y_true, elapsed


# ── Ablation study ─────────────────────────────────────────────────────────────
def run_ablation(X_train, X_test, y_train, y_test, class_names, configs):
    results = []

    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {i+1}/{len(configs)}: {cfg}")
        print(f"{'='*60}")

        acc, history, y_pred, y_true, elapsed = run_shapelets_mlp(
            X_train, X_test, y_train, y_test, cfg, class_names
        )

        label = (f"n={cfg['n_shapelets']} len={cfg['shapelet_len_frac']:.0%} "
                 f"h={cfg['hidden_sizes']}")
        plot_history(history, title=label)
        plot_confusion_matrix(y_true, y_pred, class_names,
                              title=f"Shapelets+MLP  {label}  acc={acc:.3f}")

        results.append({**cfg, "test_acc": acc, "time": elapsed})

    return results


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

    # ── Ablation configurations ────────────────────────────────────────────────
    # Ablate: number of shapelets, shapelet length, MLP hidden sizes,
    #         dropout, and learning rate.
    configs = [
        # Baseline
        {
            "n_shapelets": 20, "shapelet_len_frac": 0.1, "shapelet_iter": 50,
            "hidden_sizes": [128], "dropout": 0.3,
            "lr": 1e-3, "batch_size": 32, "num_epochs": 30,
        },
        # More shapelets
        {
            "n_shapelets": 50, "shapelet_len_frac": 0.1, "shapelet_iter": 50,
            "hidden_sizes": [128], "dropout": 0.3,
            "lr": 1e-3, "batch_size": 32, "num_epochs": 30,
        },
        # Longer shapelets
        {
            "n_shapelets": 20, "shapelet_len_frac": 0.2, "shapelet_iter": 50,
            "hidden_sizes": [128], "dropout": 0.3,
            "lr": 1e-3, "batch_size": 32, "num_epochs": 30,
        },
        # Deeper MLP
        {
            "n_shapelets": 20, "shapelet_len_frac": 0.1, "shapelet_iter": 50,
            "hidden_sizes": [256, 128], "dropout": 0.3,
            "lr": 1e-3, "batch_size": 32, "num_epochs": 30,
        },
        # Higher dropout
        {
            "n_shapelets": 20, "shapelet_len_frac": 0.1, "shapelet_iter": 50,
            "hidden_sizes": [128], "dropout": 0.5,
            "lr": 1e-3, "batch_size": 32, "num_epochs": 30,
        },
        # Best guess combo
        {
            "n_shapelets": 50, "shapelet_len_frac": 0.15, "shapelet_iter": 100,
            "hidden_sizes": [256, 128], "dropout": 0.3,
            "lr": 1e-3, "batch_size": 32, "num_epochs": 50,
        },
    ]

    results = run_ablation(X_train, X_test, y_train, y_test, class_names, configs)
    print_ablation_table(results)
