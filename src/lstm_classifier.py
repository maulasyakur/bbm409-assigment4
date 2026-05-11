"""
BBM409 Assignment 4 - LSTM Classifier
Sequence classification of lip movement landmarks using a PyTorch LSTM.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Dataset ────────────────────────────────────────────────────────────────────
class LipDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, T, 80), y: (N,)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ──────────────────────────────────────────────────────────────────────
class LipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        """
        Args:
            input_size  : number of features per timestep (80)
            hidden_size : LSTM hidden state size
            num_layers  : number of stacked LSTM layers
            num_classes : number of word classes
            dropout     : dropout probability (applied between LSTM layers)
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, T, 80)
        out, (h_n, _) = self.lstm(x)
        # Use the last hidden state of the top layer as the sequence representation
        last_hidden = h_n[-1]                   # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)   # (batch, num_classes)
        return logits


# ── Training ───────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ── Training loop ──────────────────────────────────────────────────────────────
def train_model(model, train_loader, test_loader, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Reduce LR if val loss plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
                  f"Test  loss: {test_loss:.4f}, acc: {test_acc:.3f}")

    return history


# ── Plotting helpers ───────────────────────────────────────────────────────────
def plot_history(history, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["test_loss"], label="Test")
    ax1.set_title(f"Loss — {title}")
    ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["test_acc"], label="Test")
    ax2.set_title(f"Accuracy — {title}")
    ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.show()


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


# ── Ablation study ─────────────────────────────────────────────────────────────
def run_ablation(X_train, X_test, y_train, y_test, class_names, configs):
    """
    Run multiple LSTM configurations and collect results for comparison.

    Args:
        configs : list of dicts, each with keys:
                  hidden_size, num_layers, dropout, lr, batch_size, num_epochs

    Returns:
        results : list of dicts with config + final test accuracy
    """
    results = []
    input_size = X_train.shape[2]   # 80
    num_classes = len(class_names)

    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {i+1}/{len(configs)}: {cfg}")
        print(f"{'='*60}")

        train_ds = LipDataset(X_train, y_train)
        test_ds  = LipDataset(X_test,  y_test)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"])

        model = LipLSTM(
            input_size=input_size,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_classes=num_classes,
            dropout=cfg["dropout"],
        ).to(device)

        history = train_model(
            model, train_loader, test_loader,
            num_epochs=cfg["num_epochs"],
            lr=cfg["lr"],
        )

        criterion = nn.CrossEntropyLoss()
        _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)
        print(f"\nFinal test accuracy: {test_acc:.4f}")

        plot_history(history, title=str(cfg))
        plot_confusion_matrix(y_true, y_pred, class_names,
                              title=f"Config {i+1}: acc={test_acc:.3f}")

        results.append({**cfg, "test_acc": test_acc})

    return results


def print_ablation_table(results):
    print(f"\n{'='*80}")
    print(f"{'hidden':>8} {'layers':>7} {'dropout':>8} {'lr':>8} "
          f"{'batch':>6} {'epochs':>7} {'test_acc':>10}")
    print(f"{'-'*80}")
    for r in sorted(results, key=lambda x: -x["test_acc"]):
        print(f"{r['hidden_size']:>8} {r['num_layers']:>7} {r['dropout']:>8.2f} "
              f"{r['lr']:>8.5f} {r['batch_size']:>6} {r['num_epochs']:>7} "
              f"{r['test_acc']:>10.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load prepared data
    data = np.load("prepared_data.npz", allow_pickle=True)
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    class_names = list(data["classes"])

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Classes: {class_names}")

    # ── Ablation configurations ────────────────────────────────────────────────
    # Vary one thing at a time to isolate the effect of each hyperparameter.
    # Start small and scale up — this gives you the ablation table for the report.
    configs = [
        # Baseline
        {"hidden_size": 128, "num_layers": 1, "dropout": 0.3, "lr": 1e-3, "batch_size": 32, "num_epochs": 30},
        # Larger hidden
        {"hidden_size": 256, "num_layers": 1, "dropout": 0.3, "lr": 1e-3, "batch_size": 32, "num_epochs": 30},
        # More layers
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "batch_size": 32, "num_epochs": 30},
        # Higher dropout
        {"hidden_size": 128, "num_layers": 1, "dropout": 0.5, "lr": 1e-3, "batch_size": 32, "num_epochs": 30},
        # Lower LR
        {"hidden_size": 128, "num_layers": 1, "dropout": 0.3, "lr": 1e-4, "batch_size": 32, "num_epochs": 30},
        # Larger batch
        {"hidden_size": 128, "num_layers": 1, "dropout": 0.3, "lr": 1e-3, "batch_size": 64, "num_epochs": 30},
        # Best guess combo
        {"hidden_size": 256, "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "batch_size": 32, "num_epochs": 50},
    ]

    results = run_ablation(X_train, X_test, y_train, y_test, class_names, configs)
    print_ablation_table(results)
