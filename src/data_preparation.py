"""
BBM409 Assignment 4 - Data Preparation
Loads extracted .npz files, encodes labels, pads/truncates sequences,
and splits into train/test sets.
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ── 1. Load all .npz files ─────────────────────────────────────────────────────
def load_dataset(features_root):
    """
    Walk through extracted_features/<word>/*.npz and load all sequences.

    Returns:
        sequences : list of np.ndarray, each shape (T_i, 80) — variable length
        labels    : list of str, word label for each sequence
    """
    features_root = Path(features_root)
    sequences = []
    labels = []

    for word_dir in sorted(features_root.iterdir()):
        if not word_dir.is_dir():
            continue
        word = word_dir.name

        for npz_path in sorted(word_dir.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            seq = data["normalized"]  # shape (T, 80)

            # Skip sequences that are entirely NaN (failed extraction)
            if np.all(np.isnan(seq)):
                continue

            # Replace any remaining NaN frames with zeros
            # (isolated missed detections — interpolation would be better
            #  but zeros are safe and simple)
            seq = np.nan_to_num(seq, nan=0.0)

            sequences.append(seq)
            labels.append(word)

    print(f"Loaded {len(sequences)} sequences across {len(set(labels))} classes")
    return sequences, labels


# ── 2. Analyze sequence lengths ────────────────────────────────────────────────
def analyze_lengths(sequences, plot=True):
    """
    Print length statistics and optionally plot a histogram.
    Use this to choose a good MAX_LEN for padding/truncation.
    """
    lengths = [len(s) for s in sequences]
    lengths = np.array(lengths)

    print(f"Sequence length stats:")
    print(f"  min    : {lengths.min()}")
    print(f"  max    : {lengths.max()}")
    print(f"  mean   : {lengths.mean():.1f}")
    print(f"  median : {np.median(lengths):.1f}")
    print(f"  p90    : {np.percentile(lengths, 90):.1f}")
    print(f"  p95    : {np.percentile(lengths, 95):.1f}")

    if plot:
        plt.figure(figsize=(8, 4))
        plt.hist(lengths, bins=40, edgecolor="black")
        plt.axvline(np.percentile(lengths, 95), color="red", linestyle="--",
                    label=f"95th percentile ({np.percentile(lengths, 95):.0f})")
        plt.xlabel("Sequence length (frames)")
        plt.ylabel("Count")
        plt.title("Distribution of word sequence lengths")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return lengths


# ── 3. Pad / truncate to fixed length ─────────────────────────────────────────
def pad_sequences(sequences, max_len):
    """
    Pad sequences shorter than max_len with zeros at the end,
    and truncate sequences longer than max_len from the end.

    Args:
        sequences : list of np.ndarray, each shape (T_i, 80)
        max_len   : int, fixed sequence length to use

    Returns:
        X : np.ndarray of shape (N, max_len, 80)
    """
    feature_dim = sequences[0].shape[1]  # 80
    X = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)

    for i, seq in enumerate(sequences):
        T = len(seq)
        if T >= max_len:
            X[i] = seq[:max_len]       # truncate
        else:
            X[i, :T] = seq             # pad with zeros at the end

    return X


# ── 4. Encode labels ───────────────────────────────────────────────────────────
def encode_labels(labels):
    """
    Convert string labels to integer class indices.

    Returns:
        y      : np.ndarray of int, shape (N,)
        le     : fitted LabelEncoder (use le.inverse_transform to decode)
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    return y, le


# ── 5. Train / test split ──────────────────────────────────────────────────────
def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Stratified train/test split so class distribution is balanced in both sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # preserve class distribution
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


# ── 6. Full pipeline ───────────────────────────────────────────────────────────
def prepare_data(features_root="extracted_features", test_size=0.2, max_len=None):
    """
    Run the full data preparation pipeline.

    Args:
        features_root : path to extracted_features/
        test_size     : fraction of data to use for testing
        max_len       : fixed sequence length. If None, uses the 95th percentile
                        of sequence lengths automatically.

    Returns:
        X_train, X_test : np.ndarray, shape (N, max_len, 80)
        y_train, y_test : np.ndarray, shape (N,)
        le              : LabelEncoder (to decode predictions back to words)
        max_len         : the max_len that was used (useful if auto-selected)
    """
    # Load
    sequences, labels = load_dataset(features_root)

    # Analyze lengths and choose max_len
    lengths = analyze_lengths(sequences, plot=True)
    if max_len is None:
        max_len = int(np.percentile(lengths, 95))
        print(f"\nAuto-selected MAX_LEN = {max_len} (95th percentile)")
    else:
        print(f"\nUsing MAX_LEN = {max_len}")

    covered = np.mean(lengths <= max_len) * 100
    print(f"  {covered:.1f}% of sequences fit without truncation")

    # Pad / truncate
    X = pad_sequences(sequences, max_len)
    print(f"\nPadded dataset shape: {X.shape}")  # (N, max_len, 80)

    # Encode labels
    y, le = encode_labels(labels)

    # Split
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test, le, max_len


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le, max_len = prepare_data(
        features_root="extracted_features",
        test_size=0.2,
        max_len=None,        # auto-select from data
    )

    print("\nFinal shapes:")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  y_test  : {y_test.shape}")
    print(f"  Classes : {list(le.classes_)}")

    # Save prepared data so you don't re-run this every time
    np.savez(
        "prepared_data.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classes=le.classes_,
        max_len=max_len,
    )
    print("\nSaved to prepared_data.npz")
