#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Tuple, Optional

import numpy as np

# Optional pandas/matplotlib imports with helpful errors
try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    pd = None  # Will error later with message

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # Will error later with message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train logistic/softmax regression and plot average negative log-likelihood (NLL) "
            "for training and validation over epochs."
        )
    )
    parser.add_argument("--train", required=True, help="Path to training CSV/TSV file")
    parser.add_argument("--valid", required=True, help="Path to validation CSV/TSV file")
    parser.add_argument(
        "--label-col",
        default=None,
        help=(
            "Label column name or zero-based index. If omitted, tries 'label', then 'y', then last column."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs to train (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for gradient descent (default: 0.1)",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize features using train mean/std before training",
    )
    parser.add_argument(
        "--output",
        default="nll_over_epochs.png",
        help="Output path for the generated PNG plot",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to save a CSV of per-epoch NLLs (train, valid)",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Column delimiter override. Auto-detected by extension if not provided.",
    )
    return parser.parse_args()


def _infer_sep(path: str, sep_arg: Optional[str]) -> str:
    if sep_arg is not None:
        return sep_arg
    lower = path.lower()
    if lower.endswith(".tsv") or lower.endswith(".tab"):
        return "\t"
    return ","


def _resolve_label_column(df: "pd.DataFrame", label_col: Optional[str]) -> str:
    if label_col is not None:
        # numeric index or name
        try:
            idx = int(label_col)
            return df.columns[idx]
        except Exception:
            if label_col in df.columns:
                return label_col
            raise ValueError(
                f"Specified label column '{label_col}' not found among columns: {list(df.columns)}"
            )
    for candidate in ("label", "y"):
        if candidate in df.columns:
            return candidate
    return df.columns[-1]


def _select_numeric_features(df: "pd.DataFrame", label_col_name: str) -> Tuple[np.ndarray, np.ndarray, list]:
    y = df[label_col_name].to_numpy()

    # Coerce all columns (except label) to numeric where possible; drop completely non-numeric
    features_df = df.drop(columns=[label_col_name]).copy()

    # Attempt numeric coercion; non-convertible becomes NaN, then we drop columns that are entirely NaN
    for col in list(features_df.columns):
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
    # Drop columns that are entirely NaN
    all_nan_cols = [c for c in features_df.columns if features_df[c].isna().all()]
    if len(all_nan_cols) == len(features_df.columns):
        raise ValueError("No numeric features found after coercion. Please provide numeric features.")
    if all_nan_cols:
        features_df = features_df.drop(columns=all_nan_cols)

    # Drop rows with any NaNs in features or labels
    # (Safer default; alternatively could impute)
    before_rows = len(features_df)
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    combined = features_df.join(pd.Series(y, name="__y__"))
    combined = combined.dropna(axis=0)
    y_clean = combined["__y__"].to_numpy()
    X_clean = combined.drop(columns=["__y__"]).to_numpy(dtype=np.float64)

    dropped = before_rows - len(X_clean)
    if dropped > 0:
        print(f"[info] Dropped {dropped} rows due to NaNs or infs in features/labels.")

    return X_clean, y_clean, list(features_df.columns)


def _standardize_train_valid(X_train: np.ndarray, X_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_std = (X_train - mean) / std
    X_valid_std = (X_valid - mean) / std
    return X_train_std, X_valid_std


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])


def _map_labels_to_int(y: np.ndarray) -> Tuple[np.ndarray, dict]:
    # Map arbitrary labels to {0..C-1}
    uniques = np.unique(y)
    mapping = {val: i for i, val in enumerate(uniques)}
    y_int = np.vectorize(mapping.get)(y)
    return y_int.astype(int), mapping


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    out = np.empty_like(z, dtype=np.float64)
    positive_mask = z >= 0
    negative_mask = ~positive_mask
    out[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))
    exp_z = np.exp(z[negative_mask])
    out[negative_mask] = exp_z / (1.0 + exp_z)
    return out


def logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    s = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
    out = a_max + np.log(s)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def nll_binary(Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    # Xb: (N, D+1), w: (D+1,)
    z = Xb @ w
    p = sigmoid(z)
    eps = 1e-12
    nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return float(nll)


def nll_multiclass(Xb: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
    # Xb: (N, D+1), W: (D+1, C)
    logits = Xb @ W
    lse = logsumexp(logits, axis=1, keepdims=True)
    log_probs = logits - lse
    n = Xb.shape[0]
    nll = -np.mean(log_probs[np.arange(n), y])
    return float(nll)


def train_logistic(
    X_train_b: np.ndarray,
    y_train: np.ndarray,
    X_valid_b: np.ndarray,
    y_valid: np.ndarray,
    epochs: int,
    lr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Binary logistic regression via gradient descent on average NLL
    n_features = X_train_b.shape[1]
    w = np.zeros(n_features, dtype=np.float64)

    train_nlls = np.zeros(epochs, dtype=np.float64)
    valid_nlls = np.zeros(epochs, dtype=np.float64)

    for t in range(epochs):
        z = X_train_b @ w
        p = sigmoid(z)
        # gradient of average NLL
        grad = (X_train_b.T @ (p - y_train)) / X_train_b.shape[0]
        w -= lr * grad

        train_nlls[t] = nll_binary(X_train_b, y_train, w)
        valid_nlls[t] = nll_binary(X_valid_b, y_valid, w)

    return w, train_nlls, valid_nlls


def train_softmax(
    X_train_b: np.ndarray,
    y_train: np.ndarray,
    X_valid_b: np.ndarray,
    y_valid: np.ndarray,
    num_classes: int,
    epochs: int,
    lr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Multiclass softmax regression via gradient descent on average NLL
    n_features = X_train_b.shape[1]
    W = np.zeros((n_features, num_classes), dtype=np.float64)

    train_nlls = np.zeros(epochs, dtype=np.float64)
    valid_nlls = np.zeros(epochs, dtype=np.float64)

    # One-hot representation for gradient computation
    n_train = X_train_b.shape[0]
    Y_one_hot = np.zeros((n_train, num_classes), dtype=np.float64)
    Y_one_hot[np.arange(n_train), y_train] = 1.0

    for t in range(epochs):
        logits = X_train_b @ W
        lse = logsumexp(logits, axis=1, keepdims=True)
        probs = np.exp(logits - lse)
        # gradient of average NLL
        grad = (X_train_b.T @ (probs - Y_one_hot)) / n_train
        W -= lr * grad

        train_nlls[t] = nll_multiclass(X_train_b, y_train, W)
        valid_nlls[t] = nll_multiclass(X_valid_b, y_valid, W)

    return W, train_nlls, valid_nlls


def plot_nll(train_nlls: np.ndarray, valid_nlls: np.ndarray, output_path: str) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to plot. Please install dependencies: pip install -r requirements.txt"
        )
    epochs = np.arange(1, len(train_nlls) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_nlls, label="Train NLL", color="#1f77b4")
    plt.plot(epochs, valid_nlls, label="Validation NLL", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("Average Negative Log-Likelihood")
    plt.title("Training and Validation NLL over Epochs")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.tight_layout()
    # Keep y-axis starting at zero if all values positive
    if np.all(np.isfinite(train_nlls)) and np.all(np.isfinite(valid_nlls)):
        ymin = max(0.0, min(train_nlls.min(), valid_nlls.min()) * 0.95)
        ymax = max(train_nlls.max(), valid_nlls.max()) * 1.05
        plt.ylim(ymin, ymax)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    if pd is None:
        raise RuntimeError(
            "pandas is required to load data. Please install dependencies: pip install -r requirements.txt"
        )

    if not os.path.exists(args.train):
        raise FileNotFoundError(f"Train file not found: {args.train}")
    if not os.path.exists(args.valid):
        raise FileNotFoundError(f"Validation file not found: {args.valid}")

    sep_train = _infer_sep(args.train, args.sep)
    sep_valid = _infer_sep(args.valid, args.sep)

    print(f"[info] Loading train from {args.train} (sep='{sep_train}')")
    df_train = pd.read_csv(args.train, sep=sep_train)
    print(f"[info] Loading valid from {args.valid} (sep='{sep_valid}')")
    df_valid = pd.read_csv(args.valid, sep=sep_valid)

    label_col_name = _resolve_label_column(df_train, args.label_col)
    if label_col_name not in df_valid.columns:
        # If validation lacks same label column name but has 'label'/'y'/last, resolve independently
        label_col_name_valid = _resolve_label_column(df_valid, args.label_col)
    else:
        label_col_name_valid = label_col_name

    X_train, y_train_raw, feature_names = _select_numeric_features(df_train, label_col_name)
    X_valid, y_valid_raw, _ = _select_numeric_features(df_valid, label_col_name_valid)

    # Align features by column names if possible (handles column order differences)
    # Rebuild X matrices using the same set of feature columns where possible
    common_cols = None
    try:
        # Attempt to align based on original dataframes to preserve named columns
        features_train_df = df_train.drop(columns=[label_col_name])
        features_valid_df = df_valid.drop(columns=[label_col_name_valid])
        for col in list(features_train_df.columns):
            features_train_df[col] = pd.to_numeric(features_train_df[col], errors="coerce")
        for col in list(features_valid_df.columns):
            features_valid_df[col] = pd.to_numeric(features_valid_df[col], errors="coerce")
        features_train_df = features_train_df.replace([np.inf, -np.inf], np.nan)
        features_valid_df = features_valid_df.replace([np.inf, -np.inf], np.nan)
        common_cols = [c for c in features_train_df.columns if c in features_valid_df.columns]
        # Drop all-NaN columns after alignment
        features_train_df = features_train_df[common_cols]
        features_valid_df = features_valid_df[common_cols]
        features_train_df = features_train_df.dropna(axis=0)
        features_valid_df = features_valid_df.dropna(axis=0)
        y_train_raw = df_train.loc[features_train_df.index, label_col_name].to_numpy()
        y_valid_raw = df_valid.loc[features_valid_df.index, label_col_name_valid].to_numpy()
        X_train = features_train_df.to_numpy(dtype=np.float64)
        X_valid = features_valid_df.to_numpy(dtype=np.float64)
        if len(common_cols) == 0:
            common_cols = None  # fallback to previous X matrices
    except Exception:
        common_cols = None

    if args.standardize:
        X_train, X_valid = _standardize_train_valid(X_train, X_valid)

    # Add bias term
    X_train_b = _add_bias(X_train)
    X_valid_b = _add_bias(X_valid)

    # Map labels to integers
    y_train, mapping = _map_labels_to_int(y_train_raw)
    y_valid = np.vectorize(mapping.get)(y_valid_raw)
    # Handle unseen labels in validation by mapping to nearest known (raise error instead)
    if np.any(np.isnan(y_valid)):
        raise ValueError("Validation set contains labels not seen in training set.")
    y_valid = y_valid.astype(int)

    num_classes = int(np.max(y_train)) + 1

    print(
        f"[info] Data shapes: X_train={X_train.shape}, X_valid={X_valid.shape}, classes={num_classes}"
    )

    if num_classes == 2:
        # Convert labels to 0/1 explicitly in case mapping not {0,1}
        y_train_bin = (y_train == 1).astype(np.float64)
        y_valid_bin = (y_valid == 1).astype(np.float64)
        _, train_nlls, valid_nlls = train_logistic(
            X_train_b, y_train_bin, X_valid_b, y_valid_bin, epochs=args.epochs, lr=args.lr
        )
    else:
        _, train_nlls, valid_nlls = train_softmax(
            X_train_b,
            y_train,
            X_valid_b,
            y_valid,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
        )

    # Save optional CSV
    if args.save_csv:
        out_dir = os.path.dirname(args.save_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        import csv

        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_nll", "valid_nll"])
            for i, (tr, va) in enumerate(zip(train_nlls, valid_nlls), start=1):
                writer.writerow([i, float(tr), float(va)])
        print(f"[info] Wrote NLL CSV to {args.save_csv}")

    # Plot
    out_plot = args.output
    out_dir = os.path.dirname(out_plot)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plot_nll(train_nlls, valid_nlls, out_plot)
    print(f"[done] Saved plot to {out_plot}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
