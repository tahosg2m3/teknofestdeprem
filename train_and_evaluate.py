import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EarthquakeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_and_prepare_data(
    data_path: str,
    target_col: str,
    feature_cols: Optional[List[str]],
    time_col: Optional[str],
    val_ratio: float,
    test_ratio: float,
) -> Tuple[SplitData, StandardScaler, List[str]]:
    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not in dataset: {missing}")
    else:
        excluded = {target_col}
        if time_col:
            excluded.add(time_col)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in excluded]

    if not feature_cols:
        raise ValueError("No usable feature columns found.")

    used_cols = feature_cols + [target_col]
    if time_col:
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in dataset.")
        df = df.sort_values(time_col)
        used_cols.append(time_col)

    df = df[used_cols].dropna().reset_index(drop=True)

    x = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    if len(df) < 100:
        raise ValueError("Dataset is too small. Provide at least 100 rows for stable training.")

    if time_col:
        n = len(df)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)
        train_size = n - test_size - val_size

        if train_size <= 0:
            raise ValueError("Invalid split ratios; train size became non-positive.")

        x_train, y_train = x[:train_size], y[:train_size]
        x_val, y_val = x[train_size: train_size + val_size], y[train_size: train_size + val_size]
        x_test, y_test = x[train_size + val_size:], y[train_size + val_size:]
    else:
        x_train, x_tmp, y_train, y_tmp = train_test_split(
            x, y, test_size=(val_ratio + test_ratio), random_state=42
        )
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        x_val, x_test, y_val, y_test = train_test_split(
            x_tmp, y_tmp, test_size=relative_test_ratio, random_state=42
        )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    split = SplitData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
    return split, scaler, feature_cols


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def evaluate_detection(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    y_true_evt = (y_true >= threshold).astype(int)
    y_pred_evt = (y_pred >= threshold).astype(int)

    tp = int(((y_true_evt == 1) & (y_pred_evt == 1)).sum())
    tn = int(((y_true_evt == 0) & (y_pred_evt == 0)).sum())
    fp = int(((y_true_evt == 0) & (y_pred_evt == 1)).sum())
    fn = int(((y_true_evt == 1) & (y_pred_evt == 0)).sum())

    accuracy = accuracy_score(y_true_evt, y_pred_evt)
    precision = precision_score(y_true_evt, y_pred_evt, zero_division=0)
    recall = recall_score(y_true_evt, y_pred_evt, zero_division=0)
    f1 = f1_score(y_true_evt, y_pred_evt, zero_division=0)

    false_alarm_rate = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    miss_rate = (fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy_percent": float(accuracy * 100.0),
        "false_alarm_rate_percent": float(false_alarm_rate),
        "miss_rate_percent": float(miss_rate),
    }


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    ds = TabularDataset(x, np.zeros(len(x), dtype=np.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs = []
    for xb, _ in dl:
        xb = xb.to(device)
        pred = model(xb)
        outs.append(pred.detach().cpu().numpy().ravel())
    return np.concatenate(outs)


def save_artifacts(model, scaler, feature_cols: List[str], hidden_dims: List[int], dropout: float, model_out: str):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": feature_cols,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
    }
    torch.save(payload, model_out)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate earthquake intensity model.")
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--target-col", required=True, type=str)
    parser.add_argument("--feature-cols", nargs="*", default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--event-threshold", type=float, default=3.5)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64, 32])

    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--model-out", type=str, default="artifacts/model.pt")
    parser.add_argument("--metrics-out", type=str, default="artifacts/metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()

    device = resolve_device()
    print(f"[INFO] Device: {device}")
    if str(device) == "cpu":
        print("[WARN] GPU tespit edilmedi. ROCm/PyTorch kurulumunu kontrol edin.")

    split, scaler, feature_cols = load_and_prepare_data(
        data_path=args.data_path,
        target_col=args.target_col,
        feature_cols=args.feature_cols,
        time_col=args.time_col,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_ds = TabularDataset(split.x_train, split.y_train)
    val_ds = TabularDataset(split.x_val, split.y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = EarthquakeMLP(
        input_dim=split.x_train.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"[EPOCH {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    start = time.perf_counter()
    y_pred_test = predict(model, split.x_test, device=device, batch_size=args.batch_size)
    elapsed = time.perf_counter() - start

    reg_metrics = evaluate_regression(split.y_test, y_pred_test)
    det_metrics = evaluate_detection(split.y_test, y_pred_test, args.event_threshold)

    throughput = len(split.y_test) / elapsed if elapsed > 0 else 0.0

    metrics = {
        "device": str(device),
        "num_features": len(feature_cols),
        "num_train": int(len(split.y_train)),
        "num_val": int(len(split.y_val)),
        "num_test": int(len(split.y_test)),
        "inference_time_sec": float(elapsed),
        "throughput_samples_per_sec": float(throughput),
        "regression": reg_metrics,
        "detection": det_metrics,
    }

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_artifacts(model, scaler, feature_cols, args.hidden_dims, args.dropout, args.model_out)

    print("\n===== SONUÇ =====")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n[OK] Model kaydedildi: {args.model_out}")
    print(f"[OK] Metrikler kaydedildi: {args.metrics_out}")


if __name__ == "__main__":
    main()
