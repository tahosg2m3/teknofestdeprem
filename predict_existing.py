import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class EarthquakeMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev, h),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(h),
                torch.nn.Dropout(dropout),
            ])
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Run predictions on existing earthquake records.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--target-col", default=None)
    p.add_argument("--event-threshold", type=float, default=3.5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out-csv", default="artifacts/predictions.csv")
    p.add_argument("--summary-out", default="artifacts/prediction_summary.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    feature_cols = checkpoint["feature_cols"]
    mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)

    df = pd.read_csv(args.data_path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in input CSV: {missing}")

    x = df[feature_cols].values.astype(np.float32)
    x = (x - mean) / scale

    input_dim = len(feature_cols)
    state = checkpoint["model_state_dict"]
    hidden_dims = checkpoint.get("hidden_dims", [128, 64, 32])
    dropout = checkpoint.get("dropout", 0.2)

    model = EarthquakeMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dl = DataLoader(TabularDataset(x), batch_size=args.batch_size, shuffle=False)

    preds = []
    start = time.perf_counter()
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy().ravel())
    elapsed = time.perf_counter() - start

    y_pred = np.concatenate(preds)
    df["predicted_intensity"] = y_pred
    df["predicted_event"] = (df["predicted_intensity"] >= args.event_threshold).astype(int)

    summary = {
        "device": str(device),
        "num_records": int(len(df)),
        "inference_time_sec": float(elapsed),
        "throughput_samples_per_sec": float(len(df) / elapsed if elapsed > 0 else 0.0),
        "event_threshold": args.event_threshold,
    }

    if args.target_col and args.target_col in df.columns:
        y_true_evt = (df[args.target_col].values >= args.event_threshold).astype(int)
        y_pred_evt = df["predicted_event"].values.astype(int)

        tp = int(((y_true_evt == 1) & (y_pred_evt == 1)).sum())
        tn = int(((y_true_evt == 0) & (y_pred_evt == 0)).sum())
        fp = int(((y_true_evt == 0) & (y_pred_evt == 1)).sum())
        fn = int(((y_true_evt == 1) & (y_pred_evt == 0)).sum())

        summary.update(
            {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "false_alarm_rate_percent": float(fp / (fp + tn) * 100 if (fp + tn) > 0 else 0),
                "miss_rate_percent": float(fn / (fn + tp) * 100 if (fn + tp) > 0 else 0),
                "accuracy_percent": float((tp + tn) / len(df) * 100 if len(df) > 0 else 0),
            }
        )

    df.to_csv(args.out_csv, index=False)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OK] Tahminler: {args.out_csv}")
    print(f"[OK] Özet: {args.summary_out}")


if __name__ == "__main__":
    main()
