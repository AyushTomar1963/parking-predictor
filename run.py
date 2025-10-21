#!/usr/bin/env python3
# run.py  -- project root runner for training / demo / eval / serving

import argparse
import sys
import os
from pathlib import Path
import joblib

# add project root to path so src imports work when running this file
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def train_lightgbm_from_processed(data_path: str, out_path: str, test_frac: float = 0.2, num_boost_round: int = 200):
    """
    Train a LightGBM model using a processed CSV that already has:
    Occupancy, hour_of_day, day_of_week, is_weekend, lag_1,...lag_48
    Saves a joblib file to out_path.
    """
    import pandas as pd
    import numpy as np

    # Import your LightGBM wrapper from src.models (adjust path/name if different)
    try:
        from src.models import LightGBMModel, evaluate_model
    except Exception as e:
        raise RuntimeError("Failed to import LightGBMModel from src.models: " + str(e))

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=True, infer_datetime_format=True)
    # If the file has LastUpdated as index or column, ensure index is datetime
    if 'LastUpdated' in df.columns:
        df = df.set_index(pd.to_datetime(df['LastUpdated']))

    # required columns
    required = ['Occupancy', 'hour_of_day', 'day_of_week', 'is_weekend',
                'lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in processed CSV: {missing}")

    df = df.dropna().copy()  # require no NaNs for training here
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 'hour_of_day', 'day_of_week', 'is_weekend']
    target = 'Occupancy'

    split = int(len(df) * (1 - test_frac))
    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    print(f"Training LightGBM: {len(X_train)} train samples, {len(X_test)} test samples")
    model = LightGBMModel(params={'objective': 'regression', 'metric': 'l2', 'verbosity': -1, 'seed': 42},
                          num_boost_round=num_boost_round)
    model.fit(X_train, y_train)

    # evaluate if evaluate_model exists else fallback
    try:
        metrics = evaluate_model(y_test, model.predict(X_test))
        print("Evaluation on test set:", metrics)
    except Exception:
        print("Warning: evaluate_model not found / raised error; trained model saved nonetheless.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(out_path))
    print(f"Saved trained model to {out_path}")
    return str(out_path)


def run_demo(csv_path: str, lot: str, capacity: int, save_model: bool):
    """Wrapper that invokes your existing demo script if present, else calls basic demo flow"""
    # prefer existing script
    demo_script = ROOT / 'scripts' / 'demo_pipeline.py'
    if demo_script.exists():
        print("Running scripts/demo_pipeline.py ...")
        os.execv(sys.executable, [sys.executable, str(demo_script), '--data-path', csv_path, '--lot', lot,
                                   '--capacity', str(capacity), '--save-model' if save_model else '--no-save'])
    else:
        print("demo_pipeline.py not found in scripts/, please run your notebook or scripts/demo_pipeline.py manually.")


def evaluate_models_main(data_path: str, model_dir: str, model_choice: str, plot: bool):
    eval_script = ROOT / 'scripts' / 'evaluate_models.py'
    if eval_script.exists():
        print("Running scripts/evaluate_models.py ...")
        cmd = [sys.executable, str(eval_script), '--data-path', data_path, '--model-dir', model_dir, '--model', model_choice]
        if plot:
            cmd.append('--plot')
        os.execv(sys.executable, cmd)
    else:
        print("evaluate_models.py not found. Please run scripts/evaluate_models.py manually.")


def serve_app(host: str, port: int, reload: bool = True):
    """Start FastAPI app via uvicorn (assumes app.main:app)"""
    import subprocess
    uvicorn_cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)]
    if reload:
        uvicorn_cmd.append("--reload")
    print("Starting FastAPI with:", " ".join(uvicorn_cmd))
    subprocess.run(uvicorn_cmd)


def check_path(path: str):
    p = Path(path)
    print(f"{path} -> exists: {p.exists()} (is_file: {p.is_file()}, is_dir: {p.is_dir()})")
    return p.exists()


def main():
    parser = argparse.ArgumentParser(prog="run.py", description="Run tasks for Parking Predictor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train", help="Train LightGBM from processed CSV and save joblib")
    t.add_argument("--data", required=True, help="Processed CSV with features (Occupancy + lags)")
    t.add_argument("--out", default="data/models/lightgbm_model.joblib", help="Output model path (joblib)")
    t.add_argument("--test-frac", type=float, default=0.2)
    t.add_argument("--rounds", type=int, default=200)

    # demo
    d = sub.add_parser("demo", help="Run demo pipeline (script or internal) ")
    d.add_argument("--data", default="data/raw/dataset.csv")
    d.add_argument("--lot", default="BHMBCCMKT01")
    d.add_argument("--capacity", type=int, default=600)
    d.add_argument("--save-model", action="store_true")

    # eval
    e = sub.add_parser("eval", help="Evaluate models (calls scripts/evaluate_models.py)")
    e.add_argument("--data", default="data/raw/dataset.csv")
    e.add_argument("--model-dir", default="data/models")
    e.add_argument("--model", default="all", choices=["lightgbm", "xgboost", "arima", "all"])
    e.add_argument("--plot", action="store_true")

    # serve
    s = sub.add_parser("serve", help="Run API server (uvicorn app.main:app)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-reload", dest="reload", action="store_false")

    # quick check
    c = sub.add_parser("check", help="Check existence of files/dirs")
    c.add_argument("path", help="Path to check")

    args = parser.parse_args()

    if args.cmd == "train":
        train_lightgbm_from_processed(args.data, args.out, test_frac=args.test_frac, num_boost_round=args.rounds)

    elif args.cmd == "demo":
        run_demo(args.data, args.lot, args.capacity, args.save_model)

    elif args.cmd == "eval":
        evaluate_models_main(args.data, args.model_dir, args.model, args.plot)

    elif args.cmd == "serve":
        serve_app(args.host, args.port, reload=args.reload)

    elif args.cmd == "check":
        check_path(args.path)


if __name__ == "__main__":
    main()