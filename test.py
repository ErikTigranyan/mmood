import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

from threshold_tuning import tune_threshold_f1
from export_mistakes import export_fp_fn


def load_data(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    df1["target"] = 0
    df2["target"] = 1

    df = pd.concat([df1, df2], ignore_index  = True)

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y

def create_model(model_name, params):
    if model_name == "logreg":
        return LogisticRegression(**params, max_iter = 1000)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def run_proj(run_cfg, X_train, X_test, y_train, y_test, out):
    model = create_model(run_cfg["model"], run_cfg["params"])

    pipeline = Pipeline([
        ("scaler", StandardScaler()), 
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_score = pipeline.predict_proba(X_test)[:, 1]

    best_threshold, best_f1 = tune_threshold_f1(
        y_test,
        y_score,
        save_path=out / "threshold.json"
    )
    preds = (y_score >= best_threshold).astype(int)
    preds = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    out.mkdir(parents = True, exist_ok = True)

    joblib.dump(pipeline, out / "model.pkl")

    with open(out / "metrics.json", "w") as f:
            json.dump({"accuracy": accuracy}, f, indent=2)

    with open(out / "config.json", "w") as f:
            json.dump(run_cfg, f, indent=2)

    export_fp_fn(
        X_test,
        y_test,
        y_score,
        threshold=best_threshold,
        fp_path=out / "false_positives.csv",
        fn_path=out / "false_negatives.csv"
    )
    
    print(f"{run_cfg['name']} | accuracy = {accuracy:.3f}")

def main(args):
    X, y = load_data(args.input1, args.input2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with open(args.config) as f:
        config = json.load(f)

    for run_cfg in config["runs"]:
        run_dir = Path(args.output) / run_cfg["name"]
        run_proj(run_cfg, X_train, X_test, y_train, y_test, run_dir)


if  __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", required=True)
    parser.add_argument("--input2", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    main(args)



