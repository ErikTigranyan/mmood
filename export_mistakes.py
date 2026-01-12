import pandas as pd


def export_fp_fn(
    X,
    y_true,
    y_score,
    threshold,
    fp_path,
    fn_path
):
   
    preds = (y_score >= threshold).astype(int)

    df = X.copy()
    df["y_true"] = y_true
    df["y_score"] = y_score
    df["y_pred"] = preds

    false_positives = df[(df["y_pred"] == 1) & (df["y_true"] == 0)]
    false_negatives = df[(df["y_pred"] == 0) & (df["y_true"] == 1)]

    false_positives.to_csv(fp_path, index=False)
    false_negatives.to_csv(fn_path, index=False)
