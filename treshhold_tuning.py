import numpy as np
import json
from sklearn.metrics import precision_recall_curve


def tune_threshold_f1(y_true, y_score, save_path=None):

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(
                {
                    "threshold": float(best_threshold),
                    "metric": "f1",
                    "best_f1": float(best_f1),
                },
                f,
                indent=2
            )

    return best_threshold, best_f1
