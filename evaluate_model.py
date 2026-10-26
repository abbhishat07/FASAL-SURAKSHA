"""
FASAL SURAKSHA - evaluate_model.py
Measure how good your models really are.  Three commands:

  1) Disease classifier (MobileNetV3):
       python evaluate_model.py classifier --weights models/best.pt --test_dir dataset_test --out eval_out

     test_dir layout (same as training, one folder per class):
       dataset_test/Tomato___healthy/*.jpg
       dataset_test/Tomato___Late_blight/*.jpg ...
     IMPORTANT: these images must NEVER have been used for training or tuning.

  2) Yield model (XGBoost or any regressor) from a CSV with true and predicted values:
       python evaluate_model.py regression --csv yield_test.csv --true actual_yield --pred predicted_yield

  3) Severity (DSI %) vs expert scores from a CSV:
       python evaluate_model.py severity --csv severity.csv --expert expert_pct --pred app_pct
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ───────────────────────── classifier ─────────────────────────
def expected_calibration_error(conf, correct, bins=10):
    """Gap between 'how sure the model says it is' and 'how often it is right'. 0 is perfect."""
    edges = np.linspace(0, 1, bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def eval_classifier(weights, test_dir, out_dir, min_conf=60.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                                 cohen_kappa_score, confusion_matrix, matthews_corrcoef,
                                 precision_recall_fscore_support)
    from utils.gradcam import FasalGradCAM

    os.makedirs(out_dir, exist_ok=True)
    classes = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
    model = FasalGradCAM(weights)
    if model.num_classes != len(classes):
        sys.exit(f"Model has {model.num_classes} classes but test_dir has {len(classes)} folders.")

    y_true, y_pred, probs_all, bad = [], [], [], 0
    for ci, cname in enumerate(classes):
        folder = os.path.join(test_dir, cname)
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            img = cv2.imread(os.path.join(folder, fn))
            if img is None:
                bad += 1
                continue
            idx, _, probs = model.predict(img)
            y_true.append(ci); y_pred.append(idx); probs_all.append(probs)
    if not y_true:
        sys.exit("No test images found.")
    y_true, y_pred, probs_all = np.array(y_true), np.array(y_pred), np.vstack(probs_all)
    conf = probs_all.max(1)
    correct = (y_true == y_pred).astype(float)

    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    top3 = float(np.mean([t in np.argsort(p)[::-1][:3] for t, p in zip(y_true, probs_all)]))
    summary = {
        "images_tested": int(len(y_true)), "unreadable_images_skipped": bad,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "top3_accuracy": top3,
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mean_confidence": float(conf.mean()),
        "calibration_error_ECE": expected_calibration_error(conf, correct),
        f"share_below_{int(min_conf)}pct_confidence": float((conf < min_conf / 100).mean()),
        "accuracy_when_confident": float(correct[conf >= min_conf / 100].mean()) if (conf >= min_conf / 100).any() else None,
    }
    summary = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in summary.items()}

    report = classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(os.path.join(out_dir, "per_class_report.csv"))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(6, 0.45 * len(classes))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (row-normalised)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)

    # most confused pairs
    off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)

    print(json.dumps(summary, indent=2))
    print("\nWorst 5 classes by recall:")
    per = sorted(((report[c]["recall"], c, int(report[c]["support"])) for c in classes))[:5]
    for r, c, s in per:
        print(f"  {c:45s} recall {r:.2f} (n={s})")
    print("\nMost common mistakes (count, actual -> predicted):")
    for n, a, b in off[:5]:
        print(f"  {n:4d}  {a} -> {b}")
    print(f"\nSaved to {out_dir}/: summary.json, per_class_report.csv, confusion_matrix.png")


# ───────────────────────── regression ─────────────────────────
def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nz = y_true != 0
    out = {
        "n": int(len(y_true)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_abs_error": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE_percent": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else None,
        "bias_mean_error": float(np.mean(y_pred - y_true)),   # >0 = model over-predicts
        "baseline_MAE_predict_the_mean": float(np.mean(np.abs(y_true - y_true.mean()))),
    }
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def eval_regression(csv, true_col, pred_col):
    df = pd.read_csv(csv).dropna(subset=[true_col, pred_col])
    m = regression_metrics(df[true_col], df[pred_col])
    print(json.dumps(m, indent=2))
    if m["MAE"] >= m["baseline_MAE_predict_the_mean"]:
        print("WARNING: the model is no better than always guessing the average.")


# ───────────────────────── severity vs expert ─────────────────────────
def eval_severity(csv, expert_col, pred_col):
    from scipy.stats import pearsonr, spearmanr
    df = pd.read_csv(csv).dropna(subset=[expert_col, pred_col])
    e, p = df[expert_col].astype(float).values, df[pred_col].astype(float).values
    cat = lambda x: np.digitize(x, [2, 20, 50])  # Healthy / Mild / Moderate / Severe
    m = regression_metrics(e, p)
    m["pearson_r"] = round(float(pearsonr(e, p)[0]), 4)
    m["spearman_rho"] = round(float(spearmanr(e, p)[0]), 4)
    m["category_agreement"] = round(float((cat(e) == cat(p)).mean()), 4)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classifier"); c.add_argument("--weights", required=True); c.add_argument("--test_dir", required=True)
    c.add_argument("--out", default="eval_out"); c.add_argument("--min_conf", type=float, default=60.0)
    r = sub.add_parser("regression"); r.add_argument("--csv", required=True); r.add_argument("--true", required=True); r.add_argument("--pred", required=True)
    s = sub.add_parser("severity"); s.add_argument("--csv", required=True); s.add_argument("--expert", required=True); s.add_argument("--pred", required=True)
    a = ap.parse_args()
    if a.cmd == "classifier":
        eval_classifier(a.weights, a.test_dir, a.out, a.min_conf)
    elif a.cmd == "regression":
        eval_regression(a.csv, a.true, a.pred)
    else:
        eval_severity(a.csv, a.expert, a.pred)
