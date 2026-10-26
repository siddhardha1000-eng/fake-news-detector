"""
Model Evaluation and Comparison Module
Computes:
- Precision, Recall, F1-score (binary, macro, weighted)
- Accuracy and ROC-AUC
- Confusion Matrix (TP, FP, TN, FN)
- Model Comparison Reports and Visualizations
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_single_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates a single model across all key classification metrics.

    Args:
        model: Trained scikit-learn classifier
        X_test: Transformed test feature matrix
        y_test: Ground truth labels (0 = Real, 1 = Fake)
        model_name: Name of model for reporting

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    y_proba = None
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            roc_auc = None

    # Calculate metrics
    acc = float(accuracy_score(y_test, y_pred))
    
    # Binary metrics (Class 1 = Fake News)
    prec_binary = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec_binary = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_bin = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    # Macro & Weighted metrics
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred).tolist()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Real News (0)", "Fake News (1)"],
        output_dict=True,
        zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision_fake": round(prec_binary, 4),
        "recall_fake": round(rec_binary, 4),
        "f1_fake": round(f1_bin, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": report_dict
    }


def compare_models(
    models_dict: Dict[str, Any],
    X_test: Any,
    y_test: Any
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], str]:
    """
    Evaluates and compares multiple classification models.

    Returns:
        - comparison_df: Formatted pandas DataFrame comparing all models
        - detailed_metrics: Dict mapping model_name -> detailed evaluation dict
        - best_model_name: Name of the top-performing model based on F1-macro score
    """
    detailed_metrics = {}
    summary_rows = []

    for name, model in models_dict.items():
        metrics = evaluate_single_model(model, X_test, y_test, model_name=name)
        detailed_metrics[name] = metrics
        summary_rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision (Fake)": metrics["precision_fake"],
            "Recall (Fake)": metrics["recall_fake"],
            "F1-Score (Fake)": metrics["f1_fake"],
            "F1-Score (Macro)": metrics["f1_macro"],
            "ROC-AUC": metrics["roc_auc"]
        })

    comparison_df = pd.DataFrame(summary_rows)
    # Sort models by F1-Score (Macro) descending
    comparison_df = comparison_df.sort_values(by="F1-Score (Macro)", ascending=False).reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["Model"]

    return comparison_df, detailed_metrics, best_model_name


def print_comparison_table(comparison_df: pd.DataFrame, best_model_name: str) -> None:
    """Prints a neat terminal comparison table."""
    print("\n" + "=" * 80)
    print("               MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    print(f"Top Performing Model: >>> {best_model_name} <<<")
    print("=" * 80 + "\n")
