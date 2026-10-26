"""
Machine Learning Classification Models Module
Implements:
1. Logistic Regression
2. Support Vector Machine (LinearSVC with Calibrated Probabilities)
3. Random Forest Classifier
4. Multinomial Naive Bayes (Baseline)
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of classification models configured for text classification.

    Models:
    - Logistic Regression: Effective linear classifier with L2 regularization
    - Support Vector Machine: Max-margin classifier, calibrated for reliable probability outputs
    - Random Forest: Non-linear ensemble model with bagging and feature subsampling
    - Multinomial Naive Bayes: Fast probabilistic baseline for text
    """
    # LinearSVC does not natively output predict_proba, so we wrap with CalibratedClassifierCV
    base_svc = LinearSVC(C=1.0, max_iter=2000, random_state=random_state)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svc, cv=3)

    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
            solver="liblinear"
        ),
        "Support Vector Machine": calibrated_svm,
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=30,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5)
    }

    return models


def get_model_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """Returns hyperparameter grids for optional GridSearchCV tuning."""
    return {
        "Logistic Regression": {
            "C": [0.1, 1.0, 5.0, 10.0],
            "penalty": ["l1", "l2"]
        },
        "Support Vector Machine": {
            "estimator__C": [0.1, 1.0, 5.0]
        },
        "Random Forest": {
            "n_estimators": [100, 150, 200],
            "max_depth": [20, 30, None]
        }
    }
