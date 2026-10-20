"""
Sentiment Classifier
Trains, evaluates, and persists classification models.
Supports Logistic Regression, LinearSVC, and Naive Bayes out of the box.
"""

import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)

# Sentiment label mapping used throughout the project
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class SentimentClassifier:
    """
    Wraps a sklearn classifier with training, evaluation, and persistence.

    Parameters
    ----------
    model_type : str
        One of 'logistic_regression', 'linear_svc', 'naive_bayes'.
    **model_kwargs
        Extra keyword arguments forwarded to the underlying estimator.
    """

    SUPPORTED_MODELS = {
        "logistic_regression": LogisticRegression,
        "linear_svc": LinearSVC,
        "naive_bayes": MultinomialNB,
    }

    def __init__(self, model_type: str = "logistic_regression", **model_kwargs):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"model_type must be one of {list(self.SUPPORTED_MODELS)}; "
                f"got '{model_type}'"
            )
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.model = self._build_model()
        self.classes_ = list(LABEL_MAP.values())
        self._is_trained = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        estimator_cls = self.SUPPORTED_MODELS[self.model_type]

        defaults: Dict = {}
        if self.model_type == "logistic_regression":
            defaults = {"max_iter": 1000, "C": 1.0, "solver": "lbfgs"}
        elif self.model_type == "linear_svc":
            defaults = {"max_iter": 2000, "C": 1.0}
        elif self.model_type == "naive_bayes":
            defaults = {"alpha": 0.1}

        params = {**defaults, **self.model_kwargs}
        estimator = estimator_cls(**params)

        # LinearSVC doesn't natively produce probabilities; wrap it
        if self.model_type == "linear_svc":
            estimator = CalibratedClassifierCV(estimator, cv=3)

        return estimator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, X_train, y_train) -> None:
        """Fit the model on training features and integer labels."""
        logger.info("Training %s on %d samples …", self.model_type, len(y_train))
        self.model.fit(X_train, y_train)
        self._is_trained = True
        logger.info("Training complete.")

    def predict(self, X) -> np.ndarray:
        """Return integer label predictions."""
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return class-probability matrix (n_samples × n_classes)."""
        self._check_trained()
        return self.model.predict_proba(X)

    def predict_labels(self, X) -> List[str]:
        """Return human-readable label strings."""
        preds = self.predict(X)
        return [LABEL_MAP[p] for p in preds]

    def evaluate(self, X_test, y_test) -> Dict:
        """Compute and log key evaluation metrics; return a results dict."""
        self._check_trained()
        y_pred = self.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, target_names=list(LABEL_MAP.values())
        )
        cm = confusion_matrix(y_test, y_pred)

        logger.info("── Evaluation Results ──────────────────")
        logger.info("Accuracy        : %.4f", acc)
        logger.info("F1 (macro)      : %.4f", f1_macro)
        logger.info("F1 (weighted)   : %.4f", f1_weighted)
        logger.info("\n%s", report)

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "classifier.pkl")
        joblib.dump(
            {"model_type": self.model_type, "model_kwargs": self.model_kwargs},
            path / "classifier_config.pkl",
        )
        logger.info("Classifier saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SentimentClassifier":
        path = Path(path)
        config = joblib.load(path / "classifier_config.pkl")
        obj = cls(model_type=config["model_type"], **config["model_kwargs"])
        obj.model = joblib.load(path / "classifier.pkl")
        obj._is_trained = True
        logger.info("Classifier loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
