"""
Sentiment Analysis Pipeline
End-to-end orchestrator: data loading → preprocessing → training → evaluation → inference.
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sklearn.model_selection import train_test_split, cross_val_score

from src.preprocessor import TextPreprocessor
from src.classifier import SentimentClassifier, LABEL_MAP, REVERSE_LABEL_MAP

logger = logging.getLogger(__name__)


class SentimentPipeline:
    """
    High-level pipeline that ties together preprocessing and classification.

    Typical usage
    -------------
    >>> pipeline = SentimentPipeline()
    >>> pipeline.load_data("data/sample_data.csv")
    >>> pipeline.train()
    >>> results = pipeline.evaluate()
    >>> pipeline.save("models/my_model")
    >>> prediction = pipeline.predict_single("This product is amazing!")
    >>> print(prediction)
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",
        test_size: float = 0.2,
        random_state: int = 42,
        preprocessor_kwargs: Optional[Dict] = None,
        classifier_kwargs: Optional[Dict] = None,
    ):
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor = TextPreprocessor(**(preprocessor_kwargs or {}))
        self.classifier = SentimentClassifier(
            model_type=model_type, **(classifier_kwargs or {})
        )

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.eval_results: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        csv_path: Union[str, Path],
        text_col: str = "text",
        label_col: str = "label",
    ) -> "SentimentPipeline":
        """
        Load a CSV with columns `text` (str) and `label` (str or int).
        String labels like 'positive'/'negative'/'neutral' are mapped to ints.
        """
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows from %s", len(df), csv_path)

        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"CSV must contain columns '{text_col}' and '{label_col}'. "
                f"Found: {list(df.columns)}"
            )

        df = df.dropna(subset=[text_col, label_col])
        texts = df[text_col].tolist()
        raw_labels = df[label_col].tolist()

        # Allow string or integer labels
        if isinstance(raw_labels[0], str):
            labels = [REVERSE_LABEL_MAP[l.lower()] for l in raw_labels]
        else:
            labels = [int(l) for l in raw_labels]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )
        logger.info(
            "Train size: %d | Test size: %d", len(self.X_train), len(self.X_test)
        )
        return self

    def load_data_from_lists(
        self, texts: List[str], labels: List[Union[str, int]]
    ) -> "SentimentPipeline":
        """Load data directly from Python lists (useful for testing)."""
        if isinstance(labels[0], str):
            labels = [REVERSE_LABEL_MAP[l.lower()] for l in labels]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )
        return self

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> "SentimentPipeline":
        """Preprocess training data and fit the classifier."""
        if self.X_train is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        logger.info("Preprocessing training corpus …")
        X_train_features = self.preprocessor.fit_transform(self.X_train)

        logger.info("Training classifier …")
        self.classifier.train(X_train_features, self.y_train)
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict:
        """Transform test set and evaluate the classifier."""
        if self.X_test is None:
            raise RuntimeError("No test data available.")

        logger.info("Evaluating on test set …")
        X_test_features = self.preprocessor.transform(self.X_test)
        self.eval_results = self.classifier.evaluate(X_test_features, self.y_test)
        return self.eval_results

    def cross_validate(self, cv: int = 5) -> Dict:
        """Run k-fold cross validation on the training split (re-fits each fold)."""
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        logger.info("Running %d-fold cross validation …", cv)
        # Build a lightweight sklearn pipeline for CV
        pipe = SkPipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        processed = [self.preprocessor.preprocess_text(t) for t in self.X_train]
        scores = cross_val_score(pipe, processed, self.y_train, cv=cv, scoring="f1_macro")
        logger.info(
            "CV F1 (macro): %.4f ± %.4f", scores.mean(), scores.std()
        )
        return {"cv_f1_mean": scores.mean(), "cv_f1_std": scores.std(), "cv_scores": scores.tolist()}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_single(self, text: str) -> Dict:
        """Return sentiment label + confidence for a single text."""
        features = self.preprocessor.transform([text])
        label_int = self.classifier.predict(features)[0]
        probas = self.classifier.predict_proba(features)[0]

        return {
            "text": text,
            "label": LABEL_MAP[label_int],
            "confidence": round(float(probas.max()), 4),
            "probabilities": {
                LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probas)
            },
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Return predictions for a list of texts."""
        features = self.preprocessor.transform(texts)
        label_ints = self.classifier.predict(features)
        probas = self.classifier.predict_proba(features)

        results = []
        for text, label_int, proba in zip(texts, label_ints, probas):
            results.append(
                {
                    "text": text,
                    "label": LABEL_MAP[label_int],
                    "confidence": round(float(proba.max()), 4),
                    "probabilities": {
                        LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)
                    },
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> None:
        """Save preprocessor + classifier + metadata to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.preprocessor.save(directory / "preprocessor")
        self.classifier.save(directory / "classifier")

        meta = {
            "model_type": self.model_type,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "eval_results": (
                {k: v for k, v in (self.eval_results or {}).items() if k != "confusion_matrix"}
            ),
        }
        with open(directory / "pipeline_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Pipeline saved to %s", directory)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "SentimentPipeline":
        """Restore a saved pipeline for inference."""
        directory = Path(directory)
        with open(directory / "pipeline_meta.json") as f:
            meta = json.load(f)

        obj = cls(model_type=meta["model_type"])
        obj.preprocessor = TextPreprocessor.load(directory / "preprocessor")
        obj.classifier = SentimentClassifier.load(directory / "classifier")
        obj.eval_results = meta.get("eval_results")
        logger.info("Pipeline loaded from %s", directory)
        return obj
