"""
tests/test_pipeline.py
Unit and integration tests for the NLP Sentiment Analysis pipeline.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from pathlib import Path

# Bootstrap NLTK before importing local modules
import nltk
for _r in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{_r}")
    except LookupError:
        nltk.download(_r, quiet=True)

from src.preprocessor import TextPreprocessor
from src.classifier import SentimentClassifier, LABEL_MAP, REVERSE_LABEL_MAP
from src.pipeline import SentimentPipeline
from src.data_utils import generate_sample_csv, load_csv, SAMPLE_DATA


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_texts_labels():
    texts = [t for t, _ in SAMPLE_DATA]
    labels = [REVERSE_LABEL_MAP[l] for _, l in SAMPLE_DATA]
    return texts, labels


@pytest.fixture(scope="module")
def fitted_pipeline(sample_texts_labels):
    texts, labels = sample_texts_labels
    pipeline = SentimentPipeline(model_type="logistic_regression")
    label_strs = [LABEL_MAP[l] for l in labels]
    pipeline.load_data_from_lists(texts, label_strs)
    pipeline.train()
    return pipeline


# ── TextPreprocessor tests ─────────────────────────────────────────────────────

class TestTextPreprocessor:

    def test_clean_text_lowercases(self):
        p = TextPreprocessor()
        assert p.clean_text("Hello World!") == "hello world"

    def test_clean_text_removes_urls(self):
        p = TextPreprocessor()
        result = p.clean_text("Visit https://example.com for details")
        assert "http" not in result
        assert "example" not in result

    def test_clean_text_removes_punctuation(self):
        p = TextPreprocessor()
        result = p.clean_text("Hello, world!!! How are you?")
        assert "," not in result and "!" not in result and "?" not in result

    def test_tokenize_returns_list(self):
        p = TextPreprocessor()
        tokens = p.tokenize("this is a great product")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_removes_stopwords(self):
        p = TextPreprocessor()
        tokens = p.tokenize("this is a great product")
        # 'this', 'is', 'a' are stopwords
        assert "this" not in tokens
        assert "great" in tokens or "great" in [t.lower() for t in tokens]

    def test_fit_transform_shape(self, sample_texts_labels):
        texts, _ = sample_texts_labels
        p = TextPreprocessor(max_features=500)
        X = p.fit_transform(texts)
        assert X.shape[0] == len(texts)
        assert X.shape[1] <= 500

    def test_transform_before_fit_raises(self):
        p = TextPreprocessor()
        with pytest.raises(RuntimeError):
            p.transform(["some text"])

    def test_save_and_load(self, tmp_path, sample_texts_labels):
        texts, _ = sample_texts_labels
        p = TextPreprocessor(max_features=200)
        p.fit_transform(texts)
        p.save(tmp_path / "preprocessor")

        loaded = TextPreprocessor.load(tmp_path / "preprocessor")
        result = loaded.transform(texts[:3])
        assert result.shape[0] == 3


# ── SentimentClassifier tests ──────────────────────────────────────────────────

class TestSentimentClassifier:

    def _make_simple_features(self, n=30):
        """Create trivial float features for quick tests."""
        np.random.seed(0)
        X = np.random.rand(n, 50)
        y = np.array([0, 1, 2] * (n // 3))
        return X, y

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError):
            SentimentClassifier(model_type="random_forest")

    def test_predict_before_train_raises(self):
        clf = SentimentClassifier()
        X, _ = self._make_simple_features()
        with pytest.raises(RuntimeError):
            clf.predict(X)

    def test_train_and_predict(self):
        X, y = self._make_simple_features()
        clf = SentimentClassifier(model_type="logistic_regression")
        clf.train(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba_shape(self):
        X, y = self._make_simple_features()
        clf = SentimentClassifier(model_type="logistic_regression")
        clf.train(X, y)
        probas = clf.predict_proba(X)
        assert probas.shape == (len(y), 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_labels_returns_strings(self):
        X, y = self._make_simple_features()
        clf = SentimentClassifier(model_type="logistic_regression")
        clf.train(X, y)
        labels = clf.predict_labels(X)
        assert all(l in LABEL_MAP.values() for l in labels)

    def test_save_and_load(self, tmp_path):
        X, y = self._make_simple_features()
        clf = SentimentClassifier()
        clf.train(X, y)
        clf.save(tmp_path / "clf")

        loaded = SentimentClassifier.load(tmp_path / "clf")
        preds = loaded.predict(X)
        assert preds.shape == (len(y),)


# ── SentimentPipeline integration tests ───────────────────────────────────────

class TestSentimentPipeline:

    def test_train_and_evaluate(self, fitted_pipeline):
        results = fitted_pipeline.evaluate()
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0
        assert "f1_macro" in results

    def test_predict_single_positive(self, fitted_pipeline):
        result = fitted_pipeline.predict_single("This is absolutely wonderful and amazing!")
        assert result["label"] in LABEL_MAP.values()
        assert 0.0 <= result["confidence"] <= 1.0
        assert "probabilities" in result
        assert len(result["probabilities"]) == 3

    def test_predict_single_negative(self, fitted_pipeline):
        result = fitted_pipeline.predict_single("Terrible, broken, completely useless waste of money!")
        assert result["label"] in LABEL_MAP.values()

    def test_predict_batch_length(self, fitted_pipeline):
        texts = ["great", "awful", "okay", "brilliant", "mediocre"]
        results = fitted_pipeline.predict_batch(texts)
        assert len(results) == len(texts)
        for r in results:
            assert "label" in r and "confidence" in r

    def test_save_and_load_roundtrip(self, fitted_pipeline, tmp_path):
        fitted_pipeline.save(tmp_path / "pipeline")
        loaded = SentimentPipeline.load(tmp_path / "pipeline")
        result = loaded.predict_single("I love this product so much!")
        assert result["label"] in LABEL_MAP.values()

    def test_missing_data_raises(self):
        p = SentimentPipeline()
        with pytest.raises(RuntimeError):
            p.train()


# ── Data utilities tests ───────────────────────────────────────────────────────

class TestDataUtils:

    def test_generate_sample_csv(self, tmp_path):
        path = str(tmp_path / "data.csv")
        generate_sample_csv(path, n=60)
        df = load_csv(path)
        assert len(df) == 60
        assert set(df["label"].unique()).issubset({"positive", "negative", "neutral"})

    def test_load_csv_missing_column_raises(self, tmp_path):
        import pandas as pd
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["hello"], "sentiment": ["positive"]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError):
            load_csv(str(bad_csv))   # expects 'label' column
