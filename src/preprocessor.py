"""
NLP Preprocessor
Handles text cleaning, tokenization, and TF-IDF vectorization.
"""

import re
import string
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import List, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Handles all NLP preprocessing steps:
      1. Text cleaning (lowercasing, removing punctuation/numbers/URLs)
      2. Tokenization (word-level via NLTK)
      3. Stopword removal + optional stemming/lemmatization
      4. TF-IDF vectorization
    """

    def __init__(
        self,
        use_stemming: bool = False,
        use_lemmatization: bool = True,
        max_features: int = 10_000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words("english"))

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,          # Apply log(1 + tf) scaling
            strip_accents="unicode",
        )
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Lower-case, strip URLs, punctuation, digits, extra whitespace."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)        # Remove URLs
        text = re.sub(r"<[^>]+>", " ", text)                # Remove HTML tags
        text = re.sub(r"[^a-z\s]", " ", text)               # Keep only letters
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Word-tokenize and remove stopwords; optionally stem/lemmatize."""
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif self.use_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def preprocess_text(self, text: str) -> str:
        """Clean → tokenize → rejoin into a single string for the vectorizer."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return " ".join(tokens)

    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """Apply preprocess_text to a list of documents."""
        return [self.preprocess_text(t) for t in texts]

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Preprocess + fit TF-IDF + return feature matrix (training)."""
        processed = self.preprocess_corpus(texts)
        features = self.vectorizer.fit_transform(processed)
        self._is_fitted = True
        logger.info(
            "TF-IDF fitted on %d docs | vocab size: %d",
            len(texts),
            len(self.vectorizer.vocabulary_),
        )
        return features

    def transform(self, texts: List[str]) -> np.ndarray:
        """Preprocess + transform using a fitted TF-IDF (inference)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        processed = self.preprocess_corpus(texts)
        return self.vectorizer.transform(processed)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path / "vectorizer.pkl")
        joblib.dump(
            {
                "use_stemming": self.use_stemming,
                "use_lemmatization": self.use_lemmatization,
                "is_fitted": self._is_fitted,
            },
            path / "preprocessor_config.pkl",
        )
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TextPreprocessor":
        path = Path(path)
        config = joblib.load(path / "preprocessor_config.pkl")
        obj = cls(
            use_stemming=config["use_stemming"],
            use_lemmatization=config["use_lemmatization"],
        )
        obj.vectorizer = joblib.load(path / "vectorizer.pkl")
        obj._is_fitted = config["is_fitted"]
        logger.info("Preprocessor loaded from %s", path)
        return obj
