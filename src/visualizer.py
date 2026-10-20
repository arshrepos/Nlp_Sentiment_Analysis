"""
Visualization Utilities
Generates plots for model evaluation: confusion matrix, class distribution,
feature importance, confidence histogram.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for PyCharm & CI)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CLASS_LABELS = ["negative", "neutral", "positive"]
PALETTE = ["#e74c3c", "#f39c12", "#2ecc71"]   # red / orange / green


def plot_confusion_matrix(
    confusion_matrix: List[List[int]],
    output_path: str = "output/confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """Save a styled confusion-matrix heatmap."""
    cm = np.array(confusion_matrix)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)


def plot_label_distribution(
    labels: List,
    output_path: str = "output/label_distribution.png",
    title: str = "Label Distribution",
) -> None:
    """Bar chart of label frequencies."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    unique, counts = np.unique(labels, return_counts=True)
    label_names = [CLASS_LABELS[i] for i in unique]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(label_names, counts, color=PALETTE[: len(unique)], edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_xlabel("Sentiment Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Label distribution saved to %s", output_path)


def plot_confidence_histogram(
    predictions: List[Dict],
    output_path: str = "output/confidence_histogram.png",
    title: str = "Prediction Confidence Distribution",
) -> None:
    """Histogram of prediction confidence scores by class."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in zip(CLASS_LABELS, PALETTE):
        confs = [p["confidence"] for p in predictions if p["label"] == label]
        if confs:
            ax.hist(confs, bins=20, alpha=0.65, label=label, color=color, edgecolor="white")

    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Sentiment")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Confidence histogram saved to %s", output_path)


def plot_top_tfidf_features(
    vectorizer,
    output_path: str = "output/top_features.png",
    top_n: int = 20,
    title: str = f"Top TF-IDF Features",
) -> None:
    """Bar chart of the most frequent vocabulary terms."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features = vectorizer.get_feature_names_out()
    scores = np.asarray(vectorizer.idf_).flatten() if hasattr(vectorizer, "idf_") else np.ones(len(features))

    # High IDF = rare; invert to show most common (low IDF)
    sorted_idx = np.argsort(scores)[:top_n]
    top_features = features[sorted_idx]
    top_scores = scores[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_features[::-1], top_scores[::-1], color="#3498db", edgecolor="white")
    ax.set_xlabel("IDF Score (lower = more common)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Top TF-IDF features saved to %s", output_path)
