"""
main.py — NLP Sentiment Analysis System
========================================
Run this file to execute the full pipeline:
  1. Download NLTK resources (once)
  2. Generate a sample dataset (if none exists)
  3. Preprocess → Train → Evaluate
  4. Run batch inference on example texts
  5. Save the trained model
  6. Generate visualisation plots (output/)

Usage (PyCharm: right-click → Run 'main'):
  python main.py
  python main.py --model linear_svc
  python main.py --data data/my_data.csv --model logistic_regression
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Setup logging before any local imports ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── NLTK resource bootstrap ────────────────────────────────────────────────────
import nltk

def download_nltk_resources() -> None:
    resources = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)

download_nltk_resources()

# ── Local imports (after NLTK is ready) ───────────────────────────────────────
from src.pipeline import SentimentPipeline
from src.data_utils import generate_sample_csv, dataset_summary, load_csv
from src.visualizer import (
    plot_confusion_matrix,
    plot_label_distribution,
    plot_confidence_histogram,
    plot_top_tfidf_features,
)


# ── CLI arguments ──────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLP Sentiment Analysis System")
    parser.add_argument(
        "--data",
        default="data/sample_data.csv",
        help="Path to CSV file with 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=["logistic_regression", "linear_svc", "naive_bayes"],
        help="Classifier to train.",
    )
    parser.add_argument(
        "--save-dir",
        default="models/saved_model",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualisation plots.",
    )
    return parser.parse_args(argv)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)

    # 1. Ensure data exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.info("Sample data not found. Generating at %s …", data_path)
        generate_sample_csv(str(data_path), n=300)

    df = load_csv(str(data_path))
    dataset_summary(df)

    # 2. Build and run pipeline
    pipeline = SentimentPipeline(model_type=args.model)
    pipeline.load_data(str(data_path))
    pipeline.train()

    # 3. Evaluate
    results = pipeline.evaluate()
    print("\n── Evaluation Results ──────────────────────────────────────────")
    print(f"  Accuracy        : {results['accuracy']:.4f}")
    print(f"  F1 (macro)      : {results['f1_macro']:.4f}")
    print(f"  F1 (weighted)   : {results['f1_weighted']:.4f}")
    print("\n" + results["classification_report"])

    # 4. Batch inference demo
    demo_texts = [
        "This is an absolutely wonderful product, I love it!",
        "Terrible quality, completely useless. Very disappointed.",
        "It's okay, nothing special. Does the job I suppose.",
        "Best thing I've bought all year – highly recommend!",
        "Mediocre at best. Wouldn't buy again but not the worst.",
        "Broken on arrival. Awful customer support. Never again.",
    ]

    print("── Demo Predictions ────────────────────────────────────────────")
    predictions = pipeline.predict_batch(demo_texts)
    for pred in predictions:
        bar = "█" * int(pred["confidence"] * 20)
        print(
            f"  [{pred['label'].upper():8s}] {pred['confidence']:.2%}  {bar:20s}  \"{pred['text'][:60]}\""
        )
    print()

    # 5. Save model
    pipeline.save(args.save_dir)
    print(f"  Model saved → {args.save_dir}\n")

    # 6. Visualisations
    if not args.no_plots:
        Path("output").mkdir(exist_ok=True)

        plot_confusion_matrix(results["confusion_matrix"])

        all_labels = pipeline.y_train + pipeline.y_test
        plot_label_distribution(all_labels)

        plot_confidence_histogram(predictions)

        plot_top_tfidf_features(pipeline.preprocessor.vectorizer)

        print("  Plots saved to ./output/\n")

    # 7. Load-and-infer demo (round-trip test)
    print("── Load & Infer (round-trip test) ──────────────────────────────")
    loaded_pipeline = SentimentPipeline.load(args.save_dir)
    test_text = "I am genuinely delighted with this purchase!"
    result = loaded_pipeline.predict_single(test_text)
    print(f"  Text  : {result['text']}")
    print(f"  Label : {result['label']}  ({result['confidence']:.2%} confidence)")
    print(f"  Probs : {result['probabilities']}")
    print("\n  All done!\n")


if __name__ == "__main__":
    main()
