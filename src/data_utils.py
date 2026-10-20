"""
Data Utilities
Helpers for loading, validating, and generating sample datasets.
"""

import csv
import random
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_DATA: List[Tuple[str, str]] = [
    # Positive
    ("This product is absolutely amazing! Exceeded all my expectations.", "positive"),
    ("I love this! Best purchase I've made all year.", "positive"),
    ("Fantastic quality and super fast delivery. Highly recommend!", "positive"),
    ("The customer service was outstanding and very helpful.", "positive"),
    ("Works perfectly, setup was a breeze. Very happy with this.", "positive"),
    ("Incredible value for money. Five stars without hesitation.", "positive"),
    ("The design is beautiful and the performance is top notch.", "positive"),
    ("Really impressed with the build quality. Solid and reliable.", "positive"),
    ("Great product, does exactly what it says on the tin.", "positive"),
    ("Delightful experience from start to finish. Will buy again.", "positive"),
    ("Exceeded expectations in every way. Truly outstanding.", "positive"),
    ("Brilliant! Loved every second of using this.", "positive"),
    ("The packaging was excellent and the item arrived in perfect condition.", "positive"),
    ("So happy with this purchase. It has made my life so much easier.", "positive"),
    ("Amazing features at a very reasonable price.", "positive"),
    # Negative
    ("Terrible quality, broke after just one week. Total waste of money.", "negative"),
    ("Worst product I have ever bought. Do NOT recommend.", "negative"),
    ("Completely disappointed. Nothing as described in the listing.", "negative"),
    ("Arrived damaged and customer support was useless.", "negative"),
    ("The item stopped working after three days. Very poor.", "negative"),
    ("Extremely frustrating experience. Would give zero stars if possible.", "negative"),
    ("Cheap materials, looks nothing like the photos online.", "negative"),
    ("Terrible smell out of the box and it barely functions.", "negative"),
    ("Awful product and even worse packaging. Totally broken on arrival.", "negative"),
    ("Such poor quality. Falling apart already after minimal use.", "negative"),
    ("Scam product, totally misleading description.", "negative"),
    ("Returned immediately. Waste of time and money.", "negative"),
    ("Broken on first use. Absolutely unacceptable.", "negative"),
    ("Do not buy this. Disappointed beyond words.", "negative"),
    ("Horrible experience. Regret this purchase entirely.", "negative"),
    # Neutral
    ("It does the job. Nothing special but gets the task done.", "neutral"),
    ("Average product, nothing to write home about.", "neutral"),
    ("Decent for the price, but there are better options out there.", "neutral"),
    ("It's okay. Works as expected, nothing more nothing less.", "neutral"),
    ("Fairly standard product. No major issues but no wow factor either.", "neutral"),
    ("The product is fine. I neither love nor hate it.", "neutral"),
    ("Acceptable quality. Meets basic requirements.", "neutral"),
    ("Not great, not terrible. Just a regular product.", "neutral"),
    ("Works adequately. Instructions could be clearer though.", "neutral"),
    ("Mediocre performance. Would consider alternatives next time.", "neutral"),
    ("Arrived on time. The item matches the description.", "neutral"),
    ("Standard experience overall. Nothing remarkable happened.", "neutral"),
    ("It functions as advertised, which is the minimum expectation.", "neutral"),
    ("Moderate quality. Somewhere between good and bad.", "neutral"),
    ("Passable product. Does what it should without any fuss.", "neutral"),
]


def generate_sample_csv(output_path: str = "data/sample_data.csv", n: int = 300) -> str:
    """
    Generate a synthetic CSV dataset by oversampling SAMPLE_DATA.

    Parameters
    ----------
    output_path : path where the CSV is written
    n           : total number of rows (equally split across three classes)

    Returns the output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_class = n // 3
    rows: List[Tuple[str, str]] = []

    for label in ("positive", "negative", "neutral"):
        examples = [(t, lb) for t, lb in SAMPLE_DATA if lb == label]
        sampled = random.choices(examples, k=per_class)
        rows.extend(sampled)

    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    logger.info("Generated sample CSV at %s (%d rows)", output_path, len(rows))
    return str(output_path)


def load_csv(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Load and validate a CSV dataset."""
    df = pd.read_csv(path)
    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}. Found: {list(df.columns)}")
    df = df.dropna(subset=[text_col, label_col])
    logger.info("Loaded dataset: %d rows, columns: %s", len(df), list(df.columns))
    return df


def dataset_summary(df: pd.DataFrame, label_col: str = "label") -> None:
    """Print label distribution statistics."""
    print("\n── Dataset Summary ─────────────────────")
    print(f"  Total rows : {len(df)}")
    print(f"  Label distribution:\n{df[label_col].value_counts().to_string()}")
    print("────────────────────────────────────────\n")
