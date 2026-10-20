# NLP Sentiment Analysis System

A complete, production-style machine learning pipeline that classifies text
into **positive**, **negative**, and **neutral** sentiment categories.

---

## Features

- **Text Preprocessing** — cleaning, tokenization, stopword removal, lemmatization, TF-IDF vectorization
- **Three Classifiers** — Logistic Regression, LinearSVC, Multinomial Naive Bayes
- **End-to-end Pipeline** — data loading → preprocessing → training → evaluation → inference
- **Persistence** — save and reload trained models with `joblib`
- **Visualisations** — confusion matrix, label distribution, confidence histogram, top TF-IDF features
- **Test Suite** — 20+ pytest tests covering every module

---

## Project Structure

```
nlp_sentiment_analysis/
├── main.py                  ← Entry point (run this)
├── requirements.txt
├── pytest.ini
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      ← TextPreprocessor (cleaning + TF-IDF)
│   ├── classifier.py        ← SentimentClassifier (sklearn wrapper)
│   ├── pipeline.py          ← SentimentPipeline (orchestrator)
│   ├── data_utils.py        ← CSV helpers + sample data generator
│   └── visualizer.py        ← Matplotlib/Seaborn plots
│
├── tests/
│   └── test_pipeline.py     ← pytest test suite
│
├── data/                    ← Auto-generated sample CSV goes here
├── models/                  ← Saved model artifacts
└── output/                  ← Generated plots
```

---

## Setup in PyCharm

### 1. Open the project
`File → Open` and select the `nlp_sentiment_analysis/` folder.

### 2. Create a virtual environment
`File → Settings → Project → Python Interpreter → Add Interpreter → Virtualenv`
(or use the terminal below)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline
Right-click `main.py` → **Run 'main'**

Or from the terminal:
```bash
python main.py
python main.py --model linear_svc
python main.py --model naive_bayes --no-plots
```

### 5. Run the tests
Right-click `tests/` → **Run pytest in tests**

Or from the terminal:
```bash
pytest tests/ -v
```

---

## Using your own data

Prepare a CSV with two columns:

| text                                   | label    |
|----------------------------------------|----------|
| Great product, really happy!           | positive |
| Awful, broken on arrival.              | negative |
| It's okay, nothing special.            | neutral  |

Then run:
```bash
python main.py --data path/to/your_data.csv
```

---

## CLI Options

| Flag          | Default                        | Description                                |
|---------------|--------------------------------|--------------------------------------------|
| `--data`      | `data/sample_data.csv`         | Path to input CSV                          |
| `--model`     | `logistic_regression`          | `logistic_regression / linear_svc / naive_bayes` |
| `--save-dir`  | `models/saved_model`           | Where to save trained artifacts            |
| `--no-plots`  | *(flag)*                       | Skip generating plots                      |

---

## Example Output

```
── Evaluation Results ──────────────────────────────────────────
  Accuracy        : 0.9333
  F1 (macro)      : 0.9325
  F1 (weighted)   : 0.9333

              precision  recall  f1-score  support
    negative       0.93    0.93      0.93       15
     neutral       0.93    0.93      0.93       15
    positive       0.93    0.93      0.93       15

── Demo Predictions ────────────────────────────────────────────
  [POSITIVE] 97.23%  ████████████████████  "This is an absolutely wonderful product..."
  [NEGATIVE] 96.84%  ███████████████████   "Terrible quality, completely useless..."
  [NEUTRAL ] 88.11%  █████████████████     "It's okay, nothing special..."
```
