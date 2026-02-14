# Support Ticket Topic Classification

A multi-class text classification system that categorizes customer support messages into **Billing**, **Technical**, and **Account** using TF-IDF vectorization and Logistic Regression.

## Results

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| Account | 0.89 | 0.83 | 0.86 |
| Billing | 0.90 | 0.91 | 0.91 |
| Technical | 0.93 | 0.94 | 0.93 |
| **Overall Accuracy** | | | **0.91** |

## Project Structure

```
├── Support_Ticket_Classification.ipynb   # Interactive Jupyter notebook (recommended)
├── classifier.py                         # Standalone Python script
├── twcs/
│   └── twcs.csv                          # Full Twitter Customer Support dataset
├── sample.csv                            # Small sample dataset (100 rows)
└── README.md
```

## Requirements

- Python 3.10+
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
- joblib
- Jupyter Notebook (for `.ipynb`)

### Install all dependencies

```bash
pip install pandas scikit-learn numpy matplotlib seaborn joblib notebook
```

## How to Run

### Option 1 — Jupyter Notebook (Recommended)

The notebook provides an interactive, step-by-step experience with visualizations and built-in tests.

**Step 1:** Start Jupyter Notebook from the project folder:

```bash
jupyter notebook
```

**Step 2:** In the browser, click on **`Support_Ticket_Classification.ipynb`** to open it.

**Step 3:** Run all cells: **Kernel → Restart & Run All** (or `Shift+Enter` cell by cell).

The notebook contains 10 sections:

| # | Section | What it does |
|---|---|---|
| 1 | Imports & Setup | Loads all required libraries |
| 2 | Configuration | Dataset path, sample size, category keywords |
| 3 | Helper Functions | Text preprocessing + **unit tests** to verify correctness |
| 4 | Load & Prepare Data | Loads `twcs.csv`, assigns labels, shows distribution charts |
| 5 | Text Preprocessing | Cleans text with before/after examples |
| 6 | Build & Train Model | Trains TF-IDF + Logistic Regression pipeline |
| 7 | Model Evaluation | Classification report, confusion matrix, cross-validation, top features |
| 8 | Predictions & Demo | Classifies sample messages with confidence scores |
| 9 | Interactive Testing | Type your own messages to classify in real time |
| 10 | Model Export | Saves the trained model to `support_ticket_classifier.pkl` |

> **Note:** The dataset is large (~2.8M rows). The notebook samples 50,000 inbound messages by default. Adjust `SAMPLE_SIZE` in Section 2 to use more or fewer rows.

### Option 2 — Python Script

```bash
python classifier.py
```

This runs the full pipeline (load → label → train → evaluate → demo predictions) in one go.

## Dataset

Uses the [Twitter Customer Support](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. Place `twcs.csv` inside a `twcs/` folder in the project root.

The dataset has no category labels — they are assigned automatically using keyword matching:

- **Technical** — `bug`, `crash`, `error`, `slow`, `wifi`, `battery`, `app`, `device`, ...
- **Billing** — `bill`, `charge`, `refund`, `payment`, `subscription`, `fee`, `cost`, ...
- **Account** — `account`, `password`, `login`, `profile`, `reset`, `locked`, `email`, ...

Messages matching no keywords are excluded. Priority: Technical > Billing > Account.

## Approach

| Stage | Method |
|---|---|
| **Feature Extraction** | TF-IDF Vectorization (unigrams + bigrams, top 5,000 features) |
| **Classification** | Logistic Regression (L-BFGS solver, max 1,000 iterations) |
| **Labeling** | Keyword-based heuristics (no manual annotation needed) |
