"""
Multi-Class Support Ticket Topic Classification
================================================
Categorizes customer support messages into: Billing, Technical, Account
Uses TF-IDF vectorization + Logistic Regression.
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────

DATA_PATH = "twcs/twcs.csv"

CATEGORY_KEYWORDS = {
    "Technical": [
        "bug", "crash", "error", "slow", "update", "wifi", "battery",
        "freeze", "install", "broken", "fix", "glitch", "lag",
        "disconnect", "loading", "device", "software", "hardware",
        "printer", "app", "ios", "android", "bluetooth", "screen",
        "reboot", "restart", "download", "sync", "connection",
    ],
    "Billing": [
        "bill", "charge", "refund", "payment", "price", "invoice",
        "subscription", "fee", "credit", "cost", "pay", "money",
        "overcharg", "renew", "cancel", "plan", "pricing", "receipt",
        "discount", "coupon",
    ],
    "Account": [
        "account", "password", "login", "sign in", "profile",
        "username", "email", "register", "verify", "reset",
        "locked", "access", "dm", "log in", "sign up", "settings",
        "security", "two factor", "authentication",
    ],
}


# ──────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ──────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Clean a raw tweet for classification."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)                # remove @mentions
    text = re.sub(r"#", "", text)                   # remove hash symbol
    text = re.sub(r"[^a-z\s]", "", text)            # keep only letters
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def assign_label(text: str) -> str | None:
    """
    Assign a category label based on keyword matching.
    Priority order: Technical > Billing > Account.
    Returns None if no keywords match.
    """
    cleaned = preprocess_text(text)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in cleaned for kw in keywords):
            return category
    return None


# ──────────────────────────────────────────────
# 3. LOAD & LABEL DATA
# ──────────────────────────────────────────────

def load_and_label(path: str, sample_size: int = 50000) -> pd.DataFrame:
    """Load the CSV, keep inbound customer tweets, and assign labels."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    # Keep only customer (inbound) messages
    df = df[df["inbound"] == True].copy()
    # Drop rows with missing text
    df = df.dropna(subset=["text"])
    # Sample for speed if dataset is large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} inbound messages for speed.")
    # Assign labels
    df["category"] = df["text"].apply(assign_label)
    # Drop unlabeled rows
    df = df.dropna(subset=["category"])
    # Ensure each class has at least 2 samples
    counts = df["category"].value_counts()
    valid = counts[counts >= 2].index
    df = df[df["category"].isin(valid)]
    return df


# ──────────────────────────────────────────────
# 4. BUILD & TRAIN MODEL
# ──────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """Create a TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def train_and_evaluate(df: pd.DataFrame):
    """Train the classifier and print evaluation metrics."""
    # Preprocess the text column for the model
    df["clean_text"] = df["text"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["category"]

    print(f"Dataset size: {len(df)} labeled messages")
    print(f"Label distribution:\n{y.value_counts().to_string()}\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Build and fit the pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("=" * 55)
    print("           CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=" * 55)
    print("           CONFUSION MATRIX")
    print("=" * 55)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    print()

    return pipeline


# ──────────────────────────────────────────────
# 5. PREDICTION FUNCTION
# ──────────────────────────────────────────────

def predict_category(pipeline: Pipeline, text: str) -> str:
    """Classify a single new support message."""
    cleaned = preprocess_text(text)
    prediction = pipeline.predict([cleaned])[0]
    return prediction


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Load and label
    df = load_and_label(DATA_PATH)

    # Train and evaluate
    pipeline = train_and_evaluate(df)

    # Demo predictions on new messages
    demo_messages = [
        "I was charged twice on my credit card for the same order!",
        "My app keeps crashing every time I open it after the update.",
        "I can't log into my account, it says my password is wrong.",
        "The wifi keeps disconnecting on my new phone.",
        "Can I get a refund for my last month's subscription?",
        "How do I reset my password? I forgot it.",
        "My internet is extremely slow since yesterday.",
        "I need to update my billing address on my profile.",
    ]

    print("=" * 55)
    print("           DEMO PREDICTIONS")
    print("=" * 55)
    for msg in demo_messages:
        category = predict_category(pipeline, msg)
        print(f"  [{category:>10}]  {msg}")
    print()
