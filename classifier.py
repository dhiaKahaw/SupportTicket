"""
Multi-Class Support Ticket Topic Classification
================================================
Categorizes customer support messages into: Billing, Technical, Account
Uses TF-IDF vectorization + Logistic Regression.
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────

DATA_PATH = "../twcs/twcs.csv"
OUTPUT_DIR = "output"

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

    return pipeline, X, y, X_test, y_test, y_pred, labels, cm


# ──────────────────────────────────────────────
# 4b. SAVE PLOTS
# ──────────────────────────────────────────────

def save_plots(df, pipeline, y, X_test, y_test, y_pred, labels, cm):
    """Generate and save all plots to the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    colors = ["#4A90D9", "#E8875B", "#5CB85C"]

    # ── Plot 1: Category Distribution (bar + pie) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    category_counts = df["category"].value_counts()
    category_counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white")
    axes[0].set_title("Category Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    category_counts.plot(kind="pie", ax=axes[1], colors=colors, autopct="%1.1f%%",
                          startangle=90, textprops={"fontsize": 11})
    axes[1].set_title("Category Proportions", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "category_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved: {OUTPUT_DIR}/category_distribution.png")

    # ── Plot 2: Confusion Matrix ──
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved: {OUTPUT_DIR}/confusion_matrix.png")

    # ── Plot 3: Top TF-IDF Features per Category ──
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    n_top = 10
    n_classes = len(clf.classes_)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    for i, (label, ax) in enumerate(zip(clf.classes_, axes)):
        top_idx = clf.coef_[i].argsort()[-n_top:][::-1]
        top_feats = feature_names[top_idx]
        top_weights = clf.coef_[i][top_idx]
        ax.barh(top_feats[::-1], top_weights[::-1], color=colors[i % len(colors)])
        ax.set_title(f"Top Features: {label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Weight")
    plt.suptitle("Top TF-IDF Features per Category", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "top_features.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved: {OUTPUT_DIR}/top_features.png")

    # ── Plot 4: Word Count Distribution ──
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(8, 4))
    for cat, color in zip(sorted(df["category"].unique()), colors):
        subset = df[df["category"] == cat]["word_count"]
        ax.hist(subset, bins=30, alpha=0.6, label=cat, color=color, edgecolor="white")
    ax.set_title("Word Count Distribution by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "word_count_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved: {OUTPUT_DIR}/word_count_distribution.png")


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
    pipeline, X, y, X_test, y_test, y_pred, labels, cm = train_and_evaluate(df)

    # Save all plots
    print("=" * 55)
    print("           SAVING PLOTS")
    print("=" * 55)
    save_plots(df, pipeline, y, X_test, y_test, y_pred, labels, cm)
    print()

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
