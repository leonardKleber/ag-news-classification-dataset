import re
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -----------------------------
# Configuration
# -----------------------------
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# -----------------------------
# Text preprocessing functions
# -----------------------------
def clean_text(text: str) -> str:
    """
    Apply preprocessing steps:
    - Lowercasing
    - Remove punctuation and numbers
    - Tokenization
    - Stopword removal
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize and remove stopwords
    tokens = [
        token for token in text.split()
        if token not in STOP_WORDS
    ]

    return " ".join(tokens)

# -----------------------------
# Dataset-level preprocessing
# -----------------------------
def preprocess_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load AG News dataset and apply preprocessing.
    Returns a DataFrame with:
    - label
    - raw_text
    - clean_text
    """
    df = pd.read_csv(csv_path)
    df.columns = ["label", "title", "description"]

    # Combine title and description
    df["raw_text"] = df["title"] + " " + df["description"]

    # Apply preprocessing
    df["clean_text"] = df["raw_text"].apply(clean_text)

    return df[["label", "raw_text", "clean_text"]]

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    df = preprocess_dataset("train.csv")

    print(df.head())
    print("\nSample cleaned text:\n")
    print(df.loc[0, "clean_text"])
