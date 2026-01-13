import os
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from wordcloud import WordCloud

# -----------------------------
# Configuration
# -----------------------------
TRAIN_PATH = "train.csv"
FIG_DIR = "figures"

CLASS_NAMES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

STOP_WORDS = set(ENGLISH_STOP_WORDS)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

os.makedirs(FIG_DIR, exist_ok=True)

def save_show(fig_name):
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, fig_name),
        format="png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

# -----------------------------
# Load Training Data ONLY
# -----------------------------
df = pd.read_csv(TRAIN_PATH)
df.columns = ["label", "title", "description"]

df["text"] = df["title"] + " " + df["description"]
df["class_name"] = df["label"].map(CLASS_NAMES)

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# -----------------------------
# Class Distribution
# -----------------------------
print("\nClass distribution:")
print(df["class_name"].value_counts())

sns.countplot(x="class_name", data=df)
plt.title("Class Distribution (Training Set)")
plt.xlabel("Class")
plt.ylabel("Number of Articles")
save_show("class_distribution.png")

# -----------------------------
# Text Length Analysis
# -----------------------------
df["text_length"] = df["clean_text"].apply(lambda x: len(x.split()))

print("\nText length statistics (overall):")
print("Average:", round(df["text_length"].mean(), 2))
print("Median:", df["text_length"].median())

print("\nText length per class:")
print(df.groupby("class_name")["text_length"].agg(["mean", "median"]))

sns.boxplot(x="class_name", y="text_length", data=df)
plt.title("Text Length Distribution per Class")
plt.xlabel("Class")
plt.ylabel("Number of Words")
save_show("text_length_per_class.png")

# -----------------------------
# Most Frequent Words per Class
# -----------------------------
def top_words(texts, n=20):
    words = " ".join(texts).split()
    return Counter(words).most_common(n)

for cls in CLASS_NAMES.values():
    print(f"\nTop words in class: {cls}")
    for word, count in top_words(df[df["class_name"] == cls]["clean_text"]):
        print(f"{word}: {count}")

# -----------------------------
# Word Clouds per Class
# -----------------------------
for cls in CLASS_NAMES.values():
    text = " ".join(df[df["class_name"] == cls]["clean_text"])
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud – {cls}")
    save_show(f"wordcloud_{cls.lower().replace('/', '_')}.png")

# -----------------------------
# N-gram Frequency Analysis
# -----------------------------
def plot_ngrams(texts, ngram_range, title, filename, top_n=20):
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        min_df=5
    )
    X = vectorizer.fit_transform(texts)

    freqs = zip(
        vectorizer.get_feature_names_out(),
        X.sum(axis=0).A1
    )

    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)[:top_n]
    ngrams, counts = zip(*freqs)

    sns.barplot(x=list(counts), y=list(ngrams))
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("N-gram")
    save_show(filename)

plot_ngrams(
    df["clean_text"],
    (1, 1),
    "Top Unigrams (Training Set)",
    "top_unigrams.png"
)

plot_ngrams(
    df["clean_text"],
    (2, 2),
    "Top Bigrams (Training Set)",
    "top_bigrams.png"
)

plot_ngrams(
    df["clean_text"],
    (3, 3),
    "Top Trigrams (Training Set)",
    "top_trigrams.png"
)

print("\nEDA completed successfully. PNG figures saved to ./figures/")
