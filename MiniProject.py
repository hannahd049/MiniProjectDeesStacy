import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews

fileids = movie_reviews.fileids()
raw_docs = [movie_reviews.raw(f) for f in fileids]
labels   = [f.split("/")[0] for f in fileids]          # 'pos' or 'neg'

# Split the RAW text first. Part 3.2 (BERT) and Part 4 (TinyLlama) use the raw text.
raw_train, raw_test, y_train, y_test = train_test_split(
    raw_docs, labels, test_size=0.2, random_state=42, stratify=labels)

print("Train:", len(raw_train), " Test:", len(raw_test))
print("Example (raw):", raw_train[0][:200].replace("\n", " "), "...")

def clean_text(text):
    """Lowercase, turn "n't" into " not", keep only letters, return tokens joined by spaces."""
    text = text.lower().replace("n't", " not")
    tokens = re.findall(r"[a-z]+", text)
    return " ".join(tokens)

print(clean_text("This movie isn't GOOD, but the actors were 100% brilliant!"))

X_train = [clean_text(t) for t in raw_train]
X_test  = [clean_text(t) for t in raw_test]
print("Cleaned example:", X_train[0][:150], "...")