"""
Run inference on IMDb movie reviews (unlabeled).
This script loads real-world reviews and observes model behavior.
"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import random
import json
from pathlib import Path

from scripts_lr.infer import predict   # reuse your inference pipeline

LABEL_MAP_PATH = Path("artifacts_lr") / "final_tfidf_lr_aug" / "label_map.json"
with open(LABEL_MAP_PATH, "r") as f:
    LABEL_MAP = {int(k): v for k, v in json.load(f).items()}


def decode_review(encoded_review, index_word):
    """IMDb provides reviews as sequences of integers, not text.
    Each integer corresponds to a word index in the vocabulary.
    This function decodes the integer sequences back to words."""
    return " ".join(index_word.get(i, "?") for i in encoded_review)


def main():
    print("Loading IMDb dataset...")
    (X_train, y_train), _ = imdb.load_data(num_words=20000)

    word_index = imdb.get_word_index()
    index_word = {idx + 3: word for word, idx in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"

    # Take a small sample
    indices = random.sample(range(len(X_train)), 20)
    texts = [decode_review(X_train[i], index_word) for i in indices]

    print(f"\nRunning inference on {len(texts)} IMDb reviews...\n")
    results, _, _ = predict(texts)

    for r in results:
        print("\nText:", r["text"])
        print(f"Prediction: {r['prediction_label']} (id={r['prediction_id']})")
        print("Confidence:", round(r["confidence"], 2))
        print("Status:", r["status"])
        print("Top classes:")
        top3 = sorted(r["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
        for label, prob in top3:
            print(f"  {label}: {prob:.3f}")


              


if __name__ == "__main__":
    main()
