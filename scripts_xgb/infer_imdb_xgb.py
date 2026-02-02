"""
Run inference on IMDb movie reviews using the XGBoost Pipeline.
Run from root: python -m scripts_xgb.infer_imdb_xgboost
"""
from tensorflow.keras.datasets import imdb
import random
# NOTE: We import from scripts_xgb now
from scripts_xgb.infer_xgb import predict_xgboost

def decode_review(encoded_review, index_word):
    return " ".join(index_word.get(i, "?") for i in encoded_review)

def main():
    print("Loading IMDb dataset...")
    (X_train, _), _ = imdb.load_data(num_words=20000)

    word_index = imdb.get_word_index()
    index_word = {idx + 3: word for word, idx in word_index.items()}
    index_word[0] = "<PAD>"; index_word[1] = "<START>"; index_word[2] = "<UNK>"

    indices = random.sample(range(len(X_train)), 20)
    texts = [decode_review(X_train[i], index_word) for i in indices]

    print(f"\n Running XGBoost Inference on {len(texts)} reviews...\n")
    results = predict_xgboost(texts)

    for r in results:
        print("-" * 60)
        print(f"Text: {r['text']}")
        print(f" Prediction: {r['prediction'].upper()}")
        print(f" Confidence: {r['confidence']:.2f} [{r['status']}]")
        top3 = sorted(r['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top 3:", top3)

if __name__ == "__main__":
    main()