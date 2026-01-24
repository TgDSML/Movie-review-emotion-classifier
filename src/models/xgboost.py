import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from src.data.train_dataset_EDA import load_train_with_features, load_test_with_features
from src.data.augmentation import class_balanced_augment

# Import your specific feature helpers
# We use 'try-except' in case files are named differently, but these match your previous messages
try:
    from src.features.glove import build_glove_features_from_dfs, sequences_to_vectors
    from src.features.word2vec import get_mean_vectors # Assuming you added this function to word2vec.py
except ImportError as e:
    print(f"⚠️ Warning: Could not import feature scripts. Error: {e}")

# --- 2. TRAINING FUNCTION ---
def train_xgboost(X_train, y_train, X_test, y_test, feature_name):
    print(f"\n⚡ Training XGBoost on [ {feature_name} ] ...")
    
    # Encode labels (Sad -> 0, Happy -> 1)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Initialize XGBoost
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        n_estimators=100, 
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    
    model.fit(X_train, y_train_enc)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    print(f"✅ Accuracy ({feature_name}): {acc:.2%}")
    
    return acc, model

# --- 3. MAIN CONTROLLER ---
def run_all_experiments(df_train, df_test, prefix="Original"):
    results = {}
    y_train = df_train['label']
    y_test = df_test['label']
    
    print(f"\n{'='*40}")
    print(f"  RUNNING SUITE: {prefix} Data")
    print(f"{'='*40}")

    # -----------------------------------
    # A. TF-IDF
    # -----------------------------------
    print("preparing TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(df_train['clean_text'])
    X_test_tfidf = tfidf.transform(df_test['clean_text'])
    
    acc, _ = train_xgboost(X_train_tfidf, y_train, X_test_tfidf, y_test, f"{prefix} + TF-IDF")
    results['TF-IDF'] = acc

    # -----------------------------------
    # B. Word2Vec (Averaged)
    # -----------------------------------
    print("preparing Word2Vec...")
    # Train W2V on *this* specific training set
    tokenized_text = [simple_preprocess(str(t)) for t in df_train['clean_text']]
    w2v_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=2, workers=4, sg=1)
    
    X_train_w2v = get_mean_vectors(w2v_model, df_train['clean_text'])
    X_test_w2v = get_mean_vectors(w2v_model, df_test['clean_text'])
    
    acc, _ = train_xgboost(X_train_w2v, y_train, X_test_w2v, y_test, f"{prefix} + Word2Vec")
    results['Word2Vec'] = acc

    # -----------------------------------
    # C. GloVe (Pooled)
    # -----------------------------------
    print("preparing GloVe...")
    # We use your 'build_glove_features_from_dfs' to get sequences/matrix
    # Then 'sequences_to_vectors' to flatten them for XGBoost
    X_train_seq, _, X_test_seq, _, _, embedding_matrix, _ = build_glove_features_from_dfs(
        df_train, df_test, text_col='clean_text', label_col='label'
    )
    
    X_train_glove = sequences_to_vectors(X_train_seq, embedding_matrix)
    X_test_glove = sequences_to_vectors(X_test_seq, embedding_matrix)
    
    acc, _ = train_xgboost(X_train_glove, y_train, X_test_glove, y_test, f"{prefix} + GloVe")
    results['GloVe'] = acc

    return results

if __name__ == "__main__":
    # 1. Load Data
    print("--- Loading Datasets ---")
    df_train = load_train_with_features()
    df_test = load_test_with_features()
    
    # 2. Run on ORIGINAL Data
    results_orig = run_all_experiments(df_train, df_test, prefix="Original")
    
    # 3. Run on AUGMENTED Data
    print("\n--- Generating Augmented Data ---")
    df_aug = class_balanced_augment(df_train, text_col='clean_text', label_col='label')
    results_aug = run_all_experiments(df_aug, df_test, prefix="Augmented")

    # 4. Final Scoreboard
    print("\n\n🏆 FINAL SCOREBOARD (XGBoost Accuracy)")
    print(f"{'Feature':<15} | {'Original':<10} | {'Augmented':<10} | {'Gain':<10}")
    print("-" * 55)
    
    for method in ['TF-IDF', 'Word2Vec', 'GloVe']:
        orig = results_orig.get(method, 0)
        aug = results_aug.get(method, 0)
        gain = aug - orig
        print(f"{method:<15} | {orig:.2%}     | {aug:.2%}      | {gain:+.2%}")