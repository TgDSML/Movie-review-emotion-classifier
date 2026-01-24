import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from src.data.train_dataset_EDA import load_train_with_features

def load_w2v_input():
    df = load_train_with_features()
    X_train_text = df['clean_text']
    y_train = df['label'].values
    return X_train_text, y_train 

def build_and_train_w2v_model(
        texts,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
):
    # Preprocess
    tokenized_sentences = [simple_preprocess(text) for text in texts]
    
    # Initialize AND Train (Gensim does this automatically if 'sentences' is passed)
    
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # Skip-gram is usually better for smaller datasets
        epochs=epochs 
    )
    
    # REMOVED: model.train(...) -> It was redundant!
    return model

# --- NEW FUNCTION FOR LOGISTIC REGRESSION ---
def get_mean_vectors(w2v_model, texts):
    """
    Returns [Mean_Vector, Max_Vector] concatenated. 
    New Shape: (n_samples, 200)
    """
    matrix = []
    for text in texts:
        tokens = simple_preprocess(str(text))
        valid_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        
        if valid_vectors:
            valid_vectors = np.array(valid_vectors)
            mean_vec = np.mean(valid_vectors, axis=0)
            max_vec = np.max(valid_vectors, axis=0) # Captures the strongest signal
            # Stack them together
            final_vec = np.concatenate([mean_vec, max_vec])
        else:
            # Return zeros of double size
            final_vec = np.zeros(w2v_model.vector_size * 2)
            
        matrix.append(final_vec)
    return np.array(matrix)

def fit_w2v_for_classic_ml(vector_size=100):
    X_train_text, y_train = load_w2v_input()

    # 1. Train Word2Vec
    w2v_model = build_and_train_w2v_model(
        X_train_text, 
        vector_size=vector_size
    )

    # 2. Convert Text -> Mean Vectors (The "Bridge" for Logistic Regression)
    X_train_averaged = get_mean_vectors(w2v_model, X_train_text)

    return w2v_model, X_train_averaged, y_train

if __name__ == "__main__":
    model, X_avg, y = fit_w2v_for_classic_ml()
    print("X_train (averaged) shape:", X_avg.shape) 
    # Should be (16000, 100) -> Ready for Logistic Regression!