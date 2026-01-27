import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    pass # If tensorflow isn't installed, these imports fail silently


# 1. SHARED CORE FUNCTIONS (The missing tools!)


def train_word2vec(texts, vector_size=100, window=5, min_count=2, epochs=10):
    """
    Trains a Word2Vec model on a list of texts.
    Standardized function for the project.
    """
    # Convert text to list of tokens
    tokenized_sentences = [simple_preprocess(str(text)) for text in texts]
    
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,
        epochs=epochs
    )
    return model

def build_and_train_w2v_model(texts, vector_size=100, window=5, min_count=2, workers=4, epochs=10):
    """Alias for backward compatibility."""
    return train_word2vec(texts, vector_size, window, min_count, epochs)

def get_mean_vectors(w2v_model, texts):
    """
    Returns [Mean_Vector, Max_Vector] concatenated. 
    Global helper function used by XGBoost and others.
    """
    matrix = []
    for text in texts:
        tokens = simple_preprocess(str(text))
        valid_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        
        if valid_vectors:
            valid_vectors = np.array(valid_vectors)
            mean_vec = np.mean(valid_vectors, axis=0)
            max_vec = np.max(valid_vectors, axis=0) 
            final_vec = np.concatenate([mean_vec, max_vec])
        else:
            final_vec = np.zeros(w2v_model.vector_size * 2)
            
        matrix.append(final_vec)
    return np.array(matrix)


# thodoris neural network functions


def create_embedding_matrix(tokenizer, w2v_model, embedding_dim):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    return embedding_matrix

def fit_w2v_on_dataframe(df, text_col="clean_text", label_col="label", vector_size=100, window=5, min_count=2, max_words=20000, max_len=100):
    texts = df[text_col].astype(str).tolist()
    y = df[label_col].values

    # 1) Train Word2Vec
    w2v_model = train_word2vec(texts, vector_size=vector_size, window=window, min_count=min_count)

    # 2) Fit Tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)

    # 3) Convert to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    X_seq = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # 4) Embedding Matrix
    embedding_matrix = create_embedding_matrix(tokenizer, w2v_model, embedding_dim=vector_size)

    return tokenizer, embedding_matrix, X_seq, y

def sequences_to_mean_max_vectors(X_seq, embedding_matrix):
    vectors = []
    embedding_dim = embedding_matrix.shape[1]
    for seq in X_seq:
        valid = seq[seq>0]
        if len(valid) == 0:
            mean_vec = np.zeros(embedding_dim)
            max_vec = np.zeros(embedding_dim)
        else:
            word_vecs = embedding_matrix[valid]
            mean_vec = word_vecs.mean(axis=0)
            max_vec = word_vecs.max(axis=0)
        vectors.append(np.concatenate([mean_vec, max_vec]))
    return np.vstack(vectors)

# TUNED XGBOOST FUNCTION

def fit_w2v_for_classic_ml(vector_size=300):
    """
    Trains Word2Vec specifically for XGBoost/Logistic Regression.
    Tuned with higher vector_size and epochs for better accuracy.
    """
    from src.data.train_dataset_EDA import load_train_with_features
    
    # 1. Load Data
    df = load_train_with_features()
    X_train_text = df['clean_text']
    y_train = df['label'].values
    
    # 2. Train Word2Vec (TUNED SETTINGS: 300 dim, 50 epochs)
    
    w2v_model = train_word2vec(
        X_train_text, 
        vector_size=vector_size,  
        window=5,
        min_count=2,
        epochs=50                 
    )

    # 3. Convert Text -> Mean Vectors
    # We use the GLOBAL helper function defined at the top
    X_train_averaged = get_mean_vectors(w2v_model, X_train_text)

    # 4. Return ONLY what XGBoost needs
    return w2v_model, X_train_averaged, y_train

