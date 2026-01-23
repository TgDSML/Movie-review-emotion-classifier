import os
# Disable Intel optimizations to prevent freezing/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import tensorflow as tf
from src.data.train_dataset_EDA import load_train_with_features

# Using full path for stability
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

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
    # Preprocess: Convert list of strings to list of lists of tokens
    tokenized_sentences = [simple_preprocess(text) for text in texts]
    
    # Initialize and Train
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1 # Skip-gram
    )
    
    model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=epochs)
    return model

def create_embedding_matrix(tokenizer, w2v_model, embedding_dim):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
            
    return embedding_matrix

def fit_w2v_on_clean_text_column(
        vector_size=100,
        window=5,
        min_count=2,
        max_words=20000,
        max_len=100
):
    
    X_train_text, y_train = load_w2v_input()

    # 1. Train Word2Vec Model
    w2v_model = build_and_train_w2v_model(
        X_train_text, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count
    )

    # 2. Fit Keras Tokenizer (Mapping words to Integers)
    tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(X_train_text)

    # 3. Create Sequences (The X input for the Neural Network)
    sequences = tokenizer.texts_to_sequences(X_train_text)
    X_train_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # 4. Build Embedding Matrix (The Weights for the Neural Network)
    embedding_matrix = create_embedding_matrix(tokenizer, w2v_model, embedding_dim=vector_size)

    return tokenizer, embedding_matrix, X_train_padded, y_train


def fit_w2v_on_dataframe(
    df,
    text_col="clean_text",
    label_col="label",
    vector_size=100,
    window=5,
    min_count=2,
    max_words=20000,
    max_len=100,
):
    """
    Data-driven Word2Vec pipeline:
    trains Word2Vec + builds tokenizer/sequences/embedding_matrix on the provided df.
    This is what enables augmentation/balancing experiments.
    """
    texts = df[text_col].astype(str).tolist()
    y = df[label_col].values

    # 1) Train Word2Vec model on THESE texts
    w2v_model = build_and_train_w2v_model(
        texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
    )

    # 2) Fit tokenizer on THESE texts
    tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)

    # 3) Convert to sequences + pad
    sequences = tokenizer.texts_to_sequences(texts)
    X_seq = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # 4) Create embedding matrix aligned with tokenizer indices
    embedding_matrix = create_embedding_matrix(tokenizer, w2v_model, embedding_dim=vector_size)

    return tokenizer, embedding_matrix, X_seq, y


def sequences_to_mean_max_vectors(X_seq, embedding_matrix):

    vectors = []
    embedding_dim = embedding_matrix.shape[1]

    for seq in X_seq:
        valid = seq[seq>0] # removes padding (not helpful for linear classifier)

        if len(valid) == 0:
            mean_vec = np.zeros(embedding_dim)
            max_vec = np.zeros(embedding_dim)
        else:
            word_vecs = embedding_matrix[valid]
            mean_vec = word_vecs.mean(axis=0)
            max_vec = word_vecs.max(axis=0)
        
        vectors.append(np.concatenate([mean_vec, max_vec]))

    return np.vstack(vectors)

def main():
    tokenizer, embedding_matrix, X_train, y_train = fit_w2v_on_clean_text_column()

    print("X_train (padded) shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Embedding Matrix shape:", embedding_matrix.shape)
    print("Vocabulary size:", len(tokenizer.word_index))

if __name__ == "__main__":
    main()