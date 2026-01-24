import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def build_glove_tokenizer_and_matrix(
    texts,
    glove_path="data/glove.6B.100d.txt",
    max_words=20000,
    embedding_dim=100,
):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    num_words = min(max_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_words:
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec

    return tokenizer, embedding_matrix, num_words

def build_glove_features(
    max_words=20000,
    embedding_dim=100,
    glove_path='data/glove.6B.100d.txt',
    max_len=100,
):
    train_path = "data/emotion_processed_train.csv"
    test_path  = "data/emotion_processed_test.csv"

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    texts_train = train_df["text"].astype(str).tolist()
    texts_test  = test_df["text"].astype(str).tolist()

    y_train = train_df["label"].to_numpy(dtype="int64").ravel()
    y_test  = test_df["label"].to_numpy(dtype="int64").ravel()

    tokenizer, embedding_matrix, num_words = build_glove_tokenizer_and_matrix(
        texts=texts_train,
        glove_path=glove_path,
        max_words=max_words,
        embedding_dim=embedding_dim,
    )

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(texts_train),
        maxlen=max_len,
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(texts_test),
        maxlen=max_len,
    )

    return X_train, y_train, X_test, y_test, tokenizer, embedding_matrix, num_words

def build_glove_features_from_dfs(
    train_df,
    test_df,
    text_col="clean_text",
    label_col="label",
    max_words=20000,
    embedding_dim=100,
    glove_path="data/glove.6B.100d.txt",
    max_len=100,
):
    # This function is flexible, uses the dataframes we pass 
    texts_train = train_df[text_col].astype(str).tolist()
    texts_test  = test_df[text_col].astype(str).tolist()

    y_train = train_df[label_col].to_numpy(dtype="int64").ravel()
    y_test  = test_df[label_col].to_numpy(dtype="int64").ravel()

    tokenizer, embedding_matrix, num_words = build_glove_tokenizer_and_matrix(
        texts=texts_train,
        glove_path=glove_path,
        max_words=max_words,
        embedding_dim=embedding_dim,
    )

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(texts_train),
        maxlen=max_len,
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(texts_test),
        maxlen=max_len,
    )

    return X_train, y_train, X_test, y_test, tokenizer, embedding_matrix, num_words



def sequences_to_vectors(sequences, embedding_matrix):
    embedding_dim = embedding_matrix.shape[1]
    # We will output Mean + Max (Size = 2 * dim)
    X = np.zeros((len(sequences), 2 * embedding_dim), dtype="float32")

    for i, seq in enumerate(sequences):
        vectors = []
        for idx in seq:
            if 0 < idx < embedding_matrix.shape[0]: # Valid index
                v = embedding_matrix[idx]
                if np.any(v):
                    vectors.append(v)

        if vectors:
            vectors = np.vstack(vectors)
            mean_vec = np.mean(vectors, axis=0)
            max_vec  = np.max(vectors, axis=0)
            X[i] = np.concatenate([mean_vec, max_vec])
        # Else leaves zeros
    return X



def main():
    X_train_seq, y_train_glove, X_test_seq, y_test_glove, tokenizer, embedding_matrix, num_words = (
        build_glove_features()
    )

    print("X_train_seq shape:", X_train_seq.shape)
    print("X_test_seq shape:", X_test_seq.shape)
    print("y_train_glove shape:", y_train_glove.shape)
    print("y_test_glove shape:", y_test_glove.shape)
    
    # NEW: sequences_to_vectors dimensions
    print("X_train_glove shape:", sequences_to_vectors(X_train_seq, embedding_matrix).shape)
    print("X_test_glove shape:", sequences_to_vectors(X_test_seq, embedding_matrix).shape)
    
    print("Embedding matrix shape:", embedding_matrix.shape)
    print("num_words:", num_words)

if __name__ == "__main__":
    main()


