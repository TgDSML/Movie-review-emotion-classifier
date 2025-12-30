import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def build_glove_tokenizer_and_matrix(
    texts,
    glove_path="notebooks/glove.6B.100d.txt",
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
        glove_path="notebooks/glove.6B.100d.txt",
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
    
    # Gets the size of each word vector
    # This will be the number of features for Logistic Regression
    embedding_dim = embedding_matrix.shape[1]

    # Prepares the final feature matrix, shape: (number_of_sentences, embedding_dim)
    X = np.zeros((len(sequences), embedding_dim))

    
    
    # Loop over each sentence where:
    # i is the sentence index and seq is list of token IDs for the i-th sentence
    for i, seq in enumerate(sequences):
        vectors = []
        for idx in seq:
            if idx == 0:  # padded position, idx is a word ID
                # Padded positions got to be ingored so we skip to the next one
                continue  
            v = embedding_matrix[idx]  # -> word vector
            
            # True if at least one value in v is non-zero
            # False if ALL values are zero
            if np.any(v):
                vectors.append(v)
            # So if for example a sentence has two words, we have 
            # 2 glove vectors, X is going to be 
            # vectors = [
            #    [2, 4],
            #    [4, 6]
            #    mean = [(2+4)/2, (4+6)/2] = [3, 5]

            if vectors:
                X[i] = np.mean(vectors, axis = 0)

    return X 

def main():
    X_train, y_train, X_test, y_test, tokenizer, embedding_matrix, num_words = (
        build_glove_features()
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Embedding matrix shape:", embedding_matrix.shape)
    print("num_words:", num_words)


if __name__ == "__main__":
    main()



