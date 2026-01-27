from sklearn.svm import SVC, LinearSVC  
from sklearn.metrics import classification_report, confusion_matrix
from src.features.tfidf import fit_tfidf_on_clean_text_column
from src.data.train_dataset_EDA import load_train_with_features, load_test_with_features
from src.features.tfidf import fit_tfidf_on_any_dataframe
from sklearn.preprocessing import StandardScaler
from src.data.augmentation import AugmentConfig, class_balanced_augment
from src.features.word2vec import fit_w2v_on_dataframe, sequences_to_mean_max_vectors
from src.features.glove import (
    build_glove_features_from_dfs,
    sequences_to_vectors
)


import tensorflow as tf
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

#TFIDF
def train_svm_tfidf(C: float = 10.0):
    # Train TF-IDF
    vectorizer, X_train, y_train = fit_tfidf_on_clean_text_column()

    # Test TF-IDF 
    test_df = load_test_with_features()
    X_test_text = test_df["clean_text"]   
    y_test = test_df["label"].values
    X_test = vectorizer.transform(X_test_text)

    # Model
    model = SVC(
        C=C,
        kernel="linear",
        class_weight="balanced",
    ) 

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("===== Linear SVM (TF-IDF) Train → Test =====")
    print("Train X:", X_train.shape, "Test X:", X_test.shape)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, preds))

    return model, vectorizer

#TFIDF-AUGMENTED
def train_svm_tfidf_aug(C: float = 10.0):
    train_df = load_train_with_features()
    test_df = load_test_with_features()

    aug_cfg = AugmentConfig(
        mode="eda",
        n_aug_per_sample=1,
        keep_original=True,
        seed=42
    )

    train_df_bal = class_balanced_augment(
        df=train_df,
        text_col="clean_text",
        label_col="label",
        target_per_class=None,
        config=aug_cfg
    )

    vectorizer, X_train, y_train = fit_tfidf_on_any_dataframe(train_df_bal)

    X_test = vectorizer.transform(test_df["clean_text"])
    y_test = test_df["label"].values

    model = SVC(
        C=C,
        kernel="linear",
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("===== Linear SVM (TF-IDF) AUGMENTED Train → Test =====")
    print("Train X:", X_train.shape, "Test X:", X_test.shape)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, preds))

    return model, vectorizer




def train_svm_word2vec(
    C: float = 1.0,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    max_words: int = 20000,
    max_len: int = 100,
):
    train_df = load_train_with_features()
    test_df = load_test_with_features()

    # Fit Word2Vec + tokenizer on TRAIN
    tokenizer, embedding_matrix, X_train_seq, y_train = fit_w2v_on_dataframe(
        df=train_df,
        text_col="clean_text",
        label_col="label",
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        max_words=max_words,
        max_len=max_len,
    )

    # Transform TEST with same tokenizer
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(test_df["clean_text"].astype(str).tolist()),
        maxlen=max_len,
        padding="post",
        truncating="post",
    )
    y_test = test_df["label"].values

    # Pool sequences -> dense vectors
    X_train = sequences_to_mean_max_vectors(X_train_seq, embedding_matrix)
    X_test = sequences_to_mean_max_vectors(X_test_seq, embedding_matrix)

    # Scale (important for SVM on dense features)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearSVC(C=C, class_weight="balanced")  
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("===== Linear SVM (Word2Vec mean+max) Train → Test =====")
    print("Train X:", X_train.shape, "Test X:", X_test.shape)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, preds))

    return model, tokenizer, scaler


def train_svm_word2vec_aug(
    C: float = 1.0,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    max_words: int = 20000,
    max_len: int = 100,
):
    train_df = load_train_with_features()
    test_df = load_test_with_features()

    aug_cfg = AugmentConfig(
        mode="eda",
        n_aug_per_sample=1,
        keep_original=True,
        seed=42,
    )

    # Augment TRAIN only (balanced)
    train_df_aug = class_balanced_augment(
        df=train_df,
        text_col="clean_text",
        label_col="label",
        target_per_class=None,
        config=aug_cfg,
    )

    # Fit Word2Vec + tokenizer on AUGMENTED TRAIN
    tokenizer, embedding_matrix, X_train_seq, y_train = fit_w2v_on_dataframe(
        df=train_df_aug,
        text_col="clean_text",
        label_col="label",
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        max_words=max_words,
        max_len=max_len,
    )

    # Transform TEST with same tokenizer
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(test_df["clean_text"].astype(str).tolist()),
        maxlen=max_len,
        padding="post",
        truncating="post",
    )
    y_test = test_df["label"].values

    # Pool sequences -> dense vectors
    X_train = sequences_to_mean_max_vectors(X_train_seq, embedding_matrix)
    X_test = sequences_to_mean_max_vectors(X_test_seq, embedding_matrix)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearSVC(C=C, class_weight="balanced")  
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("===== Linear SVM (Word2Vec mean+max) AUG Train → Test =====")
    print("Train X:", X_train.shape, "Test X:", X_test.shape)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, preds))

    return model, tokenizer, scaler

def train_svm_glove_aug(
    C: float = 1.0,
    max_words: int = 20000,
    embedding_dim: int = 100,
    max_len: int = 100,
    glove_path: str = "data/glove.6B.100d.txt",
):
    train_df = load_train_with_features()
    test_df = load_test_with_features()

    aug_cfg = AugmentConfig(
        mode="eda",
        n_aug_per_sample=1,
        keep_original=True,
        seed=42
    )

    train_df_aug = class_balanced_augment(
        df=train_df,
        text_col="clean_text",
        label_col="label",
        target_per_class=None,
        config=aug_cfg
    )

    X_train_seq, y_train, X_test_seq, y_test, tokenizer, embedding_matrix, num_words = (
        build_glove_features_from_dfs(
            train_df=train_df_aug,
            test_df=test_df,
            text_col="clean_text",
            label_col="label",
            max_words=max_words,
            embedding_dim=embedding_dim,
            glove_path=glove_path,
            max_len=max_len
        )
    )

    # sequences -> dense vectors for SVM
    X_train = sequences_to_vectors(X_train_seq, embedding_matrix)
    X_test = sequences_to_vectors(X_test_seq, embedding_matrix)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearSVC(C=C, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("===== Linear SVM (GloVe) AUG Train → Test =====")
    print("Train X:", X_train.shape, "Test X:", X_test.shape)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, preds))

    return model, tokenizer, embedding_matrix, scaler


if __name__ == "__main__":
    train_svm_tfidf()
    train_svm_tfidf_aug()
    train_svm_word2vec()
    train_svm_word2vec_aug()
    train_svm_glove_aug()
