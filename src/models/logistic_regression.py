from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.data.train_dataset_EDA import load_train_with_features, load_test_with_features
from src.features.tfidf import (
    fit_tfidf_on_clean_text_column,
    fit_tfidf_on_any_dataframe
)
from src.features.glove import (
    build_glove_features,
    sequences_to_vectors,
    build_glove_features_from_dfs
)
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.augmentation import augment_dataframe, AugmentConfig, class_balanced_augment



def train_lr_tfidf():
    vectorizer, X_train, y_train = fit_tfidf_on_clean_text_column()

    test_df = load_test_with_features()

    X_test_text = test_df['clean_text']
    y_test = test_df['label'].values 
    X_test = vectorizer.transform(X_test_text)

    

    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)


    print("===== Logistic Regression (TF-IDF) Train → Test =====")
    print("Train X:", X_train.shape, "Test X:", X_test.shape)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, preds))


def train_lr_glove():
    X_train_seq, y_train, X_test_seq, y_test, tokenizer, embedding_matrix, num_words = (
        build_glove_features()
    )

    # Sequences to vectors for logistic regression
    X_train = sequences_to_vectors(X_train_seq, embedding_matrix)
    X_test = sequences_to_vectors(X_test_seq, embedding_matrix)
    
    # Embeddings are dense features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced'
    )

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    print("===== Logistic Regression (Glove) Train → Test =====")
    print("Train X:", X_train.shape, "Test X:", X_test.shape)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, preds))

def train_lr_tfidf_aug():
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
        text_col='clean_text',
        label_col='label',
        target_per_class=None,
        config=aug_cfg
    )

    vectorizer, X_train, y_train = fit_tfidf_on_any_dataframe(train_df_bal)

    X_test = vectorizer.transform(test_df['clean_text'])
    y_test = test_df["label"].values

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)


    print("===== Logistic Regression (TF-IDF) AUGMENTED Train → Test =====")
    print("Train X:", X_train.shape, "Test X:", X_test.shape)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, preds))


def train_lr_glove_aug():
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

    # Optional sanity checks
    print("Original train size:", len(train_df))
    print("Balanced train size:", len(train_df_bal))
    print("Original label counts:\n", train_df["label"].value_counts())
    print("Balanced label counts:\n", train_df_bal["label"].value_counts())

    X_train_seq, y_train, X_test_seq, y_test, tokenizer, embedding_matrix, num_words = (
        build_glove_features_from_dfs(
            train_df=train_df_bal,
            test_df=test_df,
            text_col='clean_text',
            label_col='label',
            max_words=20000,
            embedding_dim=100,
            glove_path="data/glove.6B.100d.txt",
            max_len=100                                            
        )
    )

    X_train = sequences_to_vectors(X_train_seq, embedding_matrix)
    X_test = sequences_to_vectors(X_test_seq, embedding_matrix)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs'
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("===== Logistic Regression (GloVe) CLASS-BALANCED AUG Train → Test =====")
    print("Train X:", X_train.shape, "Test X:", X_test.shape)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (TEST):")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    train_lr_tfidf()
    train_lr_glove()
    train_lr_tfidf_aug()
    train_lr_glove_aug()


