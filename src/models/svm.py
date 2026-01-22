from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix

from src.data.train_dataset_EDA import load_test_with_features
from src.features.tfidf import fit_tfidf_on_clean_text_column


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


if __name__ == "__main__":
    train_svm_tfidf()
