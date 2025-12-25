from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from src.data.train_dataset_EDA import load_train_with_features, load_test_with_features
from src.features.tfidf import fit_tfidf_on_clean_text_column


def main():
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

if __name__ == "__main__":
    main()


