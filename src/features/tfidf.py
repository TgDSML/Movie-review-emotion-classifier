from src.data.train_dataset_EDA import load_train_with_features
from sklearn.feature_extraction.text import TfidfVectorizer 
from src.data.preprocess import add_clean_text_column
from src.data.train_dataset_EDA import load_train_with_features

def load_vectorizer_input():

    df = load_train_with_features()
    X_text = df['clean_text']
    y = df['label'].values
    return X_text, y


def build_tfidf_vectorizer(
        ngram_range = (1,2),
        min_df = 2,
        max_df = 0.95,
        max_features = 20000
):

    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )


def fit_tfidf_on_clean_text_column(
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95,
        max_features=20000
):
    
    X_text, y = load_vectorizer_input()

    vectorizer = build_tfidf_vectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )

    X_train = vectorizer.fit_transform(X_text)

    return vectorizer, X_train, y

def main():
    vec, X, y = fit_tfidf_on_clean_text_column()

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Vocabulary size:", len(vec.get_feature_names_out()))


if __name__ == "__main__":
    main()

