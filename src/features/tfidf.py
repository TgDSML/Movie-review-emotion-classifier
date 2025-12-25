from src.data.train_dataset_EDA import load_train_with_features
from sklearn.feature_extraction.text import TfidfVectorizer 
from src.data.preprocess import add_clean_text_column
from src.data.train_dataset_EDA import load_train_with_features

def load_vectorizer_input():

    df = load_train_with_features()
    X_train_text = df['clean_text']
    y_train = df['label'].values
    return X_train_text, y_train 


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
    
    X_train_text , y_train = load_vectorizer_input()

    vectorizer = build_tfidf_vectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )

    X_train = vectorizer.fit_transform(X_train_text)

    return vectorizer, X_train, y_train 

def main():
    vectorizer, X_train, y_train = fit_tfidf_on_clean_text_column()

    print("X shape:", X_train.shape)
    print("y shape:", y_train.shape)
    print("Vocabulary size:", len(vectorizer.get_feature_names_out()))


if __name__ == "__main__":
    main()

