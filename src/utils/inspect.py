from src.data.dataset import load_emotion_dataset, get_label_names, Label_map
from src.data.preprocess import clean_text, add_clean_text_column
from src.data.train_dataset_EDA import LengthStats
from src.data.train_dataset_EDA import (
    get_length_stats,
    load_train_with_features,
    get_label_counts,
    get_label_distribution,
    get_most_common_words_overall,
    get_most_common_words_per_label,
    find_short_or_empty_comments,
    duplicate_text
)
from src.data.EDA_visual import (
    plot_label_distribution,
    plot_length_histogram,
    plot_length_boxplot_per_label,
    plot_top_words_per_label)
import pandas as pd

def main():

    df = load_train_with_features()

    print("\n=== First 5 rows ===")
    print(df.head())

    print("=== Label counts ===")
    print(df["label"].value_counts())

    print("=== Labels with names ===")
    for label_id, count in df["label"].value_counts().items():
        print(f"{Label_map[label_id]} = {count}")

    print("\n=== Label Names from Dataset ===")
    print(get_label_names())

    sample = df.sample(5)
    

    print("\n===  5 random rows before clean text ===")
    print(sample["text"])

    
    cleaned = sample["text"].apply(clean_text)

    print("\n=== After clean ===")
    print(cleaned)

    
    stats = get_length_stats(df)

    print("=== Overall Stats ===")
    print(stats.overall)

    print("=== Per Label Stats ===")
    print(stats.per_label)

    print("=== Label Distribution (Train) ===")
    counts = get_label_counts(df)
    distribution = get_label_distribution(df)

    for label_id, count in counts.items():
        percentage = distribution[label_id]
        name = Label_map[label_id]
        print(f"{label_id}. ({name}: {count} samples -> {percentage:.2f}%)")


    print("=== TOP 20 Words Overall (Clean Text) ===")
    overall_words = get_most_common_words_overall(df, top_k=20)
    for word, count in overall_words:
        print(f"{word}: {count}")

    print("\n=== Top 10 Words per Emotion ===")    
    per_label_words = get_most_common_words_per_label(df, top_k=10)
    for label_id, words in per_label_words.items():
        print(f'\n{label_id} ({Label_map[label_id]}):')
        for words, count in words:
            print(f' {words}: {count}')

    
    short = find_short_or_empty_comments(df, max_words = 1)
    print("\n === Very Short or Empty Clean Text ===")
    print(short[['text', 'clean_text', 'clean_word_len']].head())

    dups = duplicate_text(df)
    print("\n=== Duplicated Clean Text")
    print(dups[["text", "clean_text", "count"]].head())

    plot_label_distribution(df)
    plot_length_histogram(df)
    plot_length_boxplot_per_label(df)
    plot_top_words_per_label(df)



if __name__ == "__main__":
    main()