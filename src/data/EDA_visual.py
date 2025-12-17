import matplotlib.pyplot as plt 
import pandas as pd
from src.data.dataset import Label_map
from src.data.train_dataset_EDA import get_label_counts, get_most_common_words_per_label

def plot_label_distribution(df):

    counts = get_label_counts(df)
    labels = [Label_map[i] for i in counts.index]

    plt.figure(figsize=(8,5))
    plt.bar(labels, counts.values)
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title('Label Distribution (Train)')
    plt.tight_layout()
    plt.show()


def plot_length_histogram(df):

    plt.figure(figsize=(8,5))
    plt.hist(df['clean_word_len'], bins =30)
    plt.xlabel("Cleaned text word count")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cleaned text lengths")
    plt.tight_layout()
    plt.show()

def plot_length_boxplot_per_label(df):

    plt.figure(figsize=(10,6))

    data = [df[df['label'] == i]['clean_word_len'] for i in Label_map.keys()]
    labels = [Label_map[i] for i in Label_map.keys()]

    plt.boxplot(data, labels = labels, showfliers=False)
    plt.xlabel('Emotion')
    plt.ylabel('Cleaned Word Count')
    plt.title('Distribution of Cleaned text lengths per Emotion')
    plt.tight_layout
    plt.show()

def plot_top_words_per_label(df, top_k: int = 10):

    per_label = get_most_common_words_per_label(df, top_k=top_k)
    num_labels = len(Label_map)

    plt.figure(figsize=(16,10))

    for i, label_id in enumerate(Label_map.keys(), start=1):
        words, counts = zip(*per_label[label_id])

        plt.subplot(2, 3, i)
        plt.bar(words, counts)
        plt.title(Label_map[label_id])
        plt.xticks(rotation=45)
    
    plt.tight_layout
    plt.show()
    