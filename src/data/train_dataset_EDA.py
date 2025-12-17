import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.dataset import load_emotion_dataset, Label_map, get_label_names
from src.data.preprocess import add_clean_text_column
from collections import Counter


class LengthStats:
    """
    A container that holds summary statistics about text lengths.
    
    Attributes:
    
    overall: pd.DataFrame
        Length stats computed across the entire training dataset.
        This includes summary metrics (mean, std, percentiles) for:
        - char_len: number of characters in the original text
        - word_len: number of words in the original text
        - clean_word_len: number of words after cleaning
        
    per_label: pd.DataFrame
        Length stats computed separately for each emotion label.
        For each label (e.g., joy, sadness), this includes:
        - mean length (word_len, clean_word_len)
        - standard deviation
        - median
        
        This helps us understand differences in how long texts are for each emotion
        class (e.g., anger comments might be longer)"""
    
    
    def __init__(self, overall, per_label):
        self.overall = overall
        self.per_label = per_label 


def load_train_with_features():
    # Load the train split and compute additional features needed for EDA.
    # Will return a pd.DataFrame with the train dataset with 
    # char_len, word_len, clean_text, clean_word_len

    df = load_emotion_dataset(split="train").copy()

    # raw lengths
    df["char_len"] = df["text"].astype(str).str.len()
    df["word_len"] = df["text"].astype(str).str.split().str.len()

    # cleaned text + cleaned lengths
    df = add_clean_text_column(df)
    df["clean_word_len"] = df["clean_text"].astype(str).str.split().str.len()

    return df 

def get_length_stats(df):

    #overall stats across the entire dataset
    overall = df[["char_len", "word_len", "clean_word_len"]].describe(
        percentiles=[0.1,0.25,0.5,0.75,0.9]
    )

    #stats grouped by emotion label
    per_label = df.groupby("label")[["word_len", "clean_word_len"]].agg(
        ["mean", "std", "median"]
    )

    per_label.index = [
        f"{label_id} ({Label_map[label_id]})"
        for label_id in per_label.index
    ]


    return LengthStats(overall, per_label)


def get_label_counts(df):
    
    return df['label'].value_counts().sort_index()

def get_label_distribution(df):

    counts = get_label_counts(df)
    return counts / len(df) * 100.0



def get_most_common_words_overall(df, top_k: int = 20):

    all_words = []

    for text in df['clean_text']:
        all_words.extend(str(text).split()) #split text into individual words inside a string

    counter = Counter(all_words)
    return counter.most_common(top_k)

def get_most_common_words_per_label(df, top_k: int = 10):

    results = {}

    # Loop over each label id (0-5) in order.
    # Label_map keys are: [0, 1, 2, 3, 4, 5]
    for label_id in sorted(Label_map.keys()):

        #Select rows only where the comment has this specific label
        #Example: if label_id == 0, we get only sadness comments.
        subset = df[df['label'] == label_id]['clean_text']

        words = []
        for text in subset:
            words.extend(str(text).split())

        #Counter creates a frequency dictionary
        #like, {'sad': 230, "miss": 195,...}
        counter = Counter(words)


        results[label_id] = counter.most_common(top_k)

    return results 


def find_short_or_empty_comments(df, max_words: int = 1):

    return df[df['clean_word_len'] <= max_words]

def duplicate_text(df):

    dup_counts = df['clean_text'].value_counts()
    dup_values = dup_counts[dup_counts > 1].index

    duplicates = df[df['clean_text'].isin(dup_values)].copy()
    duplicates['count'] = duplicates['clean_text'].map(dup_counts)

    return duplicates.sort_values(by="count", ascending=False)