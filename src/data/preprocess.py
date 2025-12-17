import os
import re
import pandas as pd
from src.data.dataset import load_emotion_dataset, get_label_names
from src.data.csv_loader import save_dataframes, load_dataframes

#Our dataset contains tweets (lots of noise, URLs, emojis etc)
#In order to make our model work properly using TF-IDF this is necessary.
def clean_text(text):
    
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ",text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ",text).strip()
    
    return text

def add_clean_text_column(df: pd.DataFrame):
    df = df.copy()
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    df["clean_word_len"] = df["clean_text"].apply(lambda x: len(x.split()))
    return df 


def process_and_save():
    
    train_df, test_df = load_dataframes("emotion")

    #Apply Cleaning
    train_clean = add_clean_text_column(train_df)
    test_clean = add_clean_text_column(test_df)

    #Save Processed Data (Using the helper!)
    save_dataframes(train_clean, test_clean, filename_prefix="emotion_processed")


if __name__ == "__main__":
    process_and_save()