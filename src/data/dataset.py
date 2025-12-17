import os
from datasets import load_dataset
import pandas as pd
from src.data.csv_loader import save_dataframes

Label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def load_emotion_dataset(split="train", config="split"):
    dataset = load_dataset("dair-ai/emotion", config, split=split)
    df = pd.DataFrame(dataset)
    
    return df

def get_label_names():
    dataset = load_dataset("dair-ai/emotion", "split")
    label_feature = dataset["train"].features["label"]
    
    return label_feature.names

def save_raw_data():
    
    # 1. Load the data
    train_df = load_emotion_dataset(split="train")
    test_df = load_emotion_dataset(split="test")

    # 2. Use the helper to save (it handles folder paths for you)
    # This will save: 'emotion_train.csv' and 'emotion_test.csv'
    save_dataframes(train_df, test_df, filename_prefix="emotion")
    
  

if __name__ == "__main__":
    save_raw_data()
