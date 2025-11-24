from src.data.dataset import load_emotion_dataset, get_label_names, Label_map
from src.data.preprocess import clean_text, add_clean_text_column
import pandas as pd

def main():

    df = load_emotion_dataset("train")

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

if __name__ == "__main__":
    main()