import re
import pandas as pd
from src.data.dataset import load_emotion_dataset, get_label_names

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
    
    return df 