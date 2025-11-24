from datasets import load_dataset
import pandas as pd

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





