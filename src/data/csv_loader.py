import os
import pandas as pd

def get_data_folder_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_folder = os.path.join(project_root, 'data')
    return data_folder

def save_dataframes(train_df, test_df, filename_prefix):
    data_folder = get_data_folder_paths()
    os.makedirs(data_folder, exist_ok=True)

    train_path = os.path.join(data_folder, f'{filename_prefix}_train.csv')
    test_path = os.path.join(data_folder, f'{filename_prefix}_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

def load_dataframes(filename_prefix):
    data_folder = get_data_folder_paths()
    
    train_path = os.path.join(data_folder, f'{filename_prefix}_train.csv')
    test_path = os.path.join(data_folder, f'{filename_prefix}_test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing required file: {train_path}. Did you run dataset.py first?")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df