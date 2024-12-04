import pandas as pd
import os

# Path to your dataset
dataset_dir = "./data/noise/"

# Leads to check
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Check each file in the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.feather'):
        file_path = os.path.join(dataset_dir, filename)
        df = pd.read_feather(file_path)

        # Check for missing columns
        for lead in leads:
            column_name = f"{lead}"
            if column_name not in df.columns:
                print(f"Missing column '{column_name}' in file: {file_path}")



