import pandas as pd
import os

# def update_csv_paths(csv_path, old_ext='.feather', new_ext='.csv'):
#     # Load the CSV file
#     df = pd.read_csv(csv_path)

#     # Replace file extensions
#     if 'paths' not in df.columns:
#         raise KeyError(f"'{csv_path}' does not contain a 'paths' column.")
    
#     df['paths'] = df['paths'].str.replace(old_ext, new_ext)

#     # Save the updated CSV
#     df.to_csv(csv_path, index=False)
#     print(f"Updated file paths in {csv_path} to use {new_ext} instead of {old_ext}")

# # List your dataset CSV files
# dataset_csvs = [
#     './data/segmentation/train_segmentation.csv',
#     './data/segmentation/val_segmentation.csv'
# ]

# # Update each dataset CSV
# for csv_path in dataset_csvs:
#     if os.path.exists(csv_path):
#         update_csv_paths(csv_path)
#     else:
#         print(f"Dataset CSV file {csv_path} does not exist.")


# Verify the Presence of .csv Files
def check_missing_files(csv_path):
    df = pd.read_csv(csv_path)
    missing_files = [path for path in df['paths'] if not os.path.exists(path)]
    
    if missing_files:
        print(f"Missing files in {csv_path}:")
        for file in missing_files:
            print(file)
    else:
        print(f"All files in {csv_path} exist.")

# Check for missing files
dataset_csvs = [
    './data/segmentation/train_segmentation.csv',
    './data/segmentation/val_segmentation.csv'
]

for csv_path in dataset_csvs:
    if os.path.exists(csv_path):
        check_missing_files(csv_path)
    else:
        print(f"Dataset CSV file {csv_path} does not exist.")