import pandas as pd
import os

# Path to your dataset
dataset_dir = "./data/noise/"

# Leads to check
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# List to store problematic files
problematic_files = []

# Check each file in the dataset directory
print("Checking for missing columns in dataset files...")
for filename in os.listdir(dataset_dir):
    if filename.endswith('.feather'):
        file_path = os.path.join(dataset_dir, filename)
        try:
            df = pd.read_feather(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            problematic_files.append(file_path)
            continue

        # Check for missing columns
        missing_columns = [f"{lead}_value" for lead in leads if f"{lead}_value" not in df.columns]
        if missing_columns:
            print(f"File {file_path} is missing columns: {missing_columns}")
            problematic_files.append(file_path)

# Save problematic files to a text file (optional)
with open("problematic_files.txt", "w") as f:
    for file in problematic_files:
        f.write(file + "\n")
print(f"Problematic files saved to 'problematic_files.txt'.")

# Paths to training and validation CSVs
train_csv_path = "./data/noise/train_denoise.csv"
val_csv_path = "./data/noise/val_denoise.csv"

# Remove problematic files from CSV
for csv_path in [train_csv_path, val_csv_path]:
    try:
        df = pd.read_csv(csv_path)
        original_count = len(df)
        df = df[~df['paths'].isin(problematic_files)]  # Keep rows not in problematic_files
        df.to_csv(csv_path, index=False)
        removed_count = original_count - len(df)
        print(f"Updated {csv_path}: Removed {removed_count} problematic files.")
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
