import pandas as pd

# # Path to your dataset CSV
# csv_path = "./data/segmentation/train_segmentation.csv"

# # Load the CSV file
# df = pd.read_csv(csv_path)

# # Replace `.feather` with `.csv` in the 'paths' column
# df['paths'] = df['paths'].str.replace('.feather', '.csv')

# # Save the updated CSV
# df.to_csv(csv_path, index=False)
# print("Updated file paths to use .csv successfully!")


import os

# Check if all files exist
csv_path = "./data/segmentation/train_segmentation.csv"
df = pd.read_csv(csv_path)

missing_files = []
for path in df['paths']:
    if not os.path.exists(path):
        missing_files.append(path)

if missing_files:
    print("Missing files:")
    for path in missing_files:
        print(path)
else:
    print("All files exist!")
