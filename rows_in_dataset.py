import pandas as pd

train_csv_path = "./data/noise/train_denoise.csv"
df = pd.read_csv(train_csv_path)

print(f"Number of rows in train_denoise.csv: {len(df)}")
print(df.head())