import pandas as pd
df = pd.read_feather('./data/noise/20200917190252_0normal0_3.feather')
print(df.head())
df['I_value'] = df['I']
df['II_value'] = df['II']
df['III_value'] = df['III']

print(df.columns)

# Verify Dataset Balance
# train_df = pd.read_csv("./data/noise/train_denoise.csv")
# print(train_df['lead'].value_counts())

# Verify Labels in the Dataset
data_feather = pd.read_feather('./data/noise/20200917190252_0normal0_0.feather')
print(data_feather.head())
print(data_feather['{}_label'.format('I')].value_counts())
