import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import random

class Noise_Dataset(Dataset):
    def __init__(self, csv_file, leads=None, n_max_cls=3, date_len=5000, random_crop=False, transform=None):
        """
        Initializes the dataset object.

        Args:
            csv_file (string): Path to the CSV file with dataset information.
            leads (list): List of leads to extract from each file.
            n_max_cls (int): Maximum number of classes for labels.
            date_len (int): Length of the input signal to be processed.
            random_crop (bool): Whether to randomly crop the input signal.
            transform (callable, optional): Optional transform to be applied to the sample.
        """
        self.dataframe = self.__select_lead_df(csv_file, leads)
        self.leads = leads
        self.date_len = date_len
        self.random_crop = random_crop
        self.transform = transform
        self.n_max_cls = n_max_cls

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataframe)

    def __select_lead_df(self, csv_file, leads):
        """
        Filters the dataset to include only the specified leads.

        Args:
            csv_file (str): Path to the CSV file containing dataset information.
            leads (list): List of leads to include.

        Returns:
            pd.DataFrame: Filtered dataframe containing only the specified leads.
        """
        df = pd.read_csv(csv_file)
        if 'lead' not in df.columns or 'paths' not in df.columns:
            raise KeyError("The dataset CSV file must contain 'lead' and 'paths' columns.")

        df_s = []
        for lead in leads:
            if lead not in df['lead'].values:
                print(f"Warning: Lead '{lead}' not found in dataset.")
            df_s.append(df[df['lead'] == lead])

        if not df_s:
            raise ValueError("No valid leads found in the dataset.")
        dataframe = pd.concat(df_s)
        return dataframe

    def __getitem__(self, idx):
        """
        Retrieves a sample by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Contains the input signal and corresponding labels.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file path and lead from the dataset
        data_file = os.path.join(self.dataframe.iloc[idx]['paths'])
        lead = self.dataframe.iloc[idx]['lead']

        # Load the file (CSV format only)
        if data_file.endswith('.csv'):
            data = pd.read_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}. Only .csv files are supported.")

        # Extract the signal and label
        input_signal = data.get(f'{lead}_value', data.get(lead, None))
        if input_signal is None:
            raise KeyError(f"Column '{lead}_value' or '{lead}' not found in file: {data_file}")

        label_signal = data.get(f'{lead}_label', None)
        if label_signal is None:
            label_signal = np.zeros(len(input_signal), dtype=int)  # Default to all-zero labels

        # Process the signal length
        if len(input_signal) < self.date_len:
            raise ValueError(f"Signal length ({len(input_signal)}) is shorter than required date_len ({self.date_len}).")

        if self.random_crop:
            # Randomly crop the input signal if enabled
            start_idx = random.randint(0, len(input_signal) - self.date_len)
        else:
            start_idx = 0
        input_signal = np.array(input_signal[start_idx:start_idx + self.date_len]).astype('float32').reshape(1, -1)
        label_signal = np.array(label_signal[start_idx:start_idx + self.date_len]).astype('int64').clip(0, self.n_max_cls)

        # Create a sample dictionary
        sample = {'input': input_signal, 'label': label_signal}

        # Apply transformations if any are specified
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """
    Converts ndarrays in the sample to PyTorch tensors.
    """
    def __call__(self, sample):
        input_i, label_i = sample['input'], sample['label']
        return {
            'input': torch.from_numpy(input_i),
            'label': torch.from_numpy(label_i),
        }

def get_transform(train=True):
    """
    Creates a set of transformations to apply to the dataset.

    Args:
        train (bool): Whether the transformations are for training.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    transforms = [ToTensor()]
    return T.Compose(transforms)
