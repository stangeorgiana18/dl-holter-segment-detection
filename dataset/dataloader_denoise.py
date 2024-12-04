import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
# from torch._C import device
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import warnings
warnings.filterwarnings("ignore")

from scipy.signal import find_peaks

def random_crop_noise(bw_noise_full, input_len=5000, flat_top_n=10):
    """
    Randomly crops a segment from a noise array and optionally removes peaks.

    Args:
        bw_noise_full (array): The full noise array.
        input_len (int): Length of the cropped segment.
        flat_top_n (int): Number of peaks to flatten in the cropped segment.

    Returns:
        np.array: Cropped noise segment with optional modifications.
    """
    # Randomly select a start point for cropping
    start_pt = random.randint(10, len(bw_noise_full) - input_len - 10)
    end_pt = start_pt + input_len

    # Extract the cropped noise segment
    bw_noise = bw_noise_full[start_pt:end_pt]
    assert bw_noise.shape[0] == input_len

    # Optionally remove peaks from the cropped segment
    r_peaks = []
    if flat_top_n > 0:
        r_peaks, properties = find_peaks(bw_noise, distance=100)  # Detect peaks with a minimum distance of 100
        qrs_length = 0.05 * 500
        for r_peak in r_peaks:
            before = r_peak - int(qrs_length / 2)
            after = r_peak + int(qrs_length / 2) + 1
            bw_noise[before:after] = 0  # Flatten the peaks

    # Optionally introduce noise chunks
    if random.uniform(0, 1) > 0.6:
        n_noise_chunk = random.randint(1, 6)
        bw_noise_raw = np.array([0] * len(bw_noise))
        for i in range(n_noise_chunk):
            chunk_size = random.randint(300, 600)
            start_pt = random.randint(10, len(bw_noise_raw) - chunk_size - 10)
            end_pt = start_pt + chunk_size
            bw_noise_raw[start_pt:end_pt] = bw_noise[start_pt:end_pt]

        return bw_noise_raw
    else:
        return bw_noise

class Noise_Dataset(Dataset):
    """
    Custom dataset for loading noise and normal data samples.

    Args:
        normal_csv (str): Path to the CSV file with normal data.
        noise_csv (list of str): List of paths to CSV files with noise data.
        n_sample (int): Number of samples to use.
        leads (list): Selected leads for processing.
        input_len (int): Length of each input segment.
        max_syn_noise (int): Maximum number of synthetic noise chunks.
        add_noise_ratio (float): Ratio for adding noise to normal data.
        random_ranges (list): Range for random noise ratios.
        transform (callable): Transformations to apply to samples.
        train_mode (bool): Whether the dataset is in training mode.
    """

    def __init__(self, normal_csv, noise_csv, n_sample=10000, 
                 leads=None, input_len=5000, max_syn_noise=3,
                 add_noise_ratio=0.5, random_ranges=[0.2, 0.5],
                 transform=None, train_mode=True):
        self.transform = transform
        self.leads = leads
        normal_df = pd.read_csv(normal_csv)
        # Repeat normal data to balance the dataset
        self.dataframe = pd.concat([normal_df] * 6).reset_index()

        # Load paths to noise files
        self.noise_paths = pd.read_csv(noise_csv[0])['paths'].tolist()
        self.mit_paths_csvs = self._load_MIT_csv(noise_csv)

        self.max_syn_noise = max_syn_noise
        self.n_sample = n_sample
        self.input_len = input_len
        self.add_noise_ratio = add_noise_ratio
        self.random_ranges = random_ranges
        self.train_mode = train_mode

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.dataframe)

    def _load_MIT_csv(self, noise_csv):
        """
        Load paths to specific noise files from CSVs.

        Args:
            noise_csv (list of str): List of paths to noise CSV files.

        Returns:
            list: List of lists containing paths to noise files.
        """
        bw_path, em_path, ma_path = noise_csv[1], noise_csv[2], noise_csv[3]
        bw_paths = pd.read_csv(bw_path)['paths'].tolist()
        em_paths = pd.read_csv(em_path)['paths'].tolist()
        ma_paths = pd.read_csv(ma_path)['paths'].tolist()
        return [bw_paths, em_paths, ma_paths]

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Contains the input and label data.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the data file corresponding to the index
        data_file = os.path.join(self.dataframe.iloc[idx]['paths'])
        lead = random.choice(self.leads)
        data_feather = pd.read_feather(data_file)
        data_feather_i = data_feather[lead]

        # Determine the start and end points for cropping
        if self.train_mode:
            start_pt = random.randint(1, len(data_feather_i) - self.input_len - 1)
        else:
            start_pt = 0
        end_pt = start_pt + self.input_len

        input_i = data_feather_i[start_pt:end_pt]
        label_i = data_feather_i[start_pt:end_pt]
        default_input_i = data_feather[lead + '_input'][start_pt:end_pt]

        input_i = np.array(input_i).astype('float32')
        label_i = np.array(label_i).astype('float32')
        default_input_i = np.array(default_input_i).astype('float32')

        assert len(input_i) == self.input_len
        assert len(label_i) == self.input_len

        # Add noise to the input during training
        if self.train_mode:
            if random.uniform(0, 1) > self.add_noise_ratio:
                if random.uniform(0, 1) > 0.7:
                    for noise_paths in self.mit_paths_csvs:
                        noise_path_i = random.choice(noise_paths)
                        noise_feather_i = pd.read_feather(noise_path_i)[lead]

                        if len(noise_feather_i) > len(input_i):
                            noise_crop = random_crop_noise(noise_feather_i, input_len=len(input_i), flat_top_n=30)
                        elif len(noise_feather_i) == len(input_i):
                            noise_crop = noise_feather_i
                        else:
                            noise_feather_i = pd.concat([noise_feather_i] * 3).reset_index(drop=True)
                            noise_crop = random_crop_noise(noise_feather_i, input_len=len(input_i), flat_top_n=30)

                        ratio1 = random.uniform(0.2, 0.5)
                        input_i += ratio1 * noise_crop
                        assert len(input_i) == self.input_len
                else:
                    max_n = random.randint(1, self.max_syn_noise)
                    for _ in range(max_n):
                        noise_path = random.choice(self.noise_paths)
                        noise_feather_i = pd.read_feather(noise_path)[lead]

                        if len(noise_feather_i) > len(input_i):
                            noise_crop = random_crop_noise(noise_feather_i, input_len=self.input_len, flat_top_n=30)
                        elif len(noise_feather_i) == len(input_i):
                            noise_crop = noise_feather_i
                        else:
                            noise_feather_i = pd.concat([noise_feather_i] * 3).reset_index(drop=True)
                            noise_crop = random_crop_noise(noise_feather_i, input_len=self.input_len, flat_top_n=30)

                        ratio1 = random.uniform(self.random_ranges[0], self.random_ranges[1])
                        input_i += ratio1 * noise_crop
                        assert len(input_i) == self.input_len
            else:
                input_i = default_input_i

        input_i = input_i.reshape(1, input_i.shape[0])
        label_i = label_i.reshape(1, label_i.shape[0])

        sample = {'input': input_i, 'label': label_i}

        # Apply any transformations, if specified
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        input_i, label_i = sample['input'], sample['label']
        return {
            'input': torch.from_numpy(input_i),
            'label': torch.from_numpy(label_i),
        }


def get_transform(train):
    """
    Returns a composition of transformations based on training mode.

    Args:
        train (bool): Flag indicating if the transformations are for training.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    transforms = [ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

