import os

from PIL import Image
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Sampler


class StanfordCarsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a item.
        """
        self.df = pd.read_csv(csv_file)
        self.df.class_id -= 1 # make index start with 0
        self.num_classes = len(self.df.class_id.unique())
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.filename[idx])
        image = Image.open(img_name)
        image = image.convert("RGB")
        class_id = self.df.class_id[idx]

        if self.transform:
            image = self.transform(image)

        return image, class_id


class IndexSampler(Sampler):
    """
    Args:
        dataset (Dataset): A dataset to sample from
        data_idx (array): Indexes that correspond to dataset class
        shuffle (boolean): Shuffle or not
    """
    def __init__(self, dataset, data_idx, shuffle=True):
        self.dataset = dataset
        self.data_idx = data_idx
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            rand_idx = np.random.permutation(self.data_idx)
            for i in rand_idx:
                yield i
        else:
            for i in self.data_idx:
                yield i

    def __len__(self):
        return len(self.data_idx)


def split_by_rand_pct_idx(df, valid_pct=0.2):
    "Split dataset/df randomly by return train/valid indexes"
    split = int(len(df) * valid_pct)
    rand_idx = np.random.permutation(list(range(len(df))))
    train = rand_idx[:-split]
    test = rand_idx[-split:]
    return train, test
