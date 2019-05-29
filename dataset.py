import os

from PIL import Image
import pandas as pd

from torch.utils.data import Dataset

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
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.filename[idx])
        image = Image.open(img_name)
        class_id = self.df.class_id[idx]
        item = {'image': image, 'class_id': class_id}

        if self.transform:
            item = self.transform(item)

        return item