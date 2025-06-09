import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class EmbrapaDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        """
        Args:
            csv_path (str): Đường dẫn đến CSV chứa image_path, label, label_idx
            root_dir (str): Thư mục gốc chứa các ảnh (thường là ./data/raw/embrapa/)
            transform (callable, optional): Transform áp dụng lên ảnh
        """

        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return image, label
