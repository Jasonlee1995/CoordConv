import torch, csv
import numpy as np

from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, csv_dir, canvas_size, transform=None):
        self.csv_dir = csv_dir
        self.transform = transform
        self.imgs_dir, self.coords = [], []
        with open(csv_dir, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                img_dir, x, y = row
                x, y = int(x), int(y)
                self.imgs_dir.append(img_dir)
                self.coords.append([x/canvas_size, y/canvas_size])

    def __getitem__(self, idx):
        img_dir, coord = self.imgs_dir[idx], self.coords[idx]
        img = np.load(img_dir)
        if self.transform: img = self.transform(img)
        
        return img, torch.tensor(coord)

    def __len__(self):
        return len(self.imgs_dir)