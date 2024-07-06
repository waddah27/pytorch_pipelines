import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

class MRI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.train = train
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.all_images_pths = []
        self.images_pths = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            img_dirs = os.listdir(class_dir)
            for img_pth in img_dirs:
                self.all_images_pths.append((os.path.join(class_dir, img_pth), self.class_to_idx[cls]))

        # Split the dataset into train and test sets
        train_images_pths, test_images_pths = train_test_split(self.all_images_pths, test_size=0.2, random_state=42)
        if self.train:
            self.images_pths = train_images_pths
        else:
            self.images_pths = test_images_pths

    def __len__(self):
        return len(self.images_pths)

    def __getitem__(self, idx):
        img_path, label = self.images_pths[idx]
        img = Image.open(img_path) # (C, H, W) tensor
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label