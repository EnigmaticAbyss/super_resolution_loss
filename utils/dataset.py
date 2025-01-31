import os
from torch.utils.data import Dataset
from PIL import Image

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None, hr_size=(256, 256), lr_size=(64, 64)):
        """
        Parameters:
        - hr_dir: Path to high-resolution images
        - lr_dir: Path to low-resolution images
        - transform: PyTorch transformations
        - hr_size: Target size for high-resolution images (width, height)
        - lr_size: Target size for low-resolution images (width, height)
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.hr_images = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.lr_images = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png') or f.endswith('.jpg')])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])

        # Open images
        hr_image = Image.open(hr_image_path).convert("RGB")
        lr_image = Image.open(lr_image_path).convert("RGB")

        # Resize images to a fixed size
        hr_image = hr_image.resize(self.hr_size, Image.BICUBIC)
        lr_image = lr_image.resize(self.lr_size, Image.BICUBIC)

        # Apply transformations if provided
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image
