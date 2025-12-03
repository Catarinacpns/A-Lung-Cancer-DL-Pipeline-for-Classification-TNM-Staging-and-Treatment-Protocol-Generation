import os
from PIL import Image
from torch.utils.data import Dataset


# Dataset Class
class LungCancerDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.class_map = {'A': 0, 'B': 1, 'E': 2, 'G': 3}
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.class_map[os.path.basename(file_path)[0]]  # First letter determines the label
        image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label