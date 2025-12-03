# STANDARD LIBRARIES
import os

# IMAGE PROCESSING
from PIL import Image


# NUMPY (needed only if seed_worker uses it)
import numpy as np

# PYTORCH
import torch
from torch.utils.data import Dataset, DataLoader

# CUSTOM UTILITIES
from optimization.utils.reproducibility import seed_worker
from optimization.utils.extract_patient_id import extract_patient_id


# Dataset para classificação multi-target
class LungCancerMultiTargetDataset(Dataset):
    def __init__(self, file_paths, df, transform=None):
        self.file_paths = file_paths
        self.transform = transform

        self.df = df.set_index('PatientID')  # Assume que PatientID é única

        self.valid_paths = []
        self.image_patient_map = {}

        for path in self.file_paths:
            filename = os.path.basename(path)
            patient_id = extract_patient_id(filename)
            if patient_id and patient_id in self.df.index:
                self.valid_paths.append(path)
                self.image_patient_map[path] = patient_id

    def __len__(self):
        return len(self.valid_paths)

    def __getitem__(self, idx):
        file_path = self.valid_paths[idx]
        image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        patient_id = self.image_patient_map[file_path]
        row = self.df.loc[patient_id]

        t = int(float(row['T-Stage']))
        n = int(float(row['N-Stage']))
        m = int(float(row['M-Stage']))

        targets = {
            'T': torch.tensor(t, dtype=torch.long),
            'N': torch.tensor(n, dtype=torch.long),
            'M': torch.tensor(m, dtype=torch.long)
        }

        # Metadados numéricos
        gender = row['gender']
        age = row['age']
        weight = row['weight (kg)']
        smoking = row['Smoking History']
        cancer = row['Histology']

        metadata = torch.tensor([gender, age, weight, smoking, cancer], dtype=torch.float)

        return image, metadata, targets
    
def create_dataloader(dataset, batch_size, num_workers=0, worker_init_fn=seed_worker):
    generator = torch.Generator()
    generator.manual_seed(42)  # Seed the generator
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker
    )