import pandas as pd
from pathlib import Path
import json
import numpy as np
from typing import Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


DESCRIPTOR_NAME_PATH = Path(__file__).with_name('descriptor_name.json')
with DESCRIPTOR_NAME_PATH.open() as f:
    DESCRIPTOR_NAME = json.load(f)


@dataclass
class CSVDataset(Dataset):
    direc: str | Path
    split: str
    task: Optional[str] = None
    target: Optional[str] = "target"
        
    def __post_init__(self):
        self.direc = Path(self.direc).resolve()
        if self.task:
            self.direc = self.direc / f"{self.split}_{self.task}.csv"
        else:
            self.direc = self.direc / f"{self.split}.csv"

        if not self.direc.exists():
            raise ValueError(f'{self.direc} does not exists.')
        
        self.data = pd.read_csv(self.direc)
        
        # Name of descriptors
        self.descriptors = DESCRIPTOR_NAME

    def __getitem__(self, idx):
        loc = self.data.iloc[idx]
        x = torch.tensor([loc[key] for key in self.descriptors])
        
        if self.target is None:
            y = 0.
        else:
            y = float(loc[self.target])
        
        return [x, y]
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def collate(batch):
        x_batch, y_batch = zip(*batch)
        x_batch = torch.vstack(x_batch)
        y_batch = torch.tensor(y_batch).unsqueeze(1)
        return x_batch.to(torch.float32), y_batch.to(torch.float32)

    def get_mean_and_std(self):
        batch_mean = [float(self.data[key].mean()) for key in self.descriptors]
        batch_std = [float(self.data[key].std()) for key in self.descriptors]

        if self.target is None:
            target_mean = 0
            target_std = 1
        else:
            target_mean = float(self.data[self.target].mean())
            target_std = float(self.data[self.target].std())
        
        return {
            'mean': batch_mean, 
            'std': batch_std, 
            'target_mean': target_mean, 
            'target_std': target_std,
        }

    def get_min_and_max(self):
        max = self.data.max()
        min = self.data.min()

        batch_min = [float(min[key]) for key in self.descriptors]
        batch_max = [float(max[key]) for key in self.descriptors]

        if self.target is None:
            target_min = 0.
            target_max = 1.
        else:
            target_min = float(min[self.target])
            target_max = float(max[self.target])

        return {
            'min': batch_min,
            'max': batch_max,
            'target_min': target_min,
            'target_max': target_max,
        }

@dataclass
class TextSplitDataset(Dataset):
    direc: str | Path
    split: str
    task: Optional[str] = None
    target: Optional[str] = "target"
    total_csv: str = 'total.csv'
        
    def __post_init__(self):
        self.direc = Path(self.direc).resolve()
        total_csv_dir = self.direc / self.total_csv
        if self.task:
            txt_dir = self.direc / f"{self.split}_{self.task}.txt"
        else:
            txt_dir = self.direc / f"{self.split}.txt"
        
        if not total_csv_dir.exists():
            raise ValueError(f'{total_csv_dir} does not exists.')
        if not txt_dir.exists():
            raise ValueError(f'{txt_dir} does not exists.')
        
        self.data = pd.read_csv(total_csv_dir)

        with txt_dir.open() as f:
            self.structures = [line.strip() for line in f]
        
        # Name of descriptors
        self.descriptors = DESCRIPTOR_NAME

    def __getitem__(self, idx):
        # Find the row where the filename column matches the structure name
        loc = self.data[self.data['filename'] == self.structures[idx]]
        if loc.empty:
            raise ValueError(f"No row found for structure {self.structures[idx]} in column 'filename'")
        loc = loc.iloc[0]
        x = torch.tensor([loc[key] for key in self.descriptors])

        if self.target is None:
            y = 0.
        else:
            y = float(loc[self.target])
        return [x, y]
    
    def __len__(self):
        return len(self.structures)
    
    @staticmethod
    def collate(batch):
        x_batch, y_batch = zip(*batch)
        x_batch = torch.vstack(x_batch)
        y_batch = torch.tensor(y_batch).unsqueeze(1)
        return x_batch.to(torch.float32), y_batch.to(torch.float32)

    def get_mean_and_std(self):
        batch_mean = [float(self.data[key].mean()) for key in self.descriptors]
        batch_std = [float(self.data[key].std()) for key in self.descriptors]

        if self.target is None:
            target_mean = 0
            target_std = 1
        else:
            target_mean = float(self.data[self.target].mean())
            target_std = float(self.data[self.target].std())
        
        return {
            'mean': batch_mean, 
            'std': batch_std, 
            'target_mean': target_mean, 
            'target_std': target_std,
        }

    def get_min_and_max(self):
        max = self.data.max()
        min = self.data.min()

        batch_min = [float(min[key]) for key in self.descriptors]
        batch_max = [float(max[key]) for key in self.descriptors]

        if self.target is None:
            target_min = 0.
            target_max = 1.
        else:
            target_min = float(min[self.target])
            target_max = float(max[self.target])

        return {
            'min': batch_min,
            'max': batch_max,
            'target_min': target_min,
            'target_max': target_max,
        }
    
@dataclass
class JsonSplitDataset(Dataset):
    direc: str | Path
    split: str
    task: Optional[str] = None
    target: Optional[str] = None
    total_csv: str = 'total.csv'
    filename: str = 'filename'
        
    def __post_init__(self):
        self.direc = Path(self.direc).resolve()
        total_csv_dir = self.direc / self.total_csv
        if self.task:
            json_dir = self.direc / f"{self.split}_{self.task}.json"
        else:
            json_dir = self.direc / f"{self.split}.json"
        
        if not total_csv_dir.exists():
            raise ValueError(f'{total_csv_dir} does not exists.')
        if not json_dir.exists():
            raise ValueError(f'{json_dir} does not exists.')
        
        self.data = pd.read_csv(total_csv_dir)

        with json_dir.open() as f:
            target_and_structure = json.load(f)
            target_and_structure = {k:v for k,v in target_and_structure.items() if k in self.data[self.filename].values}
            self.structures = list(target_and_structure.keys())
            self.targets = np.array(list(target_and_structure.values()))
        
        # Name of descriptors
        self.descriptors = DESCRIPTOR_NAME

    def __getitem__(self, idx):
        # Find the row where the filename column matches the structure name
        loc = self.data[self.data[self.filename] == self.structures[idx]]
        if loc.empty:
            raise ValueError(f"No row found for structure {self.structures[idx]} in column 'filename'")
        loc = loc.iloc[0]  # Get the first matching row as a Series
        x = torch.tensor([loc[key] for key in self.descriptors])
        
        if self.target is None:
            y = 0.
        else:
            y = float(self.targets[idx])
        
        return [x, y]
    
    def __len__(self):
        return len(self.structures)
    
    @staticmethod
    def collate(batch):
        x_batch, y_batch = zip(*batch)
        x_batch = torch.vstack(x_batch)
        y_batch = torch.tensor(y_batch).unsqueeze(1)
        return x_batch.to(torch.float32), y_batch.to(torch.float32)

    def get_mean_and_std(self):
        batch_mean = [float(self.data[key].mean()) for key in self.descriptors]
        batch_std = [float(self.data[key].std()) for key in self.descriptors]

        if self.target is None:
            target_mean = 0
            target_std = 1
        else:
            target_mean = float(self.targets.mean())
            target_std = float(self.targets.std())
        
        return {
            'mean': batch_mean, 
            'std': batch_std, 
            'target_mean': target_mean, 
            'target_std': target_std,
        }

    def get_min_and_max(self):
        max = self.data.max()
        min = self.data.min()

        batch_min = [float(min[key]) for key in self.descriptors]
        batch_max = [float(max[key]) for key in self.descriptors]

        if self.target is None:
            target_min = 0.
            target_max = 1.
        else:
            target_min = float(min[self.target])
            target_max = float(max[self.target])

        return {
            'min': batch_min,
            'max': batch_max,
            'target_min': target_min,
            'target_max': target_max,
        }