import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config

class AtariDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # 1. Look for the unpacked .npy files
        self.file_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npy')
        ]
        
        self.data_maps = []
        self.lengths = []
        
        print(f"🔌 Memory-Mapping {len(self.file_paths)} files directly from SSD...")
        
        for path in self.file_paths:
            # 2. MAGIC LINE: mmap_mode='r' leaves data on the disk!
            mmap_data = np.load(path, mmap_mode='r')
            self.data_maps.append(mmap_data)
            self.lengths.append(len(mmap_data))
            
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_length = self.cumulative_lengths[-1]
        
        print(f"Total frames available on disk: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[file_idx - 1]
            
        # 3. Pull EXACTLY ONE frame stack from the SSD (Zero RAM bloat)
        frames = self.data_maps[file_idx][local_idx]
        
        # 4. Transpose from (84, 84, 4) to (4, 84, 84)
        frames = np.transpose(frames, (2, 0, 1))
        
        # 5. Convert to Tensor
        tensor_frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        
        return tensor_frames