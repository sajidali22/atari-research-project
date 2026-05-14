import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config

class AtariDataset(Dataset):
    """
    A memory-efficient PyTorch Dataset for loading chunked Atari frames.
    Uses memory mapping to avoid loading massive .npy files into RAM.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # 1. Find all .npy files in the specified directory
        self.file_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npy')
        ]
        
        self.data_maps = []
        self.lengths = []
        
        print(f"📂 Mapping {len(self.file_paths)} files from {data_dir}...")
        
        # 2. Memory-map each file to keep RAM usage minimal
        for path in self.file_paths:
            mmap_data = np.load(path, mmap_mode='r')
            self.data_maps.append(mmap_data)
            self.lengths.append(len(mmap_data))
            
        # 3. Calculate cumulative lengths so we know which file an index belongs to
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_length = self.cumulative_lengths[-1]
        
        print(f"✅ Dataset ready! Total frames available: {self.total_length}")

    def __len__(self):
        """Returns the total number of frames across all files."""
        return self.total_length

    def __getitem__(self, idx):
        """Fetches a single 4-frame stack and prepares it for the neural network."""
        
        # 1. Find which file this index belongs to
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        # 2. Calculate the local index within that specific file
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[file_idx - 1]
            
        # 3. Extract the 4-frame stack from the disk
        frames = self.data_maps[file_idx][local_idx]
        
        # 4. Convert to PyTorch tensor and normalize to [0.0, 1.0]
        # Neural networks expect data in float32 format
        tensor_frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        
        return tensor_frames

if __name__ == "__main__":
    # --- Quick Test to Ensure Everything Works ---
    print("Testing the PyTorch DataLoader...")
    
    # Initialize the dataset using the training directory from config.py
    train_dataset = AtariDataset(config.TRAIN_DIR)
    
    # Create the DataLoader
    # batch_size=64 means we feed 64 stacked frames to the network at once
    # shuffle=True ensures the network doesn't memorize the sequence of a single game
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=2 # Uses multiple CPU cores to load data faster
    )
    
    # Grab one single batch to inspect it
    for batch in train_loader:
        print(f"\n📦 Batch Shape: {batch.shape}")
        print(f"   Batch Data Type: {batch.dtype}")
        print(f"   Min Pixel Value: {batch.min():.2f}")
        print(f"   Max Pixel Value: {batch.max():.2f}")
        break # We just want to check the first batch and stop