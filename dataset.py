import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config

class AtariDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # 1. Update filter to look for .npz files instead of .npy
        self.file_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npz')
        ]
        
        self.data_arrays = []
        self.lengths = []
        
        print(f"Loading {len(self.file_paths)} files from {data_dir} into RAM...")
        
        for path in self.file_paths:
            # 2. Extract the 'frames' array from the compressed dictionary
            data = np.load(path)
            frames_array = data['frames'] # Shape: (50000, 84, 84, 4)
            
            self.data_arrays.append(frames_array)
            self.lengths.append(len(frames_array))
            
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_length = self.cumulative_lengths[-1]
        
        print(f"Total frames available: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[file_idx - 1]
            
        # 3. Extract the 4-frame stack from memory (Shape: 84, 84, 4)
        frames = self.data_arrays[file_idx][local_idx]
        
        # 4. TRANSPOSE: Convert from (Height, Width, Channels) to (Channels, Height, Width)
        # This turns (84, 84, 4) into (4, 84, 84) for PyTorch
        frames = np.transpose(frames, (2, 0, 1))
        
        # 5. Convert to PyTorch tensor and normalize to [0.0, 1.0]
        tensor_frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        
        return tensor_frames

if __name__ == "__main__":
    print("Testing the PyTorch DataLoader...")
    
    # Initialize the dataset using the training directory from config.py
    train_dataset = AtariDataset(config.TRAIN_DIR)
    
    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=2
    )
    
    # Grab one single batch
    for batch in train_loader:
        print(f"\n📦 Batch Shape: {batch.shape}")
        print(f"   Batch Data Type: {batch.dtype}")
        print(f"   Min Pixel Value: {batch.min():.2f}")
        print(f"   Max Pixel Value: {batch.max():.2f}")
        break