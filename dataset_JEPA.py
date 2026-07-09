import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config  # Assumes config.TRAIN_DIR points to your custom_datasets/train folder

class AtariTransitionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # 1. Gather all compressed game archives
        self.file_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npz')
        ]
        
        # Lists to hold arrays from each file
        self.obs_arrays = []
        self.action_arrays = []
        self.next_obs_arrays = []
        self.terminal_arrays = []
        self.lengths = []
        
        print(f"Loading {len(self.file_paths)} files from {data_dir} into RAM...")
        
        for path in self.file_paths:
            data = np.load(path)
            
            # Verify keys match our verified data structure
            obs_arr = data['obs']          # Shape: (N, 84, 84, 4)
            action_arr = data['actions']    # Shape: (N,)
            next_obs_arr = data['next_obs']# Shape: (N, 84, 84, 4)
            term_arr = data['terminals']    # Shape: (N,)
            
            self.obs_arrays.append(obs_arr)
            self.action_arrays.append(action_arr)
            self.next_obs_arrays.append(next_obs_arr)
            self.terminal_arrays.append(term_arr)
            
            self.lengths.append(len(obs_arr))
            
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_length = self.cumulative_lengths[-1]
        
        print(f"Total transition steps available across all games: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which file this absolute index belongs to
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[file_idx - 1]
            
        # 2. Extract raw transition components from target file arrays
        raw_obs = self.obs_arrays[file_idx][local_idx]            # (84, 84, 4)
        raw_action = self.action_arrays[file_idx][local_idx]      # Scalar integer
        raw_next_obs = self.next_obs_arrays[file_idx][local_idx]  # (84, 84, 4)
        is_terminal = self.terminal_arrays[file_idx][local_idx]   # Boolean scalar
        
        # 3. Transpose both history and target stacks from HWC to CHW format
        obs_transposed = np.transpose(raw_obs, (2, 0, 1))          # (4, 84, 84)
        next_obs_transposed = np.transpose(raw_next_obs, (2, 0, 1)) # (4, 84, 84)
        
        # 4. Convert arrays to PyTorch Tensors and normalize pixels to [0.0, 1.0]
        s_t = torch.tensor(obs_transposed, dtype=torch.float32) / 255.0
        s_next = torch.tensor(next_obs_transposed, dtype=torch.float32) / 255.0
        
        a_t = torch.tensor(raw_action, dtype=torch.long)
        
        # Create a terminal mask: 0.0 at episode termination, otherwise 1.0
        # This prevents the predictor from penalizing predictions beyond game boundaries
        mask = torch.tensor(0.0 if is_terminal else 1.0, dtype=torch.float32)
        
        return {
            "s_t": s_t,        # Shape: [4, 84, 84]
            "a_t": a_t,        # Shape: []
            "s_next": s_next,  # Shape: [4, 84, 84]
            "mask": mask       # Shape: []
        }

if __name__ == "__main__":
    print("Testing the Refactored PyTorch Transition DataLoader...")
    
    # Simple execution test assuming config structure is valid
    try:
        train_dataset = AtariTransitionDataset(config.TRAIN_DIR)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=64, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        
        # Test extraction of a single batch
        for batch in train_loader:
            print(f"\n📦 Batch Extraction Successful:")
            print(f"   ► Context States (s_t)  : Shape {batch['s_t'].shape} | Type {batch['s_t'].dtype}")
            print(f"   ► Actions Executed (a_t): Shape {batch['a_t'].shape} | Type {batch['a_t'].dtype}")
            print(f"   ► Target States (s_next): Shape {batch['s_next'].shape} | Type {batch['s_next'].dtype}")
            print(f"   ► Terminal Masks (mask) : Shape {batch['mask'].shape} | Type {batch['mask'].dtype}")
            
            # Bound assertions
            print(f"\n🔍 Value Checkbounds:")
            print(f"   ► s_t pixel range       : [{batch['s_t'].min().item():.2f}, {batch['s_t'].max().item():.2f}]")
            print(f"   ► Encountered Actions   : {torch.unique(batch['a_t']).tolist()}")
            print(f"   ► Alive Steps in Batch  : {int(batch['mask'].sum().item())} / {len(batch['mask'])}")
            break
            
    except Exception as e:
        print(f"❌ Execution failure during dataset instantiation: {e}")