import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import config

class AtariTransitionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # 1. Identify source archives
        self.file_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npz')
        ]
        
        # Create a hidden cache directory to hold the mmap-friendly binary files
        self.cache_dir = os.path.join(data_dir, ".npy_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.obs_arrays = []
        self.action_arrays = []
        self.next_obs_arrays = []
        self.terminal_arrays = []
        self.lengths = []
        
        print(f"Verifying memory-mapped cache for {len(self.file_paths)} files...")
        
        for path in self.file_paths:
            base_name = os.path.basename(path).replace('.npz', '')
            
            # Define exact paths for the extracted .npy components
            obs_path = os.path.join(self.cache_dir, f"{base_name}_obs.npy")
            act_path = os.path.join(self.cache_dir, f"{base_name}_actions.npy")
            next_obs_path = os.path.join(self.cache_dir, f"{base_name}_next_obs.npy")
            term_path = os.path.join(self.cache_dir, f"{base_name}_terminals.npy")
            
            # If the raw binary files don't exist yet, we must extract them once
            if not all(os.path.exists(p) for p in [obs_path, act_path, next_obs_path, term_path]):
                print(f"📦 First-time setup: Unpacking {base_name}.npz into raw binary cache...")
                # Load fully into RAM just this once to extract
                data = np.load(path) 
                np.save(obs_path, data['obs'])
                np.save(act_path, data['actions'])
                np.save(next_obs_path, data['next_obs'])
                np.save(term_path, data['terminals'])
                del data # Instantly free RAM
            
            # --- TRUE MEMORY MAPPING ---
            # Now that they are standard .npy files, mmap works flawlessly!
            # The OS handles streaming this from the SSD into the GPU with zero RAM overhead.
            obs_arr = np.load(obs_path, mmap_mode='r')
            action_arr = np.load(act_path, mmap_mode='r')
            next_obs_arr = np.load(next_obs_path, mmap_mode='r')
            term_arr = np.load(term_path, mmap_mode='r')
            
            self.obs_arrays.append(obs_arr)
            self.action_arrays.append(action_arr)
            self.next_obs_arrays.append(next_obs_arr)
            self.terminal_arrays.append(term_arr)
            
            self.lengths.append(len(obs_arr))
            
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_length = self.cumulative_lengths[-1]
        
        print(f"✅ Cache verified. Total transition steps mapped: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[file_idx - 1]
            
        # --- THE IPC LEAK FIX (.copy) ---
        # When slicing a memory-mapped array, NumPy returns a "view" tied to the whole file.
        # Calling .copy() forces Python to allocate a tiny standalone block of memory just 
        # for this specific (84, 84, 4) frame, severing the ghost connection to the massive file.
        raw_obs = self.obs_arrays[file_idx][local_idx].copy()
        raw_action = self.action_arrays[file_idx][local_idx].copy()
        raw_next_obs = self.next_obs_arrays[file_idx][local_idx].copy()
        is_terminal = self.terminal_arrays[file_idx][local_idx].copy()
        
        # Transpose from HWC to CHW
        obs_transposed = np.transpose(raw_obs, (2, 0, 1))
        next_obs_transposed = np.transpose(raw_next_obs, (2, 0, 1))
        
        # Convert to Tensors
        s_t = torch.tensor(obs_transposed, dtype=torch.float32) / 255.0
        s_next = torch.tensor(next_obs_transposed, dtype=torch.float32) / 255.0
        a_t = torch.tensor(raw_action, dtype=torch.long)
        mask = torch.tensor(0.0 if is_terminal else 1.0, dtype=torch.float32)
        
        return {
            "s_t": s_t,
            "a_t": a_t,
            "s_next": s_next,
            "mask": mask
        }