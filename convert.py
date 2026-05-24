import os
import numpy as np
from tqdm import tqdm

def convert_to_mmap_format(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    print(f"📦 Unpacking {len(files)} files to Memory-Mappable .npy format...")
    
    for f in tqdm(files):
        npz_path = os.path.join(data_dir, f)
        npy_path = npz_path.replace('.npz', '.npy')
        
        # Only unpack if we haven't already
        if not os.path.exists(npy_path):
            data = np.load(npz_path)['frames']
            np.save(npy_path, data)
            
            # Optional: Delete the .npz file to save Colab disk space
            # os.remove(npz_path) 
            
    print("✅ Conversion complete! Data is ready for SSD Streaming.")

# Run this on your train directory
convert_to_mmap_format("expert_dataset/train")
# convert_to_mmap_format("expert_dataset/test") # Do the test dir too if needed