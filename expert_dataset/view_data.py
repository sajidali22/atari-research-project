import numpy as np
import matplotlib.pyplot as plt
import os

def inspect_and_visualize(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    data = np.load(file_path)
    frames = data['frames']
    
    # 1. Inspect and PRINT the shape
    print(f"--- Dataset shape: {frames.shape} ---")
    
    sample_stack = frames[4900] # Shape is (4, 84, 84)
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    
    for i in range(4):
        # Access: All rows, all columns, specifically the i-th channel
        img_data = sample_stack[:, :, i]
        
        # Now it will correctly display an 84x84 image
        axes[i].imshow(img_data, cmap='gray')
        axes[i].set_title(f"Time Step {i+1}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Point this to one of your collected files
    # Example: custom_datasets/train/PongNoFrameskip-v4_expert_50000_frames.npz
    target_file = "train/RoadRunnerNoFrameskip-v4_expert_50000_frames.npz"
    
    inspect_and_visualize(target_file)