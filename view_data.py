import numpy as np
import matplotlib.pyplot as plt
import os
import random
import config

def visualize_random_samples(game_name, dataset_type="train", num_samples=3):
    """
    Loads a .npy file and displays random 4-frame stacks.
    """
    folder = config.TRAIN_DIR if dataset_type == "train" else config.TEST_DIR
    
    # (e.g., Breakout_part1.npy)
    file_path = os.path.join(folder, f"{game_name}_part1.npy")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find file at {file_path}")
        return

    print(f"📂 Loading {file_path}...")
    data = np.load(file_path)
    print(f"✅ Data Loaded. Shape: {data.shape}") # Expecting (N, 4, 84, 84)

    for s in range(num_samples):
        idx = random.randint(0, len(data) - 1)
        sample = data[idx] # This is a (4, 84, 84) array
        
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        fig.suptitle(f"Game: {game_name} | Observation Index: {idx}", fontsize=14)
        
        for i in range(4):
            axes[i].imshow(sample[i], cmap='gray')
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_random_samples(game_name="Riverraid", dataset_type="train")