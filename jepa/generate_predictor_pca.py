import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Adjust path to import local modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset_JEPA import AtariTransitionDataset
from eval_jepa_diagnostics import load_model, EnvConditioner, extract_unbroken_sequences, GAME_ACTION_COUNTS
import games as gamelib
import config

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(config.CHECKPOINT_DIR, "vjepa_v2_best.pt"))
    parser.add_argument("--data-dir", default=config.VAL_DIR)
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "charts"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device)
    dataset = AtariTransitionDataset(args.data_dir, dtype="float32")
    
    # Extract just a few sequences to keep the plot clean
    data = extract_unbroken_sequences(dataset, GAME_ACTION_COUNTS, num_sequences=5, horizon=1, seed=42)
    s_traj = data["s_trajectory"].to(device)
    games = data["games"]
    
    cond = EnvConditioner(games, "true", device).bind(model)
    e = cond.e(batch_size=len(games))
    
    # We will collect all vectors to fit a single PCA model:
    # 1. True context vectors z_t
    # 2. True target vectors z_{t+1}
    # 3. Predicted vectors \hat{z}_{t+1}(a) for all valid actions a
    
    s_t = s_traj[:, 0]
    s_next = s_traj[:, 1]
    
    z_t = model.context_encoder(s_t, e)
    z_next = model.target_encoder(s_next, e)
    
    # Flatten representations for PCA
    z_t_flat = z_t.mean(dim=1).cpu().numpy()
    z_next_flat = z_next.mean(dim=1).cpu().numpy()
    
    all_vectors = []
    all_vectors.extend(z_t_flat)
    all_vectors.extend(z_next_flat)
    
    # Store predictions grouped by state
    preds_per_state = [[] for _ in range(len(games))]
    
    for i, game in enumerate(games):
        game_id = cond.true_ids[i].item()
        # Find valid canonical actions for this game
        mask = gamelib.valid_action_mask()[game_id]
        valid_actions = np.nonzero(mask)[0]
        
        # We need the single sample e and z_t
        e_i = e[i:i+1]
        z_t_i = z_t[i:i+1]
        
        for a in valid_actions:
            a_tensor = torch.tensor([a], dtype=torch.long, device=device)
            z_pred = model.predictor(z_t_i, a_tensor, e_i)
            z_pred_flat = z_pred.mean(dim=1).cpu().numpy()[0]
            preds_per_state[i].append(z_pred_flat)
            all_vectors.append(z_pred_flat)
            
    all_vectors = np.array(all_vectors)
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    pca.fit(all_vectors)
    
    plt.figure(figsize=(10, 8))
    
    # Plotting
    colors = sns.color_palette("tab10", len(games))
    
    for i, game in enumerate(games):
        z_t_2d = pca.transform(z_t_flat[i:i+1])[0]
        z_next_2d = pca.transform(z_next_flat[i:i+1])[0]
        
        # Plot initial state (square)
        plt.scatter(z_t_2d[0], z_t_2d[1], marker='s', color=colors[i], s=100, label=f"State {i+1} ($z_t$)")
        # Plot true next state (star)
        plt.scatter(z_next_2d[0], z_next_2d[1], marker='*', color=colors[i], s=150, edgecolor='black', label=f"True Next {i+1}")
        
        # Draw dotted line from z_t to true z_next
        plt.plot([z_t_2d[0], z_next_2d[0]], [z_t_2d[1], z_next_2d[1]], color=colors[i], linestyle=':', alpha=0.6)
        
        preds_2d = pca.transform(preds_per_state[i])
        # Plot predictions for all actions (small circles)
        plt.scatter(preds_2d[:, 0], preds_2d[:, 1], marker='o', color=colors[i], s=30, alpha=0.8)
        
        # Draw solid lines from z_t to predictions
        for j in range(preds_2d.shape[0]):
            plt.plot([z_t_2d[0], preds_2d[j, 0]], [z_t_2d[1], preds_2d[j, 1]], color=colors[i], alpha=0.3)
            
    plt.title("PCA of Predictor Outputs Across All Actions")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(args.out_dir, "pca_action_sensitivity.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved PCA chart to {plot_path}")

if __name__ == "__main__":
    main()
