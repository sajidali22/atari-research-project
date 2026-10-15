import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Adjust path to import local modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR)) # For 'models'

from dataset_JEPA import AtariTransitionDataset
from eval_jepa_diagnostics import extract_unbroken_sequences, GAME_ACTION_COUNTS
import games as gamelib
import config

from models.fsq_vae_decoder import AtariFSQVAE

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(os.path.dirname(SCRIPT_DIR), "artifacts", "checkpoints", "autoencoders", "fsq_vae_decoder", "fsq_vae_epoch_125.pth"))
    parser.add_argument("--data-dir", default=config.VAL_DIR)
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "charts"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    print(f"Loading VAE from {args.checkpoint}...")
    model = AtariFSQVAE().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = AtariTransitionDataset(args.data_dir, dtype="float32")
    
    # Extract sequences
    data = extract_unbroken_sequences(dataset, GAME_ACTION_COUNTS, num_sequences=args.num_sequences, horizon=5, seed=42)
    s_traj = data["s_trajectory"].to(device) # [B, horizon+1, 4, 84, 84]
    a_traj = data["a_trajectory"].to(device)
    games = data["games"]

    # We only care about the initial state
    s_t = s_traj[:, 0]
    a_t = a_traj[:, 0]

    # Compute latents
    print("Computing latent representations...")
    # The FSQ VAE encodes x -> z_e, then quantizes to z_q.
    # We'll map to z_q to see the actual representations the decoder/policy uses.
    z_e = model.encoder(s_t)
    z_e = model.pre_quant_proj(z_e)
    z_q, _ = model.fsq(z_e)
    
    # Pool/flatten the latent maps (z_q is likely [B, C, H_small, W_small])
    # Mean pooling across spatial dimensions is a good summary of the state representation
    z_pooled = z_q.mean(dim=[2, 3]).cpu().numpy()
    
    a_np = a_t.cpu().numpy()
    games_np = np.array(games)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_pooled)-1))
    z_2d = tsne.fit_transform(z_pooled)

    print("Generating plots...")
    # Plot 1: Colored by Game
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=games_np, palette="tab20", s=60, alpha=0.8)
    plt.title("FSQ VAE Latent Space t-SNE (Colored by Game)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot1_path = os.path.join(args.out_dir, "vae_tsne_by_game.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"Saved {plot1_path}")

    # Plot 2: Colored by Canonical Action
    plt.figure(figsize=(10, 8))
    actions_str = [f"Action {a}" for a in a_np]
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=actions_str, palette="Set1", s=60, alpha=0.8)
    plt.title("FSQ VAE Latent Space t-SNE (Colored by First Action)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot2_path = os.path.join(args.out_dir, "vae_tsne_by_action.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved {plot2_path}")

    print("Done!")

if __name__ == "__main__":
    main()
