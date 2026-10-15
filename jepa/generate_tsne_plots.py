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

from dataset_JEPA import AtariTransitionDataset
from eval_jepa_diagnostics import load_model, EnvConditioner, extract_unbroken_sequences, GAME_ACTION_COUNTS
import games as gamelib
import config

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate t-SNE plots for JEPA model latents")
    parser.add_argument("--checkpoint", default=os.path.join(config.CHECKPOINT_DIR, "vjepa_v2_best.pt"))
    parser.add_argument("--data-dir", default=config.VAL_DIR)
    parser.add_argument("--num-sequences", type=int, default=128, help="Number of sequences to extract (increases points)")
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "charts"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = AtariTransitionDataset(args.data_dir, dtype="float32")
    action_counts = GAME_ACTION_COUNTS

    # Extract sequences
    # We'll just grab the first state (s_t) and action (a_t) from each sequence to avoid highly correlated adjacent frames.
    data = extract_unbroken_sequences(dataset, action_counts, num_sequences=args.num_sequences, horizon=5, seed=42)
    s_traj = data["s_trajectory"].to(device) # [B, horizon+1, 4, 84, 84]
    a_traj = data["a_trajectory"].to(device)
    games = data["games"]

    # We only care about the initial state for clustering
    s_t = s_traj[:, 0]
    a_t = a_traj[:, 0]

    # Get environment embeddings using EnvConditioner (true env mode)
    cond = EnvConditioner(games, "true", device).bind(model)
    e = cond.e(batch_size=len(games))

    # Compute latents
    print("Computing latent representations...")
    # context_encoder outputs [B, num_patches, embed_dim]
    z_t = model.context_encoder(s_t, e)
    
    # Pool patch embeddings to get a single vector per image. Mean pooling is standard.
    z_pooled = z_t.mean(dim=1).cpu().numpy()
    
    # Move other metadata to numpy for plotting
    a_np = a_t.cpu().numpy()
    games_np = np.array(games)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_pooled)-1))
    z_2d = tsne.fit_transform(z_pooled)

    print("Generating plots...")
    # Plot 1: Colored by Game
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=games_np, palette="tab20", s=60, alpha=0.8)
    plt.title("JEPA Latent Space t-SNE (Colored by Game)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot1_path = os.path.join(args.out_dir, "tsne_by_game.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"Saved {plot1_path}")

    # Plot 2: Colored by Canonical Action
    plt.figure(figsize=(10, 8))
    # Actions are discrete, convert to strings for categorical colors
    actions_str = [f"Action {a}" for a in a_np]
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=actions_str, palette="Set1", s=60, alpha=0.8)
    plt.title("JEPA Latent Space t-SNE (Colored by First Action)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot2_path = os.path.join(args.out_dir, "tsne_by_action.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved {plot2_path}")

    print("Done!")

if __name__ == "__main__":
    main()
