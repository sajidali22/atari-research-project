import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 10, 'figure.figsize': (20, 15)})

# Standard Published Baselines (Random, Human, and Standard NatureCNN PPO)
BASELINES = {
    "Alien": {"Random": 227.8, "Human": 7127.7, "PPO": 1850.0},
    "Asteroids": {"Random": 719.1, "Human": 43189.0, "PPO": 2000.0},
    "Atlantis": {"Random": 12850.0, "Human": 29028.0, "PPO": 2300000.0},
    "BeamRider": {"Random": 363.9, "Human": 3140.0, "PPO": 2800.0},
    "Breakout": {"Random": 1.7, "Human": 30.5, "PPO": 400.0},
    "DemonAttack": {"Random": 152.1, "Human": 1971.0, "PPO": 100000.0},
    "Enduro": {"Random": 0.0, "Human": 860.5, "PPO": 750.0},
    "IceHockey": {"Random": -11.2, "Human": 0.9, "PPO": -2.0},
    "MsPacman": {"Random": 307.3, "Human": 6951.6, "PPO": 2000.0},
    "Pong": {"Random": -20.7, "Human": 14.6, "PPO": 20.9},
    "Qbert": {"Random": 163.9, "Human": 13455.0, "PPO": 14000.0},
    "Riverraid": {"Random": 1338.5, "Human": 17118.0, "PPO": 8000.0},
    "RoadRunner": {"Random": 11.5, "Human": 7845.0, "PPO": 30000.0},
    "Seaquest": {"Random": 68.4, "Human": 42054.7, "PPO": 1200.0},
    "SpaceInvaders": {"Random": 148.0, "Human": 1668.7, "PPO": 900.0}
}

def plot_comparison():
    files = glob.glob(os.path.join(HERE, "csv_data_joint", "*.csv"))
    
    fig, axes = plt.subplots(nrows=3, ncols=5, constrained_layout=True)
    axes = axes.flatten()

    for idx, file in enumerate(files):
        # Extract game name from your W&B run name
        filename = os.path.basename(file)
        game_name = filename.split("PPO_FSQ-VAE_JOINT_")[1].split("NoFrameskip")[0]
        
        df = pd.read_csv(file)
        
        df = df.sort_values(by="global_step").reset_index(drop=True)
        
        # Apply a rolling average to smooth the spiky RL data
        df['smoothed_return'] = df['charts/episodic_return'].rolling(window=50, min_periods=1).mean()
        
        ax = axes[idx]
        
        # Plot your FSQ-VAE Agent
        ax.plot(df['global_step'], df['smoothed_return'], color='#FF1493', linewidth=2, label="FSQ-VAE-JOINT (Yours)")
        
        # Plot standard baselines if we have them for this game
        if game_name in BASELINES:
            ax.axhline(y=BASELINES[game_name]["PPO"], color='blue', linestyle='--', linewidth=1.5, label="Standard PPO")
            ax.axhline(y=BASELINES[game_name]["Human"], color='green', linestyle=':', linewidth=1.5, label="Human")
            ax.axhline(y=BASELINES[game_name]["Random"], color='gray', linestyle='-.', linewidth=1.5, label="Random")
        
        ax.set_title(game_name, fontweight='bold')
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Score')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x*1e-6:g}M'))
        
        # Only add a legend to the first plot to avoid clutter
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Atari Performance: JOINT FSQ-VAE vs Standard Baselines", fontsize=20, fontweight='bold')
    plt.savefig(os.path.join(HERE, "thesis_comparison_grid_JOINT.png"), dpi=300)
    print("Saved high-resolution plot to thesis_comparison_grid_JOINT.png!")

if __name__ == "__main__":
    plot_comparison()