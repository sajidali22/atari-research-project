import gymnasium as gym
import ale_py
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from huggingface_sb3 import load_from_hub
import numpy as np
import os
from tqdm import tqdm

# Register environments
gym.register_envs(ale_py)

TRAIN_GAMES = {
    "BeamRiderNoFrameskip-v4":   ("sb3/ppo-BeamRiderNoFrameskip-v4", "ppo-BeamRiderNoFrameskip-v4.zip", PPO),
    "BreakoutNoFrameskip-v4":    ("sb3/ppo-BreakoutNoFrameskip-v4", "ppo-BreakoutNoFrameskip-v4.zip", PPO),
    "DemonAttackNoFrameskip-v4": ("julgom/dqn-DemonAttackNoFrameskip-v4", "dqn-DemonAttackNoFrameskip-v4.zip", DQN),
    "EnduroNoFrameskip-v4":      ("sb3/ppo-EnduroNoFrameskip-v4", "ppo-EnduroNoFrameskip-v4.zip", PPO),
    "MsPacmanNoFrameskip-v4":    ("sb3/ppo-MsPacmanNoFrameskip-v4", "ppo-MsPacmanNoFrameskip-v4.zip", PPO),
    "PongNoFrameskip-v4":        ("sb3/ppo-PongNoFrameskip-v4", "ppo-PongNoFrameskip-v4.zip", PPO),
    "QbertNoFrameskip-v4":       ("sb3/ppo-QbertNoFrameskip-v4", "ppo-QbertNoFrameskip-v4.zip", PPO),
    "RiverraidNoFrameskip-v4":   ("qgallouedec/ppo-RiverraidNoFrameskip-v4-3987763893", "ppo-RiverraidNoFrameskip-v4.zip", PPO),
    "RoadRunnerNoFrameskip-v4":  ("sb3/a2c-RoadRunnerNoFrameskip-v4", "a2c-RoadRunnerNoFrameskip-v4.zip", A2C),
    "SeaquestNoFrameskip-v4":    ("sb3/ppo-SeaquestNoFrameskip-v4", "ppo-SeaquestNoFrameskip-v4.zip", PPO),
    "SpaceInvadersNoFrameskip-v4":("sb3/ppo-SpaceInvadersNoFrameskip-v4", "ppo-SpaceInvadersNoFrameskip-v4.zip", PPO)
}

TEST_GAMES = {
    "AlienNoFrameskip-v4":       ("UnclearMind/dqn-AlienNoFrameskip-v4", "dqn-AlienNoFrameskip-v4.zip", DQN),
    "AsteroidsNoFrameskip-v4":   ("CS462/dqn-AsteroidsNoFrameskip-v4", "dqn-AsteroidsNoFrameskip-v4.zip", DQN),
    "AtlantisNoFrameskip-v4":    ("redjackfred/dqn-AtlantisNoFrameskip-v4", "dqn-AtlantisNoFrameskip-v4.zip", DQN),
    "IceHockeyNoFrameskip-v4":   ("Yumejichi/dqn-IceHockeyNoFrameskip-v4", "dqn-IceHockeyNoFrameskip-v4.zip", DQN)
}

def collect_data(game_id, repo_id, filename, algo_class, split_folder, total_frames=50_000):
    print(f"\n========================================")
    print(f"Starting: {game_id}")
    print(f"Downloading from: {repo_id}")
    
    # 1. Download and Load Model
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    model = algo_class.load(checkpoint)
    
    # 2. Setup Environment
    base_name = game_id.split('No')[0]
    new_game_id = f"ALE/{base_name}-v5"
    
    env = make_atari_env(
        new_game_id, n_envs=1, seed=0,
        env_kwargs={"frameskip": 1, "repeat_action_probability": 0.0}
    )
    env = VecFrameStack(env, n_stack=4)
    
    print(f"Playing emulator and collecting dynamics transitions...")
    
    # Pre-allocate separate list buffers for transitions
    collected_obs = []
    collected_actions = []
    collected_next_obs = []
    collected_terminals = []
    
    obs = env.reset()
    
    for _ in tqdm(range(total_frames), desc="Recording Steps", leave=False):
        # Predict action using the current frame stack
        action, _ = model.predict(obs, deterministic=False)
        
        # Save current state and action before environment step updates them
        current_state = obs.squeeze(0)  # Shape: [4, 84, 84]
        current_action = action[0]       # Integer scalar
        
        # Step the environment
        next_obs, _, done, info = env.step(action)
        
        if done[0]:
            real_next_state = info[0]["terminal_observation"]
        else:
            real_next_state = next_obs[0] # Using index [0] instead of squeeze(0) is always safer
            
        # Ensure it is exactly 3D (4, 84, 84) before appending
        if real_next_state.ndim == 4:
            real_next_state = real_next_state.squeeze(0)
            
        # Append data to datasets
        collected_obs.append(current_state)
        collected_actions.append(current_action)
        collected_next_obs.append(real_next_state)
        collected_terminals.append(done[0])
        
        # Update current observation pointer for next loop
        obs = next_obs
        
    env.close()
    
    # 4. Save to Disk with Compressed Named Arrays
    save_path = f"custom_datasets/{split_folder}/{game_id}_expert_{total_frames}_transitions.npz"
    np.savez_compressed(
        save_path, 
        obs=np.array(collected_obs, dtype=np.uint8),
        actions=np.array(collected_actions, dtype=np.uint8),
        next_obs=np.array(collected_next_obs, dtype=np.uint8),
        terminals=np.array(collected_terminals, dtype=np.bool_)
    )
    print(f"✅ Saved completely to: {save_path}")

if __name__ == "__main__":
    os.makedirs("custom_datasets/train", exist_ok=True)
    os.makedirs("custom_datasets/test", exist_ok=True)

    print("🚀 STARTING TRAIN SET DYNAMICS COLLECTION")
    for game_id, (repo, file, algo) in TRAIN_GAMES.items():
        collect_data(game_id, repo, file, algo, split_folder="train")

    print("\n🚀 STARTING TEST SET DYNAMICS COLLECTION")
    for game_id, (repo, file, algo) in TEST_GAMES.items():
        collect_data(game_id, repo, file, algo, split_folder="test")

    print("\n🎉 ALL DATA STACKS RUN COMPLETE SUCCESSFULLY!")