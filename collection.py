import os
import tqdm
import gymnasium as gym
import ale_py  
import numpy as np
import config 

os.environ["SDL_AUDIODRIVER"] = "dummy"

gym.register_envs(ale_py)

def collect_atari_frames(game_name, output_dir):
    """
    Collects frames from an Atari game and displays a progress bar 
    identifying the specific game being processed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Safety check: Rendering is very slow for large datasets
    if config.RENDER_GAME:
        print(f"RENDER_GAME is True.")
    
    env_id = f"ALE/{game_name}-v5"
    render = "human" if config.RENDER_GAME else None
    
    env = gym.make(env_id, frameskip=1, render_mode=render)
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=4, grayscale_obs=config.GRAYSCALE_MODE, screen_size=84, scale_obs=False
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    obs, info = env.reset()
    frames_chunk = []
    chunk_counter = 1
    
    # --- Progress Bar with Game Name ---
    pbar = tqdm.tqdm(
        total=config.TOTAL_STEPS, 
        desc=f"{game_name.ljust(15)}",
        unit="frame",
        colour="green"
    )

    for step in range(1, config.TOTAL_STEPS + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        frames_chunk.append(np.array(obs))
        
        if terminated or truncated:
            obs, info = env.reset()
            
        pbar.update(1)
            
        # Memory-safe saving
        if step % config.CHUNK_SIZE == 0 or step == config.TOTAL_STEPS:
            dataset = np.array(frames_chunk, dtype=np.uint8)
            file_name = f"{game_name}_part{chunk_counter}.npy"
            file_path = os.path.join(output_dir, file_name)
            
            np.save(file_path, dataset)
            
            pbar.write(f"   ✅ Saved {file_name}")
            
            frames_chunk = []
            chunk_counter += 1
            
    pbar.close()
    env.close()

if __name__ == "__main__":
    print(f"Starting Collection | Steps: {config.TOTAL_STEPS} | Mode: {'Gray' if config.GRAYSCALE_MODE else 'RGB'}")
    
    # Loop through Training Games
    for game in config.TRAIN_GAMES:
        collect_atari_frames(game, config.TRAIN_DIR)
        
    # Loop through Testing Games
    for game in config.TEST_GAMES:
        collect_atari_frames(game, config.TEST_DIR)
        
    print("\nAll datasets have been successfully collected and saved.")