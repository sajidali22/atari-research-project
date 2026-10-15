import time
import torch
import gymnasium as gym
import ale_py
from torch.distributions.categorical import Categorical

gym.register_envs(ale_py)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from ppo.custom.factory import PPOAgent 

def visualize_agent(checkpoint_path="artifacts/checkpoints/ppo/mspacman.pth", game_id="MsPacmanNoFrameskip-v4", max_episodes=3):
    # Ensure this matches your training hardware if possible to avoid FSQ rounding errors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")
    
    envs = make_atari_env(game_id, n_envs=1, seed=42, env_kwargs={"render_mode": "human"})
    envs = VecFrameStack(envs, n_stack=4)
    num_actions = envs.action_space.n
    
    config = {
        'extractor': {
            'arch_type': 'FSQ-VAE',
            'checkpoint_path': None,
            'feature_dim': 4 * 21 * 21, 
            'freeze_weights': True
        }
    }
    
    agent = PPOAgent(config, num_actions=num_actions).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    agent.load_state_dict(clean_dict)
    agent.eval()
    
    current_episode = 1
    obs = envs.reset()
    
    while current_episode <= max_episodes:
        obs_tensor = torch.tensor(obs, dtype=torch.float32 if "cuda" not in device.type else torch.uint8).to(device)
        if obs_tensor.dtype != torch.uint8:
            obs_tensor = obs_tensor.byte() # Force byte alignment to match training
            
        with torch.no_grad():
            features = agent.feature_extractor(obs_tensor)
            logits = agent.actor(features)
            
            # FIXED: Stochastic sampling replaces argmax
            probs = Categorical(logits=logits)
            action = probs.sample().cpu().numpy()
            
        obs, rewards, dones, infos = envs.step(action)
        time.sleep(0.016)
        
        if "episode" in infos[0]:
            true_score = infos[0]["episode"]["r"]
            print(f"🏁 Game {current_episode} Finished! Final True Score: {true_score}")
            current_episode += 1

    envs.close()

if __name__ == "__main__":
    visualize_agent()