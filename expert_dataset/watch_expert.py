import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from huggingface_sb3 import load_from_hub

# 🚨 The newest version of Gymnasium requires explicit Atari registration
gym.register_envs(ale_py)

def watch_expert_play(new_game_id, repo_id, filename, episodes=3):
    
    # 1. Download the pre-trained expert brain using the OLD name
    # checkpoint_path = load_from_hub(repo_id=hf_repo, filename=f"ppo-{old_game_id}.zip")
    checkpoint_path = load_from_hub(repo_id=repo_id, filename=filename)
    model= DQN.load(checkpoint_path)
    # model = PPO.load(checkpoint_path)
    # model = A2C.load(checkpoint_path)


    
    # 2. Setup the Atari Environment using the NEW Gymnasium naming convention
    # print(f"📺 Booting up {new_game_id} in human render mode...")
    env = make_atari_env(
        new_game_id, 
        n_envs=1, 
        seed=0, 
        env_kwargs={
            "render_mode": "human",
            "frameskip": 1,                      # Fix 1: Stop double-skipping
            "repeat_action_probability": 0.0     # Fix 2: Turn off sticky buttons
        }
    )
    env = VecFrameStack(env, n_stack=4)
    
    # 3. Watch it play!
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        print(f"▶️ Starting Episode {episode + 1}...")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            total_reward += reward[0]
            done = dones[0] 
            
        print(f"⏹️ Episode {episode + 1} finished! Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    # We pass BOTH names. Old for downloading the brain, New for booting the game!
    watch_expert_play(
        # old_game_id="AlienNoFrameskip-v4", 
        new_game_id="ALE/IceHockey-v5", 
        # hf_repo="dqn-AlienNoFrameskip-v4",
        repo_id="Yumejichi/dqn-IceHockeyNoFrameskip-v4",
        filename="dqn-IceHockeyNoFrameskip-v4.zip",
        episodes=3
    )