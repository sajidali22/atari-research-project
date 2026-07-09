import argparse
import yaml
import gym
import torch
import torch.optim as optim
import numpy as np
import time
import wandb
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from models.factory import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration yaml")
    parser.add_argument("--game", type=str, default=None, help="Command-line override for game ID")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.game:
        config['env']['id'] = args.game

    run_name = f"PPO_{config['extractor']['arch_type']}_{config['env']['id']}"
    wandb.init(project=config['logging']['project_name'], name=run_name, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct the vectorized environment matching canonical SB3 setup
    envs = make_atari_env(config['env']['id'], n_envs=config['env']['num_envs'], seed=42)
    envs = VecFrameStack(envs, n_stack=4)
    num_actions = envs.action_space.n

    agent = PPOAgent(config, num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(config['ppo']['lr']), eps=1e-5)

    # Calculate update properties
    num_envs = config['env']['num_envs']
    num_steps = config['env']['num_steps']
    batch_size = num_envs * num_steps
    minibatch_size = config['ppo']['batch_size']
    total_timesteps = config['env']['total_timesteps']
    num_updates = total_timesteps // batch_size

    # PPO Batch Storage Arrays
    obs = torch.zeros((num_steps, num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)

    for update in range(1, num_updates + 1):
        # Linear Learning Rate Schedule Adjustment
        frac = 1.0 - (update - 1.0) / num_updates
        lr_now = frac * float(config['ppo']['lr'])
        optimizer.param_groups[0]["lr"] = lr_now

        # 1. Environment Interaction Loop (Rollout Phase)
        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    wandb.log({"charts/episodic_return": item["episode"]["r"], "global_step": global_step})

        # 2. Advantage Estimation (GAE Math)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config['ppo']['gamma'] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config['ppo']['gamma'] * config['ppo']['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten arrays for optimization minibatches
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 3. Model Optimization Step
        b_inds = np.arange(batch_size)
        for epoch in range(config['ppo']['n_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds].long())
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Policy Loss
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - config['ppo']['clip_coef'], 1.0 + config['ppo']['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Head Loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Optimization step
                entropy_loss = entropy.mean()
                loss = pg_loss - config['ppo']['ent_coef'] * entropy_loss + v_loss * config['ppo']['vf_coef']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        wandb.log({"charts/learning_rate": lr_now, "charts/loss": loss.item(), "global_step": global_step})
    envs.close()
    wandb.finish()

if __name__ == "__main__":
    main()