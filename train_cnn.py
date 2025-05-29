import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
from env_cnn import DiepIOEnvBasic
from wrappers import DiepIO_CNN_Wrapper
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import random
import datetime
from typing import Dict, Any

###########################################
# Replay Buffer
###########################################

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        image_shape: tuple,
        stats_shape: tuple,
        action_dim_d: int,
        action_dim_c: int,
        device: torch.device,
    ):
        self.device = device
        self.capacity = capacity
        self.image_shape = image_shape
        self.stats_shape = stats_shape
        self.action_dim_d = action_dim_d
        self.action_dim_c = action_dim_c
        self.count = 0
        self.size = 0
        self.reward_sum = 0.0
        self.reward_square_sum = 0.0

        self.image = torch.zeros((capacity,) + image_shape, dtype=torch.float32, device=device)
        self.stats = torch.zeros((capacity,) + stats_shape, dtype=torch.float32, device=device)
        self.action_d = torch.zeros((capacity, action_dim_d), dtype=torch.int64, device=device)
        self.action_c = torch.zeros((capacity, action_dim_c), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_image = torch.zeros((capacity,) + image_shape, dtype=torch.float32, device=device)
        self.next_stats = torch.zeros((capacity,) + stats_shape, dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.values = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.returns = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(
        self,
        image: np.ndarray,
        stats: np.ndarray,
        action_d: np.ndarray,
        action_c: np.ndarray,
        reward: float,
        next_image: np.ndarray,
        next_stats: np.ndarray,
        done: float,
        value: float,
        log_prob: float,
    ):
        idx = self.count % self.capacity

        # Update reward normalization
        self.count += 1
        self.reward_sum += reward
        self.reward_square_sum += reward ** 2
        mean = self.reward_sum / self.count
        var = max(self.reward_square_sum / self.count - mean ** 2, 1e-4)
        normalized_reward = (reward - mean) / np.sqrt(var)

        self.image[idx] = torch.tensor(image, dtype=torch.float32, device=self.device)
        self.stats[idx] = torch.tensor(stats, dtype=torch.float32, device=self.device)
        self.action_d[idx] = torch.tensor(action_d, dtype=torch.int64, device=self.device)
        self.action_c[idx] = torch.tensor(action_c, dtype=torch.float32, device=self.device)
        self.rewards[idx] = torch.tensor([[normalized_reward]], dtype=torch.float32, device=self.device)
        self.next_image[idx] = torch.tensor(next_image, dtype=torch.float32, device=self.device)
        self.next_stats[idx] = torch.tensor(next_stats, dtype=torch.float32, device=self.device)
        self.dones[idx] = torch.tensor([[done]], dtype=torch.float32, device=self.device)
        self.values[idx] = torch.tensor([[value]], dtype=torch.float32, device=self.device)
        self.log_probs[idx] = torch.tensor([[log_prob]], dtype=torch.float32, device=self.device)

        self.size = min(self.size + 1, self.capacity)

    def process_trajectory(self, gamma: float, gae_lambda: float, last_value: float, is_last_terminal: bool):
        path_slice = slice(0, self.size)
        values_t = self.values[path_slice].cpu().numpy()
        rewards_t = self.rewards[path_slice].cpu().numpy()

        N = len(rewards_t)
        deltas = np.zeros(N, dtype=np.float32)
        advantages = np.zeros(N, dtype=np.float32)
        next_values = np.concatenate([values_t[1:], [[last_value]]])
        next_non_terminal = np.ones(N, dtype=np.float32)
        if is_last_terminal:
            next_non_terminal[-1] = 0
        deltas = rewards_t[:, 0] + gamma * next_values[:, 0] * next_non_terminal - values_t[:, 0]

        advantages[-1] = deltas[-1]
        for t in reversed(range(N-1)):
            advantages[t] = deltas[t] + gamma * gae_lambda * next_non_terminal[t] * advantages[t+1]

        returns = advantages + values_t[:, 0]
        self.returns[path_slice] = torch.tensor(returns[:, np.newaxis], device=self.device)
        self.advantages[path_slice] = torch.tensor(advantages[:, np.newaxis], device=self.device)

    def sample(self, batch_size: int):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            'image': self.image[indices],
            'stats': self.stats[indices],
            'action_d': self.action_d[indices],
            'action_c': self.action_c[indices],
            'rewards': self.rewards[indices],
            'next_image': self.next_image[indices],
            'next_stats': self.next_stats[indices],
            'dones': self.dones[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'returns': self.returns[indices],
            'advantages': self.advantages[indices],
        }

    def clear(self):
        self.size = 0
        self.count = 0
        self.reward_sum = 0.0
        self.reward_square_sum = 0.0

    def state_dict(self):
        return {
            'size': self.size,
            'count': self.count,
            'reward_sum': self.reward_sum,
            'reward_square_sum': self.reward_square_sum,
            'image': self.image[:self.size],
            'stats': self.stats[:self.size],
            'action_d': self.action_d[:self.size],
            'action_c': self.action_c[:self.size],
            'rewards': self.rewards[:self.size],
            'next_image': self.next_image[:self.size],
            'next_stats': self.next_stats[:self.size],
            'dones': self.dones[:self.size],
            'values': self.values[:self.size],
            'log_probs': self.log_probs[:self.size],
            'returns': self.returns[:self.size],
            'advantages': self.advantages[:self.size],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.size = state_dict['size']
        self.count = state_dict['count']
        self.reward_sum = state_dict['reward_sum']
        self.reward_square_sum = state_dict['reward_square_sum']
        self.image[:self.size] = state_dict['image']
        self.stats[:self.size] = state_dict['stats']
        self.action_d[:self.size] = state_dict['action_d']
        self.action_c[:self.size] = state_dict['action_c']
        self.rewards[:self.size] = state_dict['rewards']
        self.next_image[:self.size] = state_dict['next_image']
        self.next_stats[:self.size] = state_dict['next_stats']
        self.dones[:self.size] = state_dict['dones']
        self.values[:self.size] = state_dict['values']
        self.log_probs[:self.size] = state_dict['log_probs']
        self.returns[:self.size] = state_dict['returns']
        self.advantages[:self.size] = state_dict['advantages']

    def __len__(self):
        return self.size

###########################################
# Neural Network Models
###########################################

class FeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 128, 128)
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_size = cnn_output.shape[1]

        self.stats_fc = nn.Sequential(
            nn.Linear(11 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.device = device
        self.to(device)

    def forward(self, image, stats):
        image = image.permute(0, 3, 1, 2).float()  # No /255.0, handled in wrapper
        img_features = self.cnn(image)
        stats = stats.view(stats.size(0), -1).float()
        stats_features = self.stats_fc(stats)
        return torch.cat([img_features, stats_features], dim=-1)

class Actor(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, action_dim_d: int, action_dim_c: int, device):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Sequential(
            nn.Linear(feature_extractor.cnn_output_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.discrete_head = nn.Linear(128, action_dim_d)
        self.continuous_head = nn.Linear(128, action_dim_c)
        self.device = device
        self.to(device)

    def forward(self, image, stats):
        features = self.feature_extractor(image, stats)
        features = self.fc(features)
        discrete_logits = self.discrete_head(features)
        continuous_mean = torch.tanh(self.continuous_head(features))
        return discrete_logits, continuous_mean

class Critic(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, device):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Sequential(
            nn.Linear(feature_extractor.cnn_output_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.device = device
        self.to(device)

    def forward(self, image, stats):
        features = self.feature_extractor(image, stats)
        value = self.fc(features)
        return value

###########################################
# PPO Policy Implementation
###########################################

class PPOPolicy(nn.Module):
    def __init__(self, action_dim_d, action_dim_c, learning_rate=2.5e-4, clip_range=0.2, value_coeff=0.5,
                 entropy_coeff=0.01, initial_std=0.1, max_grad_norm=0.5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FeatureExtractor(self.device)
        self.actor = Actor(self.feature_extractor, action_dim_d, action_dim_c, self.device)
        self.critic = Critic(self.feature_extractor, self.device)
        self.log_std = nn.Parameter(torch.ones(action_dim_c) * torch.log(torch.tensor(initial_std)))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.to(self.device)

    def forward(self, image, stats):
        discrete_logits, continuous_mean = self.actor(image, stats)
        value = self.critic(image, stats)
        discrete_dists = [
            Categorical(logits=discrete_logits[:, 0:3]),
            Categorical(logits=discrete_logits[:, 3:6]),
            Categorical(logits=discrete_logits[:, 6:8]),
            Categorical(logits=discrete_logits[:, 8:]),
        ]
        std = torch.exp(self.log_std).clamp(1e-6, 50.0)
        continuous_dist = Normal(continuous_mean, std)
        return discrete_dists, continuous_dist, value

    def get_action(self, image, stats):
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        stats = torch.tensor(stats, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            discrete_dists, continuous_dist, value = self.forward(image, stats)
            discrete = torch.stack([dist.sample() for dist in discrete_dists], dim=-1)
            continuous = continuous_dist.sample()
            log_prob = sum(dist.log_prob(discrete[:, i]) for i, dist in enumerate(discrete_dists))
            log_prob += continuous_dist.log_prob(continuous).sum(dim=-1)
        action = {"d": discrete.cpu().numpy()[0], "c": continuous.cpu().numpy()[0]}
        return action, log_prob.item(), value.item()

    def get_values(self, image, stats):
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        stats = torch.tensor(stats, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, value = self.forward(image, stats)
        return value.item()

    def update(self, batch):
        image_batch = batch['image'].to(self.device)
        stats_batch = batch['stats'].to(self.device)
        action_batch = {'d': batch['action_d'].to(self.device), 'c': batch['action_c'].to(self.device)}
        log_prob_batch = batch['log_probs'].to(self.device)
        advantage_batch = batch['advantages'].to(self.device)
        return_batch = batch['returns'].to(self.device)

        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

        discrete_dists, continuous_dist, value = self.forward(image_batch, stats_batch)
        new_log_prob = sum(dist.log_prob(action_batch['d'][:, i]) for i, dist in enumerate(discrete_dists))
        new_log_prob += continuous_dist.log_prob(action_batch['c']).sum(dim=-1, keepdim=True)
        entropy = sum(dist.entropy().mean() for dist in discrete_dists) + continuous_dist.entropy().mean()

        ratio = torch.exp(new_log_prob - log_prob_batch)
        surr1 = ratio * advantage_batch
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage_batch
        pi_loss = -torch.min(surr1, surr2).mean()
        value_loss = self.value_coeff * F.mse_loss(value, return_batch)
        entropy_loss = -self.entropy_coeff * entropy
        total_loss = pi_loss + value_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        approx_kl = ((ratio - 1) - (new_log_prob - log_prob_batch)).mean()
        return pi_loss.item(), value_loss.item(), total_loss.item(), approx_kl.item(), torch.exp(self.log_std).mean().item()

###########################################
# Utility Functions
###########################################

def save_random_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

def load_random_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])

def save_checkpoint(policy, buffer, episode_count, mean_rewards, timestep, log_dir, checkpoint_path):
    checkpoint = {
        'actor_state_dict': policy.actor.state_dict(),
        'critic_state_dict': policy.critic.state_dict(),
        'feature_extractor_state_dict': policy.feature_extractor.state_dict(),
        'log_std': policy.log_std,
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'buffer': buffer.state_dict(),
        'episode_count': episode_count,
        'mean_rewards': mean_rewards,
        'timestep': timestep,
        'random_state': save_random_state(),
        'log_dir': log_dir
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved at {checkpoint_path}\n")

def load_checkpoint(policy, buffer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None, None, None, 0
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if policy is not None:
        policy.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        policy.log_std = checkpoint['log_std']
        policy.optimizer = optim.Adam(policy.parameters(), lr=2.5e-4)
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if buffer is not None:
        buffer.load_state_dict(checkpoint['buffer'])

    load_random_state(checkpoint['random_state'])
    log_dir = checkpoint.get('log_dir', 'Log')
    return checkpoint['episode_count'], checkpoint['mean_rewards'], log_dir, checkpoint['timestep']

def learn(policy, buffer, num_epochs, batch_size, writer, episode_count, timestep):
    for _ in range(num_epochs):
        batch = buffer.sample(batch_size)
        pi_loss, v_loss, total_loss, approx_kl, std = policy.update(batch)

        writer.add_scalar("train/pi_loss", pi_loss, timestep)
        writer.add_scalar("train/v_loss", v_loss, timestep)
        writer.add_scalar("train/total_loss", total_loss, timestep)
        writer.add_scalar("train/approx_kl", approx_kl, timestep)
        writer.add_scalar("train/std", std, timestep)

    return approx_kl, std

###########################################
# Training Function
###########################################

def train_ppo(
    n_tanks,
    num_episodes_per_update=10,
    max_buffer_steps=10000,
    batch_size=64,
    total_timesteps=500000,
    gamma=0.99,
    gae_lam=0.95,
    num_epochs=10,
    learning_rate=2.5e-4,
    clip_range=0.2,
    value_coeff=0.5,
    entropy_coeff=0.01,
    max_grad_norm=0.5,
    initial_std=0.1,
    save_interval=100000,
    checkpoint_path=None,
    checkpoint_dir="checkpoints",
    phase="single-agent"
):
    # Initialize environment
    env_config = {
        "n_tanks": n_tanks,
        "render_mode": False,
        "max_steps": 1000000,
        "unlimited_obs": False,
        "resize_shape": (128, 128),
        "frame_stack_size": 4,
        "skip_frames": 4
    }
    env = DiepIO_CNN_Wrapper(
        DiepIOEnvBasic(env_config),
        resize_shape=env_config["resize_shape"],
        frame_stack_size=env_config["frame_stack_size"],
        skip_frames=env_config["skip_frames"]
    )
    image_shape = (128, 128, 4)
    stats_shape = (11, 4)
    action_dim_d = 17  # MultiDiscrete([3, 3, 2, 9]) -> 17 logits
    action_dim_c = 2   # Box(-1, 1, (2,))

    # Initialize TensorBoard
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episode_count, mean_rewards, log_dir, timestep = load_checkpoint(None, None, checkpoint_path if checkpoint_path else os.path.join(checkpoint_dir, f"{phase}_checkpoint.pt"))
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("Log", f"run_{timestamp}_{phase}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    print(f"Logging to TensorBoard at {log_dir}")

    # Initialize buffer and policy
    buffer = ReplayBuffer(max_buffer_steps, image_shape, stats_shape, action_dim_d, action_dim_c, device)
    policy = PPOPolicy(
        action_dim_d=action_dim_d,
        action_dim_c=action_dim_c,
        learning_rate=learning_rate,
        clip_range=clip_range,
        value_coeff=value_coeff,
        entropy_coeff=entropy_coeff,
        initial_std=initial_std,
        max_grad_norm=max_grad_norm
    )

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        episode_count, mean_rewards, _, timestep = load_checkpoint(policy, buffer, checkpoint_path)
        print(f"Resumed training from {checkpoint_path} at episode {episode_count}, timestep {timestep}")

    # Training data
    episode_rewards = []
    mean_rewards = mean_rewards if mean_rewards else []
    timestep = timestep if timestep else 0
    episode_count = episode_count if episode_count else 0

    while timestep < total_timesteps:
        obs, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in env.possible_agents}
        dones = {agent: False for agent in env.possible_agents}
        episode_steps = 0

        while not all(dones.values()) and timestep < total_timesteps:
            actions = {}
            for agent in env.possible_agents:
                if dones[agent]:
                    continue
                image = obs[agent]["i"]
                stats = obs[agent]["s"]

                action, log_prob, value = policy.get_action(image, stats)
                actions[agent] = action

            next_obs, rewards, dones, truncations, _ = env.step(actions)

            for agent in env.possible_agents:
                if dones[agent]:
                    continue
                reward = rewards.get(agent, 0.0)
                episode_reward[agent] += reward

                buffer.push(
                    image=obs[agent]["i"],
                    stats=obs[agent]["s"],
                    action_d=action["d"],
                    action_c=action["c"],
                    reward=reward,
                    next_image=next_obs[agent]["i"],
                    next_stats=next_obs[agent]["s"],
                    done=1.0 if dones[agent] else 0.0,
                    value=value,
                    log_prob=log_prob
                )

            obs = next_obs
            episode_steps += env.skip_frames
            timestep += env.skip_frames

            if any(dones.values()) or episode_steps >= env.max_steps:
                break

        episode_count += 1
        episode_rewards.append(sum(episode_reward.values()))

        # Process trajectory
        last_values = {}
        for agent in env.possible_agents:
            if not dones[agent]:
                last_values[agent] = policy.get_values(obs[agent]["i"], obs[agent]["s"])
            else:
                last_values[agent] = 0.0
        buffer.process_trajectory(
            gamma=gamma,
            gae_lam=gae_lam,
            last_value=max(last_values.values()),
            is_last_terminal=any(dones.values())
        )

        # Update policy
        if episode_count % num_episodes_per_update == 0:
            approx_kl, std = learn(policy, buffer, num_epochs, batch_size, writer, episode_count, timestep)
            mean_reward = np.mean(episode_rewards[-num_episodes_per_update:])
            mean_rewards.append(mean_reward)
            writer.add_scalar("misc/ep_reward_mean", mean_reward, timestep)
            print(f"Timestep {timestep}/{total_timesteps} | Episode {episode_count} | Avg Reward {mean_reward:.2f} | KL {approx_kl:.4f} | STD {std:.4f}")
            buffer.clear()

        # Save checkpoint
        if timestep // save_interval > (timestep - episode_steps) // save_interval:
            checkpoint_path_phase = os.path.join(checkpoint_dir, f"{phase}_checkpoint.pt")
            save_checkpoint(
                policy, buffer, episode_count, mean_rewards, timestep, log_dir, checkpoint_path_phase
            )

    # Plot reward curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.xlabel("Update")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{phase}_reward.png"))

    # Save final checkpoint
    checkpoint_path_phase = os.path.join(checkpoint_dir, f"{phase}_checkpoint.pt")
    save_checkpoint(
        policy, buffer, episode_count, mean_rewards, timestep, log_dir, checkpoint_path_phase
    )

    writer.close()
    env.close()

###########################################
# Main Function
###########################################

if __name__ == "__main__":
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Phase 1: Single-Agent Training
    print("Starting single-agent training (n_tanks=1)...")
    train_ppo(
        n_tanks=1,
        total_timesteps=500000,
        checkpoint_dir=checkpoint_dir,
        phase="single-agent"
    )

    # Phase 2: Multi-Agent Training
    print("Starting multi-agent training (n_tanks=2)...")
    train_ppo(
        n_tanks=2,
        total_timesteps=500000,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=os.path.join(checkpoint_dir, "single-agent_checkpoint.pt"),
        phase="multi-agent"
    )

    print("Training complete.")