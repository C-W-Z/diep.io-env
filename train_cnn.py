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

        self.image = torch.zeros((capacity,) + image_shape, dtype=torch.float16, device=device)
        self.stats = torch.zeros((capacity,) + stats_shape, dtype=torch.float16, device=device)
        self.action_d = torch.zeros((capacity, action_dim_d), dtype=torch.int8, device=device)
        self.action_c = torch.zeros((capacity, action_dim_c), dtype=torch.float16, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float16, device=device)
        self.next_image = torch.zeros((capacity,) + image_shape, dtype=torch.float16, device=device)
        self.next_stats = torch.zeros((capacity,) + stats_shape, dtype=torch.float16, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float16, device=device)
        self.values = torch.zeros((capacity, 1), dtype=torch.float16, device=device)
        self.log_probs = torch.zeros((capacity, 1), dtype=torch.float16, device=device)
        self.returns = torch.zeros((capacity, 1), dtype=torch.float16, device=device)
        self.advantages = torch.zeros((capacity, 1), dtype=torch.float16, device=device)

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
        # idx = self.count % self.capacity
        idx = self.size % self.capacity

        # Update reward normalization
        # self.count += 1
        # self.reward_sum += reward
        # self.reward_square_sum += reward ** 2
        # mean = self.reward_sum / self.count
        # var = max(self.reward_square_sum / self.count - mean ** 2, 1e-4)
        # normalized_reward = (reward - mean) / np.sqrt(var)

        self.image[idx] = torch.tensor(image, dtype=torch.float16, device=self.device)
        self.stats[idx] = torch.tensor(stats, dtype=torch.float16, device=self.device)
        self.action_d[idx] = torch.tensor(action_d, dtype=torch.int8, device=self.device)
        self.action_c[idx] = torch.tensor(action_c, dtype=torch.float16, device=self.device)
        self.rewards[idx] = torch.tensor([[reward]], dtype=torch.float16, device=self.device)
        self.next_image[idx] = torch.tensor(next_image, dtype=torch.float16, device=self.device)
        self.next_stats[idx] = torch.tensor(next_stats, dtype=torch.float16, device=self.device)
        self.dones[idx] = torch.tensor([[done]], dtype=torch.float16, device=self.device)
        self.values[idx] = torch.tensor([[value]], dtype=torch.float16, device=self.device)
        self.log_probs[idx] = torch.tensor([[log_prob]], dtype=torch.float16, device=self.device)

        self.size = min(self.size + 1, self.capacity)

    def process_trajectory(self, gamma: float, gae_lambda: float, last_value: float, is_last_terminal: bool):
        path_slice = slice(0, self.size)
        # Convert to float32 for computation
        values_t = self.values[path_slice].to(torch.float32).cpu().numpy()
        rewards_t = self.rewards[path_slice].to(torch.float32).cpu().numpy()

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
        self.returns[path_slice] = torch.tensor(returns[:, np.newaxis], dtype=torch.float16, device=self.device)
        self.advantages[path_slice] = torch.tensor(advantages[:, np.newaxis], dtype=torch.float16, device=self.device)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            print("no enough data", self.size, batch_size)
            return None  # Not enough data to sample
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        # Cast to float32 for model compatibility
        return {
            'image': self.image[indices].to(torch.float32),
            'stats': self.stats[indices].to(torch.float32),
            'action_d': self.action_d[indices],  # int8 is fine
            'action_c': self.action_c[indices].to(torch.float32),
            'rewards': self.rewards[indices].to(torch.float32),
            'next_image': self.next_image[indices].to(torch.float32),
            'next_stats': self.next_stats[indices].to(torch.float32),
            'dones': self.dones[indices].to(torch.float32),
            'values': self.values[indices].to(torch.float32),
            'log_probs': self.log_probs[indices].to(torch.float32),
            'returns': self.returns[indices].to(torch.float32),
            'advantages': self.advantages[indices].to(torch.float32),
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
    def __init__(self, image_shape: tuple, stats_dim: int, device):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, image_shape[2], image_shape[0], image_shape[1])
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_size = cnn_output.shape[1]

        self.stats_fc = nn.Sequential(
            nn.Linear(stats_dim, 128),
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
    def __init__(self, image_shape, stats_dim, action_space_d, action_dim_c, lr=1e-4, clip_range=0.2, value_coeff=0.5,
                 entropy_coeff=0.01, initial_std=0.1, max_grad_norm=0.5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FeatureExtractor(image_shape, stats_dim, self.device)
        self.actor = Actor(self.feature_extractor, action_space_d, action_dim_c, self.device)
        self.critic = Critic(self.feature_extractor, self.device)
        self.log_std = nn.Parameter(torch.ones(action_dim_c) * torch.log(torch.tensor(initial_std)))
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.discrete_splits = [3, 3, 2, 9]  # For splitting logits into [3, 3, 2, 9]
        self.to(self.device)

    def forward(self, image, stats):
        discrete_logits, continuous_mean = self.actor(image, stats)
        # Split logits into 4 groups for MultiDiscrete
        split_logits = torch.split(discrete_logits, self.discrete_splits, dim=-1)
        discrete_dists = [Categorical(logits=logits) for logits in split_logits]
        std = torch.exp(self.log_std).clamp(1e-6, 50.0)
        continuous_dist = Normal(continuous_mean, std)
        value = self.critic(image, stats)
        return discrete_dists, continuous_dist, value

    def get_action(self, image, stats):
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        stats = torch.tensor(stats, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            discrete_dists, continuous_dist, value = self.forward(image, stats)
            discrete = torch.stack([dist.sample() for dist in discrete_dists], dim=-1)  # Shape (1, 4)
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
        new_log_prob = torch.zeros_like(log_prob_batch)
        for i, dist in enumerate(discrete_dists):
            new_log_prob += dist.log_prob(action_batch['d'][:, i]).unsqueeze(-1)
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

def save_checkpoint(policy, buffers, episode_count, mean_rewards, timestep, log_dir, checkpoint_path):
    checkpoint = {
        'actor_state_dict': policy.actor.state_dict(),
        'critic_state_dict': policy.critic.state_dict(),
        'feature_extractor_state_dict': policy.feature_extractor.state_dict(),
        'log_std': policy.log_std,
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'buffers': {agent: buffer.state_dict() for agent, buffer in buffers.items()},
        'episode_count': episode_count,
        'mean_rewards': mean_rewards,
        'timestep': timestep,
        'log_dir': log_dir
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved at {checkpoint_path}\n")

def load_checkpoint(policy, buffers, checkpoint_path, lr=1e-4):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, None, None, 0
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if policy is not None:
        policy.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        policy.log_std = checkpoint['log_std']
        policy.optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-5)
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if buffers is not None:
        for agent, buffer in buffers.items():
            if agent in checkpoint['buffers']:
                buffer.load_state_dict(checkpoint['buffers'][agent])

    log_dir = checkpoint.get('log_dir', 'Log')
    return checkpoint['episode_count'], checkpoint['mean_rewards'], log_dir, checkpoint['timestep']

def learn(policy: PPOPolicy, buffers: dict[str, ReplayBuffer], num_epochs, batch_size: int, writer, episode_count, timestep):
    for _ in range(num_epochs):
        all_batches = []
        for buffer in buffers.values():
            batch = buffer.sample(batch_size)  # Split batch size across agents
            if batch is not None:
                all_batches.append(batch)

        if not all_batches:
            return 0.0, 0.0  # No data to learn from

        # Concatenate batches from all agents
        combined_batch = {}
        for key in all_batches[0].keys():
            combined_batch[key] = torch.cat([batch[key] for batch in all_batches], dim=0)

        pi_loss, v_loss, total_loss, approx_kl, std = policy.update(combined_batch)
        writer.add_scalar("train/pi_loss", pi_loss, timestep)
        writer.add_scalar("train/v_loss", v_loss, timestep)
        writer.add_scalar("train/total_loss", total_loss, timestep)
        writer.add_scalar("train/approx_kl", approx_kl, timestep)
        writer.add_scalar("train/std", std, timestep)

    return approx_kl, std

###########################################
# Training Function
###########################################

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def train_ppo(
    n_tanks,
    max_buffer_size=10000,
    batch_size=32,          # per agent
    total_timesteps=500000,
    gamma=0.99,
    gae_lambda=0.95,
    num_epochs=10,
    lr=1e-4,
    clip_range=0.2,
    value_coeff=0.5,
    entropy_coeff=0.05,
    max_grad_norm=0.5,
    initial_std=1.0,
    save_interval=10,
    checkpoint_path=None,
    checkpoint_dir="checkpoints",
    phase="single-agent"
):
    # Initialize environment
    env_config = {
        "n_tanks": n_tanks,
        "render_mode": False,  # Disable rendering to save memory
        "max_steps": 4 * max_buffer_size // n_tanks,
        "resize_shape": (100, 100),
        "frame_stack_size": 4,
        "skip_frames": 4
    }
    env = DiepIO_CNN_Wrapper(
        DiepIOEnvBasic(env_config),
        resize_shape=env_config["resize_shape"],
        frame_stack_size=env_config["frame_stack_size"],
        skip_frames=env_config["skip_frames"]
    )
    image_shape = env.observation_space["i"].shape  # (100, 100, 12)
    stats_shape = env.observation_space["s"].shape  # (11, 4)
    action_space_d = env.action_space["d"].nvec.sum()  # MultiDiscrete([3, 3, 2, 9]) -> 17 logits
    action_dim_d = env.action_space["d"].shape[0]  # MultiDiscrete([3, 3, 2, 9]) -> 4 dim
    action_dim_c = env.action_space["c"].shape[0]  # Box(-1, 1, (2,))

    # Initialize TensorBoard
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episode_count, mean_rewards, log_dir, timestep = load_checkpoint(None, None, checkpoint_path, lr=lr)
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("Log", f"run_{timestamp}_{phase}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    print(f"Logging to TensorBoard at {log_dir}")

    # Initialize a ReplayBuffer for each agent
    buffers = {agent: ReplayBuffer(max_buffer_size // n_tanks, image_shape, stats_shape, action_dim_d, action_dim_c, device) for agent in env.agents}
    print_gpu_memory()
    policy = PPOPolicy(
        image_shape=image_shape,
        stats_dim=stats_shape[0] * stats_shape[1],
        action_space_d=action_space_d,
        action_dim_c=action_dim_c,
        lr=lr,
        clip_range=clip_range,
        value_coeff=value_coeff,
        entropy_coeff=entropy_coeff,
        initial_std=initial_std,
        max_grad_norm=max_grad_norm
    )

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        episode_count, mean_rewards, _, timestep = load_checkpoint(policy, buffers, checkpoint_path, lr=lr)
        print(f"Resumed training from {checkpoint_path} at episode {episode_count}, timestep {timestep}")

    # Training data
    mean_rewards = mean_rewards if mean_rewards else []
    timestep = timestep if timestep else 0
    episode_count = episode_count if episode_count else 0

    while timestep < total_timesteps:
        obs, _ = env.reset()
        episode_reward = {agent: 0.0 for agent in env.agents}
        dones = {agent: False for agent in env.agents}
        dones["__all__"] = False
        episode_steps = 0

        while not dones["__all__"] and timestep < total_timesteps:
            actions = {}
            for agent in env.agents:
                if dones[agent]:
                    continue
                image = obs[agent]["i"]
                stats = obs[agent]["s"]

                action, log_prob, value = policy.get_action(image, stats)
                actions[agent] = action

            next_obs, rewards, dones, truncations, _ = env.step(actions)

            for agent, n_obs in next_obs.items():
                # if dones[agent]:
                #     continue
                reward = rewards.get(agent, 0.0)
                episode_reward[agent] += reward

                buffers[agent].push(
                    image=obs[agent]["i"],
                    stats=obs[agent]["s"],
                    action_d=actions[agent]["d"],
                    action_c=actions[agent]["c"],
                    reward=reward,
                    next_image=n_obs["i"],
                    next_stats=n_obs["s"],
                    done=1.0 if dones[agent] else 0.0,
                    value=value,
                    log_prob=log_prob
                )

            obs = next_obs
            episode_steps += 1
            timestep += 1

            if dones["__all__"]:
                break

        episode_count += 1

        # Process trajectories for each agent
        for agent in env.agents:
            if not dones[agent]:
                last_value = policy.get_values(obs[agent]["i"], obs[agent]["s"])
            else:
                last_value = 0.0
            buffers[agent].process_trajectory(
                gamma=gamma,
                gae_lambda=gae_lambda,
                last_value=last_value,
                is_last_terminal=dones[agent]
            )

        # Learn using experiences from all agents
        approx_kl, std = learn(policy, buffers, num_epochs, batch_size, writer, episode_count, timestep)
        reward_str = ""
        for agent, reward in episode_reward.items():
            reward_str += f" Reward {agent}: {reward:.2f}"
            writer.add_scalar(f"misc/ep_reward_{agent}", reward, episode_count)
        print(f"Timestep {episode_steps} | Episode {episode_count} |" + reward_str + f" | KL {approx_kl:.4f} | STD {std:.4f}")

        # Clear buffers
        for buffer in buffers.values():
            buffer.clear()

        # Save checkpoint
        if episode_count % save_interval == 0:
            checkpoint_path_phase = os.path.join(checkpoint_dir, f"{phase}_checkpoint.pt")
            save_checkpoint(
                policy, buffers, episode_count, mean_rewards, timestep, log_dir, checkpoint_path_phase
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
        policy, buffers, episode_count, mean_rewards, timestep, log_dir, checkpoint_path_phase
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
        # checkpoint_path=os.path.join(checkpoint_dir, "single-agent_checkpoint.pt"),
        phase="single-agent"
    )

    # Phase 2: Multi-Agent Training
    # print("Starting multi-agent training (n_tanks=2)...")
    # train_ppo(
    #     n_tanks=2,
    #     total_timesteps=500000,
    #     checkpoint_dir=checkpoint_dir,
    #     # checkpoint_path=os.path.join(checkpoint_dir, "single-agent_checkpoint.pt"),
    #     phase="multi-agent"
    # )

    print("Training complete.")