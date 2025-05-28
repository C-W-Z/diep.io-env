import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN
import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Tuple, MultiDiscrete, Box
from env import DiepIOEnvBasic

# Custom Model for Tuple Action Space
class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Observation feature extractor
        self.obs_net = FullyConnectedNetwork(
            obs_space=obs_space,
            action_space=Box(-1, 1, (1,)),  # Dummy for feature extraction
            num_outputs=256,
            model_config=model_config,
            name="obs_net"
        )

        # Discrete action heads (dx, dy, shoot, skill_index)
        self.discrete_heads = nn.ModuleList([
            nn.Linear(256, n) for n in [3, 3, 2, 9]
        ])

        # Continuous action mean and log_std
        self.continuous_mean = nn.Linear(256, 2)
        self.continuous_log_std = nn.Parameter(torch.zeros(2))

        # Value function
        self.value_net = nn.Linear(256, 1)
        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        features = self.obs_net({"obs": obs})  # Extract features

        # Discrete action logits
        discrete_logits = [head(features) for head in self.discrete_heads]
        discrete_logits = torch.cat(discrete_logits, dim=-1)

        # Continuous action mean and log_std
        continuous_mean = torch.tanh(self.continuous_mean(features))
        continuous_log_std = self.continuous_log_std.expand_as(continuous_mean)

        # Combine actions
        action_out = torch.cat([discrete_logits, continuous_mean, continuous_log_std], dim=-1)

        # Compute value
        self._last_value = self.value_net(features).squeeze(-1)
        return action_out, state

    def value_function(self):
        return self._last_value

    def get_action_dist(self, model_out):
        # Split model output into discrete logits and continuous params
        discrete_logits = model_out[:, :17]  # 3 + 3 + 2 + 9 = 17
        continuous_mean = model_out[:, 17:19]
        continuous_log_std = model_out[:, 19:21]

        # Create distributions
        discrete_splits = [3, 3, 2, 9]
        discrete_logits = torch.split(discrete_logits, discrete_splits, dim=-1)
        discrete_dist = [
            torch.distributions.Categorical(logits=logits) for logits in discrete_logits
        ]
        continuous_dist = torch.distributions.Normal(continuous_mean, torch.exp(continuous_log_std))

        return discrete_dist, continuous_dist

def train():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Environment configuration
    env_config = {
        "n_tanks": 2,
        "render_mode": None,
        "max_steps": 1000000,
        "unlimited_obs": False
    }

    # Register custom model
    tune.register_env("diepio_env", lambda config: DiepIOEnvBasic(config))

    # PPO Configuration
    config = (
        PPOConfig()
        .environment(
            env="diepio_env",
            env_config=env_config
        )
        .framework("torch")
        .rollouts(num_rollout_workers=2, num_envs_per_worker=1)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            train_batch_size=2048,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
            model={
                "custom_model": CustomTorchModel,
                "custom_model_config": {}
            }
        )
        .multi_agent(
            policies={"shared_policy": (
                None,  # Use default policy class
                DiepIOEnvBasic(env_config).observation_space,
                DiepIOEnvBasic(env_config).action_space,
                {}
            )},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        .resources(num_gpus=0)
    )

    # Build and train
    algo = config.build()
    for i in range(100):
        result = algo.train()
        print(f"Iteration {i}, Mean Reward: {result['episode_reward_mean']:.4f}")
        if i % 10 == 0:
            checkpoint = algo.save()
            print(f"Checkpoint saved at: {checkpoint}")

    # Save final model
    final_checkpoint = algo.save()
    print(f"Final checkpoint: {final_checkpoint}")

    # Test
    env = DiepIOEnvBasic({"n_tanks": 2, "render_mode": "human", "max_steps": 10000})
    obs, infos = env.reset()
    total_rewards = {agent: 0.0 for agent in env._agent_ids}
    done = False

    while not done:
        actions = {}
        for agent in obs:
            action, _, _ = algo.compute_single_action(
                obs[agent], policy_id="shared_policy", explore=True
            )
            actions[agent] = action
        obs, rewards, dones, truncations, infos = env.step(actions)
        for agent in rewards:
            total_rewards[agent] += rewards[agent]
        done = dones["__all__"]
        env.render()

    print("Total rewards:", total_rewards)
    env.close()
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    train()
