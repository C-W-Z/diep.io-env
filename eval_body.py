import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from diepio.env_body import DiepIOEnvBody
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.spaces.space_utils import unbatch
import imageio
import pygame
import numpy as np

from ray.tune.registry import register_env

# Register environment
def env_creator(env_config):
    return DiepIOEnvBody(env_config)

register_env("diepio-v0", env_creator)

# 初始化 Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

# 你的 checkpoint 絕對路徑，例如：
checkpoint_path = "~/diep.io-env/ray_results/diepio_body_onlymove_dqn/checkpoint_000015"

# 載入訓練好的 policy
algo = Algorithm.from_checkpoint(checkpoint_path)
module = algo.get_module("body_policy")

env_config = {
    "n_tanks": 1,
    "render_mode": True,
    "max_steps": 5000,
    "skill_mode": [2]
}

# 設定 render 模式的環境
env = env_creator(env_config)

obs, _ = env.reset()

done = False

total_rewards = 0

frames = []

frame = pygame.surfarray.array3d(env._get_frame(for_render=True))
frame = frame.transpose([1, 0, 2])  # 轉換為 (height, width, 3)
frames.append(frame)

while not done:

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    # 模型推論，會回傳 logits
    result = module.forward_inference({"obs": obs_tensor})

    # unbatch 動作: Tensor -> np.ndarray
    action = unbatch(result['actions'])[0].detach().cpu().numpy()
    if np.random.rand() < 0.01:
        action = np.random.randint(9)

    obs, reward, done, trunc, info = env.step(action)

    if env.step_count % 2 == 0:
        frame = pygame.surfarray.array3d(env._get_frame(for_render=True))
        frame = frame.transpose([1, 0, 2])  # 轉換為 (height, width, 3)
        frames.append(frame)

    total_rewards += reward
    done |= trunc

print("Total Rewards:", total_rewards)

ray.shutdown()

pygame.quit()

# 保存為 GIF
output_path = "diepio_simulation.gif"
imageio.mimsave(output_path, frames, fps=100, optimize=True)
print(f"GIF saved to {output_path}")
