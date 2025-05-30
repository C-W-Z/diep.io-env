import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from env_new import DiepIOEnvBasic
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.spaces.space_utils import unbatch

from ray.tune.registry import register_env

# Register environment
def env_creator(env_config):
    return DiepIOEnvBasic(env_config)

register_env("diepio-v0", env_creator)

# 初始化 Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

# 你的 checkpoint 路徑，例如：
checkpoint_path = "~/ray_results/diepio_fixedobs_selfplay/PPO_diepio-v0_c8bb3_00000_0_2025-05-30_22-10-04/checkpoint_000010"

# 載入訓練好的 policy
algo = Algorithm.from_checkpoint(checkpoint_path)
module = algo.get_module("shared_policy")

# 設定 render 模式的環境
env = DiepIOEnvBasic({
    "n_tanks": 1,
    "render_mode": True,
    "max_steps": 1000000,
    "unlimited_obs": False
})

obs, _ = env.reset()

done = {"__all__": False}

total_rewards = {agent: 0.0 for agent in env._agent_ids}

while not done["__all__"]:
    actions = {}
    for agent_id, agent_obs in obs.items():
        obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)

        # 模型推論，會回傳 logits
        result = module.forward_inference({"obs": obs_tensor})
        dist_inputs = result["action_dist_inputs"]

        # 獲取 distribution class
        dist_class, _ = ModelCatalog.get_action_dist(env.action_space, config={}, framework="torch")

        # 構建 distribution，抽樣動作
        dist = dist_class(dist_inputs, env.action_space)
        sample = dist.sample()

        # unbatch 動作: TensorDict -> Dict[str, np.ndarray]
        sample_dict = unbatch(sample)[0]
        sample_dict = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in sample_dict.items()
        }
        actions[agent_id] = sample_dict

    obs, rewards, done, trunc, infos = env.step(actions)

    for agent, reward in rewards.items():
        total_rewards[agent] += reward

print("Total Rewards:", total_rewards)
