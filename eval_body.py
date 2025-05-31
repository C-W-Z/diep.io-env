import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from env_body import DiepIOEnvBody
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.spaces.space_utils import unbatch

from ray.tune.registry import register_env

# Register environment
def env_creator(env_config):
    return DiepIOEnvBody(env_config)

register_env("diepio-v0", env_creator)

# 初始化 Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

# 你的 checkpoint 路徑，例如：
checkpoint_path = "~/ray_results/diepio_body_onlymove/checkpoint_000007"

# 載入訓練好的 policy
algo = Algorithm.from_checkpoint(checkpoint_path)
module = {}
module["agent_0"] = algo.get_module("body_policy")

env_config = {
    "n_tanks": 1,
    "render_mode": True,
    "max_steps": 5000,
    "skill_mode": [2]
}

# 設定 render 模式的環境
env = env_creator(env_config)

obs, _ = env.reset()

done = {"__all__": False}

total_rewards = {agent: 0.0 for agent in env._agent_ids}

while not done["__all__"]:
    actions = {}
    for agent_id, agent_obs in obs.items():
        obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)

        # 模型推論，會回傳 logits
        result = module[agent_id].forward_inference({"obs": obs_tensor})
        dist_inputs = result["action_dist_inputs"]

        # 獲取 distribution class
        dist_class, _ = ModelCatalog.get_action_dist(env.action_space, config={}, framework="torch")

        # 構建 distribution，抽樣動作
        dist = dist_class(dist_inputs, env.action_space)
        sample = dist.sample()

        # unbatch 動作: TensorDict -> Dict[str, np.ndarray]
        sample = unbatch(sample)[0].detach().cpu().numpy()
        actions[agent_id] = sample

    obs, rewards, done, trunc, infos = env.step(actions)

    for agent, reward in rewards.items():
        total_rewards[agent] += reward

    done["__all__"] |= trunc["__all__"]

print("Total Rewards:", total_rewards)

ray.shutdown()
