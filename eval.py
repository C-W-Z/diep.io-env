import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from env_new import DiepIOEnvBasic
from wrappers import DiepIO_FixedOBS_Wrapper
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.spaces.space_utils import unbatch

from ray.tune.registry import register_env

# Register environment
def env_creator(env_config):
    return DiepIO_FixedOBS_Wrapper(env_config)

register_env("diepio-v0", env_creator)

# 初始化 Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

# 你的 checkpoint 路徑，例如：
checkpoint_path = "~/ray_results/diepio_fixedobs_only_move_aim_2agent/PPO_diepio-v0_52eea_00000_0_2025-05-31_11-42-49/checkpoint_000001"

# 載入訓練好的 policy
algo = Algorithm.from_checkpoint(checkpoint_path)
module = {}
module["agent_0"] = algo.get_module("bullet_policy")
module["agent_1"] = algo.get_module("body_policy")

env_config = {
    "n_tanks": 2,
    "render_mode": "human",
    "max_steps": 40000,
    "frame_stack_size": 1,
    "skip_frames": 4,
    "skill_mode": [1, 2]
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
        sample_dict = unbatch(sample)[0]
        sample_dict = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in sample_dict.items()
        }
        actions[agent_id] = sample_dict

    obs, rewards, done, trunc, infos = env.step(actions)

    for agent, reward in rewards.items():
        total_rewards[agent] += reward

    done["__all__"] |= trunc["__all__"]

print("Total Rewards:", total_rewards)
