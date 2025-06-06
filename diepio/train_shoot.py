# train_shoot.py

import os
import ray
from ray import tune
from ray.tune.registry import register_env
import json
import time
from tqdm import tqdm
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from .env_shoot import DiepIOEnvBasic

# 抑制 Ray 本身的 INFO / WARNING 訊息
import logging
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
# 抑制 Python 的 deprecation warnings（例如 RayDeprecationWarning）
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def env_creator(env_config):
    """
    環境建立函式。RLlib 在註冊時會呼叫這個函式來建立環境實例。
    env_config 會透過 RLlib 的 config["env_config"] 傳入，像是 n_tanks、render_mode、max_steps 等參數。
    """
    return DiepIOEnvBasic(env_config)

class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, episode, env_runner, metrics_logger, env_index, **kwargs):
        # 取得該回合的單一 reward 序列
        # rewards_list = episode.rewards  # List[float]
        # last_reward = episode.rewards[-1]  # 取得最新 timestep 的獎勵
        # 或使用 episode.get_rewards(-1)
        # print(episode.get_rewards())
        last_reward = float(sum(episode.get_rewards()['agent_0']))
        # print(last_reward)
        with open("output.txt", "a") as file:
            file.write(f"Episode {i}\tReward {last_reward:.2f}\tTime {time.time() - start_time}\n")
            file.flush()


if __name__ == "__main__":
    # 2. 初始化 Ray（確保你已經用 conda/venv 切到正確的 env）
    ray.init()

    # 3. 註冊環境，命名為 "diepio"
    register_env("diepio", env_creator)

    # 4. 建立一個臨時 env 物件（n_tanks=1），以取得 observation_space 和 action_space
    temp_env = DiepIOEnvBasic({
        "n_tanks": 1,
        "render_mode": False,
        "max_steps": 7200,
        "unlimited_obs": False
    })
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    # 5. PPO 的設定
    config = {
        "env": "diepio",    # 指到剛剛 register 的環境名稱
        "env_config": {
            "n_tanks": 1,
            "render_mode": False,
            "max_steps": 7200,
            "unlimited_obs": False,
        },
        "framework": "torch",   # 或 "tf"
        "num_gpus": 1,
        "num_workers": 0,       # 單機的情況下可設 0，全部都在 driver 上跑
        # PPO 的超參數，可依需求調整：
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "lr": 1e-3,
        "gamma": 0.99,
        # Multi-agent 設定（雖然只跑一個 agent，但因 DiepIOEnvBasic 繼承 MultiAgentEnv，須使用此介面）
        "multiagent": {
            "policies": {
                # policy_id -> (policy_class, obs_space, act_space, config)
                "policy_0": (None, obs_space, act_space, {}),
            },
            # policy_mapping_fn: 把 agent_id（如 "agent_0"）對到 "policy_0"
            "policy_mapping_fn": lambda *args, **kwargs: "policy_0",
        },
        # 訓練結果、checkpoint 等資料會存在這裡
        "local_dir": os.path.join(os.getcwd(), "rllib_results"),
        "log_level": "ERROR",
        "model": {
            "fcnet_hiddens": [512, 512],
        },
        "callbacks": MyCallbacks,
    }

    # 6. 建立 PPO 演算法物件（新版 API 用 PPO 而非 PPOTrainer）
    trainer = PPO(config=config)

    # 7. 訓練迴圈：跑 1000 次迭代 (可自行調整)
    start_time = time.time()
    max_iterations = 2000
    file = open("output.txt", "w")
    file.close()
    pbar = tqdm(range(1, max_iterations + 1))
    for i in pbar:
        result = trainer.train()
        # print(result['env_runners'].keys())
        # file.write(f"Average {result['env_runners']['episode_return_mean']}\n")
        # file.flush()

        # 每 10 次迭代存一次 checkpoint
        if i % 10 == 0:
            checkpoint_path = trainer.save()
            pbar.set_description(f"Iteration {i}, Checkpoint saved")

    # 8. 訓練結束，關閉 Ray
    ray.shutdown()
