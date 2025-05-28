import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from env import DiepIOEnvBasic
from ray.tune.registry import register_env

# 註冊環境
def env_creator(env_config):
    return DiepIOEnvBasic(env_config)

register_env("diepio-v0", env_creator)

# 所有 agent 使用同一個 policy（Self-Play）
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"

# 初始化 Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

# 初始化臨時環境以取得 obs/action space（僅一次）
temp_env = DiepIOEnvBasic({"n_tanks": 2})
obs_space = temp_env.observation_space
act_space = temp_env.action_space

# 設定 PPO 訓練參數
config = (
    PPOConfig()
    .environment(
        env="diepio-v0",
        env_config={
            "n_tanks": 2,
            "render_mode": False,
            "max_steps": 1000,
            "unlimited_obs": False
        }
    )
    .framework("torch")
    .env_runners(num_env_runners=1)  # ✅ 替代 rollouts(num_rollout_workers=1)
    .multi_agent(
        policies={"shared_policy": (None, obs_space, act_space, {})},
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["shared_policy"]
    )
    .training(
        train_batch_size=2000,
        gamma=0.99,
        lr=5e-4,
        model={"fcnet_hiddens": [256, 256]}
    )
)

# 啟動訓練
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        stop={"training_iteration": 100},
        name="diepio_selfplay",
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_at_end=True,             # ✅ 訓練結束儲存一次
            checkpoint_frequency=10,            # ✅ 每 10 次訓練儲存一次
            num_to_keep=3                       # ✅ 最多保留 3 個 checkpoint
        ),
        verbose=1
    )
)

results = tuner.fit()
