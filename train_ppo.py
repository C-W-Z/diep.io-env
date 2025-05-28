import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from env import DiepIOEnvBasic
from ray.tune.registry import register_env

# Register environment
def env_creator(env_config):
    return DiepIOEnvBasic(env_config)

register_env("diepio-v0", env_creator)

# Policy mapping function
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return "shared_policy"

# Initialize Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

# Get observation and action spaces
temp_env = DiepIOEnvBasic({"n_tanks": 2})
obs_space = temp_env.observation_space
act_space = temp_env.action_space

# Configure PPO
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True
    )
    .environment(
        env="diepio-v0",
        env_config={
            "n_tanks": 2,
            "render_mode": False,
            "max_steps": 1000000,
            "unlimited_obs": False
        }
    )
    .framework("torch")
    .resources(
        num_gpus=1,
        num_cpus_for_main_process=1, # 增加 CPU 以處理主進程
    )
    .env_runners(
        num_env_runners=1,
        num_cpus_per_env_runner=1,         # ✅ 降低 CPU 代表降低並行度 => 減少 RAM 壓力
        num_gpus_per_env_runner=0.0,
        remote_worker_envs=False,          # ✅ 改回預設，讓環境內建在主進程 => 避免多 process
        sample_timeout_s=120.0             # ✅ 放寬 timeout，避免出錯
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=0.0,
        num_cpus_per_learner=1
    )
    .multi_agent(
        policies={"shared_policy": (None, obs_space, act_space, {})},
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["shared_policy"]
    )
    .training(
        train_batch_size=512,       # ✅ 減少一次訓練的記憶體需求
        minibatch_size=128,         # ✅ 減少分割用記憶體
        gamma=0.99,
        lr=5e-4,
        model={"fcnet_hiddens": [256, 256]}
    )
)

# Start training
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        stop={"training_iteration": 100000},
        name="diepio_selfplay",
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=10,
            num_to_keep=3,
            # checkpoint_score_attribute="episode_reward_mean",
            # checkpoint_score_order="max"
        ),
        verbose=1
    )
)

results = tuner.fit()
