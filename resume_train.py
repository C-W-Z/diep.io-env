import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from diepio.env_new import DiepIOEnvBasic
from wrappers import DiepIO_FixedOBS_Wrapper
from ray.tune.registry import register_env

# Register environment
def env_creator(env_config):
    return DiepIO_FixedOBS_Wrapper(env_config)

register_env("diepio-v0", env_creator)

# Policy mapping function
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return "shared_policy"

# Initialize Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

env_config = {
    "n_tanks": 1,
    "render_mode": False,
    "max_steps": 40000,
    "frame_stack_size": 1,
    "skip_frames": 4,
}

# Get observation and action spaces
temp_env = env_creator(env_config)
obs_space = temp_env.observation_space
act_space = temp_env.action_space
temp_env.close()

print("Observation space:", obs_space)
print("Action space:", act_space)

# Configure PPO
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True
    )
    .environment(
        env="diepio-v0",
        env_config=env_config
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
        minibatch_size=64,         # ✅ 減少分割用記憶體
        gamma=0.99,
        lr=1e-4,
        model={
            # "fcnet_hiddens": [512, 512, 256],  # Deeper and wider network
            "fcnet_activation": "tanh",
            "use_lstm": True,
            "fcnet_hiddens": [256, 256],
            "lstm_cell_size": 256,
            "max_seq_len": 16,
        }
    )
)

# Start training
tuner = tune.Tuner.restore(
    path="~/ray_results/diepio_fixedobs_selfplay",  # 先前的 Tuner 輸出目錄
    trainable="PPO",
    resume_unfinished=True,        # ✅ 重新開始沒跑完的 trial
    restart_errored=True,          # ✅ 自動重跑有錯的 trial
    resume_errored=True            # ✅ 繼續上次出錯的
)

results = tuner.fit()
