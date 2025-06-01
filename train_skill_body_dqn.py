import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from env_skill_body import DiepIOEnvSkillBody
from ray.tune.registry import register_env

# Register environment
def env_creator(env_config):
    return DiepIOEnvSkillBody(env_config)

register_env("diepio-v0", env_creator)

# Policy mapping function
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return "skill_bodyt_policy"

# Initialize Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

env_config = {
    "n_tanks": 1,
    "render_mode": True,
    "max_steps": 5000,
    "skill_mode": [2],
    "skip_frames": 10,
}

# Get observation and action spaces
temp_env = env_creator(env_config)
obs_space = temp_env.observation_space
act_space = temp_env.action_space
temp_env.close()

print("Observation space:", obs_space)
print("Action space:", act_space)

# Configure
config = (
    DQNConfig()
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
        sample_timeout_s=120.0,            # ✅ 放寬 timeout，避免出錯
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=0.0,
        num_cpus_per_learner=1
    )
    .training(
        train_batch_size=64,       # ✅ 減少一次訓練的記憶體需求
        gamma=0.99,
        lr=5e-4,
        num_atoms=51,
        v_min=-10,
        v_max=100,
        n_step=5,
        noisy=True,
        epsilon=[(0, 0.25), (100, 0.01), (500, 0.01)],
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="relu",
            # use_lstm=True,
            # lstm_cell_size=256,
            # max_seq_len=20,
        )
    )
)

# Start training
tuner = tune.Tuner(
    "DQN",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        stop={"training_iteration": 1000000},
        name="diepio_body_onlyskill_dqn",
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=50,
            num_to_keep=10,
            # checkpoint_score_attribute="episode_reward_mean",
            # checkpoint_score_order="max"
        ),
        verbose=1
    )
)

results = tuner.fit()
