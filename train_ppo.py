import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from env_new import DiepIOEnvBasic
from wrappers import DiepIO_FixedOBS_Wrapper
from ray.tune.registry import register_env

# Register environment
def env_creator(env_config):
    return DiepIO_FixedOBS_Wrapper(env_config)

register_env("diepio-v0", env_creator)

# Policy mapping function
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    if agent_id == "agent_0":
        return "bullet_policy"
    return "body_policy"

# Initialize Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

env_config = {
    "n_tanks": 2,
    "render_mode": False,
    "max_steps": 40000,
    "frame_stack_size": 1,
    "skip_frames": 4,
    "skill_mode": [1, 2]
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
        policies={"bullet_policy": (None, obs_space, act_space, {}), "body_policy": (None, obs_space, act_space, {})},
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["bullet_policy", "body_policy"]
    )
    .training(
        train_batch_size=512,       # ✅ 減少一次訓練的記憶體需求
        minibatch_size=64,          # ✅ 減少分割用記憶體
        gamma=0.95,
        lr=5e-4,
        entropy_coeff=0.01,
        vf_loss_coeff=0.25,
        model={
            # "fcnet_hiddens": [512, 512, 256],  # Deeper and wider network
            "fcnet_activation": "tanh",
            "use_lstm": True,
            "fcnet_hiddens": [512, 256],
            "lstm_cell_size": 256,
            "max_seq_len": 20,
            "post_fcnet_hiddens": [128]
        },
    )
)

# Start training
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        stop={"training_iteration": 1000000},
        name="diepio_fixedobs_only_move_aim_2agent",
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=50,
            num_to_keep=3,
            # checkpoint_score_attribute="episode_reward_mean",
            # checkpoint_score_order="max"
        ),
        verbose=1
    )
)

results = tuner.fit()
