import ray
ray.init()
print("Ray initialized successfully")

from env_new import DiepIOEnvBasic
from wrappers import DiepIO_FixedOBS_Wrapper
from ray.rllib.utils.pre_checks.env import check_multiagent_environments
env = DiepIO_FixedOBS_Wrapper({
    "n_tanks": 2,
    "render_mode": False,
    "max_steps": 1000000,
    "frame_stack_size": 4,
    "skip_frames": 4,
})
check_multiagent_environments(env)

print(ray.available_resources())

ray.shutdown()
