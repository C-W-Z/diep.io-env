import ray
ray.init()
print("Ray initialized successfully")

from env import DiepIOEnvBasic
from ray.rllib.utils.pre_checks.env import check_multiagent_environments
env = DiepIOEnvBasic({"n_tanks": 2})
check_multiagent_environments(env)

print(ray.available_resources())

ray.shutdown()