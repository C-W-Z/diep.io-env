import numpy as np
import cv2
from gymnasium import Env, Wrapper, spaces
from collections import deque
from ray.rllib.env import MultiAgentEnv
from typing import Any

class DiepIO_CNN_Wrapper(Wrapper):
    def __init__(self, env: Env, resize_shape=(100, 100), frame_stack_size=4, skip_frames=4):
        super().__init__(env)
        self.resize_shape = resize_shape
        self.frame_stack_size = frame_stack_size
        self.skip_frames = skip_frames  # Number of frames to skip (apply same action)

        self.action_space = self.env.action_space
        self.action_spaces = self.env.action_spaces
        self.observation_space = spaces.Dict({
            "i": spaces.Box(    # image
                low=0,
                high=1,
                shape=(resize_shape[0], resize_shape[1], 3 * frame_stack_size),
                dtype=np.float32
            ),
            "s": spaces.Box(    # stats
                low=0,
                high=1,
                shape=(11, frame_stack_size),
                dtype=np.float32
            )  # HP/maxHP, Level, skill points, 8 stats
        })
        self.observation_spaces = {
            agent: self.observation_space for agent in self.env._agent_ids
        }

        # Stats normalization bounds
        self.stats_low = np.array([0, 1, 0] + [0] * 8)
        self.stats_scale = np.array([1, 45, 33] + [7] * 8) - self.stats_low

        self.possible_agents = self.env.possible_agents
        self.agents = self.env.agents

        # Buffers for each agent
        self.frame_buffers = {agent: deque(maxlen=frame_stack_size) for agent in self.env._agent_ids}
        self.stats_buffers = {agent: deque(maxlen=frame_stack_size) for agent in self.env._agent_ids}

    def _process_image(self, frame):
        frame = cv2.resize(frame, self.resize_shape, interpolation=cv2.INTER_AREA)  # Keep RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("test.jpg", np.transpose(frame, (1, 0, 2)))
        frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return frame

    def _process_stats(self, stats):
        normalized_stats = (stats - self.stats_low) / self.stats_scale
        return normalized_stats.astype(np.float32)[:, np.newaxis]

    def observation(self, observations):
        processed_obs = {}
        for agent, obs in observations.items():
            assert len(obs.keys()) == 2, f"obs[{agent}] keys: {obs.keys()}"

            # Process image
            frame = self._process_image(obs["i"])
            self.frame_buffers[agent].append(frame)

            # Process stats
            stats = self._process_stats(obs["s"])
            self.stats_buffers[agent].append(stats)

            # Stack frames and stats
            stacked_frame = np.concatenate(self.frame_buffers[agent], axis=-1)
            stacked_stats = np.concatenate(self.stats_buffers[agent], axis=-1)

            processed_obs[agent] = {"i": stacked_frame, "s": stacked_stats}

        return processed_obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.agents = self.env.agents

        for agent in self.env._agent_ids:
            # Clear buffers
            self.frame_buffers[agent].clear()
            self.stats_buffers[agent].clear()
            # Initialize buffers with first frame and stats
            frame = self._process_image(obs[agent]["i"])
            stats = self._process_stats(obs[agent]["s"])
            for _ in range(self.frame_stack_size):
                self.frame_buffers[agent].append(frame)
                self.stats_buffers[agent].append(stats)
            obs[agent]["i"] = np.concatenate(self.frame_buffers[agent], axis=-1)
            obs[agent]["s"] = np.concatenate(self.stats_buffers[agent], axis=-1)

        return obs, info

    def step(self, actions):
        total_rewards    = {agent: 0.0 for agent in self.env._agent_ids}
        dones            = {agent: False for agent in self.env._agent_ids}
        truncations      = {agent: False for agent in self.env._agent_ids}
        infos            = {agent: {} for agent in self.env._agent_ids}
        dones["__all__"] = False

        for f in range(self.skip_frames):
            obs, rewards, dones, truncations, infos = self.env.step(actions, skip_frame=(f < self.skip_frames - 1))

            for agent in self.env._agent_ids:
                if f > 0: # skill point should be use only once in skip_frames
                    actions[agent]["d"][3] = 0

                total_rewards[agent] += rewards[agent]

            if dones["__all__"]:
                dones["__all__"] = True
                if f < self.skip_frames - 1:
                    obs = {}
                    for agent_idx, agent in enumerate(self.env._agent_ids):
                        obs[agent] = self.env._get_obs(agent_idx)
                break

        processed_obs = self.observation(obs)

        self.agents = self.env.agents

        return processed_obs, total_rewards, dones, truncations, infos

class DiepIO_FixedOBS_Wrapper(MultiAgentEnv):
    def __init__(self, env_config: dict[str, Any] = {}):
        super(DiepIO_FixedOBS_Wrapper, self).__init__()

        from env_new import DiepIOEnvBasic

        self.env = DiepIOEnvBasic(env_config)

        self.frame_stack_size = env_config.get("frame_stack_size", 4)
        self.skip_frames = env_config.get("skip_frames", 4)  # Number of frames to skip (apply same action)

        self.skill_mode = env_config.get("skill_mode", [0 for _ in self.env._agent_ids])

        self.action_map = {
            0: [1, 1],  # NONE
            1: [1, 2],  # W
            2: [0, 1],  # A
            3: [1, 0],  # S
            4: [2, 1],  # D
            5: [0, 2],  # W+A
            6: [2, 2],  # W+D
            7: [0, 0],  # S+A
            8: [2, 0]   # S+D
        }

        self.action_space = spaces.Dict({
            "d": spaces.Discrete(9),                                       # discrete
            # "c": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32) # continuous
            "c": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # continuous
        })
        self.action_spaces =  {
            agent: self.action_space for agent in self.env._agent_ids
        }

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.env.observation_space.shape[0] * self.frame_stack_size, ),
            dtype=np.float32
        )
        self.observation_spaces = {
            agent: self.observation_space for agent in self.env._agent_ids
        }

        # Stats normalization bounds
        self.obs_low = self.env.observation_space.low
        self.obs_scale = self.env.observation_space.high - self.env.observation_space.low

        self._agent_ids = self.env._agent_ids
        self.possible_agents = self.env.possible_agents
        self.agents = self.env.agents

        # Buffers for each agent
        self.frame_buffers = {agent: deque(maxlen=self.frame_stack_size) for agent in self.env._agent_ids}

    def observation(self, observations):
        processed_obs = {}
        for agent, obs in observations.items():
            # Normalize obs
            normalized_obs = (obs - self.obs_low) / self.obs_scale

            if self.frame_stack_size > 1:
                self.frame_buffers[agent].append(normalized_obs)
                # Stack obs
                processed_obs[agent] = np.concatenate(self.frame_buffers[agent], axis=0)
            else:
                processed_obs[agent] = normalized_obs

        return processed_obs

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=None, options=None)

        # self.agents = self.env.agents

        for agent in self.env._agent_ids:
            # Clear buffers
            self.frame_buffers[agent].clear()

            # Initialize buffers with first obs
            normalized_obs = (obs[agent] - self.obs_low) / self.obs_scale

            if self.frame_stack_size > 1:
                for _ in range(self.frame_stack_size):
                    self.frame_buffers[agent].append(normalized_obs)

                obs[agent] = np.concatenate(self.frame_buffers[agent], axis=0)
            else:
                obs[agent] = normalized_obs

        return obs, info

    def step(self, actions):
        total_rewards    = {agent: 0.0 for agent in self.env._agent_ids}
        dones            = {agent: False for agent in self.env._agent_ids}
        truncations      = {agent: False for agent in self.env._agent_ids}
        infos            = {agent: {} for agent in self.env._agent_ids}
        dones["__all__"] = False
        truncations["__all__"] = False

        for i, agent in enumerate(self.env._agent_ids):
            skill_index = self.env._auto_choose_skill(agent, mode=self.skill_mode[i])

            move = self.action_map[actions[agent]["d"]]
            actions[agent]["d"] = np.concatenate((move, np.array([1, skill_index])))
            # angle = actions[agent]["c"]
            # actions[agent]["c"] = [np.cos(angle), np.sin(angle)]

        for f in range(self.skip_frames):
            obs, rewards, dones, truncations, infos = self.env.step(actions, skip_frame=(f < self.skip_frames - 1))

            for agent in self.env._agent_ids:
                if f > 0: # skill point should be use only once in skip_frames
                    actions[agent]["d"][3] = 0

                total_rewards[agent] += rewards[agent]

            if dones["__all__"]:
                if f < self.skip_frames - 1:
                    obs = {}
                    for agent_idx, agent in enumerate(self.env._agent_ids):
                        obs[agent] = self.env._get_obs(agent_idx)
                break

        processed_obs = self.observation(obs)

        # self.agents = self.env.agents

        return processed_obs, total_rewards, dones, truncations, infos

    def close(self):
        self.env.close()
        return super().close()

def test_cnn_wrapper():
    from env_cnn import DiepIOEnvBasic
    from utils import check_obs_in_space

    env_config = {
        "n_tanks": 2,
        "render_mode": "True",
        "max_steps": 1000000,
        "unlimited_obs": False
    }

    env = DiepIOEnvBasic(env_config)
    env = DiepIO_CNN_Wrapper(env)

    obs, _ = env.reset()
    print(env.observation_space)
    print(env.action_space["d"].nvec.sum(), env.action_space["c"].shape)
    for i in range(2):
        for key in obs[f"agent_{i}"].keys():
            check_obs_in_space(obs[f"agent_{i}"][key], env.observation_spaces[f"agent_{i}"][key])

    while True:
        obs, rewards, dones, truncations, infos = env.step({
            "agent_0": env.env._get_player_input(),
            "agent_1": env.env._get_random_input(),
        })
        for i in range(2):
            for key in obs[f"agent_{i}"].keys():
                check_obs_in_space(obs[f"agent_{i}"][key], env.observation_spaces[f"agent_{i}"][key])

        if dones["__all__"]:
            break
    env.close()

def test_fixedobs_wrapper():
    from utils import check_obs_in_space

    env_config = {
        "n_tanks": 1,
        "render_mode": "human",
        "max_steps": 1000000,
        "frame_stack_size": 1,
        "skip_frames": 4,
        "skill_mode": [1, 2]
    }

    env = DiepIO_FixedOBS_Wrapper(env_config)

    print(env.observation_space)
    print(env.action_space["d"], env.action_space["c"].shape)

    obs, _ = env.reset()

    for i in range(env.env.n_tanks):
        check_obs_in_space(obs[f"agent_{i}"], env.observation_spaces[f"agent_{i}"])

    dx_dy_map = {
        (1, 1): 0,  # NONE
        (1, 2): 1,  # W
        (0, 1): 2,  # A
        (1, 0): 3,  # S
        (2, 1): 4,  # D
        (0, 2): 5,  # W+A
        (2, 2): 6,  # W+D
        (0, 0): 7,  # S+A
        (2, 0): 8   # S+D
    }

    while True:
        action_0 = env.env._get_player_input()
        action_0["d"] = dx_dy_map[tuple(action_0["d"][:2])]
        # action_0["c"] = np.arctan2(action_0["c"][1], action_0["c"][0])
        # action_1 = env.env._get_random_input()
        # action_1["d"] = action_1["d"][:2]
        obs, rewards, dones, truncations, infos = env.step({
            "agent_0": action_0,
            # "agent_1": action_1,
        })
        for i in range(env.env.n_tanks):
            check_obs_in_space(obs[f"agent_{i}"], env.observation_spaces[f"agent_{i}"])

        if dones["__all__"]:
            break
    print(env.env.step_count)
    env.close()

if __name__ == "__main__":
    test_fixedobs_wrapper()
