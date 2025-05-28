import numpy as np
import cv2
from gymnasium import Env, Wrapper
from gymnasium.spaces import Dict, Box
from collections import deque

class DiepIO_CNN_Wrapper(Wrapper):
    def __init__(self, env: Env, resize_shape=(128, 128), frame_stack_size=4, skip_frames=4):
        super().__init__(env)
        self.resize_shape = resize_shape
        self.frame_stack_size = frame_stack_size
        self.skip_frames = skip_frames  # Number of frames to skip (apply same action)

        # Buffers for each agent
        self.frame_buffers = {agent: deque(maxlen=frame_stack_size) for agent in self.env.agents}
        self.stats_buffers = {agent: deque(maxlen=frame_stack_size) for agent in self.env.agents}

    def _process_image(self, frame):
        # Process image: grayscale and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        frame = cv2.resize(frame, self.resize_shape, interpolation=cv2.INTER_AREA)
        return frame.astype(np.uint8)

    def observation(self, observation):
        processed_obs = {}
        for agent, obs in observation.items():
            # Process image
            frame = self._process_image(obs["i"])
            self.frame_buffers[agent].append(frame)

            # Process stats
            stats = obs["s"]
            self.stats_buffers[agent].append(stats)

            # Pad buffers with zeros if needed
            while len(self.frame_buffers[agent]) < self.frame_stack_size:
                self.frame_buffers[agent].appendleft(np.zeros_like(frame))
            while len(self.stats_buffers[agent]) < self.frame_stack_size:
                self.stats_buffers[agent].appendleft(np.zeros_like(stats))

            # Stack frames and stats
            stacked_frame = np.concatenate(self.frame_buffers[agent], axis=-1)
            stacked_stats = np.concatenate(self.stats_buffers[agent], axis=0)

            processed_obs[agent] = {"i": stacked_frame, "s": stacked_stats}

        return processed_obs

    def reset(self, **kwargs):
        # Clear buffers
        for agent in self.frame_buffers:
            self.frame_buffers[agent].clear()
            self.stats_buffers[agent].clear()

        obs, info = self.env.reset(**kwargs)
        # Initialize buffers with first frame and stats
        for agent in obs:
            frame = self._process_image(obs[agent]["i"])
            stats = obs[agent]["s"]
            for _ in range(self.frame_stack_size):
                self.frame_buffers[agent].append(frame)
                self.stats_buffers[agent].append(stats)
            obs[agent]["i"] = np.concatenate(self.frame_buffers[agent], axis=-1)
            obs[agent]["s"] = np.concatenate(self.stats_buffers[agent], axis=0)

        return obs, info

    def step(self, actions):
        total_rewards = {agent: 0.0 for agent in self.env.agents}
        dones = {agent: False for agent in self.env.agents}
        truncations = {agent: False for agent in self.env.agents}
        infos = {agent: {} for agent in self.env.agents}
        dones["__all__"] = False

        for f in range(self.skip_frames):
            obs, rewards, step_dones, step_truncations, step_infos = self.env.step(actions)

            for agent in self.env.agents:
                if f > 0 and actions[agent]["d"][3] > 0:
                    actions[agent]["d"][3] = 0

                total_rewards[agent] += rewards[agent]
                dones[agent] |= step_dones[agent]
                truncations[agent] |= step_truncations[agent]
                # infos[agent].update(step_infos[agent])

            if step_dones["__all__"]:
                dones["__all__"] = True
                break

        processed_obs = self.observation(obs)

        return processed_obs, total_rewards, dones, truncations, infos
