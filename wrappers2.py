# wrappers.py - VERSIÓN MEJORADA
import gym
import numpy as np
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

class SkipFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done or trunc:
                break
        return obs, total_reward, done, trunc, info

class ChannelFirst(gym.ObservationWrapper):
    """SIMPLIFICADO - Sabemos que viene (H, W, 4) de FrameStack"""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Input: (84, 84, 4) -> Output: (4, 84, 84)
        h, w, c = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8
        )

    def observation(self, obs):
        #SIMPLE: (H, W, C) → (C, H, W)
        return np.moveaxis(obs, -1, 0)

class EnsureUint8(gym.ObservationWrapper):
    """Mantener - es útil"""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.observation_space.shape, dtype=np.uint8
        )

    def observation(self, obs):
        return obs.astype(np.uint8) if obs.dtype != np.uint8 else obs

def create_wrapped_env(env: gym.Env) -> gym.Env:
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)  # (H, W, 1)
    env = ResizeObservation(env, 84)                # (84, 84, 1)  
    env = FrameStack(env, num_stack=4)              # (84, 84, 4)
    env = ChannelFirst(env)                         # (4, 84, 84)
    env = EnsureUint8(env)                          # (4, 84, 84) uint8
    return env