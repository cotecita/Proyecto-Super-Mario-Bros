# wrappers.py
"""
Wrappers de preprocesamiento para Super Mario Bros (Gym/Gymnasium + nes-py).
Salida final: observaciones en formato (C, H, W) uint8 con frame stack (C=4).
"""

from typing import Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
except Exception:
    # Fallback a gym clásico si fuese necesario
    import gym
    from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


class SkipFrame(gym.Wrapper):
    """Repite la misma acción 'skip' pasos, acumulando recompensa.
    Mantiene la API de Gymnasium: step -> (obs, reward, terminated, truncated, info)
    """
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        assert skip >= 1
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        last_obs = None
        terminated = False
        truncated = False
        info = {}

        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += float(reward)
            last_obs = obs
            terminated = term
            truncated = trunc
            if term or trunc:
                break

        return last_obs, total_reward, terminated, truncated, info


class ChannelFirst(gym.ObservationWrapper):
    """Convierte observaciones a formato (C, H, W).
    Soporta (H, W, C), (H, W, N), (N, H, W, C) y (H, W, N, 1).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        obs = np.zeros(old_shape, dtype=np.uint8)
        obs_fixed = self._fix_axes(obs)
        c, h, w = obs_fixed.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

    def _fix_axes(self, obs: np.ndarray) -> np.ndarray:
        """Reordena los ejes para obtener (C, H, W) sin importar la forma original."""
        arr = np.array(obs)
        if arr.ndim == 2:
            arr = arr[None, :, :]                         # (1, H, W)
        elif arr.ndim == 3:
            if arr.shape[2] in (1, 3):                    # (H, W, C)
                arr = np.moveaxis(arr, -1, 0)             # (C, H, W)
            else:                                          # (H, W, N)
                arr = np.moveaxis(arr, -1, 0)
        elif arr.ndim == 4:
            # puede ser (N, H, W, C) o (H, W, N, 1)
            if arr.shape[-1] in (1, 3):                   # (N, H, W, 1)
                arr = arr.squeeze(-1)                     # (N, H, W)
            arr = np.moveaxis(arr, 0, 0)                  # N ya es canal
        else:
            raise ValueError(f"Forma inesperada: {arr.shape}")
        return arr

    def observation(self, obs):
        fixed = self._fix_axes(obs)
        return fixed.astype(np.uint8, copy=False)




class EnsureUint8(gym.ObservationWrapper):
    """Asegura dtype uint8 en observaciones (por seguridad y eficiencia de memoria)."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = 0
        high = 255
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=self.observation_space.shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        if obs.dtype != np.uint8:
            obs = np.clip(obs, 0, 255).astype(np.uint8)
        return obs


def create_wrapped_env(env: gym.Env) -> gym.Env:
    """Aplica la cadena de wrappers estándar para DQN con imágenes:
    - SkipFrame(4)
    - GrayScale (mantiene 1 canal)
    - Resize a 84
    - FrameStack(4)  -> (H, W, 4)
    - ChannelFirst   -> (4, H, W)
    - EnsureUint8
    """
    # 1) Acelerar con saltos de frame
    env = SkipFrame(env, skip=4)

    # 2) Escala de grises, mantiene dimensión de canal (H, W, 1)
    env = GrayScaleObservation(env, keep_dim=True)

    # 3) Redimensionar a 84x84
    env = ResizeObservation(env, 84)

    # 4) Apilar 4 frames a lo largo del eje de canales (queda (H, W, 4))
    #    Nota: en Gymnasium, FrameStack usa eje final por defecto.
    env = FrameStack(env, num_stack=4)

    # 5) Reordenar a (C, H, W) para PyTorch
    env = ChannelFirst(env)

    # 6) Asegurar uint8
    env = EnsureUint8(env)

    return env
