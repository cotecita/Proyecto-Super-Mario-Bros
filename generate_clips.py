# generate_clips.py
"""
Script para generar clips visuales de partidas del agente entrenado.
Guarda capturas de pantalla y acciones en carpetas separadas.
"""

import os
import time
from PIL import Image
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import create_wrapped_env
from agent import MarioAgent


# ==================== CONFIGURACIÓN ====================

ENV_NAME = "SuperMarioBros-1-1-v0"
EPISODES = 3
OUTPUT_DIR = "game_clips"
MODEL_PATH = "models/2025-10-16_20-00/model_50.pt"  # cambia según el modelo

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== ENTORNO ====================

env = gym_super_mario_bros.make(
    ENV_NAME,
    apply_api_compatibility=True,
    render_mode="rgb_array"  # para capturar frames
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = create_wrapped_env(env)

# ==================== AGENTE ====================

agent = MarioAgent(
    input_shape=env.observation_space.shape,
    n_actions=env.action_space.n,
    device=DEVICE
)
agent.load(MODEL_PATH)
agent.epsilon = 0.0  # sin exploración aleatoria

print(f"🎮 Acciones disponibles ({env.action_space.n}): {SIMPLE_MOVEMENT}")
print("🎥 Generando clips...")

# ==================== BUCLE DE EPISODIOS ====================

for ep in range(1, EPISODES + 1):
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    frames = []

    while not done:
        # Acción elegida por la red
        action = agent.select_action(state)
        next_state, reward, done, trunc, info = env.step(action)
        total_reward += reward

        # Guardamos frame actual (imagen RGB)
        frame_rgb = env.render()
        frames.append(Image.fromarray(frame_rgb))

        state = next_state
        time.sleep(1/60)

    # Guardar frames como PNG
    ep_dir = os.path.join(OUTPUT_DIR, f"episode_{ep}")
    os.makedirs(ep_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(ep_dir, f"frame_{i:05d}.png"))

    print(f"Episodio {ep} terminado - Recompensa total: {total_reward:.1f}")

env.close()
