#librerías necesarias
import os
import time
import torch #aprendizaje profundo
import gym_super_mario_bros #crear entorno
from nes_py.wrappers import JoypadSpace  #restringir movimientos
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT #movimientos que puede usar el agente

from wrappers import create_wrapped_env #función para preprocesar imagenes
from agent import MarioAgent #donde esta el agente DQN
from utils import timestamp #para guardar carpetas con modelos

#  CONFIGURACIÓN INICIAL

ENV_NAME = "SuperMarioBros-1-1-v0" #nivel a usar
TRAIN_MODE = True            # True: agente aprende, False: juega en base a lo aprendido
TOTAL_EPISODES = 1000  #número de episodios para el train
SAVE_INTERVAL = 100    #cada cuantos episodios se guardan los modelos
DISPLAY = True               # True para ver el juego
RENDER_SPEED = 1/60          # controla la velocidad de los frames (~60 FPS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #usar GPU de estar disponible, sino CPU
print(f"Usando dispositivo: {DEVICE}") #innecesario???????

#CREACIÓN DEL ENTORNO 

env = gym_super_mario_bros.make(   #crear entorno del nivel 1-1
    ENV_NAME,
    apply_api_compatibility=True,
    render_mode="human" if DISPLAY else "rgb_array"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT) #solo se usan los movimientos en SIMPLE_MOVEMENT
env = create_wrapped_env(env) #ajustar frames para la red-agente

print(f"Acciones disponibles ({env.action_space.n}): {SIMPLE_MOVEMENT}") #acciones que va a usar, A es salto, B correr con derecha

# INICIAR AGENTE 

agent = MarioAgent(  
    input_shape=env.observation_space.shape,       #imagenes que ve
    n_actions=env.action_space.n,        #acciones que puede realizar
    device=DEVICE                     #recurso que usa
)

# Si quiere aplicar lo aprendido (JUGAR)
if not TRAIN_MODE:
    ckpt_dir = "models/2025-10-16_20-00"     #definircarpeta e que esta el modelo
    ckpt_file = "model_50.pt"         #definir modelo
    agent.load(os.path.join(ckpt_dir, ckpt_file))     #cargar modelo       
    agent.epsilon = 0.0  # sin exploración


#  LOOP PRINCIPAL 
model_dir = os.path.join("models", timestamp()) #cómo se llama carpeta para guardar modelos del entrenamiento actual
os.makedirs(model_dir, exist_ok=True)             #crear

for ep in range(1, TOTAL_EPISODES + 1):         #resetear entorno luego de un episodio
    state, _ = env.reset()    #estado inicial
    done = False          #episodio en ejecución, es true cuando mario muere o completa el nivel         
    total_reward = 0.0      #recompensa acumulada
    steps = 0

    #bucle por cada frame del juego
    while not done: #mientras no muera/complete el nivel
        # se muestra el juego si display es True
        if DISPLAY:
            env.render()
            time.sleep(RENDER_SPEED)  # regula la velocidad visual

        # decisión y aprendizaje 
        #DECISIÓN
        action = agent.select_action(state)  #elige movimiento
        next_state, reward, done, trunc, info = env.step(action) #por cada paso retorna siguiente_imagen;recompensa por el paso;si muere/completa nivel; truncamiento del ep;indormación del entorno adicional
        total_reward += reward #recompensa acumulada
        #APRENDIZAJE
        if TRAIN_MODE:
            agent.remember(state, action, reward, next_state, done) #guarda la transición
            agent.learn() #entrena la red

        state = next_state #"Avanza"
        steps += 1

    # resumen del episodio (estadísticas)
    print(f"[Ep {ep:03d}] Recompensa: {total_reward:7.1f} | Pasos: {steps:4d} | Eps: {agent.epsilon:.3f}")

    #  guardado periódico del modelo por cada SAVE_INTERNAL
    if TRAIN_MODE and ep % SAVE_INTERVAL == 0:
        ckpt = os.path.join(model_dir, f"model_{ep}.pt")
        agent.save(ckpt)
        print(f" Modelo guardado en {ckpt}")

env.close()   #cerrar entorno
