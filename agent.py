# guardar experiencias, decide cuánto explorar, guardar pesos de la red
#define cómo aprende mario y decide qué hacer
from collections import deque, namedtuple
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent_nn import MarioNet #red neuronal



Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"]) #estructura de cada experiencia que se guarda

#almacenamiento de experiencias: todas las acciones que realiza con su respectiva recompensa, siguiente estado, ..........
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)  #guarda un número límite experiencias, se va actualizando

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))   #guardar/añadir una trancisión al buffer

    def sample(self, batch_size: int): #usa un conjunt aleatorio de experiencias para entrenar la red
        batch = random.sample(self.buffer, batch_size)
        states      = np.stack([b.state for b in batch], axis=0)          # (B, C, H, W) uint8, convierte imagenes en un tensor
        actions     = np.array([b.action for b in batch], dtype=np.int64) # (B,)
        rewards     = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.stack([b.next_state for b in batch], axis=0)
        dones       = np.array([b.done for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

#establecer hiper parametros del agente
class MarioAgent:
    def __init__(
        self,
        input_shape,
        n_actions,
        device=torch.device("cpu"),
        # Hiperparámetros 
        gamma=0.99, #importancia de la recompensa futura
        lr=1e-4,       #learning rate
        batch_size=64,     #núm de experiencias para entrenar por paso
        memory_size=100_000,     #tamaño de replay_buffer
        warmup=10_000,        #pasos necesarios antes de comenzar a entrenar
        learn_every=4,        #pasos a esperar entre cada train, durante esos pasos no entrena
        target_sync=1_000,        #cada ese num de pasos se copia los pesos de la online network (red que se entrena)a target network (copia de la online para calcular Q objetivo)
        #controlar nivel de exploración
        eps_start=1.0,        
        eps_end=0.05,
        eps_decay_steps=300_000,
        grad_clip=10.0,
    ):
        #se inicializan parámetros que permiten determinar pasos totales y aprendizaje en pasos transcurridos
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.learn_every = learn_every
        self.target_sync = target_sync
        self.grad_clip = grad_clip

        # Epsilon lineal en su decay 
        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = max(1, eps_decay_steps)
        self.total_steps = 0
        self.learn_step_counter = 0

        # Redes neuronales
        self.online = MarioNet(input_shape, n_actions).to(self.device) #red que se entrena continuamente
        self.target = MarioNet(input_shape, n_actions).to(self.device)  #red que es copia de la online pero cada cierto num de pasos y se usa para calcular Q target y equilibrar el train
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval() #

        # Optimizador y criterio
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr) #optimiza los pesos de la red online
        self.criterion = nn.SmoothL1Loss()  # Huber, evita oscilaciones en gradientes

        # Memoria
        self.replay_buffer = ReplayBuffer(memory_size)

        # Modo train por defecto
        self.online.train()

    # ----------------- Interacción -----------------

    def select_action(self, state):
        self.total_steps += 1
        # Actualiza epsilon (decaimiento lineal)
        fraction = min(1.0, self.total_steps / self.eps_decay_steps)
        self.epsilon = self.eps_start + fraction * (self.eps_end - self.eps_start)

        if random.random() < self.epsilon:     #con probabilidad eps elige accion aleatoria
            return random.randrange(self.n_actions)  

        with torch.no_grad():   #con probabilidad 1-epsilon, elige la acción con mayor Q-value.
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)  
            q = self.online(s.float())  # normaliza dentro de la red
            return int(q.argmax(dim=1).item())

    #guardar experiencias
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ----------------- Aprendizaje -----------------
    #aprende si hay suficientes experiencias en el buffer y si esta en el paso correspondiente segun lear_every
    def _can_learn(self):
        if len(self.replay_buffer) < self.warmup:
            return False
        return (self.total_steps % self.learn_every) == 0

    def learn(self):
        if not self._can_learn():
            return
        #lo que toma: experiencias
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Tensores: convierte experienias a tensores para entrenar la red
        states_t = torch.from_numpy(states).to(self.device).float()          
        next_t   = torch.from_numpy(next_states).to(self.device).float()
        actions_t = torch.from_numpy(actions).to(self.device).long().unsqueeze(1)  
        rewards_t = torch.from_numpy(rewards).to(self.device).float().unsqueeze(1)
        dones_t   = torch.from_numpy(dones).to(self.device).float().unsqueeze(1)

        # Q(s,a) actual
        q_pred = self.online(states_t).gather(1, actions_t) 

        # Q target usando red target: y = r + gamma*(1-done)*max_a' Q_target(s',a')
        with torch.no_grad():
            q_next = self.target(next_t)                      
            max_q_next, _ = q_next.max(dim=1, keepdim=True)  
            q_target = rewards_t + self.gamma * (1.0 - dones_t) * max_q_next

        #calcula la pérdida y hace backpropagation
        loss = self.criterion(q_pred, q_target) 

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        self.learn_step_counter += 1

        # Sincronización periódica de la red target
        if self.learn_step_counter % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

    # Guardar pesos de la red-modelos para continuar entrenando despues
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
                "learn_step_counter": self.learn_step_counter,
            },
            filepath,
        )

    def load(self, filepath, map_location=None):
        ckpt = torch.load(filepath, map_location=map_location or self.device)
        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = float(ckpt.get("epsilon", 0.05))
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.learn_step_counter = int(ckpt.get("learn_step_counter", 0))
        self.online.to(self.device).train()
        self.target.to(self.device).eval()
