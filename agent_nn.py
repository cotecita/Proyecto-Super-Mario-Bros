#red neuronal

import torch #libreria para redes neuronales
import torch.nn as nn
import torch.nn.functional as F


class MarioNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        # Aseguramos formato (C, H, W), c = número de canales, H: altura de la imagen, W = ancho
        if len(input_shape) == 3:
            c, h, w = input_shape
        elif len(input_shape) == 4:
            # A veces viene (H, W, N, 1) o similar, n:num de frames
            h, w, n, c_ = input_shape
            c = n * c_
        elif len(input_shape) == 2:
            # Caso raro: (H, W) sin canal → asumimos 1 canal
            h, w = input_shape
            c = 1
        else:
            raise ValueError(f"Forma de entrada inesperada: {input_shape}")

        # Debug para ver realmente qué dimensiones llegan
        print(f"[DEBUG] Input shape recibido por la red: (C={c}, H={h}, W={w})")

        # --- Bloques convolucionales ---
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_output((c, h, w))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_output(self, shape):
        #Pasa un tensor ficticio por la red conv para calcular la dimensión final
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
            return int(torch.prod(torch.tensor(o.shape[1:])))

    def forward(self, x):
        #Propagación hacia adelante
        x = x / 255.0  # normalizamos a [0,1]
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)
