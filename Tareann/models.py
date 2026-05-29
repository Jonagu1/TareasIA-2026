import torch.nn as nn

# ==========================================
# MODELO 1: SimpleMLP (Baseline lineal)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Arquitectura superficial: Solo 1 capa oculta de 32 neuronas.
        # Justificación: Establece un punto de referencia rápido para ver 
        # si el problema se puede resolver con un modelo ligero sin sobreajustarse.
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 1 neurona de salida para clasificación binaria
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# MODELO 2: MediumMLP (Complejidad Media)
# ==========================================
class MediumMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Arquitectura balanceada: 2 capas ocultas (64 -> 32).
        # Justificación: Permite a la red aprender relaciones e interacciones 
        # matemáticas más complejas entre los tipos, el nombre y las estadísticas.
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 1 neurona de salida para clasificación binaria
        )

    def forward(self, x):
        return self.network(x)
