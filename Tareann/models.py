import torch.nn as nn

# MODELO 1: SimpleMLP (Baseline lineal)

class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Arquitectura superficial: Solo 1 capa oculta de 32 neuronas.
        # Justificación: Establece un punto de referencia rápido
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # 2 neuronas de salida para clasificación binaria
        )

    def forward(self, x):
        return self.network(x)
#Seleccionada, arroja buenos numeros y no se cae altiro. COn MediumMLP el entrenamiento dura como 50 epocas y se acaba, con los modelos de más abajo lo mismo
#SimpleMLP resultó ser el mas confiable y arrojó los mayores numeros de precision y f1score.

# MODELO 2: MediumMLP (Complejidad Media)

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
            nn.Linear(32, 2) 
        )

    def forward(self, x):
        return self.network(x)


# MODELO 3: 3-Layer MLP (Arquitectura Personalizada)
#esta arquitectura fue cambiada muchas veces buscando mejorar, se subio el numero de capas con lo que bajaba la precision,
#luego se trato de bajar la complejidad y se observo que menos capas era más (EL simple funcionaba mejor que el Medium MLP)
#Así que esta fue cambiando en número de capas y neuronas hasta encontrar un buen equilibrio. 
class MLP_Tercera_Arquitectura(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 21), #Numero arbitrario porque me gusta el numero 21. igualmente con [8..15] y [25..36] no cambia mucho
            nn.ReLU(),
            nn.Linear(21, 2),
        )
#Tubo buen resultado pero no mejor que el SimpleMLP
    def forward(self, x):
        return self.network(x)