# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2
# Fase de testing


# Importar las librerí­as y otros ficheros de python
import os
import numpy as np
import random as rn
from keras.models import load_model
import environment

# Configurar las semillas para reproducibilidad
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# CONFIGURACIÓN DE LOS PARÁMETROS 
number_actions = 5
direction_boundary = (number_actions -1)/2
temperature_step = 1.5

# CONSTRUCCIÓN DEL ENTORNO CREANDO UN OBJETO DE LA CLASE ENVIRONMENT
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# CARGA DE UN MODELO PRE ENTRENADO
model = load_model("model.h5")

# ELECCIÓN DEL MODO DE ENTRENAMIENTO
train = False

# EJECUCIÓN DE UN AÑO DE SIMULACIÓN EN MODO INFERENCIA
env.train = train
current_state, _, _ = env.observe()
for timestep in range(0, 12*30*24*60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
            
    if (action < direction_boundary):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
    current_state = next_state


            
# IMPRIMIR LOS RESULTADOS DEL ENTRENAMIENTO AL FINAL DEL EPOCH
print("\n")
print(" - Energia total gastada por el sistema con IA: {:.0f} J.".format(env.total_energy_ai))
print(" - Energia total gastada por el sistema sin IA: {:.0f} J.".format(env.total_energy_noai))
print("ENERGIA AHORRADA: {:.0f} %.".format(100*(env.total_energy_noai-env.total_energy_ai)/env.total_energy_noai))



