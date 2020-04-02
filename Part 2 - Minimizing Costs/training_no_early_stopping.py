# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2
# Fase de entrenamiento de la IA

# Instalación de las librerí­as necesarias
# conda install -c conda-forge keras

# Importar las librerí­as y otros ficheros de python
import os
import numpy as np
import random as rn

import environment
import brain
import dqn

# Configurar las semillas para reproducibilidad
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# CONFIGURACIÓN DE LOS PARÁMETROS 
epsilon = 0.3
number_actions = 5
direction_boundary = (number_actions -1)/2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# CONSTRUCCIÓN DEL ENTORNO CREANDO UN OBJETO DE LA CLASE ENVIRONMENT
env = environment.Environtment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# CONSTRUCCIÓN DEL CEREBRO CREANDO UN OBJETO DE LA CLASE BRAIN
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

# CONSTRUCCIÓN DEL MODELO DQN CREANDO UN OBJETO DE LA CLASE DQN
dqn = dqn.DQN(max_memory = max_memory, discount_factor = 0.9)

# ELECCIÓN DEL MODO DE ENTRENAMIENTO
train = True

# ENTRENAR LA IA
env.train = train
model = brain.model

if (env.train):
    # INICIAR EL BUCLE DE TODAS LAS ÉPOCAS (1 Epoch = 5 Meses)
    for epoch in range(1, number_epochs):
        # INICIALIZACIÓN DE LAS VARIABLES DEL ENTORNO Y DEL BUCLE DE ENTRENAMIENTO
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)  
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        # INICIALIZACIÓN DEL BUCLE DE TIMESTEPS (Timestep = 1 minuto) EN UNA EPOCA
        while ((not game_over) and (timestep <= 5*30*24*60)):
            # EJECUTAR LA SIGUIENTE ACCIÓN POR EXPLORACIÓN
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                   
            # EJECUTAR LA SIGUIENTE ACCIÓN POR INFERENCIA
            else: 
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
            
            if (action < direction_boundary):
                direction = -1
            else:
                direction = 1
            energy_ai = abs(action - direction_boundary) * temperature_step
            
            # ACTUALIZAR EL ENTORNO Y ALCANZAR EL SIGUIENTE ESTADO
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            total_reward += reward
            
            # ALMACENAR LA NUEVA TRANSICIÓN EN LA MEMORIA
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            # OBTENER LOS DOS BLOQUES SEPARADOS DE ENTRADAS Y OBJETIVOS
            inputs, targets = dqn.get_batch(model, batch_size)
            
            # CALCULAR LA FUNCIÓN DE PÉRDIDAS UTILIZANDO TODO EL BLOQUE DE ENTRADA Y OBJETIVOS
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
            
        # IMPRIMIR LOS RESULTADOS DEL ENTRENAMIENTO AL FINAL DEL EPOCH
        print("\n")
        print("Epoch: {:03d}/{:03d}.".format(epoch, number_epochs))
        print(" - Energia total gastada por el sistema con IA: {:.0f} J.".format(env.total_energy_ai))
        print(" - Energia total gastada por el sistema sin IA: {:.0f} J.".format(env.total_energy_noai))
        # GUARDAR EL MODELO PARA SU USO FUTURO
        model.save("model.h5")


