
import sys
print(sys.path)

import itertools
import pandas as pd

# Definición de los valores para cada parámetro
num_features = [10, 20, 50]  # f: número de características
num_samples = [500, 1000, 2000]  # N: tamaño del dataset
initializations = ["Xavier", "He"]  # Estrategias de inicialización
optimizers = ["SGD", "Adam"]  # Tipos de optimizadores
learning_rates = [0.001, 0.01, 0.1]  # Valores de lr

# Crear todas las combinaciones
combinations = list(itertools.product(num_features, num_samples, initializations, optimizers, learning_rates))

# Convertir las combinaciones en un DataFrame
columns = ["Num_Features", "Num_Samples", "Initialization", "Optimizer", "Learning_Rate"]
combinations_df = pd.DataFrame(combinations, columns=columns)
combinations_df.to_csv("combinaciones_random_dataset.csv", index=False)