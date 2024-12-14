#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import xarray as xr
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time


# In[2]:


print("Versión de PyTorch:", torch.__version__)
print("Versión de CUDA usada por PyTorch:", torch.version.cuda)


# In[3]:


dataset_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
ds = xr.open_zarr(dataset_path, consolidated=True)
print(ds)


# In[4]:


print("Dimensiones del dataset:", ds.dims)


# In[71]:


subset = ds.isel(latitude=slice(0, 50), longitude=slice(0, 50), time=slice(0, 5))
print(subset)


# In[72]:


for var in subset.data_vars:
    print(f"{var}: {subset[var].shape}")


# In[73]:


variables = ['u_component_of_wind', 'v_component_of_wind', 
             'temperature', 'specific_humidity', 'mean_sea_level_pressure']

subset_selected = subset[variables]
print(subset_selected)


# In[74]:


train_idx = 3  # Timestamps para entrenamiento
val_idx = 4    # Timestamps para validación
test_idx = 5   # Timestamps para prueba

# Crear los conjuntos
train_data = subset_selected.isel(time=slice(0, train_idx))
val_data = subset_selected.isel(time=slice(train_idx, val_idx))
test_data = subset_selected.isel(time=slice(val_idx, test_idx))

# Confirmar tamaños
print(f"Train size: {train_data['time'].shape[0]} timestamps")
print(f"Val size: {val_data['time'].shape[0]} timestamps")
print(f"Test size: {test_data['time'].shape[0]} timestamps")


# In[75]:


u_mean = train_data['u_component_of_wind'].mean().values
u_std = train_data['u_component_of_wind'].std().values
v_mean = train_data['v_component_of_wind'].mean().values
v_std = train_data['v_component_of_wind'].std().values
T_mean = train_data['temperature'].mean().values
T_std = train_data['temperature'].std().values
q_mean = train_data['specific_humidity'].mean().values
q_std = train_data['specific_humidity'].std().values
p_mean = train_data['mean_sea_level_pressure'].mean().values
p_std = train_data['mean_sea_level_pressure'].std().values


# In[76]:


print(f"u_mean: {u_mean}, u_std: {u_std}")
print(f"v_mean: {v_mean}, v_std: {v_std}")
print(f"T_mean: {T_mean}, T_std: {T_std}")
print(f"q_mean: {q_mean}, q_std: {q_std}")
print(f"p_mean: {p_mean}, p_std: {p_std}")


# In[77]:


u_train = (train_data['u_component_of_wind'] - u_mean) / u_std
v_train = (train_data['v_component_of_wind'] - v_mean) / v_std
T_train = (train_data['temperature'] - T_mean) / T_std
q_train = (train_data['specific_humidity'] - q_mean) / q_std
p_train = (train_data['mean_sea_level_pressure'] - p_mean) / p_std


# In[78]:


u_val = (val_data['u_component_of_wind'] - u_mean) / u_std
v_val = (val_data['v_component_of_wind'] - v_mean) / v_std
T_val = (val_data['temperature'] - T_mean) / T_std
q_val = (val_data['specific_humidity'] - q_mean) / q_std
p_val = (val_data['mean_sea_level_pressure'] - p_mean) / p_std


# In[79]:


u_test = (test_data['u_component_of_wind'] - u_mean) / u_std
v_test = (test_data['v_component_of_wind'] - v_mean) / v_std
T_test = (test_data['temperature'] - T_mean) / T_std
q_test = (test_data['specific_humidity'] - q_mean) / q_std
p_test = (test_data['mean_sea_level_pressure'] - p_mean) / p_std


# In[80]:


u_train_tensor = torch.tensor(u_train.values, dtype=torch.float32, requires_grad=True)
v_train_tensor = torch.tensor(v_train.values, dtype=torch.float32, requires_grad=True)
T_train_tensor = torch.tensor(T_train.values, dtype=torch.float32, requires_grad=True)
q_train_tensor = torch.tensor(q_train.values, dtype=torch.float32, requires_grad=True)
p_train_tensor = torch.tensor(p_train.values, dtype=torch.float32, requires_grad=True)


# In[81]:


u_val_tensor = torch.tensor(u_val.values, dtype=torch.float32)
v_val_tensor = torch.tensor(v_val.values, dtype=torch.float32)
T_val_tensor = torch.tensor(T_val.values, dtype=torch.float32)
q_val_tensor = torch.tensor(q_val.values, dtype=torch.float32)
p_val_tensor = torch.tensor(p_val.values, dtype=torch.float32)


# In[82]:


u_test_tensor = torch.tensor(u_test.values, dtype=torch.float32)
v_test_tensor = torch.tensor(v_test.values, dtype=torch.float32)
T_test_tensor = torch.tensor(T_test.values, dtype=torch.float32)
q_test_tensor = torch.tensor(q_test.values, dtype=torch.float32)
p_test_tensor = torch.tensor(p_test.values, dtype=torch.float32)


# In[83]:


# Verificar dimensiones
print(f"u_test_tensor: {u_test_tensor.shape}")
print(f"v_test_tensor: {v_test_tensor.shape}")
print(f"T_test_tensor: {T_test_tensor.shape}")
print(f"q_test_tensor: {q_test_tensor.shape}")
print(f"p_test_tensor: {p_test_tensor.shape}")


# In[84]:


pe_test_tensor = p_test_tensor.unsqueeze(1).repeat(1, 13, 1, 1)
pe_train_tensor = p_train_tensor.unsqueeze(1).repeat(1, 13, 1, 1)
pe_val_tensor =  p_val_tensor.unsqueeze(1).repeat(1, 13, 1, 1)


# In[85]:


# Verificar valores de ejemplo
print(f"Ejemplo de u_test_tensor: {u_test_tensor[0, 0, 0, 0]}")
print(f"Ejemplo de v_test_tensor: {v_test_tensor[0, 0, 0, 0]}")
print(f"Ejemplo de T_test_tensor: {T_test_tensor[0, 0, 0, 0]}")
print(f"Ejemplo de q_test_tensor: {q_test_tensor[0, 0, 0, 0]}")
print(f"Ejemplo de pe_test_tensor: {pe_test_tensor[0, 0, 0, 0]}")


# In[86]:


print(f"Forma de u_train_tensor: {u_train_tensor.shape}")
print(f"Forma de v_train_tensor: {v_train_tensor.shape}")
print(f"Forma de T_train_tensor: {T_train_tensor.shape}")
print(f"Forma de q_train_tensor: {q_train_tensor.shape}")
print(f"Forma de pe_train_tensor: {pe_train_tensor.shape}")


# In[156]:


longitudes = subset_selected['longitude'].values
latitudes = subset_selected['latitude'].values
levels = subset_selected['level'].values
times = subset_selected['time'].values


# In[157]:


print(f"Longitudes: {longitudes.shape}, Latitudes: {latitudes.shape}, Times: {times.shape}, Levels: {levels.shape}")


# In[158]:


x_coords, y_coords, t_coords, level_coords = np.meshgrid(
    longitudes, latitudes, times[:u_train_tensor.shape[0]], levels, indexing="ij"
)


# In[159]:


t_coords_numeric = (t_coords - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')


# In[160]:


inputs = np.stack([
    x_coords.flatten(),
    y_coords.flatten(),
    t_coords_numeric.flatten(),
    level_coords.flatten()
], axis=1)


# In[161]:


inputs_tensor = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)


# In[27]:


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.layers(x)


# In[28]:


def compute_accuracy(pred, true, tolerance=0.05):
    """
    Calcula el porcentaje de predicciones dentro de un rango de tolerancia.
    """
    relative_error = torch.abs((pred - true) / true)
    accurate_predictions = torch.sum(relative_error <= tolerance).item()
    total_predictions = true.numel()
    accuracy = accurate_predictions / total_predictions
    return accuracy * 100  # Porcentaje


# In[109]:


def compute_gradients(pred, x):
    grad = torch.autograd.grad(pred.sum(), x, create_graph=True, retain_graph=True)[0]
    return grad[:, 0], grad[:, 1], grad[:, 2]  # Devuelve las derivadas parciales


# In[144]:


def loss_pde(model, x, u_true, v_true, p_true, T_true, q_true):
    pred = model(x)
    u_pred, v_pred, p_pred, T_pred, q_pred = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]

    # Calcula gradientes para cada término
    try:
        u_x, u_y, u_t = compute_gradients(u_pred, x)
        v_x, v_y, v_t = compute_gradients(v_pred, x)
        T_x, T_y, T_t = compute_gradients(T_pred, x)
        q_x, q_y, q_t = compute_gradients(q_pred, x)
        p_x, p_y, _ = compute_gradients(p_pred, x)
    except RuntimeError as e:
        print(f"Error calculando gradientes: {e}")
        print(f"x.requires_grad: {x.requires_grad}, pred.requires_grad: {pred.requires_grad}")
        raise

    # Define los términos de la pérdida
    momentum_u = u_t + u_pred * u_x + v_pred * u_y + p_x
    momentum_v = v_t + u_pred * v_x + v_pred * v_y + p_y
    continuity = u_x + v_y
    energy = T_t + u_pred * T_x + v_pred * T_y
    moisture = q_t + u_pred * q_x + v_pred * q_y

    loss_data = (
        torch.mean((u_pred - u_true) ** 2) +
        torch.mean((v_pred - v_true) ** 2) +
        torch.mean((p_pred - p_true) ** 2) +
        torch.mean((T_pred - T_true) ** 2) +
        torch.mean((q_pred - q_true) ** 2)
    )

    loss_physics = (
        torch.mean(momentum_u ** 2) +
        torch.mean(momentum_v ** 2) +
        torch.mean(continuity ** 2) +
        torch.mean(energy ** 2) +
        torch.mean(moisture ** 2)
    )

    return loss_data, loss_physics


# In[148]:


# Ajustar las dimensiones para que coincidan
u_train_flat = u_train_tensor.flatten()
v_train_flat = v_train_tensor.flatten()
T_train_flat = T_train_tensor.flatten()
q_train_flat = q_train_tensor.flatten()
p_train_flat = pe_train_tensor.flatten()

# # Normalización de las salidas
# u_min, u_max = torch.min(u_train_flat), torch.max(u_train_flat)
# v_min, v_max = torch.min(v_train_flat), torch.max(v_train_flat)
# T_min, T_max = torch.min(T_train_flat), torch.max(T_train_flat)
# q_min, q_max = torch.min(q_train_flat), torch.max(q_train_flat)
# p_min, p_max = torch.min(p_train_flat), torch.max(p_train_flat)

# u_train_flat = (u_train_flat - u_min) / (u_max - u_min + 1e-8)
# v_train_flat = (v_train_flat - v_min) / (v_max - v_min + 1e-8)
# T_train_flat = (T_train_flat - T_min) / (T_max - T_min + 1e-8)
# q_train_flat = (q_train_flat - q_min) / (q_max - q_min + 1e-8)
# p_train_flat = (p_train_flat - p_min) / (p_max - p_min + 1e-8)

print(f"u_train_flat.requires_grad: {u_train_flat.requires_grad}")
print(f"v_train_flat.requires_grad: {v_train_flat.requires_grad}")
print(f"T_train_flat.requires_grad: {T_train_flat.requires_grad}")
print(f"q_train_flat.requires_grad: {q_train_flat.requires_grad}")
print(f"p_train_flat.requires_grad: {p_train_flat.requires_grad}")

print(f"Forma de inputs_tensor: {inputs_tensor.shape}")
print(f"Forma de u_train_flat: {u_train_flat.shape}")
print(f"Forma de v_train_flat: {v_train_flat.shape}")
print(f"Forma de T_train_flat: {T_train_flat.shape}")
print(f"Forma de q_train_flat: {q_train_flat.shape}")
print(f"Forma de p_train_flat: {p_train_flat.shape}")


# In[34]:


# Lista de batch sizes a probar
batch_sizes = [16, 32, 64, 128, 256, 256, 512]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resultados almacenados
batch_results = {}

for batch_size in batch_sizes:
    print(f"Probando batch_size = {batch_size}...")
    
    # Crear DataLoader con el batch size actual
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Inicializar el modelo y mover a GPU
    model = PINN().to(device)

    # Medir tiempo para procesar un batch
    start_time = time.time()
    for batch in loader:
        inputs, u_true, v_true, p_true, T_true, q_true = [t.to(device) for t in batch]
        with torch.no_grad():  # Solo evaluación, sin retropropagación
            pred = model(inputs)
        break  # Solo medir el primer batch
    elapsed_time_batch = time.time() - start_time

    # (Opcional) Medir tiempo para procesar toda una época
    start_time_epoch = time.time()
    for batch in loader:
        inputs, u_true, v_true, p_true, T_true, q_true = [t.to(device) for t in batch]
        with torch.no_grad():
            pred = model(inputs)
    elapsed_time_epoch = time.time() - start_time_epoch

    # Almacenar resultados
    batch_results[batch_size] = {
        "time_per_batch": elapsed_time_batch,
        "time_per_epoch": elapsed_time_epoch,
    }
    print(f"Batch size {batch_size}: Tiempo por batch = {elapsed_time_batch:.4f}s, Tiempo por época = {elapsed_time_epoch:.2f}s")

# Mostrar resultados finales
print("\nResultados finales:")
for batch_size, results in batch_results.items():
    print(
        f"Batch size {batch_size}: "
        f"Tiempo por batch = {results['time_per_batch']:.4f}s, "
        f"Tiempo por época = {results['time_per_epoch']:.2f}s"
    )


# In[162]:


from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(inputs_tensor, u_train_flat, v_train_flat, p_train_flat, T_train_flat, q_train_flat)
loader = DataLoader(dataset, batch_size=256, shuffle=True)


# In[164]:


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.to(device)
inputs_tensor = inputs_tensor.to(device)
u_train_flat = u_train_flat.to(device)
v_train_flat = v_train_flat.to(device)
p_train_flat = p_train_flat.to(device)
T_train_flat = T_train_flat.to(device)
q_train_flat = q_train_flat.to(device)


# In[ ]:


epochs = 1000
loss_history = []
for epoch in range(epochs):
    model.train()  # Modo entrenamiento
    epoch_loss_total = 0.0
    start_time = time.time()

    for batch in loader:
        # Dividir los datos del batch
        inputs, u_true, v_true, p_true, T_true, q_true = [t.to(device) for t in batch]
        
        inputs = inputs.clone().detach().requires_grad_(True)
        
        optimizer.zero_grad()  # Resetear gradientes

        # Calcula ambas pérdidas
        loss_data, loss_physics = loss_pde(model, inputs, u_true, v_true, p_true, T_true, q_true)
        
        # Combina las pérdidas
        loss_total = loss_data + loss_physics

        # Retropropagación
        loss_total.backward()
        optimizer.step()

        # Acumula la pérdida total
        epoch_loss_total += loss_total.item()

    loss_history.append(epoch_loss_total / len(loader))

    # Evaluar precisión después de la época
    model.eval()  # Modo evaluación (sin dropout, etc.)
    with torch.no_grad():
        predictions = model(inputs_tensor)
        u_pred, v_pred, p_pred, T_pred, q_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3], predictions[:, 4]

        u_accuracy = compute_accuracy(u_pred, u_train_flat)
        v_accuracy = compute_accuracy(v_pred, v_train_flat)
        p_accuracy = compute_accuracy(p_pred, p_train_flat)
        T_accuracy = compute_accuracy(T_pred, T_train_flat)
        q_accuracy = compute_accuracy(q_pred, q_train_flat)

    print(f"Epoch {epoch}, Loss Total: {epoch_loss_total / len(loader):.6f}, "
          f"Accuracy: u={u_accuracy:.2f}%, v={v_accuracy:.2f}%, p={p_accuracy:.2f}%, "
          f"T={T_accuracy:.2f}%, q={q_accuracy:.2f}%, time {time.time() - start_time:.2f} s")

