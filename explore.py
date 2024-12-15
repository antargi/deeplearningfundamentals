#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import xarray as xr
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset


# In[4]:


print("Versión de PyTorch:", torch.__version__)
print("Versión de CUDA usada por PyTorch:", torch.version.cuda)


# In[11]:


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


# In[177]:


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


# In[194]:


longitudes = subset_selected['longitude'].values
latitudes = subset_selected['latitude'].values
levels = subset_selected['level'].values
times = subset_selected['time'].values


# In[195]:


# Normalizar longitudes y latitudes
longitudes_min, longitudes_max = longitudes.min(), longitudes.max()
latitudes_min, latitudes_max = latitudes.min(), latitudes.max()
levels_min, levels_max = levels.min(), levels.max()

# Normalizar tiempo (asumiendo que ya has convertido a numérico)
times_numeric = (times - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
times_min, times_max = times_numeric.min(), times_numeric.max()

longitudes_normalized = (longitudes - longitudes_min) / (longitudes_max - longitudes_min + 1e-8)
latitudes_normalized = (latitudes - latitudes_min) / (latitudes_max - latitudes_min + 1e-8)
levels_normalized = (levels - levels_min) / (levels_max - levels_min + 1e-8)
times_normalized = (times_numeric - times_min) / (times_max - times_min + 1e-8)


# In[198]:


print(f"Longitudes normalizadas: {longitudes_normalized.min()} - {longitudes_normalized.max()}")
print(f"Latitudes normalizadas: {latitudes_normalized.min()} - {latitudes_normalized.max()}")
print(f"Tiempos normalizados: {times_normalized.min()} - {times_normalized.max()}")
print(f"Niveles normalizados: {levels_normalized.min()} - {levels_normalized.max()}")


# In[157]:


print(f"Longitudes: {longitudes.shape}, Latitudes: {latitudes.shape}, Times: {times.shape}, Levels: {levels.shape}")


# In[196]:


x_coords, y_coords, t_coords, level_coords = np.meshgrid(
    longitudes_normalized, latitudes_normalized, times_normalized[:u_train_tensor.shape[0]], levels_normalized, indexing="ij"
)


# In[197]:


inputs = np.stack([
    x_coords.flatten(),
    y_coords.flatten(),
    t_coords.flatten(),
    level_coords.flatten()
], axis=1)


# In[199]:


inputs_tensor = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)


# In[245]:


class PINNWithCNN(nn.Module):
    def __init__(self):
        super(PINNWithCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 5, kernel_size=3, stride=1, padding=1)  # Cambiar a 5 canales de salida

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.conv3(x)  # Última capa genera 5 canales
        return x


# In[207]:


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128), 
            nn.Dropout(0.2),   
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.layers(x)


# In[ ]:





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


# In[204]:


def loss_pde(model, x):
    pred = model(x)
    u_pred, v_pred, p_pred, T_pred, q_pred = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]

    # Calcula gradientes para cada término
    u_x, u_y, u_t = compute_gradients(u_pred, x)
    v_x, v_y, v_t = compute_gradients(v_pred, x)
    T_x, T_y, T_t = compute_gradients(T_pred, x)
    q_x, q_y, q_t = compute_gradients(q_pred, x)
    p_x, p_y, _ = compute_gradients(p_pred, x)

    momentum_u = u_t + u_pred * u_x + v_pred * u_y + p_x
    momentum_v = v_t + u_pred * v_x + v_pred * v_y + p_y
    continuity = u_x + v_y
    energy = T_t + u_pred * T_x + v_pred * T_y
    moisture = q_t + u_pred * q_x + v_pred * q_y

    loss_physics = (
        torch.mean(momentum_u ** 2) +
        torch.mean(momentum_v ** 2) +
        torch.mean(continuity ** 2) +
        torch.mean(energy ** 2) +
        torch.mean(moisture ** 2)
    )

    loss_physics
    return loss_physics


# In[200]:


# Ajustar las dimensiones para que coincidan
u_train_flat = u_train_tensor.flatten()
v_train_flat = v_train_tensor.flatten()
T_train_flat = T_train_tensor.flatten()
q_train_flat = q_train_tensor.flatten()
p_train_flat = pe_train_tensor.flatten()

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


# In[214]:


dataset = TensorDataset(inputs_tensor, u_train_flat, v_train_flat, p_train_flat, T_train_flat, q_train_flat)
loader = DataLoader(dataset, batch_size=256, shuffle=True)


# In[219]:


def train_model(model, loader, inputs_tensor, u_train_flat, v_train_flat, p_train_flat, 
                T_train_flat, q_train_flat, optimizer, loss_pde, compute_accuracy, 
                device, alpha=1.0, beta=10.0, epochs=1000):
    model.to(device)
    inputs_tensor = inputs_tensor.to(device)
    u_train_flat = u_train_flat.to(device)
    v_train_flat = v_train_flat.to(device)
    p_train_flat = p_train_flat.to(device)
    T_train_flat = T_train_flat.to(device)
    q_train_flat = q_train_flat.to(device)

    loss_history = []
    accuracy_history = {
        "u": [],
        "v": [],
        "p": [],
        "T": [],
        "q": []
    }

    for epoch in range(epochs):
        model.train()  # Modo entrenamiento
        epoch_loss_total = 0.0
        start_time = time.time()

        for batch in loader:
            inputs, u_true, v_true, p_true, T_true, q_true = [t.to(device) for t in batch]

            optimizer.zero_grad()

            loss_total, loss_data, loss_physics = loss_pde(model, inputs, u_true, v_true, p_true, T_true, q_true, alpha, beta)
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

            # Registrar precisión
            accuracy_history["u"].append(u_accuracy)
            accuracy_history["v"].append(v_accuracy)
            accuracy_history["p"].append(p_accuracy)
            accuracy_history["T"].append(T_accuracy)
            accuracy_history["q"].append(q_accuracy)

        print(f"Epoch {epoch}, Loss Total: {epoch_loss_total / len(loader):.6f}, "
              f"Accuracy: u={u_accuracy:.2f}%, v={v_accuracy:.2f}%, p={p_accuracy:.2f}%, "
              f"T={T_accuracy:.2f}%, q={q_accuracy:.2f}%, time {time.time() - start_time:.2f} s")

    return loss_history, accuracy_history


# In[270]:


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history, accuracy_history = train_model(
    model=model,
    loader=loader,
    inputs_tensor=inputs_tensor,
    u_train_flat=u_train_flat,
    v_train_flat=v_train_flat,
    p_train_flat=p_train_flat,
    T_train_flat=T_train_flat,
    q_train_flat=q_train_flat,
    optimizer=optimizer,
    loss_pde=loss_pde,
    compute_accuracy=compute_accuracy,
    device=device,
    alpha=1.0,
    beta=10.0,
    epochs=10
)


# In[271]:


# Datos originales: (97500, 4)
batch_size = 256  # Tamaño del batch
channels = 4  # Número de características (latitud, longitud, nivel, tiempo)
height, width = 50, 50  # Dimensiones espaciales

num_points = (97500 // (50 * 50)) * (50 * 50)  # Total divisible por 2500

# Truncar inputs_tensor
inputs_tensor_truncated = inputs_tensor[:num_points]

# Reorganizar al formato 4D
inputs_tensor_cnn = inputs_tensor_truncated.view(-1, channels, height, width)
print(f"Forma truncada para CNN: {inputs_tensor_cnn.shape}")


# In[272]:


print(f"Forma de inputs_tensor_cnn: {inputs_tensor_cnn.shape}")
print(f"Forma de u_train_flat: {u_train_flat[:num_points].shape}")
print(f"Forma de v_train_flat: {v_train_flat[:num_points].shape}")
print(f"Forma de p_train_flat: {p_train_flat[:num_points].shape}")
print(f"Forma de T_train_flat: {T_train_flat[:num_points].shape}")
print(f"Forma de q_train_flat: {q_train_flat[:num_points].shape}")


# In[273]:


u_train_cnn = u_train_flat[:num_points].view(-1, 1, height, width)
v_train_cnn = v_train_flat[:num_points].view(-1, 1, height, width)
p_train_cnn = p_train_flat[:num_points].view(-1, 1, height, width)
T_train_cnn = T_train_flat[:num_points].view(-1, 1, height, width)
q_train_cnn = q_train_flat[:num_points].view(-1, 1, height, width)

# Verificar formas
print(f"Forma de u_train_cnn: {u_train_cnn.shape}")
print(f"Forma de v_train_cnn: {v_train_cnn.shape}")
print(f"Forma de p_train_cnn: {p_train_cnn.shape}")
print(f"Forma de T_train_cnn: {T_train_cnn.shape}")
print(f"Forma de q_train_cnn: {q_train_cnn.shape}")


# In[274]:


dataset_cnn = TensorDataset(
    inputs_tensor_cnn,  # Entrada para la CNN
    u_train_cnn,        # Salidas reorganizadas
    v_train_cnn,
    p_train_cnn,
    T_train_cnn,
    q_train_cnn
)

# Crear el DataLoader
loader_cnn = DataLoader(dataset_cnn, batch_size=batch_size, shuffle=True)
print(f"DataLoader creado con éxito: {len(loader_cnn)} batches")


# In[275]:


print(f"Forma del dataset:")
for data in dataset_cnn[0]:
    print(data.shape)


# In[280]:


def loss_pde_cnn_1(model, inputs, u_true, v_true, p_true, T_true, q_true):
    pred = model(inputs)
    loss_data = (
        torch.mean((pred[:, 0, :, :] - u_true[:, 0, :, :]) ** 2) +
        torch.mean((pred[:, 1, :, :] - v_true[:, 0, :, :]) ** 2) +
        torch.mean((pred[:, 2, :, :] - p_true[:, 0, :, :]) ** 2) +
        torch.mean((pred[:, 3, :, :] - T_true[:, 0, :, :]) ** 2) +
        torch.mean((pred[:, 4, :, :] - q_true[:, 0, :, :]) ** 2)
    )
    return loss_data


# In[285]:


def loss_pde_cnn(model, x, u_true, v_true, p_true, T_true, q_true, alpha=1.0, beta=1.0):
    pred = model(x)
    u_pred, v_pred, p_pred, T_pred, q_pred = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
    loss_data = (
        torch.mean((pred[:, 0, :, :] - u_true[:, 0, :, :]) ** 2) +
        torch.mean((pred[:, 1, :, :] - v_true[:, 0, :, :]) ** 2) +
        torch.mean((pred[:, 2, :, :] - p_true[:, 0, :, :]) ** 2) +
        torch.mean((pred[:, 3, :, :] - T_true[:, 0, :, :]) ** 2) +
        torch.mean((pred[:, 4, :, :] - q_true[:, 0, :, :]) ** 2)
    )
    
    # Calcula gradientes para cada término
    u_x, u_y, u_t = compute_gradients(u_pred, x)
    v_x, v_y, v_t = compute_gradients(v_pred, x)
    T_x, T_y, T_t = compute_gradients(T_pred, x)
    q_x, q_y, q_t = compute_gradients(q_pred, x)
    p_x, p_y, _ = compute_gradients(p_pred, x)

    # Define los términos de la pérdida
    momentum_u = u_t + u_pred * u_x + v_pred * u_y + p_x
    momentum_v = v_t + u_pred * v_x + v_pred * v_y + p_y
    continuity = u_x + v_y
    energy = T_t + u_pred * T_x + v_pred * T_y
    moisture = q_t + u_pred * q_x + v_pred * q_y



    loss_physics = (
        torch.mean(momentum_u ** 2) +
        torch.mean(momentum_v ** 2) +
        torch.mean(continuity ** 2) +
        torch.mean(energy ** 2) +
        torch.mean(moisture ** 2)
    )

    loss_total = alpha * loss_data + beta * loss_physics
    return loss_total, loss_data, loss_physics


# In[291]:


loss_history = []
loss_data_history = []
loss_physics_history = []
alpha = 0.5
beta = 1
model = PINNWithCNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2)
epochs = 2000

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_loss_data = 0.0
    epoch_loss_physics = 0.0

    for batch in loader_cnn:
        inputs, u_true, v_true, p_true, T_true, q_true = [t.to(device) for t in batch]

        optimizer.zero_grad()
        # Pérdida con la primera función
        loss_total, loss_data, loss_physics = loss_pde_cnn(model, inputs, u_true, v_true, p_true, T_true, q_true, alpha=1.0, beta=0)
        loss_total.backward()
        optimizer.step()
        epoch_loss += loss_total.item()
        epoch_loss_data += loss_data.item()
        epoch_loss_physics += loss_physics.item()

    loss_history.append(epoch_loss / len(loader_cnn))
    loss_data_history.append(epoch_loss_data / len(loader_cnn))
    loss_physics_history.append(epoch_loss_physics / len(loader_cnn))

    print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {epoch_loss / len(loader_cnn):.6f}, "
          f"Data Loss: {epoch_loss_data / len(loader_cnn):.6f}, "
          f"Physics Loss: {epoch_loss_physics / len(loader_cnn):.6f}")


# In[292]:


plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Total Loss')
plt.plot(loss_data_history, label='Data Loss')
plt.plot(loss_physics_history, label='Physics Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Components Over Epochs')
plt.legend()
plt.grid()
plt.show()

