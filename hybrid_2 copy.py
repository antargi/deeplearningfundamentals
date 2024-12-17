#!/usr/bin/env python
# coding: utf-8

# # Laboratorio 2 Fundamentos de Aprendizaje Profundo
# Temática: Comportamiento y predicción del clima
# 
# Nombre: Antonina Arriagada

# ### 1. Definiciones e importación librerias

# In[6]:


import torch
import xarray as xr
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset
import dask
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim


# #### 1.1 Descarga del Dataset

# Descarga de ERA5 desde 1959 a 2022 cada 6 horas. Resolución de 1440x721

# In[7]:


dataset_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
ds = xr.open_zarr(dataset_path)
print(ds)


# #### 1.2 Análisis Exploratorio

# In[ ]:


lat_min = ds['latitude'].min().values
lat_max = ds['latitude'].max().values
lon_min = ds['longitude'].min().values
lon_max = ds['longitude'].max().values

print(f"Rango de latitud en el dataset: {lat_min} a {lat_max}")
print(f"Rango de longitud en el dataset: {lon_min} a {lon_max}")


# In[ ]:


chile_ds = ds.sel(
    latitude=slice(-17.0, -56.0), 
    longitude=slice(280.0, 310.0) 
)
print(chile_ds)


# In[ ]:


lat_min = chile_ds['latitude'].min().values
lat_max = chile_ds['latitude'].max().values
lon_min = chile_ds['longitude'].min().values
lon_max = chile_ds['longitude'].max().values

print(f"Rango de latitud seleccionado: {lat_min} a {lat_max}")
print(f"Rango de longitud seleccionado: {lon_min} a {lon_max}")


# In[ ]:


for var in chile_ds.data_vars:
    print(f"{var}: {chile_ds[var].shape}")


# Selección de dirección: Cono sur Sudamérica.

# In[ ]:


land_sea = chile_ds['land_sea_mask']
plt.figure(figsize=(10, 8))
land_sea.plot(cmap="terrain", add_colorbar=True)
plt.title("Land vs Sea Mask (Chile y Cono Sur)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Gráfico para observar el comportamiento de las componentes del viento sobre la temperatura

# In[ ]:


temp = chile_ds['temperature'].isel(time=0, level=10)
u_wind = chile_ds['u_component_of_wind'].isel(time=0, level=10)
v_wind = chile_ds['v_component_of_wind'].isel(time=0, level=10)

lon, lat = np.meshgrid(chile_ds['longitude'], chile_ds['latitude'])

step = 4
lon_subset = lon[::step, ::step]
lat_subset = lat[::step, ::step]
u_wind_subset = u_wind[::step, ::step]
v_wind_subset = v_wind[::step, ::step]

plt.figure(figsize=(10, 8))

plt.pcolormesh(lon, lat, temp, cmap="coolwarm", shading="auto")
plt.colorbar(label="Temperatura [K]")
plt.quiver(lon_subset, lat_subset, u_wind_subset, v_wind_subset, scale=300, color='black', alpha=0.7)
plt.title("Temperatura y Vectores de Viento al nivel 8 - Chile y Cono Sur (Submuestreo)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Temperatura en el nivel 10

# In[ ]:


temperature = chile_ds['temperature'].isel(time=0, level=10)
plt.figure(figsize=(10, 8))
temperature.plot(cmap="coolwarm", add_colorbar=True)
plt.title("Temperatura a nivel 10 (Primer Paso Temporal) - Chile y Cono Sur")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Gráfico para observar la relación entre la humedad específica, temperatura y viento.

# In[ ]:


temperature = chile_ds['temperature'].isel(time=0, level=10)
specific_humidity = chile_ds['specific_humidity'].isel(time=0, level=10) 
u_wind = chile_ds['u_component_of_wind'].isel(time=0, level=10)
v_wind = chile_ds['v_component_of_wind'].isel(time=0, level=10)
lon, lat = np.meshgrid(chile_ds['longitude'], chile_ds['latitude'])

step = 4
lon_subset = lon[::step, ::step]
lat_subset = lat[::step, ::step]
u_wind_subset = u_wind[::step, ::step]
v_wind_subset = v_wind[::step, ::step]

plt.figure(figsize=(10, 8))
plt.pcolormesh(lon, lat, specific_humidity, cmap="YlGnBu", shading="auto")
plt.colorbar(label="Humedad Específica [kg/kg]")
contour = plt.contour(lon, lat, temperature, levels=10, colors='red', linewidths=0.8)
plt.clabel(contour, inline=True, fontsize=8, fmt="%.0f K")
plt.quiver(lon_subset, lat_subset, u_wind_subset, v_wind_subset, scale=500, color='black', alpha=0.8)
plt.title("Humedad Específica, Temperatura y Viento - Chile y Cono Sur")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Gráfico para observar la temperatura a 2m de la superficie

# In[ ]:


temp_2m = chile_ds['2m_temperature'].isel(time=0)
plt.figure(figsize=(10, 8))
temp_2m.plot(cmap="coolwarm", add_colorbar=True)
plt.title("Temperatura a 2 metros (Primer Paso Temporal) - Chile y Cono Sur")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Componientes del viento a 10 metros

# In[ ]:


u_wind = chile_ds['10m_u_component_of_wind'].isel(time=0)
v_wind = chile_ds['10m_v_component_of_wind'].isel(time=0)
lon, lat = np.meshgrid(chile_ds['longitude'], chile_ds['latitude'])
plt.figure(figsize=(10, 8))
plt.quiver(lon, lat, u_wind, v_wind, scale=200, color='blue')
plt.title("Vectores de Viento a 10 metros (Primer Paso Temporal) - Chile y Cono Sur")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Gráfico para observar la relación entre las componentes del viento sobre la temperatura a 2 y 10 metros respectivamente. El viento debería converger hacia las zonas azules.

# In[ ]:


temp_2m = chile_ds['2m_temperature'].isel(time=0)
u_wind = chile_ds['10m_u_component_of_wind'].isel(time=0)
v_wind = chile_ds['10m_v_component_of_wind'].isel(time=0)

lon, lat = np.meshgrid(chile_ds['longitude'], chile_ds['latitude'])

step = 4
lon_subset = lon[::step, ::step]
lat_subset = lat[::step, ::step]
u_wind_subset = u_wind[::step, ::step]
v_wind_subset = v_wind[::step, ::step]

plt.figure(figsize=(10, 8))

plt.pcolormesh(lon, lat, temp_2m, cmap="coolwarm", shading="auto")
plt.colorbar(label="Temperatura [K]")
plt.quiver(lon_subset, lat_subset, u_wind_subset, v_wind_subset, scale=300, color='black', alpha=0.7)
plt.title("Temperatura y Vectores de Viento a 10 metros - Chile y Cono Sur (Submuestreo)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Gráfico para observar la presión de la superficie y la relación con las vectores del viento a 10 metros.

# In[ ]:


surface_pressure = chile_ds['surface_pressure'].isel(time=0)
u_wind = chile_ds['10m_u_component_of_wind'].isel(time=0)
v_wind = chile_ds['10m_v_component_of_wind'].isel(time=0)
lon, lat = np.meshgrid(chile_ds['longitude'], chile_ds['latitude'])
step = 4
lon_subset = lon[::step, ::step]
lat_subset = lat[::step, ::step]
u_wind_subset = u_wind[::step, ::step]
v_wind_subset = v_wind[::step, ::step]

plt.figure(figsize=(10, 8))
plt.pcolormesh(lon, lat, surface_pressure, cmap="viridis", shading="auto")
plt.colorbar(label="Presión en Superficie [Pa]")
plt.quiver(lon_subset, lat_subset, u_wind_subset, v_wind_subset, scale=300, color='black', alpha=0.7)
plt.title("Presión en Superficie y Vectores de Viento a 10 metros - Chile y Cono Sur")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Gráfico para observar la humedad específica en el nivel 0 con las variables anteriormente mencionadas, componenentes de viento a nivel 0 y temperatura a 2m.

# In[ ]:


temp_2m = chile_ds['2m_temperature'].isel(time=0)
specific_humidity = chile_ds['specific_humidity'].isel(time=0, level=0)  # Humedad específica en el mismo nivel
u_wind = chile_ds['u_component_of_wind'].isel(time=0, level=0)
v_wind = chile_ds['v_component_of_wind'].isel(time=0, level=0)
lon, lat = np.meshgrid(chile_ds['longitude'], chile_ds['latitude'])

step = 4
lon_subset = lon[::step, ::step]
lat_subset = lat[::step, ::step]
u_wind_subset = u_wind[::step, ::step]
v_wind_subset = v_wind[::step, ::step]

plt.figure(figsize=(10, 8))
plt.pcolormesh(lon, lat, specific_humidity, cmap="YlGnBu", shading="auto")
plt.colorbar(label="Humedad Específica [kg/kg]")
contour = plt.contour(lon, lat, temp_2m, levels=10, colors='red', linewidths=0.8)
plt.clabel(contour, inline=True, fontsize=8, fmt="%.0f K")
plt.quiver(lon_subset, lat_subset, u_wind_subset, v_wind_subset, scale=500, color='black', alpha=0.8)
plt.title("Humedad Específica, Temperatura y Viento - Chile y Cono Sur")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Geopotencial en el nivel 10 igual que temperatura, humedad especifica y comp de vientos.

# In[ ]:


geopotential = chile_ds['geopotential'].isel(time=0, level=10)  # Geopotential en nivel 10
temperature = chile_ds['temperature'].isel(time=0, level=10)
specific_humidity = chile_ds['specific_humidity'].isel(time=0, level=10)
u_wind = chile_ds['u_component_of_wind'].isel(time=0, level=10)
v_wind = chile_ds['v_component_of_wind'].isel(time=0, level=10)

lon, lat = np.meshgrid(chile_ds['longitude'], chile_ds['latitude'])

step = 4
lon_subset = lon[::step, ::step]
lat_subset = lat[::step, ::step]
u_wind_subset = u_wind[::step, ::step]
v_wind_subset = v_wind[::step, ::step]

plt.figure(figsize=(10, 8))
plt.pcolormesh(lon, lat, geopotential, cmap="viridis", shading="auto")
plt.colorbar(label="Geopotential [m²/s²]")
contour = plt.contour(lon, lat, temperature, levels=10, colors='red', linewidths=0.8)
plt.clabel(contour, inline=True, fontsize=8, fmt="%.0f K")
plt.quiver(lon_subset, lat_subset, u_wind_subset, v_wind_subset, scale=500, color='black', alpha=0.8)
plt.title("Geopotential, Temperatura y Viento - Nivel 10")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# Selección de subdataset y variables

# In[15]:


chile_ds = ds.sel(
    latitude=slice(-17.0, -56.0), 
    longitude=slice(280.0, 310.0) 
)
variables = ['u_component_of_wind', 'v_component_of_wind', 
             'temperature', 'specific_humidity', 'geopotential']
subset_selected_ch = chile_ds[variables].isel(level=10)
ds = subset_selected_ch.sel(time=slice("2010-01-01", "2010-01-31"))
print(ds)


# In[ ]:


from dask.diagnostics import ProgressBar

with ProgressBar():
    subset_selected_ch.to_zarr("dataset/chile_2010_january.zarr", consolidated=True)


# In[6]:


ds = xr.open_zarr("dataset/chile_2010_january.zarr")
print(ds)


# #### 1.3 Preprocesamiento de datos

# In[11]:


def preprocess_dataset(ds, variables, train_time_scale, val_time_scale, test_time_scale):
    train_vars = [ds[var].sel(time=train_time_scale).values for var in variables]  # Entrenamiento
    val_vars = [ds[var].sel(time=val_time_scale).values for var in variables]     # Validación
    test_vars = [ds[var].sel(time=test_time_scale).values for var in variables]   # Prueba

    train_data = np.array(train_vars)
    val_data = np.array(val_vars)
    test_data = np.array(test_vars)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    lat, lon = train_data.shape[2], train_data.shape[3]

    means = train_data.mean(dim=(1, 2, 3), keepdim=True)
    stds = train_data.std(dim=(1, 2, 3), keepdim=True)
    
    train_data = (train_data - means) / stds
    val_data = (val_data - means) / stds
    test_data = (test_data - means) / stds

    return train_data, val_data, test_data, lat, lon, means, stds


# In[42]:


train_time_scale = slice("2010-01-01", "2010-01-20")
val_time_scale = slice("2010-01-21", "2010-01-25")
test_time_scale = slice("2010-01-26", "2010-01-31")


# In[43]:


variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']


# In[44]:


train_data, val_data, test_data, lat, lon, means, stds = preprocess_dataset(
    ds, 
    variables, 
    train_time_scale, 
    val_time_scale, 
    test_time_scale
)

train_data = train_data.permute(1, 0, 2, 3)
val_data = val_data.permute(1, 0, 2, 3)
test_data = test_data.permute(1, 0, 2, 3)

print("Train Data Shape:", train_data.shape)
print("Validation Data Shape:", val_data.shape)
print("Test Data Shape:", test_data.shape)


# In[45]:


train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)
test_dataset = TensorDataset(test_data)

batch_size = 8  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for batch in train_loader:
    print("Train Batch Shape:", batch[0].shape)
    break


# In[46]:


class SimpleClimateModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleClimateModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# In[47]:


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=5):
    model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_data = batch[0].to(device)  # Los datos de entrada
            output = model(input_data)       # Salida del modelo
            loss = loss_fn(output, input_data)  # Comparar con entrada (autoencoder)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_data = batch[0].to(device)
                output = model(input_data)
                loss = loss_fn(output, input_data)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
        
        # Imprimir pérdidas
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses


# In[48]:


epochs = 100
input_channels = 5  # Número de variables
output_channels = 5  # Salida igual a la entrada
model = SimpleClimateModel(input_channels, output_channels)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_losses_1, val_losses_1 = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs)


# In[31]:


plt.plot(train_losses_1, label='Train Loss')
plt.plot(val_losses_1, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[49]:


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_data = batch[0].to(device)
            output = model(input_data)
            loss = loss_fn(output, input_data)
            total_loss += loss.item()
    return total_loss / len(data_loader)


# In[50]:


test_loss = evaluate_model(model, test_loader, loss_fn, device)
print(f"Test Loss: {test_loss:.4f}")


# In[51]:


def mean_absolute_error(model, data_loader, device):
    model.eval()
    total_mae = 0
    with torch.no_grad():
        for batch in data_loader:
            input_data = batch[0].to(device)
            output = model(input_data)
            mae = torch.abs(output - input_data).mean().item()
            total_mae += mae
    return total_mae / len(data_loader)

test_mae = mean_absolute_error(model, test_loader, device)
print(f"Mean Absolute Error (MAE): {test_mae:.4f}")


# In[55]:


def generate_spatial_embeddings(latitudes, longitudes):
    """
    Genera embeddings trigonométricos para latitud y longitud.
    """
    lat_rad = torch.tensor(np.radians(np.ravel(latitudes)), dtype=torch.float32)  # Convertir a 1D y radianes
    lon_rad = torch.tensor(np.radians(np.ravel(longitudes)), dtype=torch.float32)

    # Expandir dimensiones para generar una cuadrícula completa
    sin_lat = torch.sin(lat_rad).unsqueeze(1).expand(-1, len(lon_rad))  # Shape: (lat, lon)
    cos_lat = torch.cos(lat_rad).unsqueeze(1).expand(-1, len(lon_rad))
    sin_lon = torch.sin(lon_rad).unsqueeze(0).expand(len(lat_rad), -1)  # Shape: (lat, lon)
    cos_lon = torch.cos(lon_rad).unsqueeze(0).expand(len(lat_rad), -1)

    # Combinar embeddings en una cuadrícula de 4 canales
    spatial_embeddings = torch.stack([
        sin_lat * cos_lon, sin_lat * sin_lon,
        cos_lat * cos_lon, cos_lat * sin_lon
    ], dim=0)  # Shape: (4, lat, lon)

    return spatial_embeddings


# In[ ]:


def generate_temporal_embeddings(time_steps):
    """
    Genera embeddings temporales usando funciones trigonométricas.
    """
    day_of_year = (time_steps.dayofyear / 365.0) * 2 * np.pi
    hour_of_day = (time_steps.hour / 24.0) * 2 * np.pi

    # Embeddings seno y coseno
    sin_day = torch.sin(torch.tensor(day_of_year, dtype=torch.float32)).unsqueeze(1)
    cos_day = torch.cos(torch.tensor(day_of_year, dtype=torch.float32)).unsqueeze(1)
    sin_hour = torch.sin(torch.tensor(hour_of_day, dtype=torch.float32)).unsqueeze(1)
    cos_hour = torch.cos(torch.tensor(hour_of_day, dtype=torch.float32)).unsqueeze(1)

    # Combinar embeddings
    temporal_embeddings = torch.cat([sin_day, cos_day, sin_hour, cos_hour], dim=1)  # Shape: (time, 4)
    return temporal_embeddings


# In[56]:


latitudes = ds.latitude.values
longitudes = ds.longitude.values
spatial_embeddings = generate_spatial_embeddings(latitudes, longitudes)
print("Spatial Embeddings Shape:", spatial_embeddings.shape)  # Esperado: (4, lat, lon)


# In[57]:


time_steps = ds.time.to_index()  # Convertir a datetime
temporal_embeddings = generate_temporal_embeddings(time_steps)
print("Temporal Embeddings Shape:", temporal_embeddings.shape) 


# In[58]:


# Índices del tiempo de entrenamiento
train_time_indices = range(len(train_data))  # 80 pasos de tiempo

# Ajustar embeddings espaciales para el tiempo de entrenamiento
spatial_embeddings_expanded = spatial_embeddings.unsqueeze(0).expand(len(train_time_indices), -1, -1, -1)  # Shape: (80, 4, 157, 121)

# Ajustar embeddings temporales para el tiempo de entrenamiento
temporal_embeddings_expanded = temporal_embeddings[train_time_indices].unsqueeze(-1).unsqueeze(-1)  # (80, 4, 1, 1)
temporal_embeddings_expanded = temporal_embeddings_expanded.expand(-1, -1, len(ds.latitude), len(ds.longitude))  # Shape: (80, 4, 157, 121)

# Combinar datos climáticos con embeddings espaciales y temporales
combined_data = torch.cat([train_data, spatial_embeddings_expanded, temporal_embeddings_expanded], dim=1)
print("Combined Data Shape:", combined_data.shape)  # Esperado: (80, 13, 157, 121)


# In[59]:


class ClimateModelWithEmbeddings(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ClimateModelWithEmbeddings, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# In[60]:


combined_dataset = TensorDataset(combined_data)
combined_train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)


# In[61]:


spatial_embeddings_val = spatial_embeddings.unsqueeze(0).expand(len(val_data), -1, -1, -1) 

temporal_embeddings_val = temporal_embeddings[:len(val_data)].unsqueeze(-1).unsqueeze(-1)  
temporal_embeddings_val = temporal_embeddings_val.expand(-1, -1, len(ds.latitude), len(ds.longitude))  

combined_val_data = torch.cat([val_data, spatial_embeddings_val, temporal_embeddings_val], dim=1) 

combined_val_dataset = TensorDataset(combined_val_data)
combined_val_loader = DataLoader(combined_val_dataset, batch_size=8, shuffle=False)


# In[62]:


combined_val_data.shape


# In[63]:


# Ajustar embeddings temporales para el conjunto de prueba
temporal_embeddings_test = temporal_embeddings[:len(test_data)].unsqueeze(-1).unsqueeze(-1)  # (24, 4, 1, 1)
temporal_embeddings_test = temporal_embeddings_test.expand(-1, -1, len(ds.latitude), len(ds.longitude))  # (24, 4, 157, 121)

# Combinar datos climáticos con embeddings espaciales y temporales para prueba
spatial_embeddings_test = spatial_embeddings.unsqueeze(0).expand(len(test_data), -1, -1, -1)  # (24, 4, 157, 121)
combined_test_data = torch.cat([test_data, spatial_embeddings_test, temporal_embeddings_test], dim=1)  # (24, 13, 157, 121)

# Crear DataLoader para prueba
combined_test_dataset = TensorDataset(combined_test_data)
combined_test_loader = DataLoader(combined_test_dataset, batch_size=8, shuffle=False)


# In[64]:


combined_test_data.shape


# In[69]:


input_channels = 13  # 5 variables climáticas + 4 embeddings espaciales + 4 embeddings temporales
output_channels = 13  # Reconstruir todos los canales
model = ClimateModelWithEmbeddings(input_channels, output_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # Pérdida basada en el error cuadrático medio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import matplotlib.pyplot as plt

# Inicializar listas para guardar las pérdidas
train_losses = []
val_losses = []

epochs = 100  # Número de épocas
for epoch in range(epochs):
    # Modo entrenamiento
    model.train()
    train_loss = 0
    for batch in combined_train_loader:
        optimizer.zero_grad()
        input_data = batch[0].to(device)  # Usar datos combinados
        output = model(input_data)  # Pasada hacia adelante
        loss = loss_fn(output, input_data)  # Pérdida basada en la reconstrucción
        loss.backward()  # Gradiente hacia atrás
        optimizer.step()  # Actualizar los parámetros del modelo
        train_loss += loss.item()
    
    # Pérdida de entrenamiento promedio
    avg_train_loss = train_loss / len(combined_train_loader)
    train_losses.append(avg_train_loss)

    # Validación
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in combined_val_loader:  # Usar el nuevo val_loader combinado
            input_data = batch[0].to(device)  # Datos combinados
            output = model(input_data)
            loss = loss_fn(output, input_data)
            val_loss += loss.item()
    
    # Pérdida promedio de validación
    avg_val_loss = val_loss / len(combined_val_loader)
    val_losses.append(avg_val_loss)
    
    # Imprimir pérdidas de la época
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# Graficar Training y Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()


# In[70]:


model.eval()
test_loss = 0
with torch.no_grad():
    for batch in combined_test_loader:  # Usar el DataLoader combinado del conjunto de prueba
        input_data = batch[0].to(device)  # Datos combinados
        output = model(input_data)
        loss = loss_fn(output, input_data)  # Pérdida basada en la reconstrucción
        test_loss += loss.item()

avg_test_loss = test_loss / len(combined_test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")


# In[71]:


# Función para calcular el gradiente de un tensor en dirección espacial
def compute_gradient(x, dim):
    return torch.gradient(x, dim=dim)[0]

# Función que incorpora las restricciones físicas de advección
def physics_loss(temperature, u, v, dx, dy):
    # Gradientes de la temperatura en las direcciones espaciales
    grad_T_x = compute_gradient(temperature, 3)  # Gradiente en la dirección x (lon)
    grad_T_y = compute_gradient(temperature, 2)  # Gradiente en la dirección y (lat)
    
    # Cálculo de la advección (producto de la velocidad y gradiente)
    advection_x = u * grad_T_x
    advection_y = v * grad_T_y
    
    # Cálculo de la ecuación de advección
    loss = torch.mean((advection_x + advection_y)**2)  # La pérdida debe ser pequeña
    return loss

# Modificar la función de pérdida para incluir la física
def combined_loss(output, input_data, u, v, dx, dy):
    data_loss = nn.MSELoss()(output, input_data)  # Pérdida basada en la reconstrucción
    pde_loss = physics_loss(output, u, v, dx, dy)  # Pérdida basada en la física
    return data_loss + pde_loss  # Combinamos ambas pérdidas


# In[72]:


def get_wind_components(input_data):
    u = input_data[:, 3, :, :].unsqueeze(1)  # Agregar dimensión de canal
    v = input_data[:, 4, :, :].unsqueeze(1)  # Agregar dimensión de canal
    return u, v


# In[73]:


input_channels = 13  # 5 variables climáticas + 4 embeddings espaciales + 4 embeddings temporales
output_channels = 13  # Reconstruir todos los canales
model = ClimateModelWithEmbeddings(input_channels, output_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # Pérdida basada en el error cuadrático medio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Entrenamiento con PINNs
epochs = 100  # Número de épocas
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in combined_train_loader:
        optimizer.zero_grad()
        input_data = batch[0].to(device)  # Usar datos combinados
        output = model(input_data)  # Pasada hacia adelante
        
        # Asumiendo que tenemos los valores de viento (u, v)
        u, v = get_wind_components(input_data)  # Función para obtener u y v (debe ser definida)
        dx, dy = 1, 1  # Si es necesario, ajusta estos valores según la resolución espacial

        # Usar la función de pérdida combinada
        loss = combined_loss(output, input_data, u, v, dx, dy)
        loss.backward()  # Gradiente hacia atrás
        optimizer.step()  # Actualizar los parámetros del modelo
        train_loss += loss.item()
    
    # Pérdida promedio de entrenamiento para la época actual
    avg_train_loss = train_loss / len(combined_train_loader)
    
    # Evaluación
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in combined_val_loader:  # Usar el DataLoader combinado de validación
            input_data = batch[0].to(device)
            output = model(input_data)
            
            u, v = get_wind_components(input_data)  # Obtener u, v
            loss = combined_loss(output, input_data, u, v, dx, dy)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(combined_val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")


# In[74]:


# Evaluación en el conjunto de prueba
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in combined_test_loader:
        input_data = batch[0].to(device)  # Datos combinados del conjunto de prueba
        output = model(input_data)  # Predicción del modelo
        
        u, v = get_wind_components(input_data)  # Obtener componentes del viento
        dx, dy = 1, 1  # Ajustar si es necesario según la resolución espacial
        
        # Pérdida combinada (reconstrucción + física)
        loss = combined_loss(output, input_data, u, v, dx, dy)
        test_loss += loss.item()

# Pérdida promedio del conjunto de prueba
avg_test_loss = test_loss / len(combined_test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")


# In[75]:


# Elegir un batch del conjunto de prueba
batch = next(iter(combined_test_loader))
input_data = batch[0].to(device)  # Datos reales combinados
output = model(input_data).detach().cpu()  # Predicción del modelo (sin gradientes)

# Seleccionar una variable (ejemplo: temperatura en el canal 2)
variable_index = 2  # Índice de la variable climática a visualizar
real_data = input_data[:, variable_index, :, :].cpu().numpy()  # Datos reales
predicted_data = output[:, variable_index, :, :].numpy()  # Predicciones del modelo

# Elegir un paso de tiempo específico
time_step = 0  # Puedes cambiar el índice del batch aquí

# Visualizar los datos reales y predicciones
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Valores Reales")
plt.imshow(real_data[time_step], cmap="coolwarm")
plt.colorbar(label="Valor")

plt.subplot(1, 2, 2)
plt.title("Predicciones del Modelo")
plt.imshow(predicted_data[time_step], cmap="coolwarm")
plt.colorbar(label="Valor")

plt.tight_layout()
plt.show()


# In[76]:


def create_prediction_dataset(data, lags=1):
    """
    Crea un dataset donde la entrada es data[t-lags:t] y la salida es data[t].
    """
    X = []
    y = []
    for t in range(lags, len(data)):
        X.append(data[t-lags:t])  # Pasos anteriores como entrada
        y.append(data[t])         # Paso actual como salida (predicción)

    X = torch.stack(X)  # Convertir a tensor
    y = torch.stack(y)
    return X, y

# Generar dataset de predicción
lags = 1  # Número de pasos anteriores
train_X, train_y = create_prediction_dataset(train_data, lags)
val_X, val_y = create_prediction_dataset(val_data, lags)
test_X, test_y = create_prediction_dataset(test_data, lags)

print("Train Input Shape:", train_X.shape)  # (steps, lags, channels, lat, lon)
print("Train Target Shape:", train_y.shape)  # (steps, channels, lat, lon)


# In[77]:


class ClimatePredictionModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ClimatePredictionModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Combinar lags (promediar o sumar a través del tiempo)
        x = torch.mean(x, dim=1)  # Promediar a través de la dimensión de lags
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# In[78]:


# Crear Datasets y DataLoaders
train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)
test_dataset = TensorDataset(test_X, test_y)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Entrenar el modelo
epochs = 20
model = ClimatePredictionModel(input_channels=5, output_channels=5)  # 5 canales climáticos
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")


# In[79]:


model.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader):.4f}")


# In[80]:


# Elegir un lote de prueba
batch = next(iter(test_loader))
X_batch, y_batch = batch
X_batch, y_batch = X_batch.to(device), y_batch.to(device)

# Generar predicciones
model.eval()
with torch.no_grad():
    predictions = model(X_batch)

# Graficar valores reales vs predichos
import matplotlib.pyplot as plt

# Seleccionar un canal y paso de tiempo
time_step = 0
channel = 0  # Selecciona una variable climática (ejemplo: temperatura)

real_data = y_batch[time_step, channel].cpu().numpy()
predicted_data = predictions[time_step, channel].cpu().numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Valores Reales")
plt.imshow(real_data, cmap="coolwarm")
plt.colorbar(label="Valor")

plt.subplot(1, 2, 2)
plt.title("Valores Predichos")
plt.imshow(predicted_data, cmap="coolwarm")
plt.colorbar(label="Valor")

plt.tight_layout()
plt.show()

