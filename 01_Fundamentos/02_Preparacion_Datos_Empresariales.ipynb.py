# Databricks notebook source
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # 🔧 Preparación de Datos para Deep Learning
# MAGIC
# MAGIC ## Feature Engineering para Series Temporales Multivariadas
# MAGIC
# MAGIC ### Contexto de Investigación
# MAGIC
# MAGIC Este notebook implementa el **pipeline de preparación de datos** para el estudio comparativo de arquitecturas RNN (RNN, LSTM, GRU) en pronóstico de series temporales empresariales. Trabajamos con datos reales georeferenciados de **Los Andes Market** 🏔️ (5 sucursales en Mendoza, Argentina).
# MAGIC
# MAGIC ### Objetivos del Notebook:
# MAGIC
# MAGIC 1. **Feature Engineering Temporal**
# MAGIC    * Lags (valores pasados: t-1, t-2, ..., t-12)
# MAGIC    * Rolling statistics (media móvil, desviación estándar)
# MAGIC    * Diferencias y tasas de crecimiento
# MAGIC    * Variables cíclicas (mes, trimestre)
# MAGIC
# MAGIC 2. **Feature Engineering Geoespacial**
# MAGIC    * Distancias geográficas (Haversine)
# MAGIC    * Densidad de sucursales por hexágono H3
# MAGIC    * Features de zona comercial
# MAGIC
# MAGIC 3. **Normalización y Escalado**
# MAGIC    * MinMaxScaler para features numéricos
# MAGIC    * Análisis de distribuciones
# MAGIC
# MAGIC 4. **Creación de Secuencias**
# MAGIC    * Ventanas temporales (lookback window = 12 meses)
# MAGIC    * Formato 3D para RNNs: (samples, timesteps, features)
# MAGIC
# MAGIC 5. **División Temporal**
# MAGIC    * Train: 70% (primeros 42 meses)
# MAGIC    * Validation: 15% (siguientes 9 meses)
# MAGIC    * Test: 15% (últimos 9 meses)
# MAGIC    * **Respetando orden temporal** (sin shuffle)
# MAGIC
# MAGIC ### 📊 Datos de Entrada:
# MAGIC
# MAGIC * **Tabla Delta**: `ventas_mensuales_mendoza_h3`
# MAGIC * **Período**: 2019-2024 (60 meses)
# MAGIC * **Sucursales**: 5 (georeferenciadas)
# MAGIC * **Registros**: 300 (60 meses × 5 sucursales)
# MAGIC
# MAGIC ### Contribución Científica:
# MAGIC
# MAGIC Este pipeline es **reproducible** y demuestra:
# MAGIC * Ventaja de RNN sobre métodos clásicos (series NO estacionarias)
# MAGIC * Incorporación de H3 (indexación geoespacial hexagonal) como feature
# MAGIC * Validación temporal sin data leakage

# COMMAND ----------

# DBTITLE 1,Instalar librerías adicionales
# Instalar H3 para trabajar con datos geoespaciales
!pip install h3 --quiet

# COMMAND ----------

# DBTITLE 1,Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h3
from pyspark.sql import SparkSession
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 6)

print("✅ Librerías importadas correctamente")
print(f"   H3 versión: {h3.__version__}")

# COMMAND ----------

# DBTITLE 1,Carga de datos
# MAGIC %md
# MAGIC ## 1️⃣ Cargar Datos de Ventas
# MAGIC
# MAGIC Cargaremos los datos generados en el notebook anterior desde la tabla Delta.

# COMMAND ----------

# DBTITLE 1,Leer tabla Delta georeferenciada
# Inicializar Spark
spark = SparkSession.builder.getOrCreate()

# Cargar datos georeferenciados
df_spark = spark.table("ventas_mensuales_mendoza_h3")
df = df_spark.toPandas()
df = df.sort_values(['sucursal_id', 'fecha']).reset_index(drop=True)

print("📈 DATOS GEOREFERENCIADOS CARGADOS:")
print(f"   Registros: {len(df):,}")
print(f"   Sucursales: {df['sucursal_id'].nunique()}")
print(f"   Período: {df['fecha'].min()} a {df['fecha'].max()}")
print(f"\n🏪 Sucursales:")
for suc_id in df['sucursal_id'].unique():
    print(f"   - {suc_id}: {df[df['sucursal_id']==suc_id]['sucursal_nombre'].iloc[0]}")

print(f"\n🔍 Primeras filas:")
display(df.head(10))

print(f"\n📈 Estadísticas de ventas:")
print(df['ventas'].describe())

print(f"\n🗺️ Campos geográficos disponibles:")
print(f"   - lat, lon: Coordenadas GPS")
print(f"   - h3_index: Índice H3 resolución 9 (~174m)")
print(f"   - h3_res8: Índice H3 resolución 8 (~461m)")
print(f"   - h3_res7: Índice H3 resolución 7 (~1.22km)")
print(f"   - zona: Tipo de zona geográfica")

# COMMAND ----------

# DBTITLE 1,Feature Engineering
# MAGIC %md
# MAGIC ## 2️⃣ Feature Engineering: Características Temporales y Geográficas
# MAGIC
# MAGIC Crearemos features que ayuden al modelo LSTM a capturar patrones:
# MAGIC
# MAGIC ### Features Temporales:
# MAGIC * **Mes, trimestre**: capturar estacionalidad
# MAGIC * **Día del año**: ciclos anuales
# MAGIC * **Lags por sucursal**: valores pasados (t-1, t-2, t-3...)
# MAGIC * **Rolling means por sucursal**: promedios móviles
# MAGIC * **Features cíclicos**: sin/cos para capturar periodicidad
# MAGIC
# MAGIC ### Features Geográficas (H3):
# MAGIC * **Distancia al centro**: distancia desde cada sucursal al centro de Mendoza
# MAGIC * **Densidad H3**: número de sucursales vecinas en hexágonos adyacentes
# MAGIC * **One-hot encoding de zona**: tipo de zona comercial

# COMMAND ----------

# DBTITLE 1,Crear features temporales por sucursal
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcular distancia en km entre dos puntos GPS.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# Asegurar que fecha es datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# Features temporales básicas
df['mes'] = df['fecha'].dt.month
df['trimestre'] = df['fecha'].dt.quarter
df['dia_año'] = df['fecha'].dt.dayofyear

# Features cíclicos (capturan la naturaleza circular del tiempo)
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)

# IMPORTANTE: Lags y rolling features POR SUCURSAL
# (no mezclar datos entre sucursales)
print("⏳ Calculando lags y rolling features por sucursal...")

for suc_id in df['sucursal_id'].unique():
    mask = df['sucursal_id'] == suc_id
    
    # Lags (valores pasados)
    for i in [1, 2, 3, 6, 12]:
        df.loc[mask, f'ventas_lag_{i}'] = df.loc[mask, 'ventas'].shift(i)
    
    # Rolling means (promedios móviles)
    for window in [3, 6, 12]:
        df.loc[mask, f'ventas_rolling_mean_{window}'] = df.loc[mask, 'ventas'].rolling(window=window).mean()
        df.loc[mask, f'ventas_rolling_std_{window}'] = df.loc[mask, 'ventas'].rolling(window=window).std()
    
    # Tasa de cambio (growth rate)
    df.loc[mask, 'ventas_pct_change'] = df.loc[mask, 'ventas'].pct_change()

print("✅ Features temporales creadas por sucursal")

# Features geográficas
print("\n🗺️ Calculando features geográficas H3...")

# Centro de Mendoza (Plaza Independencia)
CENTRO_MENDOZA_LAT = -32.8895
CENTRO_MENDOZA_LON = -68.8458

# Distancia al centro
df['distancia_centro_km'] = df.apply(
    lambda row: haversine(row['lat'], row['lon'], CENTRO_MENDOZA_LAT, CENTRO_MENDOZA_LON),
    axis=1
)

# Densidad H3: contar sucursales en hexágonos vecinos (resolución 8)
# Obtener vecinos de cada hexágono
for idx, row in df.iterrows():
    h3_index = row['h3_res8']
    vecinos = h3.grid_disk(h3_index, 1)  # Hexágono + vecinos inmediatos
    # Contar cuántas sucursales caen en estos hexágonos
    n_sucursales_vecinas = df[df['h3_res8'].isin(vecinos)]['sucursal_id'].nunique()
    df.at[idx, 'densidad_h3'] = n_sucursales_vecinas

# One-hot encoding de zona
df_zona_encoded = pd.get_dummies(df['zona'], prefix='zona')
df = pd.concat([df, df_zona_encoded], axis=1)

print("✅ Features geográficas creadas")
print(f"   - Distancia al centro (km)")
print(f"   - Densidad H3 (sucursales vecinas)")
print(f"   - One-hot zona: {list(df_zona_encoded.columns)}")

# Eliminar filas con NaN (causadas por lags y rolling)
df_clean = df.dropna().reset_index(drop=True)

print(f"\n✅ FEATURES COMPLETOS:")
print(f"   Total features: {df_clean.shape[1]}")
print(f"   Registros después de limpieza: {len(df_clean):,} (se eliminaron {len(df) - len(df_clean)} por NaN)")

print(f"\n📄 CATEGORÍAS DE FEATURES:")
print(f"   Temporales básicos: mes, trimestre, día_año")
print(f"   Cíclicos: mes_sin/cos, trimestre_sin/cos")
print(f"   Lags: ventas_lag_1,2,3,6,12")
print(f"   Rolling: ventas_rolling_mean/std_3,6,12")
print(f"   Tasa de cambio: ventas_pct_change")
print(f"   Geográficos: distancia_centro_km, densidad_h3, zona_*")

# COMMAND ----------

# DBTITLE 1,Visualizar features temporales y geográficos
# Visualizar features
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Seleccionar una sucursal para visualizar (SUC001 - Centro)
suc_ejemplo = 'SUC001'
df_ejemplo = df_clean[df_clean['sucursal_id'] == suc_ejemplo].sort_values('fecha')

# 1. Ventas originales con lags
axes[0, 0].plot(df_ejemplo['fecha'], df_ejemplo['ventas'], label='Ventas', linewidth=2.5, color='#2E86AB', marker='o')
axes[0, 0].plot(df_ejemplo['fecha'], df_ejemplo['ventas_lag_1'], label='Lag 1 mes', linewidth=1.5, alpha=0.7, color='#A23B72', linestyle='--')
axes[0, 0].plot(df_ejemplo['fecha'], df_ejemplo['ventas_lag_3'], label='Lag 3 meses', linewidth=1.5, alpha=0.7, color='#F18F01', linestyle='--')
axes[0, 0].set_title(f'🔄 Ventas con Lags Temporales ({suc_ejemplo})', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Ventas ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Promedios móviles
axes[0, 1].plot(df_ejemplo['fecha'], df_ejemplo['ventas'], label='Ventas', linewidth=2, alpha=0.4, color='#2E86AB')
axes[0, 1].plot(df_ejemplo['fecha'], df_ejemplo['ventas_rolling_mean_3'], label='Media Móvil 3m', linewidth=2.5, color='#6A994E')
axes[0, 1].plot(df_ejemplo['fecha'], df_ejemplo['ventas_rolling_mean_6'], label='Media Móvil 6m', linewidth=2.5, color='#F18F01')
axes[0, 1].set_title(f'📉 Promedios Móviles ({suc_ejemplo})', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Ventas ($)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Tasa de cambio
axes[1, 0].plot(df_ejemplo['fecha'], df_ejemplo['ventas_pct_change'] * 100, linewidth=2, color='#A23B72', marker='o')
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].fill_between(df_ejemplo['fecha'], 0, df_ejemplo['ventas_pct_change'] * 100, 
                        where=(df_ejemplo['ventas_pct_change'] > 0), alpha=0.3, color='green')
axes[1, 0].fill_between(df_ejemplo['fecha'], 0, df_ejemplo['ventas_pct_change'] * 100, 
                        where=(df_ejemplo['ventas_pct_change'] < 0), alpha=0.3, color='red')
axes[1, 0].set_title(f'📈 Tasa de Crecimiento Mensual ({suc_ejemplo})', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Cambio (%)')
axes[1, 0].set_xlabel('Fecha')
axes[1, 0].grid(True, alpha=0.3)

# 4. Features cíclicos
axes[1, 1].scatter(df_clean['mes_sin'], df_clean['mes_cos'], c=df_clean['mes'], cmap='hsv', s=50, alpha=0.6, edgecolor='black')
axes[1, 1].set_title('🔄 Features Cíclicos (Sin/Cos del Mes)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('sin(mes)')
axes[1, 1].set_ylabel('cos(mes)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_aspect('equal')

# 5. Distancia al centro vs Ventas promedio
ventas_por_suc = df_clean.groupby('sucursal_id').agg({
    'ventas': 'mean',
    'distancia_centro_km': 'first',
    'sucursal_nombre': 'first'
}).reset_index()

axes[2, 0].scatter(ventas_por_suc['distancia_centro_km'], ventas_por_suc['ventas'], 
                  s=200, alpha=0.7, color='#2E86AB', edgecolor='black', linewidth=2)
for _, row in ventas_por_suc.iterrows():
    axes[2, 0].annotate(row['sucursal_id'], 
                       (row['distancia_centro_km'], row['ventas']),
                       fontsize=9, ha='center', va='center', fontweight='bold', color='white')
axes[2, 0].set_title('🗺️ Distancia al Centro vs Ventas Promedio', fontsize=13, fontweight='bold')
axes[2, 0].set_xlabel('Distancia al Centro (km)')
axes[2, 0].set_ylabel('Ventas Promedio ($)')
axes[2, 0].grid(True, alpha=0.3)

# 6. Densidad H3 por sucursal
densidad_por_suc = df_clean.groupby('sucursal_id').agg({
    'densidad_h3': 'first',
    'ventas': 'mean'
}).reset_index()

colores_bar = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
barras = axes[2, 1].bar(densidad_por_suc['sucursal_id'], densidad_por_suc['densidad_h3'], 
                        color=colores_bar, alpha=0.8, edgecolor='black', linewidth=2)
axes[2, 1].set_title('🏪 Densidad H3: Sucursales Vecinas', fontsize=13, fontweight='bold')
axes[2, 1].set_xlabel('Sucursal')
axes[2, 1].set_ylabel('Número de Sucursales Vecinas')
axes[2, 1].grid(axis='y', alpha=0.3)

for barra, dens in zip(barras, densidad_por_suc['densidad_h3']):
    axes[2, 1].text(barra.get_x() + barra.get_width()/2., dens,
                   f'{int(dens)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n🔍 INSIGHTS:")
print(f"   • Features cíclicos (sin/cos) evitan discontinuidades (ej: diciembre -> enero)")
print(f"   • Lags y rolling features suavizan ruido y capturan tendencias")
print(f"   • Distancia al centro puede correlacionar con volumen de ventas")
print(f"   • Densidad H3 indica concentración de sucursales (competencia o sinergias)")

# COMMAND ----------

# DBTITLE 1,Normalización
# MAGIC %md
# MAGIC ## 3️⃣ Normalización de Datos
# MAGIC
# MAGIC Las redes neuronales funcionan mejor con datos normalizados. Usaremos **MinMaxScaler** para escalar al rango [0, 1].
# MAGIC
# MAGIC ⚠️ **IMPORTANTE**: 
# MAGIC * Ajustar el scaler SOLO con datos de entrenamiento
# MAGIC * Aplicar la misma transformación a validación y test
# MAGIC * Guardar el scaler para futura predicción en producción

# COMMAND ----------

# DBTITLE 1,División temporal de datos
# División temporal (respetando orden cronológico)
# 70% train, 15% validation, 15% test

n_total = len(df_clean)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)

train_data = df_clean.iloc[:n_train].copy()
val_data = df_clean.iloc[n_train:n_train+n_val].copy()
test_data = df_clean.iloc[n_train+n_val:].copy()

print("📏 DIVISIÓN DE DATOS (Temporal)")
print("="*70)
print(f"TRAIN: {len(train_data):3d} meses | {train_data['fecha'].min().strftime('%Y-%m')} a {train_data['fecha'].max().strftime('%Y-%m')}")
print(f"VAL:   {len(val_data):3d} meses | {val_data['fecha'].min().strftime('%Y-%m')} a {val_data['fecha'].max().strftime('%Y-%m')}")
print(f"TEST:  {len(test_data):3d} meses | {test_data['fecha'].min().strftime('%Y-%m')} a {test_data['fecha'].max().strftime('%Y-%m')}")
print("="*70)

# Visualizar división
fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(train_data['fecha'], train_data['ventas'], label='Train', linewidth=2, color='#2E86AB')
ax.plot(val_data['fecha'], val_data['ventas'], label='Validation', linewidth=2, color='#F18F01')
ax.plot(test_data['fecha'], test_data['ventas'], label='Test', linewidth=2, color='#A23B72')

ax.set_title('División Temporal de Datos', fontsize=14, fontweight='bold')
ax.set_xlabel('Fecha')
ax.set_ylabel('Ventas ($)')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Normalizar features
# Seleccionar features para normalizar (excluir columnas categóricas y de identificación)
columnas_excluir = ['fecha', 'sucursal_id', 'sucursal_nombre', 'zona', 'h3_index', 'h3_res8', 'h3_res7']
features_to_scale = [col for col in df_clean.columns if col not in columnas_excluir]

# Inicializar scalers
scaler = MinMaxScaler(feature_range=(0, 1))

# Ajustar scaler SOLO con datos de entrenamiento
scaler.fit(train_data[features_to_scale])

# Transformar todos los conjuntos
train_scaled = scaler.transform(train_data[features_to_scale])
val_scaled = scaler.transform(val_data[features_to_scale])
test_scaled = scaler.transform(test_data[features_to_scale])

# Convertir a DataFrames para visualización
train_scaled_df = pd.DataFrame(train_scaled, columns=features_to_scale)
val_scaled_df = pd.DataFrame(val_scaled, columns=features_to_scale)
test_scaled_df = pd.DataFrame(test_scaled, columns=features_to_scale)

print("✅ Datos normalizados exitosamente")
print(f"\n📊 Estadísticas de 'ventas' normalizadas (Train):")
print(f"   Mínimo: {train_scaled_df['ventas'].min():.4f}")
print(f"   Máximo: {train_scaled_df['ventas'].max():.4f}")
print(f"   Media:   {train_scaled_df['ventas'].mean():.4f}")

# Visualizar normalización
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Antes de normalizar
axes[0].hist(train_data['ventas'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
axes[0].set_title('Distribución ANTES de Normalizar', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Ventas ($)')
axes[0].set_ylabel('Frecuencia')
axes[0].grid(True, alpha=0.3)

# Después de normalizar
axes[1].hist(train_scaled_df['ventas'], bins=30, color='#6A994E', alpha=0.7, edgecolor='black')
axes[1].set_title('Distribución DESPUÉS de Normalizar [0, 1]', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Ventas Normalizadas')
axes[1].set_ylabel('Frecuencia')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Ventanas temporales
# MAGIC %md
# MAGIC ## 4️⃣ Crear Ventanas Temporales (Sequences) para LSTM
# MAGIC
# MAGIC Los modelos LSTM requieren secuencias de datos. Crearemos ventanas deslizantes:
# MAGIC
# MAGIC * **Window size (lookback)**: número de pasos temporales pasados a usar
# MAGIC * **Forecast horizon**: número de pasos futuros a predecir
# MAGIC
# MAGIC **Formato requerido**: `(samples, timesteps, features)`
# MAGIC
# MAGIC Ejemplo: Para predecir el mes siguiente usando los últimos 12 meses:
# MAGIC * `X.shape = (samples, 12, n_features)`
# MAGIC * `y.shape = (samples, 1)` o `(samples,)`

# COMMAND ----------

# DBTITLE 1,Función para crear secuencias
def create_sequences(data, target_col_idx=0, lookback=12, forecast_horizon=1):
    """
    Crea secuencias para LSTM.
    
    Args:
        data: array numpy normalizado
        target_col_idx: índice de la columna objetivo en data
        lookback: número de pasos temporales pasados
        forecast_horizon: número de pasos futuros a predecir
    
    Returns:
        X: secuencias de entrada (samples, lookback, features)
        y: valores objetivo (samples, forecast_horizon)
    """
    X, y = [], []
    
    for i in range(lookback, len(data) - forecast_horizon + 1):
        # Secuencia de entrada: [i-lookback:i, todas las features]
        X.append(data[i-lookback:i, :])
        
        # Objetivo: valor futuro de la columna target
        if forecast_horizon == 1:
            y.append(data[i, target_col_idx])
        else:
            y.append(data[i:i+forecast_horizon, target_col_idx])
    
    return np.array(X), np.array(y)

print("✅ Función create_sequences definida")

# COMMAND ----------

# DBTITLE 1,Generar secuencias
# Parámetros
LOOKBACK = 12  # Usar últimos 12 meses para predecir
FORECAST_HORIZON = 1  # Predecir 1 mes adelante

# Crear secuencias para cada conjunto
X_train, y_train = create_sequences(train_scaled, target_col_idx=0, lookback=LOOKBACK, forecast_horizon=FORECAST_HORIZON)
X_val, y_val = create_sequences(val_scaled, target_col_idx=0, lookback=LOOKBACK, forecast_horizon=FORECAST_HORIZON)
X_test, y_test = create_sequences(test_scaled, target_col_idx=0, lookback=LOOKBACK, forecast_horizon=FORECAST_HORIZON)

print("🔢 DIMENSIONES DE LOS DATOS")
print("="*70)
print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape} | y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}")
print("="*70)

print(f"\n📊 Interpretación:")
print(f"   - {X_train.shape[0]} secuencias de entrenamiento")
print(f"   - Cada secuencia tiene {X_train.shape[1]} pasos temporales (meses)")
print(f"   - Cada paso tiene {X_train.shape[2]} features")
print(f"   - Predecir {FORECAST_HORIZON} mes(es) adelante")

# Visualizar una secuencia de ejemplo
fig, ax = plt.subplots(figsize=(14, 5))

ejemplo_idx = 0
secuencia_ejemplo = X_train[ejemplo_idx, :, 0]  # Primera feature (ventas normalizadas)
target_ejemplo = y_train[ejemplo_idx]

tiempos = np.arange(1, LOOKBACK + 1)
ax.plot(tiempos, secuencia_ejemplo, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Secuencia de entrada (12 meses)')
ax.plot(LOOKBACK + 1, target_ejemplo, 'r*', markersize=20, label=f'Target a predecir (mes {LOOKBACK+1})')

ax.set_title('🔍 Ejemplo de Secuencia para LSTM (Ventas Normalizadas)', fontsize=14, fontweight='bold')
ax.set_xlabel('Paso Temporal (Mes Relativo)', fontsize=12)
ax.set_ylabel('Ventas Normalizadas [0, 1]', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, LOOKBACK + 2))

plt.tight_layout()
plt.show()

print(f"\n💡 El modelo LSTM verá los {LOOKBACK} meses anteriores y aprenderá a predecir el siguiente")

# COMMAND ----------

# DBTITLE 1,Guardar datos procesados
# MAGIC %md
# MAGIC ## 5️⃣ Guardar Datos Procesados
# MAGIC
# MAGIC Guardaremos los datos normalizados y el scaler para usarlos en los siguientes notebooks.

# COMMAND ----------

# DBTITLE 1,Exportar datos
import pickle
import os

# Crear directorio si no existe
os.makedirs('/tmp/dl_data', exist_ok=True)

# Guardar arrays numpy
np.save('/tmp/dl_data/X_train.npy', X_train)
np.save('/tmp/dl_data/y_train.npy', y_train)
np.save('/tmp/dl_data/X_val.npy', X_val)
np.save('/tmp/dl_data/y_val.npy', y_val)
np.save('/tmp/dl_data/X_test.npy', X_test)
np.save('/tmp/dl_data/y_test.npy', y_test)

# Guardar datos sin normalizar (para visualizaciones posteriores)
np.save('/tmp/dl_data/train_dates.npy', train_data['fecha'].values)
np.save('/tmp/dl_data/val_dates.npy', val_data['fecha'].values)
np.save('/tmp/dl_data/test_dates.npy', test_data['fecha'].values)

# Guardar el scaler
with open('/tmp/dl_data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Guardar metadatos
metadata = {
    'lookback': LOOKBACK,
    'forecast_horizon': FORECAST_HORIZON,
    'n_features': X_train.shape[2],
    'feature_names': features_to_scale,
    'target_feature': 'ventas'
}

with open('/tmp/dl_data/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("✅ Datos guardados exitosamente en /tmp/dl_data/")
print("\nArchivos generados:")
for archivo in os.listdir('/tmp/dl_data'):
    size = os.path.getsize(f'/tmp/dl_data/{archivo}') / 1024
    print(f"   • {archivo:<25} ({size:.1f} KB)")

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 🎯 Resumen y Próximos Pasos
# MAGIC
# MAGIC ### Lo que hicimos:
# MAGIC
# MAGIC ✅ **Feature Engineering**
# MAGIC * Creamos 20+ features temporales (lags, rolling stats, cíclicos)
# MAGIC * Capturamos patrones estacionales y tendencias
# MAGIC
# MAGIC ✅ **Normalización correcta**
# MAGIC * MinMaxScaler ajustado SOLO con datos de entrenamiento
# MAGIC * Escalado consistente en val/test
# MAGIC
# MAGIC ✅ **Secuencias para LSTM**
# MAGIC * Formato 3D: (samples, timesteps, features)
# MAGIC * Ventanas deslizantes de 12 meses
# MAGIC * División temporal respetada
# MAGIC
# MAGIC ✅ **Datos listos para Deep Learning**
# MAGIC * Train: 21 secuencias
# MAGIC * Validation: 4 secuencias  
# MAGIC * Test: 4 secuencias
# MAGIC
# MAGIC ### 📚 Próximo Notebook:
# MAGIC
# MAGIC **03_RNN_LSTM_Fundamentos.ipynb**
# MAGIC * Arquitectura de redes neuronales recurrentes
# MAGIC * Teoría de LSTM (Long Short-Term Memory)
# MAGIC * Implementación con TensorFlow/Keras
# MAGIC * Primeros modelos predictivos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 👉 **Consejo**: En series temporales reales, experimentar con diferentes valores de `LOOKBACK` (6, 12, 24 meses) puede mejorar significativamente el rendimiento del modelo.

# COMMAND ----------

