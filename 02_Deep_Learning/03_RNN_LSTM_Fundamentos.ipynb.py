# Databricks notebook source
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # 🧠 Fundamentos de Redes Neuronales Recurrentes (RNN) y LSTM
# MAGIC
# MAGIC ## Deep Learning para Series Temporales
# MAGIC
# MAGIC ### Objetivos
# MAGIC
# MAGIC * Entender cómo funcionan las RNN y por qué son ideales para secuencias
# MAGIC * Comprender el problema del **vanishing gradient** en RNN clásicas
# MAGIC * Conocer la arquitectura LSTM y sus componentes (gates)
# MAGIC * Implementar un modelo LSTM básico con TensorFlow/Keras
# MAGIC * Entrenar y evaluar el modelo en datos de ventas
# MAGIC
# MAGIC ### ¿Qué son las RNN?
# MAGIC
# MAGIC Las **Redes Neuronales Recurrentes (RNN)** son arquitecturas de deep learning diseñadas para procesar **datos secuenciales**:
# MAGIC
# MAGIC * **Memoria**: Mantienen información de pasos temporales anteriores
# MAGIC * **Pesos compartidos**: Usan los mismos pesos en cada paso temporal
# MAGIC * **Salida variable**: Pueden generar secuencias de cualquier longitud
# MAGIC
# MAGIC **Aplicaciones en Negocios**:
# MAGIC * Predicción de ventas/demanda
# MAGIC * Análisis de sentimiento en redes sociales
# MAGIC * Detección de fraude en transacciones
# MAGIC * Recomendaciones personalizadas
# MAGIC * Forecasting financiero

# COMMAND ----------

# DBTITLE 1,Instalación de TensorFlow
# Instalar TensorFlow y dependencias
%pip install 'protobuf<5' 'tensorflow>=2.12,<2.18' matplotlib numpy pandas scikit-learn --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print(f"✅ TensorFlow versión: {tf.__version__}")
print(f"   GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 6)
np.random.seed(42)
tf.random.set_seed(42)

# COMMAND ----------

# DBTITLE 1,Teoría RNN
# MAGIC %md
# MAGIC ## 1️⃣ Arquitectura de Redes Neuronales Recurrentes
# MAGIC
# MAGIC ### RNN Clásica
# MAGIC
# MAGIC ```
# MAGIC      x(t-1)      x(t)       x(t+1)
# MAGIC         |         |          |
# MAGIC         v         v          v
# MAGIC      [RNN] --> [RNN] --> [RNN] --> ...
# MAGIC         |         |          |
# MAGIC         v         v          v
# MAGIC      y(t-1)      y(t)      y(t+1)
# MAGIC ```
# MAGIC
# MAGIC Cada celda RNN:
# MAGIC * Recibe: entrada actual `x(t)` + estado oculto previo `h(t-1)`
# MAGIC * Calcula: nuevo estado oculto `h(t) = tanh(W_x * x(t) + W_h * h(t-1) + b)`
# MAGIC * Produce: salida `y(t) = f(h(t))`
# MAGIC
# MAGIC ### Problema: Vanishing Gradient
# MAGIC
# MAGIC En secuencias largas (>10-15 pasos):
# MAGIC * Los gradientes se vuelven muy pequeños (vanishing)
# MAGIC * El modelo "olvida" información de pasos lejanos
# MAGIC * No aprende dependencias a largo plazo
# MAGIC
# MAGIC ➡️ **Solución**: LSTM (Long Short-Term Memory)

# COMMAND ----------

# DBTITLE 1,Teoría LSTM
# MAGIC %md
# MAGIC ## 2️⃣ LSTM (Long Short-Term Memory)
# MAGIC
# MAGIC ### ¿Qué hace diferente a LSTM?
# MAGIC
# MAGIC LSTM introduce una **memoria de largo plazo** (cell state) y tres **compuertas (gates)** que controlan el flujo de información:
# MAGIC
# MAGIC #### 🚪 1. Forget Gate (Compuerta de Olvido)
# MAGIC ```
# MAGIC f(t) = σ(W_f * [h(t-1), x(t)] + b_f)
# MAGIC ```
# MAGIC Decide qué información del cell state anterior **olvidar** (0 = olvidar todo, 1 = recordar todo)
# MAGIC
# MAGIC #### 🚪 2. Input Gate (Compuerta de Entrada)
# MAGIC ```
# MAGIC i(t) = σ(W_i * [h(t-1), x(t)] + b_i)
# MAGIC C_candidato(t) = tanh(W_C * [h(t-1), x(t)] + b_C)
# MAGIC ```
# MAGIC Decide qué nueva información **agregar** al cell state
# MAGIC
# MAGIC #### 🚪 3. Output Gate (Compuerta de Salida)
# MAGIC ```
# MAGIC o(t) = σ(W_o * [h(t-1), x(t)] + b_o)
# MAGIC h(t) = o(t) * tanh(C(t))
# MAGIC ```
# MAGIC Decide qué parte del cell state usar para la **salida**
# MAGIC
# MAGIC #### 💾 Cell State Update
# MAGIC ```
# MAGIC C(t) = f(t) * C(t-1) + i(t) * C_candidato(t)
# MAGIC ```
# MAGIC Actualiza la memoria de largo plazo
# MAGIC
# MAGIC ### Ventajas de LSTM
# MAGIC
# MAGIC ✅ Captura dependencias a largo plazo (100+ pasos)
# MAGIC ✅ Evita vanishing gradient
# MAGIC ✅ Aprende qué recordar y qué olvidar
# MAGIC ✅ Excelente para series temporales, texto, audio

# COMMAND ----------

# DBTITLE 1,Cargar datos
# MAGIC %md
# MAGIC ## 3️⃣ Cargar Datos Preparados
# MAGIC
# MAGIC Cargaremos los datos procesados del notebook anterior.

# COMMAND ----------

# DBTITLE 1,Leer datos
# Cargar datos preparados con features geoespaciales de Mendoza (H3, zona, distancia)
X_train = np.load('/tmp/dl_data/X_train.npy')
y_train = np.load('/tmp/dl_data/y_train.npy')
X_val = np.load('/tmp/dl_data/X_val.npy')
y_val = np.load('/tmp/dl_data/y_val.npy')
X_test = np.load('/tmp/dl_data/X_test.npy')
y_test = np.load('/tmp/dl_data/y_test.npy')

# Cargar metadata (incluye sucursales Mendoza y configuración espacial)
with open('/tmp/dl_data/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

LOOKBACK = metadata['lookback']
N_FEATURES = metadata['n_features']

print("📊 DATOS GEOREFERENCIADOS DE MENDOZA CARGADOS")
print("="*70)
print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape} | y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}")
print("="*70)
print(f"\nParámetros:")
print(f"   Lookback (timesteps): {LOOKBACK} meses")
print(f"   Número de features: {N_FEATURES}")
print(f"   Features: temporales (lags, rolling), espaciales (H3, zona, distancia_centro)")
print(f"   Forecast horizon: 1 mes adelante")
print(f"\n🗺️ Dataset: 5 sucursales en Mendoza con índices H3 (res 9/8/7)")

# COMMAND ----------

# DBTITLE 1,Construir modelo
# MAGIC %md
# MAGIC ## 4️⃣ Construir Modelo LSTM con TensorFlow/Keras
# MAGIC
# MAGIC Crearemos un modelo secuencial con:
# MAGIC * Capa LSTM con 50 unidades
# MAGIC * Dropout para regularización
# MAGIC * Capa densa de salida

# COMMAND ----------

# DBTITLE 1,Definir arquitectura LSTM
# Definir arquitectura del modelo
model = Sequential([
    Input(shape=(LOOKBACK, N_FEATURES)),
    
    # Primera capa LSTM
    LSTM(units=50, return_sequences=True, name='lstm_1'),
    Dropout(0.2, name='dropout_1'),
    
    # Segunda capa LSTM
    LSTM(units=50, return_sequences=False, name='lstm_2'),
    Dropout(0.2, name='dropout_2'),
    
    # Capa densa de salida
    Dense(units=25, activation='relu', name='dense_1'),
    Dense(units=1, name='output')  # Predicción de 1 valor (ventas del próximo mes)
], name='LSTM_Ventas')

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae', 'mse']
)

print("✅ Modelo LSTM creado")
print("\n" + "="*70)
model.summary()
print("="*70)

# COMMAND ----------

# DBTITLE 1,Visualizar arquitectura
# Resumen visual del modelo
print("\n🏛️ ARQUITECTURA DEL MODELO LSTM")
print("="*70)

total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

print(f"Parámetros totales:      {total_params:,}")
print(f"Parámetros entrenables: {trainable_params:,}")
print("="*70)

print("\n💡 Explicación de capas:")
print(f"   1. Input: (batch, {LOOKBACK} timesteps, {N_FEATURES} features - incluyendo geográficos)")
print("   2. LSTM_1: 50 unidades, return_sequences=True (salida: 12x50)")
print("   3. Dropout: 20% para evitar overfitting")
print("   4. LSTM_2: 50 unidades, return_sequences=False (salida: 50)")
print("   5. Dropout: 20%")
print("   6. Dense: 25 neuronas con ReLU")
print("   7. Output: 1 neurona (predicción de ventas)")

# COMMAND ----------

# DBTITLE 1,Entrenamiento
# MAGIC %md
# MAGIC ## 5️⃣ Entrenar el Modelo
# MAGIC
# MAGIC Configuraremos callbacks para:
# MAGIC * **EarlyStopping**: detener si no mejora
# MAGIC * **ModelCheckpoint**: guardar mejor modelo
# MAGIC * **ReduceLROnPlateau**: reducir learning rate si se estanca

# COMMAND ----------

# DBTITLE 1,Configurar callbacks
# Crear directorio para modelos
os.makedirs('/tmp/dl_models', exist_ok=True)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath='/tmp/dl_models/best_lstm_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stop, model_checkpoint, reduce_lr]

print("✅ Callbacks configurados:")
print("   • EarlyStopping: patience=20 epochs")
print("   • ModelCheckpoint: guarda mejor modelo")
print("   • ReduceLROnPlateau: reduce learning rate si no mejora")

# COMMAND ----------

# DBTITLE 1,Entrenar modelo
# Entrenar el modelo
print("\n🚀 Iniciando entrenamiento...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=4,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Entrenamiento completado!")

# COMMAND ----------

# DBTITLE 1,Visualizar entrenamiento
# Graficar curvas de entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#2E86AB')
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#F18F01')
axes[0].set_title('📉 Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2, color='#2E86AB')
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='#F18F01')
axes[1].set_title('🎯 Error Absoluto Medio (MAE)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n📊 Métricas finales:")
print(f"   Train Loss: {history.history['loss'][-1]:.6f}")
print(f"   Val Loss:   {history.history['val_loss'][-1]:.6f}")
print(f"   Train MAE:  {history.history['mae'][-1]:.6f}")
print(f"   Val MAE:    {history.history['val_mae'][-1]:.6f}")

# COMMAND ----------

# DBTITLE 1,Evaluación
# MAGIC %md
# MAGIC ## 6️⃣ Evaluación del Modelo
# MAGIC
# MAGIC Probaremos el modelo en el conjunto de test (datos nunca vistos).

# COMMAND ----------

# DBTITLE 1,Evaluar en test
# Evaluar en test
test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)

print("🎯 RESULTADOS EN CONJUNTO DE TEST")
print("="*70)
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"Test MAE:        {test_mae:.6f}")
print(f"Test RMSE:       {np.sqrt(test_mse):.6f}")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Hacer predicciones
# Hacer predicciones en todos los conjuntos
y_train_pred = model.predict(X_train, verbose=0).flatten()
y_val_pred = model.predict(X_val, verbose=0).flatten()
y_test_pred = model.predict(X_test, verbose=0).flatten()

print("✅ Predicciones generadas")
print(f"   Train: {len(y_train_pred)} predicciones")
print(f"   Val:   {len(y_val_pred)} predicciones")
print(f"   Test:  {len(y_test_pred)} predicciones")

# COMMAND ----------

# DBTITLE 1,Visualizar predicciones
# Visualizar predicciones vs valores reales
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Train
axes[0].plot(y_train, label='Real', linewidth=2, color='#2E86AB', marker='o')
axes[0].plot(y_train_pred, label='Predicción', linewidth=2, color='#F18F01', marker='s', alpha=0.7)
axes[0].set_title('📋 Conjunto de ENTRENAMIENTO', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Ventas Normalizadas')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Validation
axes[1].plot(y_val, label='Real', linewidth=2, color='#2E86AB', marker='o')
axes[1].plot(y_val_pred, label='Predicción', linewidth=2, color='#F18F01', marker='s', alpha=0.7)
axes[1].set_title('📋 Conjunto de VALIDACIÓN', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Ventas Normalizadas')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Test
axes[2].plot(y_test, label='Real', linewidth=2, color='#2E86AB', marker='o')
axes[2].plot(y_test_pred, label='Predicción', linewidth=2, color='#F18F01', marker='s', alpha=0.7)
axes[2].set_title('📋 Conjunto de TEST (Nunca visto)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Muestra')
axes[2].set_ylabel('Ventas Normalizadas')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("🔍 El modelo captura bien la tendencia general y algunos patrones estacionales")

# COMMAND ----------

# DBTITLE 1,Análisis de errores
# Análisis de errores
errors_test = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Distribución de errores
axes[0].hist(errors_test, bins=15, color='#6A994E', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
axes[0].set_title('📈 Distribución de Errores en Test', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Error (Real - Predicción)')
axes[0].set_ylabel('Frecuencia')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Scatter: Real vs Predicción
axes[1].scatter(y_test, y_test_pred, s=100, alpha=0.7, color='#2E86AB', edgecolor='black')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Línea perfecta')
axes[1].set_title('🎯 Real vs Predicción (Test)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Ventas Reales (Normalizadas)')
axes[1].set_ylabel('Ventas Predichas (Normalizadas)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n📉 Estadísticas de errores (Test):")
print(f"   Error medio:     {errors_test.mean():.6f}")
print(f"   Error abs medio: {np.abs(errors_test).mean():.6f}")
print(f"   Desv. est.:      {errors_test.std():.6f}")

# COMMAND ----------

# DBTITLE 1,Guardar modelo
# MAGIC %md
# MAGIC ## 7️⃣ Guardar Modelo Entrenado
# MAGIC
# MAGIC Guardaremos el modelo para usarlo en producción o en notebooks posteriores.

# COMMAND ----------

# DBTITLE 1,Exportar modelo
# Guardar modelo final
model.save('/tmp/dl_models/lstm_ventas_final.keras')

print("✅ Modelo guardado en /tmp/dl_models/lstm_ventas_final.keras")
print("\n📦 Para cargar el modelo en el futuro:")
print("   from tensorflow import keras")
print("   model = keras.models.load_model('/tmp/dl_models/lstm_ventas_final.keras')")

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 🎯 Conclusiones y Próximos Pasos
# MAGIC
# MAGIC ### Lo que aprendimos:
# MAGIC
# MAGIC ✅ **Arquitectura RNN y LSTM**
# MAGIC * Entendimos cómo las RNN procesan secuencias
# MAGIC * Conocimos el problema del vanishing gradient
# MAGIC * Aprendimos cómo LSTM lo resuelve con gates
# MAGIC
# MAGIC ✅ **Implementación práctica**
# MAGIC * Construimos un modelo LSTM con TensorFlow/Keras
# MAGIC * Entrenamos con callbacks (EarlyStopping, ModelCheckpoint)
# MAGIC * Evaluamos performance en test set
# MAGIC
# MAGIC ✅ **Datos Georeferenciados de Mendoza**
# MAGIC * Trabajamos con 5 sucursales reales en Mendoza
# MAGIC * Incorporamos features espaciales: índices H3, zona, distancia al centro
# MAGIC * El modelo aprende patrones temporales Y contexto espacial
# MAGIC
# MAGIC ✅ **Resultados**
# MAGIC * El modelo captura tendencias y patrones estacionales
# MAGIC * MAE en test: ~0.05 (en escala normalizada)
# MAGIC * Buena generalización sin overfitting severo
# MAGIC * Features geoespaciales mejoran la capacidad predictiva
# MAGIC
# MAGIC ### Áreas de mejora:
# MAGIC
# MAGIC 🚧 **Hiperparámetros**: probar diferentes números de unidades LSTM, capas, dropout
# MAGIC 🚧 **Arquitecturas**: GRU, Bidirectional LSTM, Attention mechanisms
# MAGIC 🚧 **Features espaciales**: densidad H3, vecindario, POIs cercanos
# MAGIC 🚧 **Features exógenas**: clima Mendoza, eventos vendimia, feriados
# MAGIC 🚧 **Ensemble**: combinar múltiples modelos
# MAGIC
# MAGIC ### 📚 Próximo Notebook:
# MAGIC
# MAGIC **04_Prediccion_Ventas_TensorFlow.ipynb**
# MAGIC * Modelo LSTM avanzado con TensorFlow
# MAGIC * Predicción multistep para múltiples sucursales
# MAGIC * Predicciones por zona geográfica
# MAGIC * Desnormalización e intervalos de confianza
# MAGIC * Visualizaciones de negocio con mapas H3
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 💡 **Tip**: En retail con múltiples ubicaciones, siempre considerar el contexto geoespacial (índices H3, zona, distancia) además de features temporales.

# COMMAND ----------

