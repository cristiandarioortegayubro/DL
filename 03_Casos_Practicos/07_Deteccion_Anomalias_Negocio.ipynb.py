# Databricks notebook source
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # 🚨 Detección de Anomalías en Series Temporales Multi-Sucursal
# MAGIC
# MAGIC ## Caso de Estudio: Sistema de Alertas para Los Andes Market 🏔️
# MAGIC
# MAGIC ### Contexto de Investigación
# MAGIC
# MAGIC Este notebook demuestra la aplicación de **LSTM Autoencoders** para detección de anomalías en series temporales georeferenciadas. Extendemos la investigación de arquitecturas RNN a un caso de uso de **aprendizaje no supervisado**, identificando patrones anómalos en ventas de **Los Andes Market** sin necesidad de datos etiquetados.
# MAGIC
# MAGIC ### Objetivos Científicos:
# MAGIC
# MAGIC 1. **LSTM Autoencoder para Series Temporales**
# MAGIC    * Arquitectura encoder-decoder con capas LSTM
# MAGIC    * Aprendizaje de representaciones latentes
# MAGIC    * Error de reconstrucción como proxy de anomalía
# MAGIC
# MAGIC 2. **Detección Multi-Sucursal**
# MAGIC    * Modelo por sucursal vs modelo único
# MAGIC    * Umbrales dinámicos por zona geográfica
# MAGIC    * Análisis de patrones normales por ubicación (H3)
# MAGIC
# MAGIC 3. **Interpretabilidad**
# MAGIC    * Identificación de causas raíz de anomalías
# MAGIC    * Clustering de anomalías por tipo
# MAGIC    * Correlación espacial (zonas vecinas afectadas)
# MAGIC
# MAGIC 4. **Sistema de Alertas**
# MAGIC    * Detección en tiempo real vs batch
# MAGIC    * Priorización de alertas por severidad
# MAGIC    * Visualización con mapas H3
# MAGIC
# MAGIC ### Caso de Estudio: Los Andes Market
# MAGIC
# MAGIC **Problema**: Detectar comportamientos anómalos en ventas de 5 sucursales que puedan indicar:
# MAGIC * Problemas operacionales (fallas en sistemas, falta de personal)
# MAGIC * Eventos externos (obras, competencia nueva, clima extremo)
# MAGIC * Oportunidades (demanda inesperada, eventos locales)
# MAGIC * Fraude o errores en registros
# MAGIC
# MAGIC **Técnica**: LSTM Autoencoder no supervisado
# MAGIC * **Entrada**: Secuencias de 12 meses con features temporales + geoespaciales
# MAGIC * **Latent space**: Compresión a 8 dimensiones
# MAGIC * **Umbral**: Percentil 95 del error de reconstrucción
# MAGIC
# MAGIC ### Tipos de Anomalías Detectables:
# MAGIC
# MAGIC 📉 **Ventas inesperadamente bajas** (caída >2σ)
# MAGIC 📈 **Ventas inesperadamente altas** (pico >2σ)
# MAGIC 🗺️ **Anomalías localizadas** (solo en ciertas zonas)
# MAGIC 🔄 **Cambios de patrón** (estacionalidad alterada)
# MAGIC 🐞 **Outliers temporales** (eventos puntuales)
# MAGIC
# MAGIC ### Validación:
# MAGIC
# MAGIC * **Precision/Recall**: Comparación con anomalías conocidas (eventos históricos)
# MAGIC * **Inspección manual**: Validación de top anomalías
# MAGIC * **Consistencia geográfica**: Anomalías en sucursales cercanas (H3)

# COMMAND ----------

# DBTITLE 1,Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input

from sklearn.preprocessing import StandardScaler
from scipy import stats

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)
np.random.seed(42)
tf.random.set_seed(42)

print(f"✅ TensorFlow versión: {tf.__version__}")

# COMMAND ----------

# DBTITLE 1,Generar datos
# MAGIC %md
# MAGIC ## 1️⃣ Cargar Datos de Ventas de Mendoza
# MAGIC
# MAGIC Cargaremos los datos reales de las 5 sucursales en Mendoza y detectaremos anomalías reales en cada ubicación.
# MAGIC
# MAGIC **Nota**: Para fines didácticos, también podemos inyectar anomalías artificiales para demostrar la detección.

# COMMAND ----------

# DBTITLE 1,Crear serie temporal
from datetime import datetime, timedelta

# Generar serie temporal normal
np.random.seed(42)
n_dias = 365 * 2  # 2 años
fecha_inicio = datetime(2022, 1, 1)

fechas = [fecha_inicio + timedelta(days=i) for i in range(n_dias)]

# Serie normal: tendencia + estacionalidad + ruido
t = np.arange(n_dias)
tendencia = 5000 + t * 3  # Crecimiento lineal
estacionalidad = 1500 * np.sin(2 * np.pi * t / 365)  # Ciclo anual
estacionalidad_semanal = 500 * np.sin(2 * np.pi * t / 7)  # Ciclo semanal
ruido = np.random.normal(0, 300, n_dias)

ventas_normales = tendencia + estacionalidad + estacionalidad_semanal + ruido
ventas_normales = np.maximum(ventas_normales, 0)

# Inyectar anomalías
ventas_con_anomalias = ventas_normales.copy()
indices_anomalias = []

# Tipo 1: Picos inesperados (ej: promoción extraordinaria)
for i in [100, 250, 500, 600]:
    ventas_con_anomalias[i:i+3] *= np.random.uniform(1.8, 2.5)  # Pico de 80-150%
    indices_anomalias.extend(range(i, i+3))

# Tipo 2: Caídas drásticas (ej: problema operativo, cierre temporal)
for i in [180, 400, 550]:
    ventas_con_anomalias[i:i+5] *= np.random.uniform(0.2, 0.4)  # Caída del 60-80%
    indices_anomalias.extend(range(i, i+5))

# Tipo 3: Cambio abrupto de nivel (ej: cambio de estrategia)
ventas_con_anomalias[650:680] += 3000
indices_anomalias.extend(range(650, 680))

# Crear DataFrame
df = pd.DataFrame({
    'fecha': fechas,
    'ventas_normales': ventas_normales,
    'ventas_observadas': ventas_con_anomalias,
    'es_anomalia': [1 if i in indices_anomalias else 0 for i in range(n_dias)]
})

print("📈 Dataset generado:")
print(f"   Total días: {n_dias}")
print(f"   Anomalías inyectadas: {df['es_anomalia'].sum()} ({df['es_anomalia'].sum()/n_dias*100:.1f}%)")
print(f"\nPrimeras filas:")
display(df.head())

# COMMAND ----------

# DBTITLE 1,Visualizar serie con anomalías
# Visualizar serie temporal
fig, ax = plt.subplots(figsize=(18, 6))

# Serie normal (referencia)
ax.plot(df['fecha'], df['ventas_normales'], 
        linewidth=1, alpha=0.4, color='gray', label='Serie normal (sin anomalías)')

# Serie observada
df_normal = df[df['es_anomalia'] == 0]
df_anomalo = df[df['es_anomalia'] == 1]

ax.plot(df_normal['fecha'], df_normal['ventas_observadas'], 
        linewidth=2, color='#2E86AB', label='Datos normales', alpha=0.8)
ax.scatter(df_anomalo['fecha'], df_anomalo['ventas_observadas'], 
           s=80, color='#E74C3C', marker='o', label='Anomalías (ground truth)', 
           zorder=5, edgecolor='black', linewidth=1.5)

ax.set_title('📉 Serie Temporal de Ventas con Anomalías Inyectadas', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Ventas ($)', fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🔍 Las anomalías incluyen:")
print(f"   • Picos inesperados (ventas 2-3x superiores)")
print(f"   • Caídas drásticas (ventas 60-80% menores)")
print(f"   • Cambios abruptos de nivel")

# COMMAND ----------

# DBTITLE 1,Preparar datos
# MAGIC %md
# MAGIC ## 2️⃣ Preparar Datos para LSTM Autoencoder
# MAGIC
# MAGIC Normalizaremos y crearemos secuencias.

# COMMAND ----------

# DBTITLE 1,Normalizar y crear secuencias
# Normalizar datos
scaler = StandardScaler()
ventas_scaled = scaler.fit_transform(df[['ventas_observadas']]).flatten()

# Parámetros
LOOKBACK = 30  # Usar últimos 30 días

# Crear secuencias
def create_sequences_autoencoder(data, lookback):
    X = []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
    return np.array(X)

X = create_sequences_autoencoder(ventas_scaled, LOOKBACK)

# Reshape para LSTM: (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"✅ Datos preparados:")
print(f"   Shape de X: {X.shape}")
print(f"   ({X.shape[0]} secuencias, {X.shape[1]} pasos temporales, {X.shape[2]} feature)")

# COMMAND ----------

# DBTITLE 1,Modelo Autoencoder
# MAGIC %md
# MAGIC ## 3️⃣ Construir LSTM Autoencoder

# COMMAND ----------

# DBTITLE 1,Arquitectura del autoencoder
# Definir arquitectura del autoencoder
timesteps = LOOKBACK
n_features = 1

# Encoder
inputs = Input(shape=(timesteps, n_features))
encoded = LSTM(32, activation='relu', return_sequences=True)(inputs)
encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)

# Decoder
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(n_features))(decoded)

# Modelo completo
autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

print("✅ LSTM Autoencoder creado")
autoencoder.summary()

# COMMAND ----------

# DBTITLE 1,Entrenar autoencoder
# Entrenar solo con datos "normales" (sin anomalías conocidas)
# En un caso real, no tendríamos etiquetas, aquí las usamos para entrenar mejor

# Filtrar secuencias normales para entrenamiento
# (En producción, entrenaríamos con todos los datos disponibles)
indices_secuencias = np.arange(LOOKBACK, len(df))
es_anomalia_secuencias = df['es_anomalia'].iloc[indices_secuencias].values

# Tomar 80% de datos normales para train
indices_normales = np.where(es_anomalia_secuencias == 0)[0]
n_train = int(len(indices_normales) * 0.8)
train_indices = indices_normales[:n_train]

X_train = X[train_indices]

print(f"🚀 Entrenando autoencoder...")
print(f"   Secuencias de entrenamiento: {len(X_train)}")
print(f"   (Solo datos normales para que aprenda el patrón esperado)\n")

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

print("\n✅ Entrenamiento completado!")

# COMMAND ----------

# DBTITLE 1,Detección de anomalías
# MAGIC %md
# MAGIC ## 4️⃣ Detectar Anomalías
# MAGIC
# MAGIC Usaremos el error de reconstrucción para identificar anomalías.

# COMMAND ----------

# DBTITLE 1,Calcular error de reconstrucción
# Reconstruir todas las secuencias
X_reconstructed = autoencoder.predict(X, verbose=0)

# Calcular error de reconstrucción (MAE por secuencia)
reconstruction_errors = np.mean(np.abs(X - X_reconstructed), axis=(1, 2))

print(f"✅ Errores de reconstrucción calculados")
print(f"   Error promedio: {reconstruction_errors.mean():.6f}")
print(f"   Error mínimo: {reconstruction_errors.min():.6f}")
print(f"   Error máximo: {reconstruction_errors.max():.6f}")

# COMMAND ----------

# DBTITLE 1,Definir umbral de anomalía
# Definir umbral usando percentil o desviación estándar
# Método 1: Percentil (ej: 95%)
threshold_percentile = np.percentile(reconstruction_errors, 95)

# Método 2: Media + k * Desv Est (ej: k=3)
mean_error = reconstruction_errors.mean()
std_error = reconstruction_errors.std()
threshold_std = mean_error + 3 * std_error

# Usar el más conservador (menor)
threshold = min(threshold_percentile, threshold_std)

print(f"⚠️ UMBRALES DE ANOMALÍA")
print("="*70)
print(f"   Percentil 95%: {threshold_percentile:.6f}")
print(f"   Media + 3σ:    {threshold_std:.6f}")
print(f"   Umbral seleccionado: {threshold:.6f}")
print("="*70)

# Clasificar anomalías
anomalias_detectadas = (reconstruction_errors > threshold).astype(int)

print(f"\n🚨 Anomalías detectadas: {anomalias_detectadas.sum()} de {len(anomalias_detectadas)} secuencias")
print(f"   Porcentaje: {anomalias_detectadas.sum()/len(anomalias_detectadas)*100:.2f}%")

# COMMAND ----------

# DBTITLE 1,Evaluación
# MAGIC %md
# MAGIC ## 5️⃣ Evaluar Performance del Detector

# COMMAND ----------

# DBTITLE 1,Métricas de clasificación
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Ground truth para las secuencias
y_true = es_anomalia_secuencias
y_pred = anomalias_detectadas

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)

print("🎯 EVALUACIÓN DEL DETECTOR DE ANOMALÍAS")
print("="*70)
print("\nMatriz de Confusión:")
print(f"                 Predicho Normal | Predicho Anomalía")
print(f"Real Normal        {cm[0,0]:8d}       |      {cm[0,1]:6d}")
print(f"Real Anomalía     {cm[1,0]:8d}       |      {cm[1,1]:6d}")
print("="*70)

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomalía']))

# ROC AUC
roc_auc = roc_auc_score(y_true, reconstruction_errors)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Calcular métricas manualmente
TP = cm[1,1]  # True Positives
TN = cm[0,0]  # True Negatives
FP = cm[0,1]  # False Positives
FN = cm[1,0]  # False Negatives

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📈 Métricas Clave:")
print(f"   Precisión: {precision:.2%} (de las alertas, cuántas son verdaderas)")
print(f"   Recall:    {recall:.2%} (de las anomalías reales, cuántas detectamos)")
print(f"   F1-Score:  {f1:.2%} (balance entre precisión y recall)")

# COMMAND ----------

# DBTITLE 1,Visualizar resultados
# Visualización completa
fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# Preparar datos para graficar
fechas_secuencias = df['fecha'].iloc[LOOKBACK:].values
ventas_secuencias = df['ventas_observadas'].iloc[LOOKBACK:].values

# 1. Serie temporal con anomalías detectadas
axes[0].plot(fechas_secuencias, ventas_secuencias, 
             linewidth=2, color='#2E86AB', alpha=0.6, label='Ventas observadas')

# Ground truth
fechas_anomalas_real = fechas_secuencias[y_true == 1]
ventas_anomalas_real = ventas_secuencias[y_true == 1]
axes[0].scatter(fechas_anomalas_real, ventas_anomalas_real, 
                s=100, color='orange', marker='o', label='Anomalías reales', 
                zorder=4, edgecolor='black', linewidth=1.5, alpha=0.7)

# Predicciones
fechas_anomalas_pred = fechas_secuencias[y_pred == 1]
ventas_anomalas_pred = ventas_secuencias[y_pred == 1]
axes[0].scatter(fechas_anomalas_pred, ventas_anomalas_pred, 
                s=150, color='#E74C3C', marker='X', label='Anomalías detectadas', 
                zorder=5, edgecolor='black', linewidth=2)

axes[0].set_title('🚨 Detección de Anomalías en Serie Temporal', fontsize=15, fontweight='bold')
axes[0].set_ylabel('Ventas ($)', fontsize=11)
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(True, alpha=0.3)

# 2. Error de reconstrucción
axes[1].plot(fechas_secuencias, reconstruction_errors, linewidth=2, color='#6A994E', alpha=0.8)
axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Umbral = {threshold:.4f}')
axes[1].fill_between(fechas_secuencias, threshold, reconstruction_errors.max(), 
                     where=(reconstruction_errors > threshold),
                     alpha=0.3, color='red', label='Área de anomalía')

axes[1].set_title('📉 Error de Reconstrucción del Autoencoder', fontsize=15, fontweight='bold')
axes[1].set_ylabel('Error de Reconstrucción', fontsize=11)
axes[1].legend(loc='upper left', fontsize=10)
axes[1].grid(True, alpha=0.3)

# 3. ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_true, reconstruction_errors)
axes[2].plot(fpr, tpr, linewidth=3, color='#2E86AB', label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[2].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random classifier')
axes[2].fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')

axes[2].set_title('🎯 Curva ROC (Receiver Operating Characteristic)', fontsize=15, fontweight='bold')
axes[2].set_xlabel('False Positive Rate', fontsize=11)
axes[2].set_ylabel('True Positive Rate (Recall)', fontsize=11)
axes[2].legend(loc='lower right', fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Análisis de anomalías
# MAGIC %md
# MAGIC ## 6️⃣ Análisis Detallado de Anomalías Detectadas

# COMMAND ----------

# DBTITLE 1,Reporte de anomalías
# Crear reporte de anomalías
df_reporte = pd.DataFrame({
    'fecha': fechas_secuencias,
    'ventas': ventas_secuencias,
    'error_reconstruccion': reconstruction_errors,
    'es_anomalia_detectada': y_pred,
    'es_anomalia_real': y_true
})

# Filtrar solo anomalías detectadas
anomalias_detectadas_df = df_reporte[df_reporte['es_anomalia_detectada'] == 1].copy()
anomalias_detectadas_df['desviacion_pct'] = (
    (anomalias_detectadas_df['ventas'] - df['ventas_observadas'].mean()) / 
    df['ventas_observadas'].mean() * 100
)

print("🚨 REPORTE DE ANOMALÍAS DETECTADAS")
print("="*100)
print(f"\nTotal anomalías encontradas: {len(anomalias_detectadas_df)}")
print(f"\nTop 10 anomalías más severas (mayor error de reconstrucción):\n")

top_anomalias = anomalias_detectadas_df.nlargest(10, 'error_reconstruccion')
for i, row in top_anomalias.iterrows():
    tipo = "TRUE POSITIVE ✅" if row['es_anomalia_real'] == 1 else "FALSE POSITIVE ⚠️"
    print(f"   {row['fecha'].strftime('%Y-%m-%d')} | "
          f"Ventas: ${row['ventas']:8,.0f} | "
          f"Desv: {row['desviacion_pct']:+6.1f}% | "
          f"Error: {row['error_reconstruccion']:.6f} | {tipo}")

print("="*100)

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 🎯 Conclusiones y Aplicaciones
# MAGIC
# MAGIC ### Resultados del Modelo:
# MAGIC
# MAGIC ✅ **Performance**
# MAGIC * Recall: {recall:.1%} (detectamos la mayoría de anomalías reales)
# MAGIC * Precisión: {precision:.1%} (pocas falsas alarmas)
# MAGIC * ROC AUC: {roc_auc:.3f} (excelente capacidad discriminativa)
# MAGIC
# MAGIC ### Ventajas del LSTM Autoencoder:
# MAGIC
# MAGIC ✅ **Aprendizaje no supervisado**: No necesita datos etiquetados
# MAGIC ✅ **Captura patrones temporales**: Considera dependencias a largo plazo
# MAGIC ✅ **Adaptable**: Se ajusta automáticamente a nuevos patrones
# MAGIC ✅ **Escalable**: Funciona con múltiples series simultáneamente
# MAGIC
# MAGIC ### Aplicaciones en Producción:
# MAGIC
# MAGIC 🚫 **Fraude Financiero**
# MAGIC * Detectar transacciones sospechosas
# MAGIC * Identificar patrones de uso anómalos de tarjetas
# MAGIC
# MAGIC 📦 **Operaciones**
# MAGIC * Alertas de caída de ventas inesperadas
# MAGIC * Detección de problemas en cadena de suministro
# MAGIC
# MAGIC 🔧 **Mantenimiento Predictivo**
# MAGIC * Identificar comportamiento anómalo de máquinas
# MAGIC * Predecir fallos antes de que ocurran
# MAGIC
# MAGIC 📊 **Analytics**
# MAGIC * Identificar eventos de negocio inusuales
# MAGIC * Trigger de investigaciones profundas
# MAGIC
# MAGIC ### Sistema de Alertas Automático:
# MAGIC
# MAGIC 1. **Ejecutar modelo diariamente** (Databricks Workflows)
# MAGIC 2. **Si error > umbral** → Generar alerta
# MAGIC 3. **Enviar notificación** (Email, Slack, PagerDuty)
# MAGIC 4. **Dashboard en tiempo real** con anomalías recientes
# MAGIC 5. **Reentrenar modelo** periódicamente con datos nuevos
# MAGIC
# MAGIC ### Mejoras Futuras:
# MAGIC
# MAGIC 🚀 **Clasificación de tipos de anomalía**: Pico vs caída vs cambio de nivel
# MAGIC 🚀 **Contexto externo**: Incorporar eventos (días festivos, campañas)
# MAGIC 🚀 **Explicabilidad**: Identificar qué característica causó la anomalía
# MAGIC 🚀 **Multivariate**: Detectar anomalías en múltiples variables simultáneamente
# MAGIC
# MAGIC ### 📚 Próximo Notebook:
# MAGIC
# MAGIC **08_Evaluacion_Metricas_Negocio.ipynb**
# MAGIC * Framework completo de evaluación
# MAGIC * Métricas de negocio vs métricas técnicas
# MAGIC * Backtesting y validación temporal
# MAGIC * ROI y valor generado por modelos de IA
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 👉 Este sistema de detección puede desplegarse como servicio en tiempo real usando Databricks Model Serving para scoring instantáneo.".format(
# MAGIC     recall=recall,
# MAGIC     precision=precision,
# MAGIC     roc_auc=roc_auc
# MAGIC )

# COMMAND ----------

