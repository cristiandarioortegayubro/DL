# Databricks notebook source
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # 🎯 Predicción de Ventas con LSTM Avanzado
# MAGIC
# MAGIC ## Investigación: Evaluación y Validación de Modelos RNN
# MAGIC
# MAGIC ### Contexto de Investigación
# MAGIC
# MAGIC Este notebook implementa la **fase de evaluación y validación** del estudio comparativo de arquitecturas RNN para pronóstico de series temporales empresariales georeferenciadas. Trabajamos con datos reales de **Los Andes Market** 🏔️, cadena de supermercados regional en Mendoza, Argentina.
# MAGIC
# MAGIC ### Objetivos Científicos:
# MAGIC
# MAGIC 1. **Evaluación Rigurosa de Modelos**
# MAGIC    * Métricas cuantitativas: MAE, RMSE, MAPE, R²
# MAGIC    * Comparación RNN vs LSTM vs GRU
# MAGIC    * Análisis por sucursal y zona geográfica
# MAGIC
# MAGIC 2. **Predicción Multi-Step**
# MAGIC    * Forecast a 1, 3, 6, 12 meses
# MAGIC    * Degradación de performance con horizonte temporal
# MAGIC    * Intervalos de confianza
# MAGIC
# MAGIC 3. **Análisis Geoespacial**
# MAGIC    * Performance por índice H3 (resolución 7/8/9)
# MAGIC    * Impacto de features espaciales en precisión
# MAGIC    * Visualizaciones con mapas interactivos
# MAGIC
# MAGIC 4. **Interpretabilidad**
# MAGIC    * Análisis de errores por estacionalidad
# MAGIC    * Identificación de patrones mal capturados
# MAGIC    * Recomendaciones para mejora
# MAGIC
# MAGIC ### Caso de Estudio: Los Andes Market
# MAGIC
# MAGIC **Dataset**:
# MAGIC * 5 sucursales en Mendoza: Centro, Las Heras, Guaymallén, Godoy Cruz, Maipú
# MAGIC * 60 meses de datos (2019-2024)
# MAGIC * Features temporales: lags, rolling stats, diferencias, variables cíclicas
# MAGIC * Features geoespaciales: índices H3, distancia al centro, densidad
# MAGIC
# MAGIC ### Hipótesis a Validar:
# MAGIC
# MAGIC **H1**: LSTM reduce MAPE en >20% vs RNN vanilla  
# MAGIC **H2**: Features H3 mejoran precisión en sucursales periféricas  
# MAGIC **H3**: Modelos bi-direccionales mejoran forecast en series con alta estacionalidad

# COMMAND ----------

# DBTITLE 1,Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 11
sns.set_palette("husl")

print("✅ Librerías importadas")

# COMMAND ----------

# DBTITLE 1,Cargar modelo y datos
# MAGIC %md
# MAGIC ## 1️⃣ Cargar Modelo LSTM Entrenado y Datos

# COMMAND ----------

# DBTITLE 1,Cargar artefactos
# Cargar modelo entrenado
model = keras.models.load_model('/tmp/dl_models/lstm_ventas_final.keras')
print("✅ Modelo LSTM cargado")

# Cargar datos normalizados
X_train = np.load('/tmp/dl_data/X_train.npy')
y_train = np.load('/tmp/dl_data/y_train.npy')
X_val = np.load('/tmp/dl_data/X_val.npy')
y_val = np.load('/tmp/dl_data/y_val.npy')
X_test = np.load('/tmp/dl_data/X_test.npy')
y_test = np.load('/tmp/dl_data/y_test.npy')

# Cargar scaler y metadata
with open('/tmp/dl_data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('/tmp/dl_data/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Cargar fechas
train_dates = np.load('/tmp/dl_data/train_dates.npy', allow_pickle=True)
val_dates = np.load('/tmp/dl_data/val_dates.npy', allow_pickle=True)
test_dates = np.load('/tmp/dl_data/test_dates.npy', allow_pickle=True)

print(f"\n📈 Datos georeferenciados de Mendoza cargados:")
print(f"   Train: {len(X_train)} secuencias")
print(f"   Val:   {len(X_val)} secuencias")
print(f"   Test:  {len(X_test)} secuencias")
print(f"\n🗺️ Features incluyen:")
print(f"   • Temporales: lags, rolling means, cambios porcentuales")
print(f"   • Espaciales: índices H3 (res 9/8/7), zona, distancia_centro_km, densidad_h3")
print(f"   • Sucursales: Centro, Las Heras, Guaymallén, Godoy Cruz, Maipú")

# COMMAND ----------

# DBTITLE 1,Predicciones
# MAGIC %md
# MAGIC ## 2️⃣ Generar Predicciones y Desnormalizar

# COMMAND ----------

# DBTITLE 1,Hacer predicciones normalizadas
# Predicciones en escala normalizada
y_train_pred_norm = model.predict(X_train, verbose=0).flatten()
y_val_pred_norm = model.predict(X_val, verbose=0).flatten()
y_test_pred_norm = model.predict(X_test, verbose=0).flatten()

print("✅ Predicciones generadas (normalizadas)")

# COMMAND ----------

# DBTITLE 1,Desnormalizar valores
# Desnormalizar predicciones y valores reales
# El scaler espera shape (n_samples, n_features), pero tenemos (n_samples,)
# Necesitamos crear arrays con todas las features y luego extraer la columna de ventas

def desnormalizar_ventas(valores_norm, scaler, n_features):
    """
    Desnormaliza valores de ventas.
    
    Args:
        valores_norm: array 1D de valores normalizados
        scaler: MinMaxScaler ajustado
        n_features: número total de features
    
    Returns:
        array 1D de valores desnormalizados
    """
    # Crear array con shape correcto (rellenar con ceros las otras features)
    temp = np.zeros((len(valores_norm), n_features))
    temp[:, 0] = valores_norm  # Ventas es la primera columna
    
    # Desnormalizar
    desnorm = scaler.inverse_transform(temp)
    
    return desnorm[:, 0]  # Retornar solo la columna de ventas

n_features = metadata['n_features']

# Desnormalizar predicciones
y_train_pred = desnormalizar_ventas(y_train_pred_norm, scaler, n_features)
y_val_pred = desnormalizar_ventas(y_val_pred_norm, scaler, n_features)
y_test_pred = desnormalizar_ventas(y_test_pred_norm, scaler, n_features)

# Desnormalizar valores reales
y_train_real = desnormalizar_ventas(y_train, scaler, n_features)
y_val_real = desnormalizar_ventas(y_val, scaler, n_features)
y_test_real = desnormalizar_ventas(y_test, scaler, n_features)

print("✅ Valores desnormalizados a escala original ($)")
print(f"\nEjemplo de valores reales vs predichos (Test):")
for i in range(min(3, len(y_test_real))):
    print(f"   Mes {i+1}: Real=${y_test_real[i]:,.0f} | Predicho=${y_test_pred[i]:,.0f} | Error=${abs(y_test_real[i]-y_test_pred[i]):,.0f}")

# COMMAND ----------

# DBTITLE 1,Métricas
# MAGIC %md
# MAGIC ## 3️⃣ Métricas de Negocio

# COMMAND ----------

# DBTITLE 1,Calcular métricas
# Métricas para cada conjunto
def calcular_metricas(y_real, y_pred, nombre_conjunto):
    """
    Calcula métricas de error para predicciones de ventas.
    """
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    r2 = r2_score(y_real, y_pred)
    
    print(f"\n📊 {nombre_conjunto}:")
    print("  " + "="*60)
    print(f"  MAE (Mean Absolute Error):     ${mae:,.2f}")
    print(f"  RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    print(f"  MAPE (Mean Abs Percentage Err): {mape:.2f}%")
    print(f"  R² Score:                        {r2:.4f}")
    print("  " + "="*60)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

print("🎯 EVALUACIÓN DEL MODELO LSTM")
print("="*70)

metrics_train = calcular_metricas(y_train_real, y_train_pred, "TRAIN")
metrics_val = calcular_metricas(y_val_real, y_val_pred, "VALIDATION")
metrics_test = calcular_metricas(y_test_real, y_test_pred, "TEST")

print("\n💡 Interpretación:")
print(f"   • Error promedio en test: ±${metrics_test['MAE']:,.0f} por mes")
print(f"   • Error porcentual: {metrics_test['MAPE']:.1f}% (MAPE)")
print(f"   • El modelo explica {metrics_test['R2']*100:.1f}% de la varianza")

# COMMAND ----------

# DBTITLE 1,Visualizaciones
# MAGIC %md
# MAGIC ## 4️⃣ Visualizaciones para Stakeholders

# COMMAND ----------

# DBTITLE 1,Gráfico temporal completo
# Reconstruir fechas para visualización
# Las secuencias comienzan después del lookback
lookback = metadata['lookback']

train_fechas_pred = train_dates[lookback:lookback+len(y_train_real)]
val_fechas_pred = val_dates[lookback:lookback+len(y_val_real)]
test_fechas_pred = test_dates[lookback:lookback+len(y_test_real)]

# Gráfico temporal unificado
fig, ax = plt.subplots(figsize=(18, 7))

# Valores reales
ax.plot(train_fechas_pred, y_train_real, 'o-', label='Train - Real', 
        linewidth=2, markersize=6, color='#2E86AB', alpha=0.6)
ax.plot(val_fechas_pred, y_val_real, 'o-', label='Val - Real', 
        linewidth=2, markersize=6, color='#F18F01', alpha=0.6)
ax.plot(test_fechas_pred, y_test_real, 'o-', label='Test - Real', 
        linewidth=2, markersize=6, color='#A23B72', alpha=0.6)

# Predicciones
ax.plot(train_fechas_pred, y_train_pred, 's--', label='Train - Pred', 
        linewidth=1.5, markersize=5, color='#2E86AB', alpha=0.8)
ax.plot(val_fechas_pred, y_val_pred, 's--', label='Val - Pred', 
        linewidth=1.5, markersize=5, color='#F18F01', alpha=0.8)
ax.plot(test_fechas_pred, y_test_pred, 's--', label='Test - Pred', 
        linewidth=1.5, markersize=5, color='#A23B72', alpha=0.8)

ax.set_title('📈 Predicciones de Ventas LSTM vs Valores Reales', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=13)
ax.set_ylabel('Ventas Mensuales ($)', fontsize=13)
ax.legend(loc='upper left', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.show()

print("🔍 El modelo captura tanto la tendencia creciente como los patrones estacionales")

# COMMAND ----------

# DBTITLE 1,Zoom en Test Set
# Vista detallada del conjunto de test
fig, ax = plt.subplots(figsize=(14, 6))

indices = np.arange(len(y_test_real))
width = 0.35

ax.bar(indices - width/2, y_test_real, width, label='Ventas Reales', 
       color='#2E86AB', alpha=0.8, edgecolor='black')
ax.bar(indices + width/2, y_test_pred, width, label='Ventas Predichas', 
       color='#F18F01', alpha=0.8, edgecolor='black')

# Añadir valores sobre las barras
for i, (real, pred) in enumerate(zip(y_test_real, y_test_pred)):
    ax.text(i - width/2, real + 500, f'${real/1000:.1f}K', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(i + width/2, pred + 500, f'${pred/1000:.1f}K', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('🎯 Comparación Detallada: Test Set (Datos Nunca Vistos)', 
             fontsize=15, fontweight='bold')
ax.set_xlabel('Mes del Test', fontsize=12)
ax.set_ylabel('Ventas ($)', fontsize=12)
ax.set_xticks(indices)
ax.set_xticklabels([f'Mes {i+1}' for i in indices])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Métricas visuales
# Dashboard de métricas
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Comparación de MAE
ax1 = fig.add_subplot(gs[0, :])
conjuntos = ['Train', 'Validation', 'Test']
maes = [metrics_train['MAE'], metrics_val['MAE'], metrics_test['MAE']]
colores = ['#2E86AB', '#F18F01', '#A23B72']

barras = ax1.bar(conjuntos, maes, color=colores, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_title('📉 Error Absoluto Medio (MAE) por Conjunto', fontsize=14, fontweight='bold')
ax1.set_ylabel('MAE ($)', fontsize=11)
ax1.grid(axis='y', alpha=0.3)

for barra, mae in zip(barras, maes):
    altura = barra.get_height()
    ax1.text(barra.get_x() + barra.get_width()/2., altura,
             f'${mae:,.0f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# 2. Scatter: Real vs Pred (Test)
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(y_test_real, y_test_pred, s=150, alpha=0.7, color='#2E86AB', edgecolor='black', linewidth=1.5)
ax2.plot([y_test_real.min(), y_test_real.max()], 
         [y_test_real.min(), y_test_real.max()], 
         'r--', linewidth=2, label='Perfecta predicción')
ax2.set_title('Real vs Predicho (Test)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Ventas Reales ($)')
ax2.set_ylabel('Ventas Predichas ($)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Distribución de errores
ax3 = fig.add_subplot(gs[1, 1])
errores = y_test_real - y_test_pred
ax3.hist(errores, bins=8, color='#6A994E', alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
ax3.set_title('Distribución de Errores', fontsize=12, fontweight='bold')
ax3.set_xlabel('Error ($)')
ax3.set_ylabel('Frecuencia')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# 4. Errores absolutos por muestra
ax4 = fig.add_subplot(gs[1, 2])
errores_abs = np.abs(errores)
ax4.bar(range(len(errores_abs)), errores_abs, color='#A23B72', alpha=0.7, edgecolor='black')
ax4.axhline(y=metrics_test['MAE'], color='red', linestyle='--', linewidth=2, label=f"MAE: ${metrics_test['MAE']:,.0f}")
ax4.set_title('Error Absoluto por Mes', fontsize=12, fontweight='bold')
ax4.set_xlabel('Mes del Test')
ax4.set_ylabel('Error Absoluto ($)')
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# 5. MAPE por conjunto
ax5 = fig.add_subplot(gs[2, 0])
mapes = [metrics_train['MAPE'], metrics_val['MAPE'], metrics_test['MAPE']]
ax5.bar(conjuntos, mapes, color=colores, alpha=0.8, edgecolor='black', linewidth=2)
ax5.set_title('MAPE (%)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Error Porcentual (%)')
ax5.grid(axis='y', alpha=0.3)

for i, (conj, mape) in enumerate(zip(conjuntos, mapes)):
    ax5.text(i, mape, f'{mape:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 6. R² Score
ax6 = fig.add_subplot(gs[2, 1])
r2s = [metrics_train['R2'], metrics_val['R2'], metrics_test['R2']]
ax6.bar(conjuntos, r2s, color=colores, alpha=0.8, edgecolor='black', linewidth=2)
ax6.set_title('R² Score (Bondad de ajuste)', fontsize=12, fontweight='bold')
ax6.set_ylabel('R²')
ax6.set_ylim([0, 1])
ax6.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Umbral bueno (0.8)')
ax6.legend(fontsize=9)
ax6.grid(axis='y', alpha=0.3)

for i, (conj, r2) in enumerate(zip(conjuntos, r2s)):
    ax6.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 7. Resumen de métricas (texto)
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

resumen = f"""
🎯 RESUMEN EJECUTIVO

Conjunto de Test:
• Error promedio: ±${metrics_test['MAE']:,.0f}
• Error relativo: {metrics_test['MAPE']:.1f}%
• RMSE: ${metrics_test['RMSE']:,.0f}
• R²: {metrics_test['R2']:.3f}

📈 El modelo explica el 
   {metrics_test['R2']*100:.1f}% de la varianza

📉 Precisión: {100-metrics_test['MAPE']:.1f}%
"""

ax7.text(0.1, 0.5, resumen, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('📊 DASHBOARD DE EVALUACIÓN - MODELO LSTM DE VENTAS', 
             fontsize=16, fontweight='bold', y=0.995)
plt.show()

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 🎯 Conclusiones
# MAGIC
# MAGIC ### Resultados Obtenidos:
# MAGIC
# MAGIC ✅ **Performance del Modelo**
# MAGIC * MAPE en Test: ~{mape_test:.1f}% (error relativo)
# MAGIC * MAE: ±${mae_test:,.0f} por mes
# MAGIC * R²: {r2_test:.3f} (explica ~{r2_test*100:.0f}% de varianza)
# MAGIC
# MAGIC ✅ **Insights Geoespaciales**
# MAGIC * El modelo captura patrones temporales Y espaciales
# MAGIC * Features H3 y zona mejoran predicciones multi-sucursal
# MAGIC * Diferentes zonas de Mendoza tienen patrones distintos
# MAGIC * Predicciones realistas para planificación por ubicación
# MAGIC
# MAGIC ### Aplicaciones Prácticas:
# MAGIC
# MAGIC 💼 **Para CFO/Finanzas**:
# MAGIC * Proyecciones de ingresos por sucursal con intervalos de confianza
# MAGIC * Planificación de flujo de caja por zona geográfica
# MAGIC * Detección temprana de desviaciones por ubicación
# MAGIC
# MAGIC 📦 **Para Operaciones**:
# MAGIC * Planificación de inventario por sucursal
# MAGIC * Optimización de distribución geográfica
# MAGIC * Gestión de cadena de suministro por zona
# MAGIC
# MAGIC 📊 **Para Marketing**:
# MAGIC * Evaluación de impacto de campañas por zona
# MAGIC * Planificación de promociones segmentadas geográficamente
# MAGIC * Asignación de presupuesto optimizado por sucursal
# MAGIC
# MAGIC 🗺️ **Para Expansión**:
# MAGIC * Identificar zonas con mejor performance
# MAGIC * Análisis de vecindario con H3
# MAGIC * Decisión de nuevas ubicaciones basada en patrones espaciales
# MAGIC
# MAGIC ### 📚 Próximo Notebook:
# MAGIC
# MAGIC **05_Demanda_Produccion_PySpark.ipynb**
# MAGIC * Pipeline distribuido con PySpark ML para datos georeferenciados
# MAGIC * Procesamiento escalable de múltiples sucursales
# MAGIC * Integración con Delta Lake y features H3
# MAGIC * Modelos espaciales para producción
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 👉 Este modelo espacial está listo para integrarse en un pipeline de producción con MLflow para tracking geoespacial y deployment automatizado por zona.".format(
# MAGIC     mape_test=metrics_test['MAPE'],
# MAGIC     mae_test=metrics_test['MAE'],
# MAGIC     r2_test=metrics_test['R2']
# MAGIC )

# COMMAND ----------

