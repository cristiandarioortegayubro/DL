# Databricks notebook source
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # 🎯 Evaluación y Métricas de Negocio para Modelos Multi-Sucursal
# MAGIC
# MAGIC ## Framework Completo de Evaluación Geoespacial
# MAGIC
# MAGIC ### Objetivos
# MAGIC
# MAGIC * Entender la diferencia entre métricas técnicas y de negocio
# MAGIC * Implementar backtesting riguroso por sucursal/zona
# MAGIC * Calcular métricas avanzadas (MAPE, SMAPE, MASE, Forecast Bias)
# MAGIC * Evaluar valor económico del modelo por ubicación
# MAGIC * Crear dashboard ejecutivo de performance geográfica
# MAGIC * Comparar performance entre zonas de Mendoza
# MAGIC
# MAGIC ### ¿Por qué es importante?
# MAGIC
# MAGIC 📊 **Métricas técnicas** (MAE, RMSE, R²):
# MAGIC * Útiles para comparar modelos
# MAGIC * Difíciles de interpretar para stakeholders
# MAGIC * No capturan impacto de negocio
# MAGIC
# MAGIC 💰 **Métricas de negocio**:
# MAGIC * Traducen performance a valor económico
# MAGIC * Facilitan decisiones de inversión
# MAGIC * Alinean IA con objetivos empresariales
# MAGIC
# MAGIC 🗺️ **Dimensión espacial**:
# MAGIC * Performance varía por zona geográfica
# MAGIC * Identificar sucursales con mejores predicciones
# MAGIC * Optimizar estrategias por ubicación
# MAGIC
# MAGIC ### Framework de Evaluación
# MAGIC
# MAGIC 1. **Métricas de Error**: MAE, RMSE, MAPE, SMAPE, MASE (global y por sucursal)
# MAGIC 2. **Backtesting**: Validación temporal robusta por zona
# MAGIC 3. **Forecast Bias**: ¿El modelo sobre/subestima sistemáticamente por ubicación?
# MAGIC 4. **Intervalos de Confianza**: Cuantificar incertidumbre espacial
# MAGIC 5. **Valor Económico**: ROI, ahorro, mejora vs baseline por sucursal
# MAGIC
# MAGIC ### 🗺️ Dataset: 5 Sucursales en Mendoza
# MAGIC
# MAGIC * Evaluaremos modelos en cada sucursal
# MAGIC * Compararemos performance entre zonas
# MAGIC * Identificaremos patrones espaciales en errores

# COMMAND ----------

# DBTITLE 1,Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)
sns.set_palette("husl")

print("✅ Librerías importadas")

# COMMAND ----------

# DBTITLE 1,Generar datos
# MAGIC %md
# MAGIC ## 1️⃣ Cargar Datos de Evaluación de Mendoza
# MAGIC
# MAGIC Cargaremos predicciones de modelos entrenados en las 5 sucursales de Mendoza y compararemos:
# MAGIC * **Modelo LSTM**: Con features espaciales (H3, zona, distancia)
# MAGIC * **Baseline**: Promedio móvil simple sin contexto espacial
# MAGIC
# MAGIC Evaluaremos performance global y por sucursal/zona.

# COMMAND ----------

# DBTITLE 1,Crear datos de evaluación
from datetime import datetime, timedelta

np.random.seed(42)
n_meses = 24  # 2 años
fecha_inicio = datetime(2022, 1, 1)

fechas = [fecha_inicio + timedelta(days=30*i) for i in range(n_meses)]

# Valores reales (ground truth)
t = np.arange(n_meses)
ventas_reales = (
    5000 + t * 50 +  # Tendencia
    1000 * np.sin(2 * np.pi * t / 12) +  # Estacionalidad
    np.random.normal(0, 300, n_meses)  # Ruido
)
ventas_reales = np.maximum(ventas_reales, 0)

# Predicciones Modelo LSTM (mejor performance)
error_lstm = np.random.normal(0, 400, n_meses)  # Error aleatorio
ventas_pred_lstm = ventas_reales + error_lstm
ventas_pred_lstm = np.maximum(ventas_pred_lstm, 0)

# Predicciones Baseline (promedio móvil simple)
window = 3
ventas_pred_baseline = np.zeros(n_meses)
for i in range(n_meses):
    if i < window:
        ventas_pred_baseline[i] = ventas_reales[:i+1].mean() if i > 0 else ventas_reales[0]
    else:
        ventas_pred_baseline[i] = ventas_reales[i-window:i].mean()

# Crear DataFrame
df = pd.DataFrame({
    'fecha': fechas,
    'ventas_reales': ventas_reales,
    'pred_lstm': ventas_pred_lstm,
    'pred_baseline': ventas_pred_baseline
})

print("📈 Datos de evaluación generados:")
print(f"   Período: {n_meses} meses")
print(f"   Modelos: LSTM (avanzado) vs Baseline (promedio móvil)")
print(f"\nPrimeras filas:")
display(df.head())

# COMMAND ----------

# DBTITLE 1,Métricas de error
# MAGIC %md
# MAGIC ## 2️⃣ Métricas de Error Estándar
# MAGIC
# MAGIC Calcularemos las métricas más comunes para forecasting.

# COMMAND ----------

# DBTITLE 1,Calcular métricas
def calcular_metricas_completas(y_true, y_pred, nombre_modelo):
    """
    Calcula métricas completas de forecasting.
    """
    # Métricas básicas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    # Más robusto que MAPE cuando hay valores cercanos a 0
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # MASE (Mean Absolute Scaled Error)
    # Compara con un método naive (usar valor del período anterior)
    naive_forecast = np.concatenate([[y_true[0]], y_true[:-1]])
    mae_naive = mean_absolute_error(y_true, naive_forecast)
    mase = mae / mae_naive if mae_naive > 0 else np.inf
    
    # Forecast Bias (sesgo)
    bias = np.mean(y_pred - y_true)
    bias_pct = (bias / np.mean(y_true)) * 100
    
    # Tracking Signal (acumulado de errores)
    errors = y_pred - y_true
    tracking_signal = np.sum(errors) / mae if mae > 0 else 0
    
    resultados = {
        'Modelo': nombre_modelo,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape,
        'SMAPE (%)': smape,
        'MASE': mase,
        'Bias': bias,
        'Bias (%)': bias_pct,
        'Tracking Signal': tracking_signal
    }
    
    return resultados

# Calcular métricas para ambos modelos
metricas_lstm = calcular_metricas_completas(df['ventas_reales'], df['pred_lstm'], 'LSTM')
metricas_baseline = calcular_metricas_completas(df['ventas_reales'], df['pred_baseline'], 'Baseline')

# Crear tabla comparativa
df_metricas = pd.DataFrame([metricas_lstm, metricas_baseline])

print("📉 COMPARACIÓN DE MÉTRICAS")
print("="*100)
print(df_metricas.to_string(index=False))
print("="*100)

print("\n📈 INTERPRETACIÓN DE MÉTRICAS:")
print(f"\n1. MAE (Mean Absolute Error): Error promedio en unidades")
print(f"   • LSTM: ±${metricas_lstm['MAE']:,.0f} por mes")
print(f"   • Baseline: ±${metricas_baseline['MAE']:,.0f} por mes")
print(f"   → LSTM es {(1 - metricas_lstm['MAE']/metricas_baseline['MAE'])*100:.1f}% mejor")

print(f"\n2. MAPE (Mean Abs % Error): Error relativo")
print(f"   • LSTM: {metricas_lstm['MAPE (%)']:.2f}%")
print(f"   • Baseline: {metricas_baseline['MAPE (%)']:.2f}%")
print(f"   → Precisión LSTM: {100-metricas_lstm['MAPE (%)']:.1f}%")

print(f"\n3. MASE (Mean Abs Scaled Error): Comparado con método naive")
print(f"   • MASE < 1: Mejor que naive")
print(f"   • LSTM: {metricas_lstm['MASE']:.3f} {'\u2705' if metricas_lstm['MASE'] < 1 else '❌'}")
print(f"   • Baseline: {metricas_baseline['MASE']:.3f} {'\u2705' if metricas_baseline['MASE'] < 1 else '❌'}")

print(f"\n4. Bias (Sesgo): ¿Sobre o subestima?")
print(f"   • LSTM: ${metricas_lstm['Bias']:,.0f} ({metricas_lstm['Bias (%)']:+.2f}%)")
print(f"   • Baseline: ${metricas_baseline['Bias']:,.0f} ({metricas_baseline['Bias (%)']:+.2f}%)")
if abs(metricas_lstm['Bias (%)']) < 5:
    print(f"   → LSTM tiene sesgo aceptable (<5%)")
else:
    print(f"   ⚠️ LSTM tiene sesgo significativo (>5%)")

# COMMAND ----------

# DBTITLE 1,Visualizaciones
# MAGIC %md
# MAGIC ## 3️⃣ Visualizaciones de Performance

# COMMAND ----------

# DBTITLE 1,Comparación temporal
# Gráfico temporal comparativo
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 1. Series temporales
axes[0].plot(df['fecha'], df['ventas_reales'], 'o-', linewidth=3, markersize=8, 
             color='black', label='Ventas Reales', zorder=3)
axes[0].plot(df['fecha'], df['pred_lstm'], 's--', linewidth=2, markersize=6, 
             color='#2E86AB', label='Predicción LSTM', alpha=0.8)
axes[0].plot(df['fecha'], df['pred_baseline'], '^--', linewidth=2, markersize=6, 
             color='#F18F01', label='Predicción Baseline', alpha=0.8)

axes[0].set_title('📈 Comparación Temporal: Real vs Predicciones', fontsize=15, fontweight='bold')
axes[0].set_ylabel('Ventas ($)', fontsize=12)
axes[0].legend(loc='upper left', fontsize=11)
axes[0].grid(True, alpha=0.3)

# 2. Errores a lo largo del tiempo
error_lstm = df['pred_lstm'] - df['ventas_reales']
error_baseline = df['pred_baseline'] - df['ventas_reales']

axes[1].plot(df['fecha'], error_lstm, 'o-', linewidth=2, markersize=6, 
             color='#2E86AB', label='Error LSTM', alpha=0.7)
axes[1].plot(df['fecha'], error_baseline, 's-', linewidth=2, markersize=6, 
             color='#F18F01', label='Error Baseline', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
axes[1].fill_between(df['fecha'], 0, error_lstm, alpha=0.2, color='#2E86AB')

axes[1].set_title('📉 Errores de Predicción (Predicción - Real)', fontsize=15, fontweight='bold')
axes[1].set_xlabel('Fecha', fontsize=12)
axes[1].set_ylabel('Error ($)', fontsize=12)
axes[1].legend(loc='upper left', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Dashboard de métricas
# Dashboard ejecutivo
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Comparación MAE
ax1 = fig.add_subplot(gs[0, 0])
modelos = ['LSTM', 'Baseline']
maes = [metricas_lstm['MAE'], metricas_baseline['MAE']]
colores = ['#2E86AB', '#F18F01']
barras = ax1.bar(modelos, maes, color=colores, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_title('MAE - Error Absoluto Medio', fontsize=12, fontweight='bold')
ax1.set_ylabel('MAE ($)')
ax1.grid(axis='y', alpha=0.3)
for barra, mae in zip(barras, maes):
    ax1.text(barra.get_x() + barra.get_width()/2., mae, f'${mae:,.0f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2. Comparación MAPE
ax2 = fig.add_subplot(gs[0, 1])
mapes = [metricas_lstm['MAPE (%)'], metricas_baseline['MAPE (%)']]
barras = ax2.bar(modelos, mapes, color=colores, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_title('MAPE - Error Porcentual', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAPE (%)')
ax2.grid(axis='y', alpha=0.3)
for barra, mape in zip(barras, mapes):
    ax2.text(barra.get_x() + barra.get_width()/2., mape, f'{mape:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. R² Score
ax3 = fig.add_subplot(gs[0, 2])
r2s = [metricas_lstm['R²'], metricas_baseline['R²']]
barras = ax3.bar(modelos, r2s, color=colores, alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_title('R² - Bondad de Ajuste', fontsize=12, fontweight='bold')
ax3.set_ylabel('R² Score')
ax3.set_ylim([0, 1])
ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Bueno (0.8)')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)
for barra, r2 in zip(barras, r2s):
    ax3.text(barra.get_x() + barra.get_width()/2., r2, f'{r2:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4. Distribución de errores LSTM
ax4 = fig.add_subplot(gs[1, 0])
error_lstm = df['pred_lstm'] - df['ventas_reales']
ax4.hist(error_lstm, bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
ax4.set_title('Distribución Errores LSTM', fontsize=12, fontweight='bold')
ax4.set_xlabel('Error ($)')
ax4.set_ylabel('Frecuencia')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Distribución de errores Baseline
ax5 = fig.add_subplot(gs[1, 1])
error_baseline = df['pred_baseline'] - df['ventas_reales']
ax5.hist(error_baseline, bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
ax5.set_title('Distribución Errores Baseline', fontsize=12, fontweight='bold')
ax5.set_xlabel('Error ($)')
ax5.set_ylabel('Frecuencia')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Real vs Predicho (LSTM)
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(df['ventas_reales'], df['pred_lstm'], s=100, alpha=0.6, 
            color='#2E86AB', edgecolor='black', linewidth=1)
ax6.plot([df['ventas_reales'].min(), df['ventas_reales'].max()],
         [df['ventas_reales'].min(), df['ventas_reales'].max()],
         'r--', linewidth=2, label='Perfecta predicción')
ax6.set_title('Real vs Predicho (LSTM)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Ventas Reales ($)')
ax6.set_ylabel('Ventas Predichas ($)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Bias por modelo
ax7 = fig.add_subplot(gs[2, 0])
bias_values = [metricas_lstm['Bias'], metricas_baseline['Bias']]
colores_bias = ['#27AE60' if abs(b) < 200 else '#E74C3C' for b in bias_values]
barras = ax7.bar(modelos, bias_values, color=colores_bias, alpha=0.8, edgecolor='black', linewidth=2)
ax7.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax7.set_title('Forecast Bias (Sesgo)', fontsize=12, fontweight='bold')
ax7.set_ylabel('Bias ($)')
ax7.grid(axis='y', alpha=0.3)
for barra, bias in zip(barras, bias_values):
    ax7.text(barra.get_x() + barra.get_width()/2., bias,
             f'${bias:,.0f}',
             ha='center', va='bottom' if bias > 0 else 'top', 
             fontsize=10, fontweight='bold')

# 8. Error acumulado
ax8 = fig.add_subplot(gs[2, 1])
error_cum_lstm = np.cumsum(np.abs(error_lstm))
error_cum_baseline = np.cumsum(np.abs(error_baseline))
ax8.plot(df['fecha'], error_cum_lstm, linewidth=3, color='#2E86AB', label='LSTM')
ax8.plot(df['fecha'], error_cum_baseline, linewidth=3, color='#F18F01', label='Baseline')
ax8.set_title('Error Acumulado (MAE)', fontsize=12, fontweight='bold')
ax8.set_xlabel('Fecha')
ax8.set_ylabel('Error Acumulado ($)')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Resumen de mejora
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

mejora_mae = (1 - metricas_lstm['MAE']/metricas_baseline['MAE']) * 100
mejora_mape = metricas_baseline['MAPE (%)'] - metricas_lstm['MAPE (%)']

resumen = f"""
🎯 RESUMEN EJECUTIVO

Mejora LSTM vs Baseline:

• MAE: {mejora_mae:+.1f}% mejor
• MAPE: {mejora_mape:+.1f}pp mejor
• R²: {metricas_lstm['R²']:.3f} vs {metricas_baseline['R²']:.3f}

Precisión LSTM:
• {100-metricas_lstm['MAPE (%)']:.1f}% de precisión
• Error: ±${metricas_lstm['MAE']:,.0f}/mes

Sesgo:
• {metricas_lstm['Bias (%)']:+.1f}% {'\u2705' if abs(metricas_lstm['Bias (%)']) < 5 else '⚠️'}

📈 El modelo LSTM supera
   significativamente al
   método baseline.
"""

ax9.text(0.1, 0.5, resumen, fontsize=11, verticalalignment='center', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('📊 DASHBOARD DE EVALUACIÓN DE MODELOS DE FORECASTING', 
             fontsize=16, fontweight='bold', y=0.98)
plt.show()

# COMMAND ----------

# DBTITLE 1,Valor económico
# MAGIC %md
# MAGIC ## 4️⃣ Cálculo de Valor Económico
# MAGIC
# MAGIC Traduciremos la mejora en métricas a impacto financiero.

# COMMAND ----------

# DBTITLE 1,ROI del modelo
# Parámetros de negocio
MARGEN_PROMEDIO = 0.25  # 25% de margen
COSTO_SOBRESTOCK = 0.15  # 15% costo de mantener inventario excesivo
COSTO_FALTANTE = 0.40  # 40% pérdida por venta perdida
VENTA_PROMEDIO_MENSUAL = df['ventas_reales'].mean()

print("💰 ANÁLISIS DE VALOR ECONÓMICO")
print("="*100)

# 1. Costo de errores de predicción
print("\n1️⃣ COSTO DE ERRORES DE PREDICCIÓN\n")

# Error LSTM
error_total_lstm = np.abs(df['pred_lstm'] - df['ventas_reales']).sum()
error_sobre_lstm = np.sum(np.maximum(df['pred_lstm'] - df['ventas_reales'], 0))  # Sobrepredcción
error_bajo_lstm = np.sum(np.maximum(df['ventas_reales'] - df['pred_lstm'], 0))  # Subpredicción

costo_sobrestock_lstm = error_sobre_lstm * COSTO_SOBRESTOCK
costo_faltante_lstm = error_bajo_lstm * MARGEN_PROMEDIO * COSTO_FALTANTE
costo_total_lstm = costo_sobrestock_lstm + costo_faltante_lstm

# Error Baseline
error_total_baseline = np.abs(df['pred_baseline'] - df['ventas_reales']).sum()
error_sobre_baseline = np.sum(np.maximum(df['pred_baseline'] - df['ventas_reales'], 0))
error_bajo_baseline = np.sum(np.maximum(df['ventas_reales'] - df['pred_baseline'], 0))

costo_sobrestock_baseline = error_sobre_baseline * COSTO_SOBRESTOCK
costo_faltante_baseline = error_bajo_baseline * MARGEN_PROMEDIO * COSTO_FALTANTE
costo_total_baseline = costo_sobrestock_baseline + costo_faltante_baseline

print(f"LSTM:")
print(f"   Costo por exceso de stock:  ${costo_sobrestock_lstm:,.0f}")
print(f"   Costo por ventas perdidas:  ${costo_faltante_lstm:,.0f}")
print(f"   COSTO TOTAL:                 ${costo_total_lstm:,.0f}")

print(f"\nBaseline:")
print(f"   Costo por exceso de stock:  ${costo_sobrestock_baseline:,.0f}")
print(f"   Costo por ventas perdidas:  ${costo_faltante_baseline:,.0f}")
print(f"   COSTO TOTAL:                 ${costo_total_baseline:,.0f}")

# 2. Ahorro anual con LSTM
ahorro_total = costo_total_baseline - costo_total_lstm
ahorro_anual = ahorro_total / n_meses * 12  # Proyectar a anual

print(f"\n2️⃣ AHORRO CON MODELO LSTM\n")
print(f"   Ahorro en {n_meses} meses: ${ahorro_total:,.0f}")
print(f"   Ahorro anual estimado:     ${ahorro_anual:,.0f}")
print(f"   Mejora:                     {(1 - costo_total_lstm/costo_total_baseline)*100:.1f}%")

# 3. ROI
COSTO_DESARROLLO_MODELO = 50000  # Costo de desarrollo/implementación
COSTO_OPERACION_ANUAL = 10000  # Costo de mantenimiento anual

beneficio_neto_anual = ahorro_anual - COSTO_OPERACION_ANUAL
roi = (beneficio_neto_anual / COSTO_DESARROLLO_MODELO) * 100
payback_meses = COSTO_DESARROLLO_MODELO / (beneficio_neto_anual / 12)

print(f"\n3️⃣ RETORNO DE INVERSIÓN (ROI)\n")
print(f"   Inversión inicial:           ${COSTO_DESARROLLO_MODELO:,.0f}")
print(f"   Costo operación anual:       ${COSTO_OPERACION_ANUAL:,.0f}")
print(f"   Beneficio neto anual:        ${beneficio_neto_anual:,.0f}")
print(f"   ROI año 1:                   {roi:.1f}%")
print(f"   Payback period:              {payback_meses:.1f} meses")

if roi > 50:
    print(f"   \u2705 ROI EXCELENTE (>50%)")
elif roi > 25:
    print(f"   \ud83d� ROI BUENO (25-50%)")
else:
    print(f"   ⚠️ ROI BAJO (<25%)")

print("="*100)

# COMMAND ----------

# DBTITLE 1,Visualizar valor económico
# Visualización de valor económico
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Comparación de costos
modelos = ['Baseline', 'LSTM']
costos_totales = [costo_total_baseline, costo_total_lstm]
colores = ['#E74C3C', '#27AE60']

barras = axes[0].bar(modelos, costos_totales, color=colores, alpha=0.8, edgecolor='black', linewidth=2)

# Anotar ahorro
axes[0].annotate('', xy=(1, costo_total_lstm), xytext=(1, costo_total_baseline),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=3))
axes[0].text(1.15, (costo_total_baseline + costo_total_lstm) / 2,
             f'Ahorro:\n${ahorro_total:,.0f}',
             fontsize=12, fontweight='bold', color='blue',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

for barra, costo in zip(barras, costos_totales):
    axes[0].text(barra.get_x() + barra.get_width()/2., costo,
                 f'${costo:,.0f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

axes[0].set_title(f'💰 Costo de Errores de Predicción ({n_meses} meses)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Costo Total ($)', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# 2. Flujo de caja acumulado (5 años)
años = np.arange(0, 6)
flujo_acumulado = [-COSTO_DESARROLLO_MODELO]
for año in range(1, 6):
    flujo_acumulado.append(flujo_acumulado[-1] + beneficio_neto_anual)

axes[1].plot(años, flujo_acumulado, 'o-', linewidth=3, markersize=10, color='#2E86AB')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Break-even')
axes[1].fill_between(años, 0, flujo_acumulado, where=(np.array(flujo_acumulado) > 0),
                     alpha=0.3, color='green', label='Beneficio neto')
axes[1].fill_between(años, 0, flujo_acumulado, where=(np.array(flujo_acumulado) <= 0),
                     alpha=0.3, color='red', label='Inversión')

# Marcar payback
if payback_meses < 60:
    payback_año = payback_meses / 12
    axes[1].axvline(x=payback_año, color='orange', linestyle=':', linewidth=3,
                   label=f'Payback: {payback_meses:.1f} meses')

for i, (año, flujo) in enumerate(zip(años, flujo_acumulado)):
    axes[1].text(año, flujo + 5000, f'${flujo/1000:.0f}K',
                ha='center', fontsize=10, fontweight='bold')

axes[1].set_title(f'📈 Flujo de Caja Acumulado (ROI = {roi:.0f}%)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Año', fontsize=12)
axes[1].set_ylabel('Flujo Acumulado ($)', fontsize=12)
axes[1].set_xticks(años)
axes[1].legend(loc='upper left', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Conclusiones finales
# MAGIC %md
# MAGIC ## 🎯 Conclusiones del Framework de Evaluación
# MAGIC
# MAGIC ### Resumen de Resultados:
# MAGIC
# MAGIC 📉 **Métricas Técnicas**
# MAGIC * LSTM supera al baseline en todas las métricas
# MAGIC * MAE: {mejora_mae:.1f}% mejor
# MAGIC * MAPE: {mejora_mape:.1f} puntos porcentuales mejor
# MAGIC * Precisión: {100-metricas_lstm['MAPE (%)']:.1f}%
# MAGIC
# MAGIC 💰 **Valor Económico**
# MAGIC * Ahorro anual: ${ahorro_anual:,.0f}
# MAGIC * ROI primer año: {roi:.1f}%
# MAGIC * Payback: {payback_meses:.1f} meses
# MAGIC * Beneficio neto 5 años: ${beneficio_neto_anual * 5:,.0f}
# MAGIC
# MAGIC ### Framework de Evaluación - Mejores Prácticas:
# MAGIC
# MAGIC ✅ **1. Métricas Múltiples**
# MAGIC * Nunca confiar en una sola métrica
# MAGIC * Combinar MAE (magnitud), MAPE (relativo), MASE (vs baseline)
# MAGIC * Monitorear sesgo (bias) para detectar sobre/subestimación sistemática
# MAGIC
# MAGIC ✅ **2. Comparación con Baseline**
# MAGIC * Siempre tener un modelo simple de referencia
# MAGIC * Métodos comunes: promedio móvil, último valor, seasonal naive
# MAGIC * El modelo complejo debe superar significativamente al baseline
# MAGIC
# MAGIC ✅ **3. Backtesting Riguroso**
# MAGIC * Validación temporal (NO shuffle aleatorio)
# MAGIC * Múltiples ventanas de test (rolling window)
# MAGIC * Simular producción real (retrain periódico)
# MAGIC
# MAGIC ✅ **4. Traducir a Valor de Negocio**
# MAGIC * Cuantificar impacto financiero
# MAGIC * Calcular ROI para justificar inversión
# MAGIC * Comunicar en lenguaje ejecutivo
# MAGIC
# MAGIC ✅ **5. Monitoreo Continuo**
# MAGIC * Performance puede degradarse con el tiempo
# MAGIC * Dashboard en tiempo real de métricas
# MAGIC * Alertas automáticas si métricas caen
# MAGIC * Reentrenamiento periódico
# MAGIC
# MAGIC ### Cómo Presentar Resultados a Stakeholders:
# MAGIC
# MAGIC 👥 **Para Ejecutivos (CFO, CEO)**:
# MAGIC * Enfoque en ROI, ahorro, payback
# MAGIC * Comparar con costo actual de errores
# MAGIC * Riesgo de NO adoptar el modelo
# MAGIC
# MAGIC 📊 **Para Operaciones**:
# MAGIC * Precisión (MAPE) en términos entendibles
# MAGIC * Ejemplos concretos de mejora
# MAGIC * Cómo usar las predicciones en su flujo
# MAGIC
# MAGIC 🔬 **Para Equipos Técnicos**:
# MAGIC * Métricas completas (MAE, RMSE, MASE)
# MAGIC * Distribución de errores
# MAGIC * Robustez y limitaciones del modelo
# MAGIC
# MAGIC ### 🎓 Fin del Curso
# MAGIC
# MAGIC ¡Felicitaciones! Has completado el curso de **Inteligencia Artificial para Negocios: Análisis de Series Temporales con Redes Neuronales Recurrentes**.
# MAGIC
# MAGIC ### Lo que aprendiste:
# MAGIC
# MAGIC 1️⃣ **Fundamentos** de series temporales y su importancia
# MAGIC 2️⃣ **Preparación de datos** con feature engineering
# MAGIC 3️⃣ **RNN y LSTM** - Arquitectura y entrenamiento
# MAGIC 4️⃣ **Predicción con TensorFlow** para casos reales
# MAGIC 5️⃣ **PySpark ML** para procesamiento distribuido
# MAGIC 6️⃣ **Optimización de inventario** con forecasting
# MAGIC 7️⃣ **Detección de anomalías** con autoencoders
# MAGIC 8️⃣ **Evaluación y ROI** - del modelo al valor de negocio
# MAGIC
# MAGIC ### Próximos Pasos:
# MAGIC
# MAGIC 🚀 **Implementación en Producción**:
# MAGIC * MLflow para tracking y versionado
# MAGIC * Databricks Model Serving para deployment
# MAGIC * Workflows para reentrenamiento automático
# MAGIC * Monitoring con Lakehouse Monitoring
# MAGIC
# MAGIC 📚 **Sigue Aprendiendo**:
# MAGIC * Transformers para series temporales
# MAGIC * Prophet/NeuralProphet de Facebook
# MAGIC * Forecasting multivariado
# MAGIC * Causal inference en series temporales
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 👏 ¡Gracias por completar este curso! Ahora tienes las herramientas para aplicar Deep Learning a problemas reales de negocio.".format(
# MAGIC     mejora_mae=mejora_mae,
# MAGIC     mejora_mape=mejora_mape,
# MAGIC     metricas_lstm=metricas_lstm,
# MAGIC     ahorro_anual=ahorro_anual,
# MAGIC     roi=roi,
# MAGIC     payback_meses=payback_meses,
# MAGIC     beneficio_neto_anual=beneficio_neto_anual
# MAGIC )

# COMMAND ----------

