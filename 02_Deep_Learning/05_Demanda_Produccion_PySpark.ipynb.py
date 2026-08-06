# Databricks notebook source
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # ⚡ Análisis de Demanda con PySpark ML
# MAGIC
# MAGIC ## Procesamiento Distribuido de Series Temporales Georeferenciadas
# MAGIC
# MAGIC ### Objetivos
# MAGIC
# MAGIC * Procesar series temporales georeferenciadas a gran escala con PySpark
# MAGIC * Usar PySpark ML para feature engineering distribuido (temporal + espacial H3)
# MAGIC * Implementar modelo de forecasting con Spark MLlib
# MAGIC * Integrar con Delta Lake para datos con índices H3
# MAGIC * Comparar con enfoque TensorFlow/LSTM
# MAGIC * Procesar múltiples sucursales simultáneamente
# MAGIC
# MAGIC ### ¿Cuándo usar PySpark?
# MAGIC
# MAGIC ✅ **Ideal para**:
# MAGIC * Múltiples series temporales simultáneas (ej: ventas por sucursal/zona/producto)
# MAGIC * Datos georeferenciados que no caben en memoria de un solo nodo
# MAGIC * Integración con lakehouse (Delta Lake + H3)
# MAGIC * Pipelines de producción escalables
# MAGIC * Feature engineering distribuido (temporal + espacial)
# MAGIC * Análisis de vecindario con H3
# MAGIC
# MAGIC ❌ **NO ideal para**:
# MAGIC * Modelos de deep learning complejos (usar TensorFlow/PyTorch)
# MAGIC * Series temporales muy largas individuales
# MAGIC * Experimentación rápida (pandas es más ágil)
# MAGIC
# MAGIC ### Arquitectura del Pipeline
# MAGIC
# MAGIC ```
# MAGIC Delta Lake (ventas_mensuales_mendoza_h3) → PySpark DataFrame → 
# MAGIC Feature Engineering (temporal + H3) → Model Training → 
# MAGIC Predictions → Delta Lake
# MAGIC ```
# MAGIC
# MAGIC ### 🗺️ Dataset: 5 Sucursales Reales en Mendoza
# MAGIC
# MAGIC * Centro Comercial, Las Heras, Guaymallén, Godoy Cruz, Maipú
# MAGIC * Con índices H3 (res 9/8/7), zona, distancia al centro

# COMMAND ----------

# DBTITLE 1,Configuración de Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Inicializar Spark
spark = SparkSession.builder \
    .appName("Demanda_Produccion_LSTM") \
    .getOrCreate()

print(f"✅ Spark iniciado: versión {spark.version}")
print(f"   Ejecutores disponibles: {spark.sparkContext.defaultParallelism}")

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# COMMAND ----------

# DBTITLE 1,Generación de datos
# MAGIC %md
# MAGIC ## 1️⃣ Cargar Dataset de Ventas Georeferenciadas de Mendoza
# MAGIC
# MAGIC Cargaremos los datos reales de **5 sucursales en Mendoza** con índices H3 desde Delta Lake para demostrar el poder de procesamiento distribuido con datos espaciales.

# COMMAND ----------

# DBTITLE 1,Crear datos de demanda
# Cargar datos georeferenciados de ventas en Mendoza desde Delta Lake
df_spark = spark.table("ventas_mensuales_mendoza_h3")

# Renombrar columnas para mantener consistencia con el resto del notebook
# 'ventas' -> 'demanda', 'sucursal_nombre' -> 'ubicacion'
df_spark = df_spark.withColumnRenamed('ventas', 'demanda') \
                   .withColumnRenamed('sucursal_nombre', 'ubicacion') \
                   .withColumnRenamed('año', 'anio')

# Extraer información de las sucursales
sucursales_info = df_spark.select('sucursal_id', 'ubicacion', 'zona', 'lat', 'lon', 
                                   'h3_index', 'h3_res8', 'h3_res7').distinct().collect()

print("🗺️ DATASET DE VENTAS GEOREFERENCIADAS DE MENDOZA")
print("="*70)
print(f"Total registros: {df_spark.count():,}")
print(f"Sucursales: {df_spark.select('sucursal_id').distinct().count()}")
print(f"Período: {df_spark.select(F.min('fecha'), F.max('fecha')).first()}")
print(f"Demanda total (5 años): ${df_spark.agg(F.sum('demanda')).first()[0]:,.0f}")
print("="*70)

print("\n📍 SUCURSALES EN MENDOZA:")
for suc in sucursales_info:
    print(f"  {suc['sucursal_id']} - {suc['ubicacion']}")
    print(f"     Zona: {suc['zona']}")
    print(f"     Coords: ({suc['lat']:.4f}, {suc['lon']:.4f})")
    print(f"     H3 (res 9): {suc['h3_index']}")
    print()

print("📋 SCHEMA DEL DATASET:")
df_spark.printSchema()

print("\n🔍 PRIMERAS FILAS:")
display(df_spark.limit(10))

# COMMAND ----------

# DBTITLE 1,Análisis exploratorio
# MAGIC %md
# MAGIC ## 2️⃣ Análisis Exploratorio con PySpark

# COMMAND ----------

# DBTITLE 1,Estadísticas agregadas
# Análisis por sucursal
print("🏪 VENTAS TOTALES POR SUCURSAL (5 AÑOS)")
demanda_sucursal = df_spark.groupBy('sucursal_id', 'ubicacion', 'zona') \
    .agg(
        F.sum('demanda').alias('ventas_total'),
        F.avg('demanda').alias('ventas_promedio'),
        F.min('demanda').alias('ventas_min'),
        F.max('demanda').alias('ventas_max')
    ) \
    .orderBy(F.desc('ventas_total'))

display(demanda_sucursal)

# Análisis por zona geográfica
print("\n🗺️ VENTAS TOTALES POR ZONA GEOGRÁFICA")
demanda_zona = df_spark.groupBy('zona') \
    .agg(
        F.sum('demanda').alias('ventas_total'),
        F.avg('demanda').alias('ventas_promedio'),
        F.count('*').alias('meses_totales')
    ) \
    .orderBy(F.desc('ventas_total'))

display(demanda_zona)

# Distribución mensual
print("\n📅 VENTAS PROMEDIO POR MES DEL AÑO")
demanda_mes = df_spark.groupBy('mes') \
    .agg(
        F.avg('demanda').alias('ventas_promedio')
    ) \
    .orderBy('mes')

display(demanda_mes)

# COMMAND ----------

# DBTITLE 1,Visualización con pandas
# Convertir a pandas para visualización
df_pd = df_spark.select('fecha', 'sucursal_id', 'ubicacion', 'zona', 'demanda').toPandas()

# Obtener lista de sucursales
sucursales = df_pd['sucursal_id'].unique()
colores_zona = {'Centro Comercial': '#E63946', 'Residencial Norte': '#457B9D', 
                'Corredor Comercial': '#2A9D8F', 'Zona Comercial Sur': '#E9C46A', 
                'Suburbano': '#F4A261'}

# Gráfico de series temporales por sucursal
fig, axes = plt.subplots(len(sucursales), 1, figsize=(16, 12), sharex=True)

for i, suc_id in enumerate(sorted(sucursales)):
    df_suc = df_pd[df_pd['sucursal_id'] == suc_id].sort_values('fecha')
    zona = df_suc['zona'].iloc[0]
    ubicacion = df_suc['ubicacion'].iloc[0]
    color = colores_zona.get(zona, '#457B9D')
    
    axes[i].plot(df_suc['fecha'], df_suc['demanda'], 
                linewidth=2.5, alpha=0.8, color=color, label=zona)
    axes[i].fill_between(df_suc['fecha'], df_suc['demanda'], alpha=0.2, color=color)
    
    axes[i].set_title(f'🏪 {suc_id} - {ubicacion}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Ventas ($)', fontsize=10)
    axes[i].legend(loc='upper left', fontsize=9)
    axes[i].grid(True, alpha=0.3)
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

axes[-1].set_xlabel('Fecha', fontsize=11)
plt.suptitle('🗺️ Series Temporales de Ventas por Sucursal en Mendoza', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n📊 Las series muestran:")
print("   • Tendencia creciente en todas las sucursales")
print("   • Estacionalidad (picos en vendimia y fiestas)")
print("   • Diferencias por zona geográfica")

# COMMAND ----------

# DBTITLE 1,Feature Engineering
# MAGIC %md
# MAGIC ## 3️⃣ Feature Engineering Distribuido con PySpark
# MAGIC
# MAGIC Crearemos features usando funciones de ventana (Window functions) de Spark.

# COMMAND ----------

# DBTITLE 1,Crear features temporales
# Definir ventanas para calcular lags y rolling features
# Ventana particionada por sucursal, ordenada por fecha
window_spec = Window.partitionBy('sucursal_id').orderBy('fecha')

print("🔧 FEATURE ENGINEERING DISTRIBUIDO CON PYSPARK")
print("="*70)
print("📅 Features temporales (calculadas con Window functions):")
print("   • Lags: 1, 2, 3, 6 meses")
print("   • Rolling means: 3 y 6 meses")
print("   • Rolling std: 3 meses")
print("   • Componentes cíclicos: sin/cos del mes")
print("\n🗺️ Features espaciales (ya incluidas en el dataset):")
print("   • Índices H3: res 9 (~174m), res 8 (~461m), res 7 (~1.22km)")
print("   • Zona: categoría de ubicación")
print("   • Coordenadas: lat, lon")
print("="*70)

# Crear features
df_features = df_spark \
    .withColumn('mes_sin', F.sin(2 * 3.14159 * F.col('mes') / 12)) \
    .withColumn('mes_cos', F.cos(2 * 3.14159 * F.col('mes') / 12)) \
    .withColumn('demanda_lag_1', F.lag('demanda', 1).over(window_spec)) \
    .withColumn('demanda_lag_2', F.lag('demanda', 2).over(window_spec)) \
    .withColumn('demanda_lag_3', F.lag('demanda', 3).over(window_spec)) \
    .withColumn('demanda_lag_6', F.lag('demanda', 6).over(window_spec)) \
    .withColumn('demanda_rolling_mean_3', 
                F.avg('demanda').over(window_spec.rowsBetween(-3, -1))) \
    .withColumn('demanda_rolling_mean_6', 
                F.avg('demanda').over(window_spec.rowsBetween(-6, -1))) \
    .withColumn('demanda_rolling_std_3', 
                F.stddev('demanda').over(window_spec.rowsBetween(-3, -1)))

# Eliminar filas con nulls (causados por lags)
df_clean = df_features.na.drop()

print(f"✅ Features creadas")
print(f"   Registros antes: {df_spark.count():,}")
print(f"   Registros después: {df_clean.count():,}")
print(f"\nColumnas:")
for col in df_clean.columns:
    print(f"   • {col}")

display(df_clean.limit(5))

# COMMAND ----------

# DBTITLE 1,Modelo de ML
# MAGIC %md
# MAGIC ## 4️⃣ Modelo de Machine Learning con Spark MLlib
# MAGIC
# MAGIC Usaremos **Gradient Boosted Trees (GBT)**, un método ensemble muy efectivo para forecasting.

# COMMAND ----------

# DBTITLE 1,Preparar datos para ML
# Codificar variables categóricas (producto, ubicacion)
from pyspark.ml.feature import StringIndexer

indexer_producto = StringIndexer(inputCol='producto', outputCol='producto_idx')
indexer_ubicacion = StringIndexer(inputCol='ubicacion', outputCol='ubicacion_idx')

df_indexed = indexer_producto.fit(df_clean).transform(df_clean)
df_indexed = indexer_ubicacion.fit(df_indexed).transform(df_indexed)

# Seleccionar features para el modelo
feature_cols = [
    'producto_idx', 'ubicacion_idx', 'año', 'mes',
    'mes_sin', 'mes_cos',
    'demanda_lag_1', 'demanda_lag_2', 'demanda_lag_3', 'demanda_lag_6',
    'demanda_rolling_mean_3', 'demanda_rolling_mean_6', 'demanda_rolling_std_3'
]

# Ensamblar features en un vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_raw')

# Escalar features
scaler = StandardScaler(inputCol='features_raw', outputCol='features', 
                        withStd=True, withMean=True)

# Aplicar transformaciones
df_assembled = assembler.transform(df_indexed)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

print("✅ Datos preparados para ML")
print(f"   Features: {len(feature_cols)}")
print(f"   Registros: {df_scaled.count():,}")

# COMMAND ----------

# DBTITLE 1,Dividir train/test
# División temporal: 80% train, 20% test
# Usar año como criterio (primeros 3 años train, último año test)

df_train = df_scaled.filter(F.col('año') < 2023)
df_test = df_scaled.filter(F.col('año') >= 2023)

print("📏 DIVISIÓN DE DATOS")
print("="*70)
print(f"Train: {df_train.count():,} registros (2020-2022)")
print(f"Test:  {df_test.count():,} registros (2023)")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Entrenar modelo GBT
# Gradient Boosted Trees Regressor
gbt = GBTRegressor(
    featuresCol='features',
    labelCol='demanda',
    maxIter=50,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

print("🚀 Entrenando modelo Gradient Boosted Trees...")
print("   (Esto puede tomar unos minutos con Spark)\n")

gbt_model = gbt.fit(df_train)

print("✅ Modelo entrenado!")
print(f"   Número de árboles: {gbt_model.getNumTrees}")
print(f"   Profundidad máxima: {gbt_model.getMaxDepth()}")

# COMMAND ----------

# DBTITLE 1,Evaluación
# MAGIC %md
# MAGIC ## 5️⃣ Evaluación del Modelo

# COMMAND ----------

# DBTITLE 1,Hacer predicciones
# Predicciones en train y test
predictions_train = gbt_model.transform(df_train)
predictions_test = gbt_model.transform(df_test)

print("✅ Predicciones generadas")

# COMMAND ----------

# DBTITLE 1,Calcular métricas
# Evaluadores
evaluator_rmse = RegressionEvaluator(labelCol='demanda', predictionCol='prediction', metricName='rmse')
evaluator_mae = RegressionEvaluator(labelCol='demanda', predictionCol='prediction', metricName='mae')
evaluator_r2 = RegressionEvaluator(labelCol='demanda', predictionCol='prediction', metricName='r2')

# Métricas en Train
train_rmse = evaluator_rmse.evaluate(predictions_train)
train_mae = evaluator_mae.evaluate(predictions_train)
train_r2 = evaluator_r2.evaluate(predictions_train)

# Métricas en Test
test_rmse = evaluator_rmse.evaluate(predictions_test)
test_mae = evaluator_mae.evaluate(predictions_test)
test_r2 = evaluator_r2.evaluate(predictions_test)

print("🎯 RESULTADOS DEL MODELO GBT")
print("="*70)
print(f"TRAIN:")
print(f"   RMSE: {train_rmse:.2f}")
print(f"   MAE:  {train_mae:.2f}")
print(f"   R²:   {train_r2:.4f}")
print(f"\nTEST:")
print(f"   RMSE: {test_rmse:.2f}")
print(f"   MAE:  {test_mae:.2f}")
print(f"   R²:   {test_r2:.4f}")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Visualizar predicciones
# Convertir a pandas para visualización
test_pd = predictions_test.select('fecha', 'producto', 'ubicacion', 'demanda', 'prediction').toPandas()

# Visualizar para un producto y ubicación específicos
producto_ejemplo = 'ProductoA'
ubicacion_ejemplo = 'Tienda_Norte'

test_filtrado = test_pd[
    (test_pd['producto'] == producto_ejemplo) & 
    (test_pd['ubicacion'] == ubicacion_ejemplo)
].sort_values('fecha')

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(test_filtrado['fecha'], test_filtrado['demanda'], 
        'o-', label='Demanda Real', linewidth=2.5, markersize=8, color='#2E86AB')
ax.plot(test_filtrado['fecha'], test_filtrado['prediction'], 
        's--', label='Predicción GBT', linewidth=2, markersize=7, color='#F18F01', alpha=0.8)

ax.fill_between(test_filtrado['fecha'], 
                test_filtrado['demanda'], 
                test_filtrado['prediction'],
                alpha=0.2, color='gray', label='Área de error')

ax.set_title(f'📈 Predicciones vs Real: {producto_ejemplo} - {ubicacion_ejemplo} (Test 2023)', 
             fontsize=15, fontweight='bold')
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Demanda', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n📉 Error promedio para {producto_ejemplo} - {ubicacion_ejemplo}:")
error_medio = (test_filtrado['demanda'] - test_filtrado['prediction']).abs().mean()
print(f"   MAE: {error_medio:.2f} unidades")

# COMMAND ----------

# DBTITLE 1,Análisis por producto
# Error por producto
error_por_producto = test_pd.copy()
error_por_producto['error_abs'] = (error_por_producto['demanda'] - error_por_producto['prediction']).abs()

error_stats = error_por_producto.groupby('producto').agg({
    'error_abs': ['mean', 'std', 'min', 'max']
}).round(2)

error_stats.columns = ['MAE', 'Std_Error', 'Error_Min', 'Error_Max']
error_stats = error_stats.sort_values('MAE')

print("📈 ERROR POR PRODUCTO (Test Set)")
print("="*70)
print(error_stats.to_string())
print("="*70)

# Visualizar
fig, ax = plt.subplots(figsize=(12, 6))

productos_ordenados = error_stats.index
maes = error_stats['MAE'].values

colores = ['#2E86AB' if mae < 40 else '#F18F01' if mae < 50 else '#A23B72' 
           for mae in maes]

barras = ax.bar(productos_ordenados, maes, color=colores, alpha=0.8, edgecolor='black', linewidth=2)

for barra, mae in zip(barras, maes):
    altura = barra.get_height()
    ax.text(barra.get_x() + barra.get_width()/2., altura,
            f'{mae:.1f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('🎯 Error Absoluto Medio (MAE) por Producto', fontsize=14, fontweight='bold')
ax.set_xlabel('Producto', fontsize=12)
ax.set_ylabel('MAE (unidades)', fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Guardar resultados
# MAGIC %md
# MAGIC ## 6️⃣ Guardar Predicciones en Delta Lake

# COMMAND ----------

# DBTITLE 1,Exportar a Delta
# Seleccionar columnas relevantes
predictions_final = predictions_test.select(
    'fecha', 'producto', 'ubicacion', 'año', 'mes',
    'demanda', 'prediction'
).withColumnRenamed('demanda', 'demanda_real') \
 .withColumnRenamed('prediction', 'demanda_predicha')

# Guardar como Delta Table
tabla_predicciones = "predicciones_demanda_gbt"

predictions_final.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable(tabla_predicciones)

print(f"✅ Predicciones guardadas en: {tabla_predicciones}")
print(f"   Total registros: {predictions_final.count():,}")

# Verificar
display(spark.table(tabla_predicciones).limit(10))

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 🎯 Conclusiones
# MAGIC
# MAGIC ### Resultados con PySpark ML:
# MAGIC
# MAGIC ✅ **Performance**
# MAGIC * MAE en Test: ~{mae:.1f} unidades
# MAGIC * RMSE: ~{rmse:.1f}
# MAGIC * R²: {r2:.3f}
# MAGIC
# MAGIC ✅ **Ventajas de PySpark**
# MAGIC * Procesa múltiples series simultáneamente
# MAGIC * Escalable a millones de productos/ubicaciones
# MAGIC * Feature engineering distribuido
# MAGIC * Integración nativa con Delta Lake
# MAGIC * Ideal para producción empresarial
# MAGIC
# MAGIC ### Comparación: LSTM vs GBT
# MAGIC
# MAGIC | Aspecto | LSTM (TensorFlow) | GBT (PySpark) |
# MAGIC |---------|-------------------|---------------|
# MAGIC | Complejidad | Alta | Media |
# MAGIC | Datos necesarios | Más (miles de puntos) | Menos (cientos) |
# MAGIC | Interpretabilidad | Baja (caja negra) | Media (importancia features) |
# MAGIC | Escalabilidad | Limitada (GPU) | Alta (cluster) |
# MAGIC | Patrones temporales | Excelente | Bueno |
# MAGIC | Tiempo entrenamiento | Largo | Rápido |
# MAGIC | Producción | Más complejo | Más simple |
# MAGIC
# MAGIC ### Recomendaciones:
# MAGIC
# MAGIC 🎯 **Usar LSTM cuando**:
# MAGIC * Series muy largas (años de datos diarios)
# MAGIC * Patrones temporales complejos
# MAGIC * No hay muchas features exógenas
# MAGIC * Se necesita máxima precisión
# MAGIC
# MAGIC 🎯 **Usar GBT cuando**:
# MAGIC * Múltiples series cortas
# MAGIC * Muchas features disponibles
# MAGIC * Se necesita rapidez e interpretabilidad
# MAGIC * Infraestructura Spark disponible
# MAGIC
# MAGIC ### 📚 Próximo Notebook:
# MAGIC
# MAGIC **06_Analisis_Inventario_Multiproducto.ipynb**
# MAGIC * Forecasting para múltiples SKUs
# MAGIC * Optimización de inventario
# MAGIC * Análisis de punto de reorden
# MAGIC * Safety stock inteligente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 👉 En producción, considerar un **ensemble** (combinar LSTM + GBT) para aprovechar lo mejor de ambos mundos.".format(
# MAGIC     mae=test_mae,
# MAGIC     rmse=test_rmse,
# MAGIC     r2=test_r2
# MAGIC )

# COMMAND ----------

