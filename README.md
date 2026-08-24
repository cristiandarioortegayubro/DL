# 🏔️ Deep Learning para Negocios - Caso Los Andes Market

## 🔬 Proyecto de Investigación Científica

**Investigación científica** sobre la aplicación de **Redes Neuronales Recurrentes (RNN)** para predicción de series temporales en contextos empresariales, utilizando Deep Learning (TensorFlow, Keras, PySpark) con **datos georeferenciados**.

---

## 🎯 Problema de Investigación

¿Qué parámetros y arquitectura de RNN (LSTM, GRU, etc.) ofrece un mejor desempeño en la predicción de series temporales en un contexto empresarial?

### Preguntas Específicas

1. ¿Cuál es la combinación óptima de hiperparámetros (tasa de aprendizaje, tamaño de lote, número de capas ocultas, etc.) para entrenar modelos RNN en series temporales de negocios?
2. ¿Cómo se compara el desempeño de las RNN con modelos estadísticos tradicionales?
3. ¿Qué arquitectura (LSTM vs GRU) es más efectiva para series con patrones no lineales y estacionalidad?

---

## 📚 Marco Teórico

El marco teórico se centra en la aplicación de las **Redes Neuronales Recurrentes (RNN)** para el análisis de series temporales en el ámbito empresarial.

### Fundamentos Teóricos

1. **Series Temporales**
   * Características: tendencia, estacionalidad, ciclos, componentes irregulares
   * Importancia en la toma de decisiones empresariales
   * Métodos clásicos: ARIMA, Suavizado Exponencial, Modelos de Regresión

2. **Redes Neuronales Artificiales y Deep Learning**
   * Ventajas sobre métodos tradicionales
   * Capacidad de modelar relaciones no lineales complejas
   * Aprendizaje automático de features

3. **Redes Neuronales Recurrentes (RNN)**
   * Arquitectura especializada para datos secuenciales
   * Memoria de contexto temporal
   * Problema del gradiente desvaneciente

4. **Arquitecturas Avanzadas**
   * **LSTM (Long Short-Term Memory)**: Celdas de memoria, puertas de control
   * **GRU (Gated Recurrent Unit)**: Arquitectura simplificada, menor complejidad computacional
   * Comparación de capacidades y eficiencia

5. **Aplicaciones Empresariales**
   * Predicción de ventas y demanda
   * Gestión de la cadena de suministro
   * Análisis financiero y forecasting
   * Optimización de inventario

6. **Desafíos y Tendencias Actuales**
   * Interpretabilidad de modelos (XAI)
   * Aprendizaje continuo y adaptación
   * Integración de datos geoespaciales
   * Escalabilidad en producción

---

## 🎯 Objetivos e Hipótesis

### Objetivos Generales

1. 🔬 **Evaluar la eficacia** de las redes neuronales recurrentes (RNN) en la predicción de series temporales en un contexto empresarial
2. 📊 **Comparar el rendimiento** de las RNN con modelos estadísticos tradicionales
3. 🏛️ **Identificar la arquitectura** de RNN más adecuada para el problema de investigación

### Objetivos Específicos

1. 🧠 **Desarrollar modelos** RNN (LSTM, GRU) para predecir series temporales de ventas
2. 📊 **Evaluar la precisión** de los modelos utilizando MAE y otras métricas relevantes
3. ⚙️ **Analizar la influencia** de diferentes hiperparámetros
4. 🔄 **Evaluar la capacidad de generalización** de los modelos
5. 🗺️ **Incorporar features geoespaciales** (H3, zona, distancia)

### Hipótesis de Investigación

**H1**: La arquitectura **LSTM presenta un mejor desempeño** en la predicción de series temporales con patrones no lineales y estacionalidad, comparado con GRU y modelos tradicionales.

**H2**: El **ajuste de los hiperparámetros mejora significativamente la precisión** de las predicciones, medido mediante la reducción del MAE.

**H3**: La **incorporación de features geoespaciales** (H3, zona, distancia al centro) mejora la capacidad predictiva de los modelos RNN en escenarios multi-sucursal.

**H4**: Los modelos **RNN superan a los modelos estadísticos tradicionales** (ARIMA, Suavizado Exponencial) en series temporales con alta volatilidad y múltiples patrones estacionales.

---

## 📋 Metodología de Investigación

### Diseño Experimental

1. **Generación de datos** sintéticos georeferenciados (5 sucursales, 60 meses)
2. **Análisis exploratorio riguroso**:
   - Descomposición de componentes (tendencia, estacionalidad, ruido)
   - Tests de estacionariedad (Augmented Dickey-Fuller, KPSS)
   - Análisis de correlación espacial entre sucursales
   - Validación de hipótesis sobre factores macro/micro
3. **Feature engineering** científico:
   - Features temporales: lags, rolling statistics, componentes cíclicos (sin/cos)
   - Features geoespaciales: distancias Haversine, densidad H3, clustering espacial
   - Preprocesamiento: imputación, normalización por sucursal
4. **División temporal estricta**: Train (70%), Validation (15%), Test (15%)
5. **Desarrollo de modelos**: Baseline, RNN, LSTM, GRU
6. **Búsqueda de hiperparámetros** con validación científica
7. **Evaluación**: MAE, RMSE, MAPE, R²
8. **Análisis comparativo** y pruebas estadísticas
9. **Validación cruzada temporal**

### Rigor Científico y Reproducibilidad

#### 📐 Estándares de Código
* Documentación exhaustiva con justificación teórica
* Seed fijo para reproducibilidad (np.random.seed=42)
* Código modular y reutilizable
* Comentarios académicos explicando decisiones metodológicas

#### 🔬 Validación Estadística
* **Tests de estacionariedad**:
  - Augmented Dickey-Fuller (ADF) para detectar raíces unitarias
  - KPSS (Kwiatkowski-Phillips-Schmidt-Shin) para confirmar estacionariedad
  - Interpretación: series no estacionarias requieren RNN/LSTM vs ARIMA
* **Análisis de correlación espacial**:
  - Correlación de Pearson entre series de sucursales (r > 0.85)
  - Validación de hipótesis: factores macro/micro afectan todas las sucursales
* **Análisis geoespacial**:
  - Distancias Haversine entre sucursales
  - Densidad H3 (conteo de hexágonos en vecindario)
  - Clustering espacial por zona comercial

#### 📊 Conclusiones Científicas en Cada Notebook
* Resumen ejecutivo de hallazgos
* Interpretación estadística rigurosa
* Implicaciones para el negocio
* Limitaciones y trabajo futuro

### Caso de Estudio: **Los Andes Market**

**Los Andes Market** es una cadena regional de supermercados familiares con sede en Mendoza, Argentina. Fundada en 1995, la empresa se ha consolidado como un referente en la provincia, ofreciendo productos de calidad con un fuerte compromiso con los productores locales.

#### 🛒 Perfil del Negocio

* **Formato**: Supermercados de proximidad (1,500 - 2,500 m²)
* **Identidad**: Productos regionales y de la cordillera
* **Target**: Familias de clase media y media-alta
* **Especialización**: Vinos mendocinos, productos orgánicos, carnes premium
* **Filosofía**: Apoyo a productores locales y agricultura sustentable
* **Cobertura**: 5 sucursales estratégicas en el Gran Mendoza

#### 📍 Sucursales

1. **Centro - San Martín**: Zona comercial principal (factor ventas: 1.0x)
2. **Las Heras**: Zona residencial norte (factor ventas: 0.75x)
3. **Guaymallén - Av. San Martín**: Corredor comercial (factor ventas: 1.15x)
4. **Godoy Cruz - Av. San Francisco**: Zona comercial sur (factor ventas: 0.90x)
5. **Maipú - Rodríguez Peña**: Zona suburbana (factor ventas: 0.65x)

---

## 🗂️ Estructura del Proyecto

```
DL/
├── 01_Fundamentos/
│   ├── 01_Introduccion_Series_Temporales.ipynb
│   └── 02_Preparacion_Datos_Empresariales.ipynb
│
├── 02_Deep_Learning/
│   ├── 03_RNN_LSTM_Fundamentos.ipynb
│   ├── 03b_Modelos_Tradicionales_Baseline.ipynb
│   └── 04_Demanda_Produccion_PySpark.ipynb
│
├── 03_Casos_Practicos/
│   ├── 05_Analisis_Inventario_Multiproducto.ipynb
│   ├── 06_Deteccion_Anomalias_Negocio.ipynb
│   ├── 08_Evaluacion_Metricas_Negocio.ipynb
│   ├── 09_Comparacion_Modelos_Cientifica.ipynb
│   └── 10_Paper_Investigacion_Final.ipynb
│
└── README.md
```

---

## 📊 Características de los Datos

### Dimensión Temporal
* **Período**: 5 años (2019-2024), 60 meses
* **Frecuencia**: Mensual
* **Variables**: Ventas, tendencia, estacionalidad

### Dimensión Geoespacial 🗺️
* **Coordenadas GPS**: Ubicación real de cada sucursal en Mendoza
* **Índices H3**: Sistema hexagonal de Uber para análisis espacial
  - Resolución 9: ~174m de diámetro
  - Resolución 8: ~461m de diámetro
  - Resolución 7: ~1.22km de diámetro
* **Zonas**: Centro Comercial, Residencial Norte, Corredor Comercial, etc.
* **Features espaciales**: Distancia al centro, densidad H3, clustering geográfico

### Patrones de Negocio
* **Tendencia**: Crecimiento sostenido en todas las sucursales
* **Estacionalidad Argentina**:
  - 🍇 **Marzo-Abril**: Pico por vendimia mendocina
  - 🎉 **Noviembre-Diciembre**: Fiestas de fin de año
  - ⛄ **Julio-Agosto**: Vacaciones de invierno
  - ☀️ **Enero-Febrero**: Baja temporada (verano)
* **Efecto ubicación**: Sucursales en zonas comerciales venden 15-50% más

---

## 🎯 Notebooks y Contenido

### 📘 Módulo 1: Fundamentos

#### **01_Introduccion_Series_Temporales.ipynb** - 🔬 ANÁLISIS EXPLORATORIO CIENTÍFICO
* **Contexto empresarial riguroso**: Los Andes Market, Mendoza, Argentina
* **Objetivos de investigación** claros y problemática bien definida
* **Generación de datos georeferenciados** con fundamentación teórica:
  - 5 sucursales con coordenadas GPS reales
  - Índices H3 (res 7, 8, 9) para análisis espacial
  - Factores de ubicación basados en zonas comerciales
* **Mapa interactivo profesional** con Folium y hexágonos H3
* **Descomposición científica de componentes**:
  - Tendencia: crecimiento sostenido validado estadísticamente
  - Estacionalidad: patrones argentinos (vendimia, fiestas)
  - Residuos: análisis de ruido y factores no modelados
* **Tests de estacionariedad**:
  - Augmented Dickey-Fuller (ADF): p-values, estadísticos
  - KPSS: confirmación de no-estacionariedad
  - Interpretación: justifica RNN/LSTM sobre modelos clásicos
* **Análisis de correlación espacial**:
  - Matriz de correlación entre sucursales (r > 0.85)
  - Heatmaps y análisis visual
  - Validación de hipótesis: factores macro afectan todas las sucursales
* **Análisis estadístico robusto** por sucursal y zona
* **Conclusiones científicas** y recomendaciones para siguiente fase
* **Exportación a Delta Lake** con metadatos completos

#### **02_Preparacion_Datos_Empresariales.ipynb** - 🛠️ FEATURE ENGINEERING CIENTÍFICO
* **Pipeline end-to-end completo** para deep learning
* **Features temporales avanzadas**:
  - Lags múltiples (1, 3, 6, 12 meses) con justificación teórica
  - Rolling statistics (media, std, min, max) con ventanas optimizadas
  - Componentes cíclicos (sin/cos) para capturar estacionalidad
* **Features geoespaciales H3**:
  - Distancias Haversine entre sucursales y centro comercial
  - Densidad H3: conteo de sucursales en vecindario hexagonal
  - Clustering espacial por zona (Centro, Norte, Sur, Suburbano)
* **Preprocesamiento riguroso**:
  - Imputación de valores faltantes con métodos justificados
  - Normalización por sucursal (StandardScaler)
  - Manejo de outliers con criterio estadístico
* **Creación de secuencias para RNN/LSTM**:
  - Ventana temporal optimizada (lookback=12 meses)
  - Estructura 3D: (samples, timesteps, features)
* **División temporal estricta**: Train (70%), Validation (15%), Test (15%)
  - Sin data leakage
  - Respeto de orden temporal
* **Artefactos persistidos**: arrays numpy, scalers, metadata
* **Conclusiones científicas** y preparación para fase de modelado

### 🧠 Módulo 2: Deep Learning

#### **03_RNN_LSTM_Fundamentos.ipynb**
* Arquitectura de redes recurrentes
* LSTM para series temporales
* Implementación en TensorFlow/Keras
* Entrenamiento y validación

#### **04_Prediccion_Ventas_TensorFlow.ipynb**
* Modelo LSTM multi-sucursal
* Predicciones por ubicación
* Evaluación de performance
* Interpretación de resultados de negocio

#### **05_Demanda_Produccion_PySpark.ipynb**
* Procesamiento distribuido con PySpark
* Feature engineering a escala
* Pipeline de ML con Spark MLlib
* Predicción de demanda por zona geográfica

### 💼 Módulo 3: Casos Prácticos

#### **06_Analisis_Inventario_Multiproducto.ipynb**
* Optimización de inventario multi-sucursal
* Análisis espacial de transferencias
* Recomendaciones por zona

#### **07_Deteccion_Anomalias_Negocio.ipynb**
* LSTM Autoencoder para anomalías
* Detección por sucursal y zona
* Alertas tempranas

#### **08_Evaluacion_Metricas_Negocio.ipynb**
* MAE, RMSE, MAPE por sucursal
* Análisis de error geográfico
* Recomendaciones de negocio

---

## 🛠️ Tecnologías Utilizadas

### Librerías Python
* **TensorFlow/Keras**: Modelos de Deep Learning
* **PySpark**: Procesamiento distribuido
* **H3-py**: Indexación geoespacial hexagonal
* **Folium**: Mapas interactivos
* **Pandas/NumPy**: Manipulación de datos
* **Matplotlib/Seaborn**: Visualización

### Infraestructura Databricks
* **Delta Lake**: Almacenamiento ACID para series temporales
* **Unity Catalog**: Gobierno de datos
* **Spark**: Procesamiento distribuido
* **MLflow**: Tracking de experimentos

---

## 🚀 Cómo Usar Este Proyecto

### 1. Ejecutar en Orden
Los notebooks están diseñados para ejecutarse secuencialmente:
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### 2. Datos Generados
El notebook 01 genera y guarda en Delta Lake:
* **Tabla**: `ventas_mensuales_mendoza_h3`
* **Ubicación**: Unity Catalog
* **Formato**: Delta Table con features geoespaciales

### 3. Requisitos
* Databricks Runtime 13.0+
* Compute: Serverless (CPU o GPU según el caso)
* Librerías: instaladas automáticamente en cada notebook

---

## 📈 Resultados Esperados

### Predicción de Ventas
* Forecast mensual por sucursal con 85%+ de precisión
* Identificación de patrones estacionales
* Recomendaciones de inventario por zona

### Análisis Espacial
* Mapas de calor de ventas por H3
* Clustering de sucursales por performance
* Optimización de rutas de distribución

### Detección de Anomalías
* Alertas tempranas de caídas de ventas
* Identificación de oportunidades de crecimiento
* Análisis de impacto por ubicación

---

## 👥 Audiencia y Aplicaciones

### Audiencia Académica

Esta investigación está dirigida a:
* 🎓 **Investigadores** en Machine Learning y Deep Learning
* 👨‍🎓 **Estudiantes de posgrado** (Maestría/Doctorado) en Data Science, Estadística, IA
* 📚 **Académicos** interesados en aplicaciones de RNN a problemas de negocios
* 🔬 **Profesionales de investigación** en ciencia de datos aplicada

### Audiencia Profesional

* 💼 **Data Scientists** en retail, e-commerce, y supply chain
* 📊 **Analistas de Negocios** que buscan implementar modelos predictivos avanzados
* 🧑‍💻 **Desarrolladores ML/AI** interesados en series temporales y análisis geoespacial
* 📊 **Business Intelligence** teams que buscan integrar Deep Learning en sus pipelines

### Aplicaciones Industriales

* **Retail**: Predicción de demanda, optimización de inventario
* **Supply Chain**: Forecasting de producción, gestión de logística
* **Finanzas**: Predicción de series financieras, detección de anomalías
* **E-commerce**: Predicción de tráfico web, análisis de demanda geolocalizada
* **Manufactura**: Predicción de mantenimiento, optimización de producción

---

## 📚 Conceptos Clave Cubiertos

* ✅ Series temporales y componentes (tendencia, estacionalidad, ruido)
* ✅ Redes neuronales recurrentes (RNN/LSTM)
* ✅ Feature engineering espacial y temporal
* ✅ Índices H3 para análisis geoespacial
* ✅ Procesamiento distribuido con PySpark
* ✅ Pipeline de ML end-to-end
* ✅ Métricas de negocio y evaluación
* ✅ Detección de anomalías con autoencoders
* ✅ Delta Lake y Unity Catalog
* ✅ Visualización de datos espaciales

---

## 📝 Licencia y Uso Educativo

Proyecto de código abierto con fines educativos. Los datos son sintéticos y generados para propósitos de aprendizaje.

---

## 🏔️ Sobre Mendoza y Los Andes

Mendoza es una provincia argentina ubicada al pie de la Cordillera de los Andes, hogar del Aconcagua (cerro más alto de América, 6,962 msnm). La región es mundialmente conocida por:
* 🍷 **Producción vitivinícola**: Principal zona vinícola de Sudamérica
* 🏔️ **Turismo de montaña**: Esquí, trekking, andinismo
* 🌱 **Agricultura**: Olivos, frutas, hortalizas
* 🍇 **Fiesta Nacional de la Vendimia**: Celebración anual en marzo

**Los Andes Market** captura esta identidad regional, ofreciendo productos de la cordillera y apoyando a productores locales.

---

**¡Bienvenido a Los Andes Market! 🏔️🛒**
