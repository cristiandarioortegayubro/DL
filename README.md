# 🏔️ Deep Learning para Negocios - Caso Los Andes Market

## 📋 Descripción del Proyecto

Proyecto educativo de **Inteligencia Artificial aplicada a Negocios** utilizando Deep Learning (TensorFlow, Keras, PySpark) para análisis predictivo de series temporales con **datos georeferenciados**.

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
│   ├── 04_Prediccion_Ventas_TensorFlow.ipynb
│   └── 05_Demanda_Produccion_PySpark.ipynb
│
├── 03_Casos_Practicos/
│   ├── 06_Analisis_Inventario_Multiproducto.ipynb
│   ├── 07_Deteccion_Anomalias_Negocio.ipynb
│   └── 08_Evaluacion_Metricas_Negocio.ipynb
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

#### **01_Introduccion_Series_Temporales.ipynb**
* Conceptos básicos de series temporales
* Generación de datos sintéticos georeferenciados
* Mapa interactivo con Folium y hexágonos H3
* Descomposición: tendencia, estacionalidad, ruido
* Análisis estadístico por sucursal y zona
* Exportación a Delta Lake

#### **02_Preparacion_Datos_Empresariales.ipynb**
* Feature engineering espacial y temporal
* Lags y rolling windows
* Normalización y escalado
* Creación de sequences para LSTM
* División train/validation/test temporal

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

## 👥 Audiencia

Este proyecto está diseñado para:
* 📚 **Estudiantes** de Data Science, ML, e IA aplicada a negocios
* 💼 **Profesionales** de Retail, Supply Chain, y Business Analytics
* 🧑‍💻 **Desarrolladores** interesados en series temporales y análisis geoespacial
* 📊 **Analistas de Negocios** que buscan entender modelos predictivos

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
