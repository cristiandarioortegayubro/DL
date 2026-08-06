# Databricks notebook source
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # 📈 Introducción al Análisis de Series Temporales
# MAGIC
# MAGIC ## Inteligencia Artificial para Negocios
# MAGIC
# MAGIC ### Objetivos de Aprendizaje
# MAGIC
# MAGIC * Comprender qué son las series temporales y su importancia en negocios
# MAGIC * Identificar componentes: tendencia, estacionalidad, ruido
# MAGIC * Visualizar y explorar datos temporales
# MAGIC * Preparar datos para modelos de Deep Learning
# MAGIC
# MAGIC ### ¿Qué es una Serie Temporal?
# MAGIC
# MAGIC Una **serie temporal** es una secuencia de observaciones ordenadas en el tiempo. En negocios, ejemplos incluyen:
# MAGIC
# MAGIC * Ventas diarias/mensuales
# MAGIC * Demanda de productos
# MAGIC * Niveles de inventario
# MAGIC * Precios de mercado
# MAGIC * Tráfico web
# MAGIC * Producción industrial
# MAGIC
# MAGIC ### Caso de Estudio: **Los Andes Market** 🏔️
# MAGIC
# MAGIC #### Sobre la Empresa
# MAGIC
# MAGIC **Los Andes Market** es una cadena regional de supermercados familiares con sede en Mendoza, Argentina. Fundada en 1995, la empresa se ha consolidado como un referente en la provincia, ofreciendo productos de calidad con un fuerte compromiso con los productores locales.
# MAGIC
# MAGIC **Características del Negocio**:
# MAGIC * 🛒 **Formato**: Supermercados de proximidad (1,500 - 2,500 m²)
# MAGIC * 🏔️ **Identidad**: Productos regionales y de la cordillera
# MAGIC * 👨‍👩‍👧‍👦 **Target**: Familias de clase media y media-alta
# MAGIC * 🍇 **Especialización**: Vinos mendocinos, productos orgánicos, carnes premium
# MAGIC * 🌱 **Filosofía**: Apoyo a productores locales y agricultura sustentable
# MAGIC * 📍 **Cobertura**: 5 sucursales estratégicas en el Gran Mendoza
# MAGIC
# MAGIC En este notebook trabajaremos con datos de ventas de **Los Andes Market**, analizando el comportamiento de sus 5 sucursales en diferentes zonas de Mendoza.

# COMMAND ----------

# DBTITLE 1,Instalación de librerías
# Instalación de librerías necesarias
# H3 es el sistema de indexación geoespacial hexagonal de Uber
!pip install matplotlib seaborn pandas numpy h3 folium --quiet

# COMMAND ----------

# DBTITLE 1,Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h3
import folium
from datetime import datetime, timedelta

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

print("✅ Librerías importadas correctamente")
print(f"   H3 versión: {h3.__version__}")

# COMMAND ----------

# DBTITLE 1,Generación de datos
# MAGIC %md
# MAGIC ## 1️⃣ Generación de Datos Empresariales Sintéticos - Los Andes Market
# MAGIC
# MAGIC Crearemos series temporales de **ventas mensuales** para la cadena de supermercados **Los Andes Market** con 5 sucursales estratégicas en el Gran Mendoza, Argentina.
# MAGIC
# MAGIC ### Características de los datos:
# MAGIC
# MAGIC 📍 **Dimensión Geográfica**:
# MAGIC * 5 sucursales en diferentes zonas de Mendoza
# MAGIC * Coordenadas GPS reales
# MAGIC * Índices H3 (resolución 9: ~174m de diámetro)
# MAGIC
# MAGIC 📈 **Componentes Temporales**:
# MAGIC * **Tendencia creciente**: el negocio crece con el tiempo
# MAGIC * **Estacionalidad**: picos en vendimia (Mar-Abr) y fiestas (Nov-Dic)
# MAGIC * **Efecto ubicación**: sucursales en zonas comerciales venden más
# MAGIC * **Ruido aleatorio**: variabilidad natural del mercado

# COMMAND ----------

# DBTITLE 1,Crear datos de ventas georeferenciados
# Configuración
np.random.seed(42)
n_meses = 60  # 5 años de datos
fecha_inicio = datetime(2019, 1, 1)

# Definir sucursales de "Los Andes Market" en Mendoza con coordenadas reales
sucursales = [
    {
        'id': 'SUC001',
        'nombre': 'Centro - San Martín',
        'lat': -32.8895,
        'lon': -68.8458,
        'zona': 'Centro Comercial',
        'factor_base': 1.0  # Sucursal de referencia
    },
    {
        'id': 'SUC002', 
        'nombre': 'Las Heras',
        'lat': -32.8523,
        'lon': -68.8272,
        'zona': 'Residencial Norte',
        'factor_base': 0.75  # Zona residencial, menos tráfico
    },
    {
        'id': 'SUC003',
        'nombre': 'Guaymallén - Av. San Martín',
        'lat': -32.9003,
        'lon': -68.7953,
        'zona': 'Corredor Comercial',
        'factor_base': 1.15  # Alto tráfico comercial
    },
    {
        'id': 'SUC004',
        'nombre': 'Godoy Cruz - Av. San Francisco',
        'lat': -32.9287,
        'lon': -68.8503,
        'zona': 'Zona Comercial Sur',
        'factor_base': 0.90
    },
    {
        'id': 'SUC005',
        'nombre': 'Maipú - Rodríguez Peña',
        'lat': -32.9833,
        'lon': -68.7914,
        'zona': 'Suburbano',
        'factor_base': 0.65  # Zona más alejada
    }
]

# Calcular índices H3 (resolución 9: ~174m diámetro por hexágono)
for sucursal in sucursales:
    sucursal['h3_index'] = h3.latlng_to_cell(sucursal['lat'], sucursal['lon'], 9)
    sucursal['h3_res8'] = h3.latlng_to_cell(sucursal['lat'], sucursal['lon'], 8)  # ~461m
    sucursal['h3_res7'] = h3.latlng_to_cell(sucursal['lat'], sucursal['lon'], 7)  # ~1.22km

print("🏔️ SUCURSALES DE 'LOS ANDES MARKET' EN MENDOZA:")
print("="*100)
for suc in sucursales:
    print(f"\n{suc['id']} - {suc['nombre']}")
    print(f"   Ubicación: ({suc['lat']:.4f}, {suc['lon']:.4f})")
    print(f"   Zona: {suc['zona']}")
    print(f"   H3 (res 9): {suc['h3_index']}")
    print(f"   Factor ventas: {suc['factor_base']:.2f}x")
print("="*100)

# Generar series temporales para cada sucursal
data_records = []

for sucursal in sucursales:
    for i in range(n_meses):
        fecha = fecha_inicio + timedelta(days=30*i)
        mes = fecha.month
        
        # Base de ventas según ubicación (en pesos argentinos)
        base_ventas = 800000 * sucursal['factor_base']
        
        # Tendencia de crecimiento (crece más rápido en zonas comerciales)
        crecimiento_mensual = 8000 * sucursal['factor_base']
        tendencia = base_ventas + (i * crecimiento_mensual)
        
        # Estacionalidad argentina: picos en vendimia (Mar-Abr) y fiestas (Nov-Dic)
        estacionalidad = 0
        if mes in [3, 4]:  # Vendimia y otoño
            estacionalidad = 150000 * sucursal['factor_base']
        elif mes in [11, 12]:  # Fiestas de fin de año
            estacionalidad = 200000 * sucursal['factor_base']
        elif mes in [7, 8]:  # Vacaciones de invierno
            estacionalidad = 80000 * sucursal['factor_base']
        elif mes in [1, 2]:  # Verano (baja temporada)
            estacionalidad = -50000 * sucursal['factor_base']
        
        # Ruido aleatorio
        ruido = np.random.normal(0, 50000 * sucursal['factor_base'])
        
        # Ventas totales
        ventas = tendencia + estacionalidad + ruido
        ventas = max(ventas, 0)  # No negativas
        
        data_records.append({
            'fecha': fecha,
            'sucursal_id': sucursal['id'],
            'sucursal_nombre': sucursal['nombre'],
            'zona': sucursal['zona'],
            'lat': sucursal['lat'],
            'lon': sucursal['lon'],
            'h3_index': sucursal['h3_index'],
            'h3_res8': sucursal['h3_res8'],
            'h3_res7': sucursal['h3_res7'],
            'ventas': ventas,
            'tendencia': tendencia,
            'estacionalidad': estacionalidad
        })

# Crear DataFrame
df_ventas = pd.DataFrame(data_records)

print("\n📈 DATOS DE VENTAS GENERADOS:")
print(f"   Período: {df_ventas['fecha'].min().strftime('%Y-%m')} a {df_ventas['fecha'].max().strftime('%Y-%m')}")
print(f"   Sucursales: {df_ventas['sucursal_id'].nunique()}")
print(f"   Total registros: {len(df_ventas):,}")
print(f"   Ventas promedio: ${df_ventas['ventas'].mean():,.0f}")
print(f"   Ventas totales (5 años): ${df_ventas['ventas'].sum():,.0f}")

# Resumen por sucursal
print("\n📊 VENTAS PROMEDIO POR SUCURSAL:")
ventas_por_suc = df_ventas.groupby(['sucursal_id', 'sucursal_nombre'])['ventas'].agg(['mean', 'sum']).round(0)
ventas_por_suc.columns = ['Promedio Mensual', 'Total 5 Años']
print(ventas_por_suc.to_string())

print("\n🔍 Primeras filas:")
display(df_ventas.head(10))

# COMMAND ----------

# DBTITLE 1,Mapa interactivo de sucursales (H3)
# Crear mapa interactivo de Mendoza con sucursales y hexágonos H3
import folium
from folium import plugins

# Centro del mapa (Mendoza capital)
centro_mendoza = [-32.8895, -68.8458]

# Crear mapa base
mapa = folium.Map(
    location=centro_mendoza,
    zoom_start=12,
    tiles='OpenStreetMap'
)

# Colores para cada sucursal
colores = ['red', 'blue', 'green', 'purple', 'orange']

# Agregar marcadores de sucursales
for i, suc in enumerate(sucursales):
    # Calcular ventas promedio de esta sucursal
    ventas_suc = df_ventas[df_ventas['sucursal_id'] == suc['id']]['ventas'].mean()
    
    # Popup con información
    popup_html = f"""
    <div style="font-family: Arial; width: 220px;">
        <h3 style="color: #2E5090; margin-bottom: 5px;">🏔️ Los Andes Market</h3>
        <h4 style="margin-top: 0;">{suc['nombre']}</h4>
        <b>ID:</b> {suc['id']}<br>
        <b>Zona:</b> {suc['zona']}<br>
        <b>Ventas promedio:</b> ${ventas_suc:,.0f}<br>
        <hr>
        <b>H3 (res 9):</b><br>
        <small>{suc['h3_index']}</small>
    </div>
    """
    
    folium.Marker(
        location=[suc['lat'], suc['lon']],
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=suc['nombre'],
        icon=folium.Icon(color=colores[i], icon='store', prefix='fa')
    ).add_to(mapa)
    
    # Dibujar hexágono H3 (resolución 9)
    hex_boundary = h3.cell_to_boundary(suc['h3_index'])
    hex_coords = [[lat, lon] for lat, lon in hex_boundary]
    
    folium.Polygon(
        locations=hex_coords,
        color=colores[i],
        weight=2,
        fill=True,
        fillColor=colores[i],
        fillOpacity=0.2,
        popup=f"H3: {suc['h3_index']}"
    ).add_to(mapa)

# Agregar controles
folium.LayerControl().add_to(mapa)

print("🗺️ MAPA INTERACTIVO - SUCURSALES DE 'LOS ANDES MARKET' EN MENDOZA")
print("   Marcadores: Ubicación de cada sucursal")
print("   Hexágonos: Índice H3 (resolución 9: ~174m diámetro)")
print("   Haz clic en los marcadores para ver detalles\n")

# Mostrar mapa
mapa

# COMMAND ----------

# DBTITLE 1,Visualización de componentes
# MAGIC %md
# MAGIC ## 2️⃣ Visualización de Componentes
# MAGIC
# MAGIC Descompondremos la serie temporal en sus componentes fundamentales para entender el comportamiento del negocio.

# COMMAND ----------

# DBTITLE 1,Gráfico de series temporales por sucursal
# Gráfico de series temporales por sucursal
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 1. Ventas totales (todas las sucursales)
ventas_totales = df_ventas.groupby('fecha')['ventas'].sum()
axes[0].plot(ventas_totales.index, ventas_totales.values, linewidth=3, color='#2E86AB', label='Ventas Totales')
axes[0].fill_between(ventas_totales.index, ventas_totales.values, alpha=0.3, color='#2E86AB')
axes[0].set_title('📈 Ventas Totales Mensuales - Todas las Sucursales (2019-2024)', fontsize=15, fontweight='bold')
axes[0].set_ylabel('Ventas ($)', fontsize=12)
axes[0].legend(loc='upper left', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

# 2. Series por sucursal
colores_suc = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
for i, suc_id in enumerate(df_ventas['sucursal_id'].unique()):
    datos_suc = df_ventas[df_ventas['sucursal_id'] == suc_id]
    nombre_suc = datos_suc['sucursal_nombre'].iloc[0]
    axes[1].plot(datos_suc['fecha'], datos_suc['ventas'], linewidth=2.5, 
                color=colores_suc[i], label=nombre_suc, alpha=0.8)

axes[1].set_title('🏪 Ventas por Sucursal', fontsize=15, fontweight='bold')
axes[1].set_xlabel('Fecha', fontsize=12)
axes[1].set_ylabel('Ventas ($)', fontsize=12)
axes[1].legend(loc='upper left', fontsize=10, ncol=2)
axes[1].grid(True, alpha=0.3)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.show()

print("📈 Observaciones:")
print("   • Las sucursales del centro comercial (SUC001, SUC003) tienen mayores ventas")
print("   • Todas muestran picos en vendimia (Mar-Abr) y fiestas (Nov-Dic)")
print("   • Tendencia de crecimiento en todas las ubicaciones")

# COMMAND ----------

# DBTITLE 1,Descomposición de componentes (agregado)
# Descomposición en componentes (usando ventas agregadas)
df_agregado = df_ventas.groupby('fecha').agg({
    'ventas': 'sum',
    'tendencia': 'sum',
    'estacionalidad': 'sum'
}).reset_index()

fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# 1. Serie Original Agregada
axes[0].plot(df_agregado['fecha'], df_agregado['ventas'], linewidth=2.5, color='#2E86AB', marker='o', markersize=4)
axes[0].set_title('📈 Serie Temporal Original (Ventas Totales)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Ventas ($)', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

# 2. Tendencia Agregada
axes[1].plot(df_agregado['fecha'], df_agregado['tendencia'], linewidth=3, color='#A23B72')
axes[1].fill_between(df_agregado['fecha'], df_agregado['tendencia'], alpha=0.3, color='#A23B72')
axes[1].set_title('📈 Componente de Tendencia (Crecimiento del negocio)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Tendencia ($)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

# 3. Estacionalidad Agregada
axes[2].plot(df_agregado['fecha'], df_agregado['estacionalidad'], linewidth=2.5, color='#F18F01', marker='s', markersize=4)
axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[2].fill_between(df_agregado['fecha'], 0, df_agregado['estacionalidad'], 
                     where=(df_agregado['estacionalidad'] > 0), alpha=0.3, color='green', label='Temporada alta')
axes[2].fill_between(df_agregado['fecha'], 0, df_agregado['estacionalidad'], 
                     where=(df_agregado['estacionalidad'] < 0), alpha=0.3, color='red', label='Temporada baja')
axes[2].set_title('🍇 Componente Estacional (Vendimia, Fiestas, Vacaciones)', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Estacionalidad ($)', fontsize=11)
axes[2].legend(loc='upper right', fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# 4. Residuos (Ruido)
residuos = df_agregado['ventas'] - df_agregado['tendencia'] - df_agregado['estacionalidad']
axes[3].plot(df_agregado['fecha'], residuos, linewidth=1.5, color='#6A994E', alpha=0.7)
axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[3].fill_between(df_agregado['fecha'], 0, residuos, alpha=0.2, color='#6A994E')
axes[3].set_title('📉 Residuos (Ruido aleatorio y factores no modelados)', fontsize=13, fontweight='bold')
axes[3].set_xlabel('Fecha', fontsize=12)
axes[3].set_ylabel('Residuos ($)', fontsize=11)
axes[3].grid(True, alpha=0.3)
axes[3].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.show()

print("\n🔍 INTERPRETACIÓN DE COMPONENTES:")
print(f"   • Tendencia: Crecimiento promedio de ${df_agregado['tendencia'].diff().mean():,.0f}/mes")
print(f"   • Estacionalidad máxima: ${df_agregado['estacionalidad'].max():,.0f} (Nov-Dic)")
print(f"   • Estacionalidad mínima: ${df_agregado['estacionalidad'].min():,.0f} (Ene-Feb)")
print(f"   • Desv. estándar residuos: ${residuos.std():,.0f}")

# COMMAND ----------

# DBTITLE 1,Análisis estadístico
# MAGIC %md
# MAGIC ## 3️⃣ Análisis Estadístico Básico
# MAGIC
# MAGIC Calcularemos estadísticas descriptivas para entender el comportamiento de las ventas.

# COMMAND ----------

# DBTITLE 1,Estadísticas por sucursal y zona
# Añadir columnas de tiempo
df_ventas['año'] = pd.DatetimeIndex(df_ventas['fecha']).year
df_ventas['mes'] = pd.DatetimeIndex(df_ventas['fecha']).month
df_ventas['trimestre'] = pd.DatetimeIndex(df_ventas['fecha']).quarter

print("📈 ESTADÍSTICAS POR SUCURSAL")
print("="*100)
estadisticas_suc = df_ventas.groupby(['sucursal_id', 'sucursal_nombre', 'zona'])['ventas'].agg([
    ('Total', 'sum'),
    ('Promedio', 'mean'),
    ('Mínimo', 'min'),
    ('Máximo', 'max'),
    ('Desv_Std', 'std')
]).round(0)
print(estadisticas_suc.to_string())
print("="*100)

# Estadísticas anuales agregadas
print("\n📊 VENTAS ANUALES AGREGADAS (TODAS LAS SUCURSALES)")
print("="*100)
estadisticas_anuales = df_ventas.groupby('año')['ventas'].agg([
    ('Total', 'sum'),
    ('Promedio_Mensual', 'mean')
]).round(0)
print(estadisticas_anuales.to_string())
print("="*100)

# Tasa de crecimiento anual
ventas_anuales = df_ventas.groupby('año')['ventas'].sum()
crecimiento = ventas_anuales.pct_change() * 100

print("\n📈 TASA DE CRECIMIENTO ANUAL:")
for año, tasa in crecimiento.items():
    if not np.isnan(tasa):
        print(f"   {año}: {tasa:+.1f}%")

# Análisis geográfico
print("\n🗺️ ANÁLISIS POR ZONA GEOGRÁFICA:")
ventas_zona = df_ventas.groupby('zona')['ventas'].agg(['sum', 'mean', 'count']).round(0)
ventas_zona.columns = ['Total', 'Promedio', 'Meses']
ventas_zona['%_Participación'] = (ventas_zona['Total'] / ventas_zona['Total'].sum() * 100).round(1)
print(ventas_zona.to_string())

# COMMAND ----------

# DBTITLE 1,Análisis estacional y mapas de calor
# Análisis de patrones mensuales
ventas_por_mes = df_ventas.groupby('mes')['ventas'].mean()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Patrón estacional mensual
meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
colores = ['#E63946' if m in [1,2] else '#2A9D8F' if m in [3,4] else '#F4A261' if m in [7,8] else '#E9C46A' if m in [11,12] else '#457B9D' for m in range(1,13)]

barras = axes[0,0].bar(meses_nombres, ventas_por_mes, color=colores, alpha=0.8, edgecolor='black')
axes[0,0].axhline(y=ventas_por_mes.mean(), color='red', linestyle='--', linewidth=2, label='Promedio Anual')
axes[0,0].set_title('📅 Patrón Estacional: Ventas Promedio por Mes', fontsize=13, fontweight='bold')
axes[0,0].set_ylabel('Ventas Promedio ($)', fontsize=11)
axes[0,0].legend(fontsize=10)
axes[0,0].grid(axis='y', alpha=0.3)
axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

for barra in barras:
    altura = barra.get_height()
    axes[0,0].text(barra.get_x() + barra.get_width()/2., altura,
            f'${altura/1000:.0f}K',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# 2. Heatmap: Ventas por Mes y Sucursal
ventas_pivot = df_ventas.pivot_table(values='ventas', index='mes', columns='sucursal_id', aggfunc='mean')
sns.heatmap(ventas_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0,1], cbar_kws={'label': 'Ventas ($)'})
axes[0,1].set_title('🔥 Mapa de Calor: Ventas por Mes y Sucursal', fontsize=13, fontweight='bold')
axes[0,1].set_xlabel('Sucursal', fontsize=11)
axes[0,1].set_ylabel('Mes', fontsize=11)
axes[0,1].set_yticklabels(meses_nombres, rotation=0)

# 3. Comparación de zonas
ventas_zona_mes = df_ventas.groupby(['mes', 'zona'])['ventas'].mean().unstack()
for zona in ventas_zona_mes.columns:
    axes[1,0].plot(meses_nombres, ventas_zona_mes[zona], marker='o', linewidth=2, label=zona, markersize=6)

axes[1,0].set_title('🗺️ Estacionalidad por Zona Geográfica', fontsize=13, fontweight='bold')
axes[1,0].set_xlabel('Mes', fontsize=11)
axes[1,0].set_ylabel('Ventas Promedio ($)', fontsize=11)
axes[1,0].legend(loc='best', fontsize=9)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# 4. Boxplot por trimestre
df_ventas['trimestre_nombre'] = df_ventas['trimestre'].map({1: 'Q1\n(Ene-Mar)', 2: 'Q2\n(Abr-Jun)', 3: 'Q3\n(Jul-Sep)', 4: 'Q4\n(Oct-Dic)'})
sns.boxplot(data=df_ventas, x='trimestre_nombre', y='ventas', palette='Set2', ax=axes[1,1])
axes[1,1].set_title('📋 Distribución de Ventas por Trimestre', fontsize=13, fontweight='bold')
axes[1,1].set_xlabel('Trimestre', fontsize=11)
axes[1,1].set_ylabel('Ventas ($)', fontsize=11)
axes[1,1].grid(axis='y', alpha=0.3)
axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

plt.tight_layout()
plt.show()

print("\n📅 MESES CON MAYORES VENTAS:")
top_meses = ventas_por_mes.nlargest(3)
for i, (mes, valor) in enumerate(top_meses.items(), 1):
    print(f"   {i}. {meses_nombres[mes-1]}: ${valor:,.0f}")

print("\n📅 MESES CON MENORES VENTAS:")
bottom_meses = ventas_por_mes.nsmallest(3)
for i, (mes, valor) in enumerate(bottom_meses.items(), 1):
    print(f"   {i}. {meses_nombres[mes-1]}: ${valor:,.0f}")

# COMMAND ----------

# DBTITLE 1,Guardar datos
# MAGIC %md
# MAGIC ## 4️⃣ Exportar Datos para Próximos Notebooks
# MAGIC
# MAGIC Guardaremos estos datos para utilizarlos en los siguientes notebooks de la serie.

# COMMAND ----------

# DBTITLE 1,Guardar como Delta Table con datos georeferenciados
# Convertir a Spark DataFrame y guardar como Delta Table
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

# Seleccionar columnas relevantes
df_para_guardar = df_ventas[[
    'fecha', 'sucursal_id', 'sucursal_nombre', 'zona',
    'lat', 'lon', 'h3_index', 'h3_res8', 'h3_res7',
    'ventas', 'tendencia', 'estacionalidad',
    'año', 'mes', 'trimestre'
]]

# Convertir pandas a Spark DataFrame
df_spark = spark.createDataFrame(df_para_guardar)

# Guardar como tabla Delta
tabla_destino = "ventas_mensuales_mendoza_h3"

df_spark.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(tabla_destino)

print(f"✅ Datos guardados en tabla Delta: {tabla_destino}")
print(f"   Total registros: {df_spark.count():,}")
print(f"   Sucursales: {df_spark.select('sucursal_id').distinct().count()}")
print(f"   Período: {df_ventas['fecha'].min().strftime('%Y-%m')} a {df_ventas['fecha'].max().strftime('%Y-%m')}")

print("\n📋 Esquema de la tabla:")
df_spark.printSchema()

print("\n🔍 Primeras filas guardadas:")
display(spark.table(tabla_destino).limit(10))

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 🎯 Conclusiones y Próximos Pasos
# MAGIC
# MAGIC ### Lo que aprendimos:
# MAGIC
# MAGIC ✅ **Conceptos básicos de series temporales**
# MAGIC * Identificamos tendencia, estacionalidad y ruido
# MAGIC * Visualizamos patrones temporales
# MAGIC * Calculamos métricas estadísticas
# MAGIC
# MAGIC ✅ **Dimensión Geográfica con H3**
# MAGIC * 5 sucursales de "Los Andes Market" en diferentes zonas de Mendoza
# MAGIC * Coordenadas GPS reales y índices H3 (resolución 9, 8 y 7)
# MAGIC * Mapa interactivo con Folium
# MAGIC * Análisis espacial de performance
# MAGIC
# MAGIC ✅ **Insights de negocio para "Los Andes Market"**
# MAGIC * La cadena crece consistentemente en todas las sucursales
# MAGIC * Patrones estacionales argentinos:
# MAGIC   - **Marzo-Abril**: Pico por vendimia mendocina 🍇
# MAGIC   - **Noviembre-Diciembre**: Fiestas de fin de año 🎉
# MAGIC   - **Julio-Agosto**: Vacaciones de invierno ⛄
# MAGIC   - **Enero-Febrero**: Baja temporada (verano) ☀️
# MAGIC * Sucursales en zonas comerciales (Centro, Guaymallén) tienen 15-50% más ventas
# MAGIC * Variabilidad geográfica: ubicación impacta significativamente en volumen
# MAGIC
# MAGIC ### 🎓 Capacidades Implementadas:
# MAGIC
# MAGIC 🗺️ **Georeferenciación**:
# MAGIC * Coordenadas lat/lon de cada sucursal
# MAGIC * Índices H3 en 3 resoluciones (9, 8, 7)
# MAGIC * Mapas interactivos con Folium
# MAGIC * Análisis por zona geográfica
# MAGIC
# MAGIC 📊 **Análisis Avanzado**:
# MAGIC * Series múltiples (una por sucursal)
# MAGIC * Heatmaps de ventas por mes y ubicación
# MAGIC * Comparación de performance espacial
# MAGIC * Datos realistas para supermercado regional en Mendoza
# MAGIC
# MAGIC ### 📚 Próximo Notebook:
# MAGIC
# MAGIC **02_Preparacion_Datos_Empresariales.ipynb**
# MAGIC * Feature engineering para Deep Learning
# MAGIC * **Features geográficas** (distancia al centro, densidad H3)
# MAGIC * Normalización y escalado
# MAGIC * Creación de ventanas temporales (sequences)
# MAGIC * División train/validation/test **por sucursal**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 💡 **Tip empresarial**: Las series temporales requieren dividir los datos respetando el orden temporal (no shuffle aleatorio como en otros problemas de ML).
# MAGIC
# MAGIC 🗺️ **Tip geoespacial**: Los índices H3 permiten agregar datos por proximidad geográfica, detectar clusters de clientes, y crear features espaciales para modelos de ML.