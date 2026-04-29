# Clustering de Modos de Consumo Energético — BPC Estación de Bombeo

**Diplomado Ciencia de Datos | Proyecto 6**  
**IDC Ingeniería de Confiabilidad S.A.S. — División de Analítica Predictiva**

---

## Descripción del Proyecto

Desarrollo de un modelo de analítica descriptiva y clustering para identificar y agrupar los diferentes **modos de consumo energético** de una Bomba con motor eléctrico de media tensión (~4.2 kV) en una estación de bombeo de hidrocarburos.

El análisis permite:
- Identificar patrones operativos distintos (modos de consumo)
- Determinar el **punto de mejor eficiencia (BEP)** de la bomba
- Detectar desviaciones respecto a la operación óptima histórica
- Proporcionar al operador un mapa de operación interactivo

---

## Estructura del Proyecto

```
pump-clustering-bpc/
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb       # Preprocesamiento + Análisis Exploratorio
│   ├── 02_clustering.ipynb              # Pipeline de clustering + validación
│   └── 03_efficiency_3d_viz.ipynb       # Análisis de eficiencia + viz 3D
├── src/
│   ├── preprocessing.py                 # Funciones de limpieza y filtrado
│   ├── clustering_utils.py              # Validación y selección de clusters
│   └── visualization.py                 # Gráficos estáticos e interactivos
├── outputs/
│   ├── figures/                         # Gráficos exportados (.png, .svg)
│   └── html/                            # Dashboards interactivos (.html)
├── requirements.txt
└── README.md
```

---

## Dataset

| Atributo | Detalle |
|----------|---------|
| Activo | Bomba centrífuga con motor eléctrico de media tensión |
| Registros | 255,307 filas |
| Variables | 23 columnas |
| Período | 2024-06-01 a 2024-08-01 |
| Frecuencia | ~14 segundos |

### Cómo obtener los datos

Los datos **no se incluyen** en el repositorio (información confidencial de cliente real).  
Descarga los archivos desde el repositorio privado del diplomado y colócalos en la raíz del proyecto:

```
Data_EBR_processed (1).csv
Pesos Ponderados - Proyecto6 (1).xlsx
```

---

## Instalación y Configuración

### Prerequisitos
- Python 3.10+
- VS Code con extensión Jupyter (recomendado)

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Abrir en VS Code

```bash
code .
```

Selecciona el kernel de Python correcto al abrir cada notebook.

---

## Metodología

### 1. Preprocesamiento
- Conversión de timestamps UTC
- Eliminación de períodos de paro (RPM < 100, Potencia = 0)
- Detección y documentación de anomalías físicas (potencia sin flujo, flujo sin presión)
- Análisis de varianza y correlación para selección de features
- Normalización/estandarización para clustering

### 2. Reducción de Dimensionalidad
- **PCA**: para interpretabilidad y varianza explicada
- **UMAP**: para visualización no lineal de clusters

### 3. Clustering
- **K-Means**: eficiente, centroides interpretables
- **DBSCAN**: robusto a ruido, sin número predefinido de clusters
- **Agglomerative Hierarchical**: dendrograma para presentación a gerencia

### 4. Validación
- Coeficiente de Silhouette
- Índice Davies-Bouldin
- Índice Calinski-Harabasz
- Coherencia física de centroides

### 5. Análisis de Eficiencia
- Método 1: Mínimo de Potencia/Flujo (proxy de eficiencia hidráulica)
- Método 2: Máximo del índice de desempeño ponderado (pesos EPI)
- Método 3: BEP estimado sobre curva H-Q (ajuste polinomial)

---

## Visualizaciones

Los outputs interactivos se guardan en `outputs/html/`:
- `mapa_operacion_2d.html` — Scatter 2D UMAP coloreado por cluster
- `mapa_operacion_3d.html` — Scatter 3D Plotly (RPM × Flujo × Potencia)
- `pump_3d_clusters.html` — Modelo 3D de bomba con modo operativo superpuesto
- `efficiency_surface.html` — Superficie de eficiencia energética

---

## Equipo

| Nombre | Rol |
|--------|-----|
| Nicolás Tejada | Estudiante — Analítica de Datos |

**Instructor/Sponsor:** IDC Ingeniería de Confiabilidad S.A.S.

---

*Uso académico exclusivo. Datos confidenciales de cliente real — no distribuir.*
