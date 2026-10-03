# Resumen metodológico y resultados actuales del pipeline

## 1. Cambio metodológico principal

El pipeline actual ya no usa directamente el texto completo como entrada única para SBERT. Se incorporó un procedimiento de chunking antes de la vectorización semántica.

### Antes
- Se tomaba el texto completo de cada unidad discursiva.
- Se enviaba directamente a Sentence-BERT mediante `model.encode(texto_completo)`.
- Este enfoque presentaba un problema importante para discursos largos, porque el modelo solo procesa una porción limitada de tokens y el resto del texto podía quedar fuera del embedding.

### Ahora
- El texto se divide en segmentos (chunks) usando el tokenizador del modelo.
- Se fijó una ventana de 120 tokens con solapamiento de 20 tokens.
- Cada chunk genera un embedding independiente.
- Los embeddings parciales se agregan mediante mean pooling para obtener un embedding final por documento.
- Este embedding final se usa para la similitud coseno, la proyección UMAP y el clustering.

## 2. Cambios en los parámetros del pipeline

### Vectorización semántica
- Modelo: `paraphrase-multilingual-MiniLM-L12-v2`
- Dimensión del embedding: 384
- Estrategia nueva: chunking por tokens + mean pooling

### UMAP
- Antes: `n_neighbors=8`, `min_dist=0.1`, fijo para todos los ejes.
- Ahora: se realizó una búsqueda automática sobre:
  - `n_neighbors`: 5, 8 y 12
  - `min_dist`: 0.0, 0.1 y 0.25
- La mejor configuración se seleccionó por eje en función de la calidad del clustering.

### HDBSCAN
- Antes: `min_cluster_size=3`, `min_samples=2`, fijo para todos los ejes.
- Ahora: se evaluaron combinaciones de:
  - `min_cluster_size`: 3, 4 y 5
  - `min_samples`: 2 y 3
- La mejor combinación se eligió por eje según Silhouette y Davies-Bouldin.

### Validación adicional
- Se incorporó una validación con Ward y K-Means para comprobar la estabilidad de los agrupamientos.
- Ward se usó para estimar un número razonable de clusters.
- K-Means se evaluó como comparación externa frente a HDBSCAN.

## 3. Resultados actuales reales del pipeline

Los resultados actuales se obtuvieron con los embeddings generados tras el chunking y con la búsqueda de hiperparámetros. Estos resultados deben interpretarse como una validación experimental del nuevo pipeline, no como una confirmación definitiva de una estructura de clusters muy fuerte.

### Resultados del clustering final

La tabla siguiente resume los resultados del clustering obtenido con la configuración final seleccionada para cada eje, en lugar de centrarse únicamente en el proceso de comparación entre hiperparámetros. Los valores se tomaron de [resultados_clustering/metricas_ward_kmeans.csv](resultados_clustering/metricas_ward_kmeans.csv).

| Eje temático | K estimada por Ward | Clusters HDBSCAN | Puntos de ruido | Silhouette HDBSCAN | Davies-Bouldin HDBSCAN | ARI K-Means vs HDBSCAN |
|---|---:|---:|---:|---:|---:|---:|
| Educación, innovación y tecnología | 4 | 10 | 0 | 0.321 | 0.656 | 0.102 |
| Empleo, desarrollo y emprendimiento | 2 | 10 | 1 | 0.314 | 0.687 | 0.164 |
| Integridad pública y lucha contra la corrupción | 2 | 2 | 1 | 0.396 | 0.987 | 0.311 |
| Seguridad ciudadana y lucha contra la criminalidad | 2 | 11 | 1 | 0.482 | 0.554 | 0.111 |

### Interpretación breve
- Los valores actuales son claramente más consistentes que los obtenidos con la configuración fija anterior.
- Sin embargo, los resultados siguen mostrando una estructura de clusters moderada, no una separación extremadamente clara en todos los ejes.
- Esto sugiere que la mejora metodológica sí impactó el análisis, pero que la naturaleza semántica del corpus también limita la separación entre grupos.

## 4. Proyecciones UMAP + HDBSCAN por eje temático

Las siguientes proyecciones muestran la distribución bidimensional obtenida con UMAP y los clusters detectados con HDBSCAN para cada eje temático, usando la mejor configuración seleccionada por el proceso de búsqueda.

### Educación, innovación y tecnología
![Proyección UMAP + HDBSCAN](resultados_clustering/proyeccion_umap_hdbscan_educacion_innovacion_y_tecnologia.png)

### Empleo, desarrollo y emprendimiento
![Proyección UMAP + HDBSCAN](resultados_clustering/proyeccion_umap_hdbscan_empleo_desarrollo_y_emprendimiento.png)

### Integridad pública y lucha contra la corrupción
![Proyección UMAP + HDBSCAN](resultados_clustering/proyeccion_umap_hdbscan_integridad_publica_y_lucha_contra_la_corrupcion.png)

### Seguridad ciudadana y lucha contra la criminalidad
![Proyección UMAP + HDBSCAN](resultados_clustering/proyeccion_umap_hdbscan_seguridad_ciudadana_y_lucha_contra_la_criminalidad.png)

## 5. Archivos actuales del flujo

- Embeddings y corpus procesado:
  - [corpus_total_con_embeddings.csv](corpus_total_con_embeddings.csv)
  - [embeddings_sbert.npy](embeddings_sbert.npy)
- Resultados de clustering y validación:
  - [resultados_clustering/metricas_ward_kmeans.csv](resultados_clustering/metricas_ward_kmeans.csv)
  - [resultados_clustering/grid_busqueda_educacion_innovacion_y_tecnologia.csv](resultados_clustering/grid_busqueda_educacion_innovacion_y_tecnologia.csv)
  - [resultados_clustering/grid_busqueda_empleo_desarrollo_y_emprendimiento.csv](resultados_clustering/grid_busqueda_empleo_desarrollo_y_emprendimiento.csv)
  - [resultados_clustering/grid_busqueda_integridad_publica_y_lucha_contra_la_corrupcion.csv](resultados_clustering/grid_busqueda_integridad_publica_y_lucha_contra_la_corrupcion.csv)
  - [resultados_clustering/grid_busqueda_seguridad_ciudadana_y_lucha_contra_la_criminalidad.csv](resultados_clustering/grid_busqueda_seguridad_ciudadana_y_lucha_contra_la_criminalidad.csv)
- Resultados de similitud y calor:
  - [resultados](resultados)
  - [graficos](graficos)

## 6. Recomendación para el paper

En el texto metodológico se recomienda describir el nuevo procedimiento de la siguiente manera:

> Se incorporó un procedimiento de chunking antes de la generación de embeddings con SBERT. Cada intervención discursiva fue segmentada en bloques de 120 tokens con solapamiento de 20 tokens, generando embeddings parciales que posteriormente fueron agregados mediante mean pooling para obtener un embedding final por documento. Este procedimiento busca preservar la información semántica contenida en textos largos y reducir el sesgo de truncamiento del modelo. Posteriormente, los embeddings fueron proyectados con UMAP y agrupados con HDBSCAN, utilizando una búsqueda de hiperparámetros para seleccionar la configuración más estable por eje temático.
