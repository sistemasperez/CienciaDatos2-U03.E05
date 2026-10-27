# Declaración de uso de inteligencia artificial · INF-8239

## Herramientas utilizadas
- Antigravity IDE / Gemini 3.8 Flash (High) para asistencia en desarrollo, revisión metodológica, estructuración de pruebas y redacción técnica de reportes.
- Python 3.12 y suite científica (`uv`, `pandas`, `numpy`, `scikit-learn`, `pytest`, `matplotlib`, `seaborn`).

## Propósito de cada uso
- **Depuración e infraestructura:** Verificación de ejecución de scripts, particionamiento temporal y configuración de entorno virtual reproducible con `uv`.
- **Implementación de métricas de ranking:** Estructuración de funciones vectorizadas para `Precision@K`, `Recall@K`, `NDCG@K`, `HitRate@K` y `Catalog Coverage`.
- **Análisis de Pareto y Hibridación:** Asistencia en la exploración paramétrica de $\alpha$ y visualización de la frontera de Pareto.
- **Documentación y Dataset Card:** Síntesis técnica de características del dataset MovieLens, contratos de datos y consideraciones éticas / licencias.

## Prompts o decisiones relevantes
- Decisión de corte temporal: Se utilizó `temporal_leave_one_out` para preservar la causalidad temporal y evitar data leakage en la evaluación.
- Normalización en recomendador híbrido: Min-max scaling acotado para poner en escala común los puntajes de factorización matricial (predicción de rating continuo) y similitud de contenido TF-IDF (coseno en $[0, 1]$).
- Manejo de cold-start: Definición de fallback no personalizado basado en popularidad bayesiana ponderada (`weighted_popularity`) para mitigar la ausencia de historial.

## Elementos verificados por el equipo
- Verificación matemática y código de `ndcg_at_k`, `precision_at_k`, `recall_at_k`.
- Validación de que ningún dato crudo (`data/raw/`) sea subido al repositorio para cumplir con la licencia de GroupLens MovieLens.
- Ejecución completa y exitosa de la suite de pruebas unitarias (`pytest`).
- Comprobación de que la función de predicción y las métricas computen valores finitos sin fugas de datos.

## Cambios realizados sobre resultados generados
- Ajuste del cálculo de NDCG para manejar casos de corte a $K$ con relevancia binaria o implícita en leave-one-out.
- Modificación de `.gitignore` para preservar los reportes generados (`reports/`) asegurando la reproducibilidad del repositorio.

## Responsabilidad asumida
El estudiante asume la responsabilidad total por la autoría, veracidad, rigor conceptual, conclusiones y defensa de todo el código y documentación presentados en este proyecto.
