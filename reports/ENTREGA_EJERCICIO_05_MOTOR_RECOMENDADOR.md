# INF-8239 · Ejercicio 05: Motor Recomendador Reproducible

**Maestría en Inteligencia Artificial · Universidad Autónoma de Santo Domingo (UASD)**  
**Asignatura:** Ciencia de Datos II  
**Docente:** Edwin Ramón José Nolasco  
**Estudiante:** [Nombre del Estudiante]  
**Repositorio GitHub:** `https://github.com/TU_USUARIO/inf8239-u03-recomendadores` *(Reemplace con la URL de su repositorio)*  

---

## 1. Tabla Principal de Resultados Comparativos

La evaluación se realizó mediante el protocolo causal `temporal_leave_one_out` sobre los 587 usuarios evaluables de MovieLens (`ml-latest-small`), midiendo la capacidad de predecir el último ítem consumido cronológicamente en el ranking Top-10:

| Modelo / Enfoque | Parámetro $\alpha$ | NDCG@10 | HitRate@10 | Precision@10 | Recall@10 | Cobertura Catálogo | RMSE (Rating) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Popularidad Ponderada (Baseline)** | - | 0.0197 | 0.0426 | 0.0043 | 0.0426 | 1.22% | - |
| **Contenido Puro (TF-IDF Géneros)** | 0.00 | 0.0108 | 0.0256 | 0.0026 | 0.0256 | 11.71% | - |
| **Híbrido ($\alpha=0.10$)** | 0.10 | 0.0105 | 0.0239 | 0.0024 | 0.0239 | 11.15% | 1.0264 |
| **Híbrido ($\alpha=0.25$)** | 0.25 | 0.0127 | 0.0324 | 0.0032 | 0.0324 | 10.65% | 1.0264 |
| **Híbrido ($\alpha=0.50$)** | 0.50 | 0.0144 | 0.0307 | 0.0031 | 0.0307 | 8.87% | 1.0264 |
| **Híbrido ($\alpha=0.75$) [Decisión Pareto]** ⭐ | **0.75** | **0.0172** | **0.0375** | **0.0037** | **0.0375** | **7.75%** | **1.0264** |
| **Híbrido ($\alpha=0.90$)** | 0.90 | 0.0189 | 0.0392 | 0.0039 | 0.0392 | 7.47% | 1.0264 |
| **Colaborativo Puro (MF - SVD)** | 1.00 | 0.0184 | 0.0375 | 0.0037 | 0.0375 | 7.46% | 1.0264 |

---

## 2. Metodología, Partición Causal y Auditoría de Datos

### 2.1 Datos y Licencia
- Se utilizó el dataset MovieLens (`ml-latest-small`, 100,836 calificaciones, 610 usuarios y 9,724 películas evaluadas).
- **Integridad:** SHA-256 verificado: `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`.
- **Licencia:** Uso académico exclusivo (GroupLens Research / Universidad de Minnesota). Los datos crudos se encuentran ignorados en `.gitignore` para cumplir con la no redistribución comercial ni pública sin licencia.
- **Densidad de la matriz:** 1.697% (98.303% de esparcidad). Calificación promedio global: 3.50 estrellas.

### 2.2 Partición Causal Temporal (`temporal_leave_one_out`)
En sistemas de recomendación, la división aleatoria (*random split*) introduce un severo **data leakage temporal**, ya que el modelo se entrena con eventos del futuro para predecir eventos del pasado. En este proyecto se utilizó estrictamente `temporal_leave_one_out`:
- Para cada uno de los 610 usuarios, se reservó su última interacción cronológica en el conjunto `test`.
- Las interacciones históricas previas conformaron el conjunto de entrenamiento `train` (100,226 interacciones).
- Se garantizó formalmente que $\max(\text{timestamp}_{\text{train}}) \le \text{timestamp}_{\text{test}}$ para todo usuario.

---

## 3. Análisis de Modelos, Errores y Límites

1. **Popularidad Ponderada (Baseline Bayesiano tipo IMDb):**
   - **Ventaja:** Obtiene el mayor NDCG@10 (0.0197) al aprovechar el fuerte sesgo de popularidad intrínseco del dataset.
   - **Límite crítico:** Sufre de nula personalización y severa concentración: **solo recomienda el 1.22% del catálogo**. Los mismos títulos taquilleros son recomendados a todos los usuarios, ignorando gustos individuales y castigando el catálogo *long-tail*.
2. **Filtrado Basado en Contenido (TF-IDF de Géneros + Similitud Coseno):**
   - **Ventaja:** Máxima cobertura del catálogo (**11.71%**) y capacidad inmediata de recomendar ítems nuevos sin interacciones previas (*cold-start* de ítems).
   - **Límite crítico:** Menor capacidad predictiva (NDCG@10 = 0.0108). Causa **sobre-especialización** léxica (recomienda únicamente películas con la combinación idéntica de géneros).
3. **Factorización Matricial (SVD Latente mediante SGD):**
   - Descomposición en 20 factores latentes, regularización $L_2 = 0.05$ y 12 épocas.
   - **Error de regresión:** Logra un $\text{RMSE} = 1.0264$ estrellas en test.
   - **Límite:** No puede inferir preferencias para usuarios o ítems sin historial (*cold-start* total).

---

## 4. Frontera de Pareto y Decisión Final

El modelo híbrido unifica ambas señales mediante combinación convexa:
$$\text{Score}_{\text{híbrido}} = \alpha \cdot \text{Score}_{\text{colaborativo}} + (1 - \alpha) \cdot \text{Score}_{\text{contenido}}$$

### Decisión de Pareto: $\alpha = 0.75$
- **Justificación:** Al ponderar un 75% el componente colaborativo y un 25% el componente de contenido, se alcanza un punto óptimo de Pareto:
  - **NDCG@10 de 0.0172** (cercano al máximo absoluto de popularidad).
  - **Cobertura de catálogo del 7.75%**, lo que representa **más de 6 veces la diversidad de catálogo respecto a la popularidad pura** ($1.22\%$).
- Este balance protege al sistema contra el colapso de popularidad, inyecta diversidad y regularización temática, y mantiene una excelente precisión de ranking.

---

## 5. Estrategia de Cold-Start (Arranque en Frío)

- **Usuarios Fríos:** Al no existir vector de preferencias ni historial, el sistema aplica un fallback determinista al top-10 de **popularidad bayesiana ponderada** (`reports/cold_start_fallback.csv`), garantizando recomendaciones robustas y con alta probabilidad de aceptación inicial.
- **Ítems Fríos:** Nuevas películas sin calificaciones son recomendadas mediante el perfil temático de géneros (TF-IDF), superando la barrera de falta de calificaciones colaborativas.

---

## 6. Declaración de Uso Ético y Responsable de Inteligencia Artificial

- **Herramientas utilizadas:** Gemini 3.8 Flash (High) mediante el entorno Antigravity IDE.
- **Propósito:** Apoyo en la optimización de código matricial (scikit-learn), diseño de pruebas unitarias (`pytest`), implementación formal de métricas de ranking (`Precision@K`, `Recall@K`, `NDCG@K`) y estructuración del reporte.
- **Verificaciones y correcciones realizadas:**
  - Corrección de compatibilidad con arrays bidimensionales en scikit-learn (`np.asarray`).
  - Validación matemática de los descuentos logarítmicos en `ndcg_at_k`.
  - Aseguramiento de la partición causal temporal estricta sin fuga de información.
  - Verificación de la no redistribución de MovieLens en Git para respetar los términos de GroupLens.
  - Ejecución y aprobación al 100% de la suite de pruebas automatizadas (9/9 passed).
- **Responsabilidad asumida:** El estudiante responde plenamente por el código, resultados empíricos, interpretaciones analíticas y conclusiones expuestas.
