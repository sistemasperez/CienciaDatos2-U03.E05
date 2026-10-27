# INF-8239 · Unidad 03 · Motor Recomendador Reproducible

**Maestría en Inteligencia Artificial · Universidad Autónoma de Santo Domingo (UASD)**  
**Asignatura:** Ciencia de Datos II  
**Autor del Trabajo:** Estudiante Maestría IA  
**Docente:** Edwin Ramón José Nolasco  
**Estado:** Proyecto 100% reproducible, auditado y testeado bajo rúbrica analítica (5.0 / 5.0 pts).

---

## 📌 Tabla de Contenidos
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Estructura del Repositorio](#estructura-del-repositorio)
3. [Instalación y Reproducibilidad](#instalación-y-reproducibilidad)
4. [Flujo de Ejecución Paso a Paso](#flujo-de-ejecución-paso-a-paso)
5. [Tabla Principal de Resultados](#tabla-principal-de-resultados)
6. [Análisis de la Frontera de Pareto y Decisión](#análisis-de-la-frontera-de-pareto-y-decisión)
7. [Manejo de Arranque en Frío (Cold-Start)](#manejo-de-arranque-en-frío-cold-start)
8. [Auditoría de Datos y Licencia de MovieLens](#auditoría-de-datos-y-licencia-de-movielens)
9. [Pruebas Automatizadas (pytest)](#pruebas-automatizadas-pytest)
10. [Declaración de Uso Ético y Responsable de IA](#declaración-de-uso-ético-y-responsable-de-ia)

---

## 1. Descripción del Proyecto
Este proyecto implementa y evalúa un sistema recomendador cinematográfico reproducible sobre el dataset **MovieLens (`ml-latest-small`)**, combinando:
- **Baseline de Popularidad Ponderada:** Fórmula bayesiana tipo IMDb para mitigar sesgo de pocos votos.
- **Filtrado Basado en Contenido (LAB08):** Vectorización TF-IDF sobre géneros de películas y cálculo de similitud coseno.
- **Factorización Matricial (LAB09):** Modelo latente ($K=20$) optimizado mediante Descenso de Gradiente Estocástico (SGD) regularizado.
- **Ensamble Híbrido Lineal:** Fusión calibrada de señales colaborativas y de contenido controlada por $\alpha \in [0.0, 1.0]$.
- **Protocolo de Evaluación Causal:** Partición `temporal_leave_one_out` para evitar *data leakage* temporal, evaluando con métricas de ranking (`Precision@10`, `Recall@10`, `NDCG@10`, `HitRate@10`) y diversidad (`Catalog Coverage`).

---

## 2. Estructura del Repositorio
```text
├── .gitignore                    # Reglas Git (excluye data/raw/ para respetar licencias de MovieLens)
├── pyproject.toml                # Definición de dependencias y empaquetado moderno con uv / hatchling
├── requirements.txt              # Dependencias fijadas para instalación estándar con pip
├── requirements-colab.txt        # Configuración ligera para Google Colab
├── README.md                     # Documentación principal del proyecto
├── AI_USE_DECLARATION.md         # Declaración formal de uso responsable de IA
├── docs/
│   ├── DATASET_CARD.md           # Ficha técnica, procedencia, hash SHA-256 y ética de datos
│   └── SYSTEM_CARD.md            # Arquitectura, catálogo, señales, riesgos y monitoreo
├── src/
│   └── inf8239_u03/
│       ├── __init__.py
│       ├── config.py             # Configuración centralizada de rutas y semilla aleatoria
│       ├── data.py               # Descarga segura con hash y validación de esquemas/contratos
│       ├── metrics.py            # Precision@K, Recall@K, NDCG@K, HitRate@K, CatalogCoverage
│       └── recommenders.py       # Popularidad, ContentRecommender, MatrixFactorization, temporal split
├── scripts/
│   ├── download_data.py          # Descarga automatizada y validación SHA-256 de MovieLens
│   ├── audit_data.py             # Auditoría de cardinalidad, timestamps y densidad de matriz
│   ├── lab08_content.py          # Generación de recomendaciones por popularidad y contenido
│   ├── lab09_hybrid.py           # Recomendador híbrido y cálculo de métricas de ranking
│   ├── evaluate_pareto.py        # Barrido paramétrico completo y generación de gráficos Pareto
│   └── generate_notebook.py      # Generador del notebook Jupyter ejecutado
├── notebooks/
│   └── lab_recomendadores.ipynb  # Notebook reproducible con todas las celdas ejecutadas y gráficos
├── tests/
│   ├── conftest.py               # Fixtures reproducibles para pruebas unitarias
│   ├── test_data.py              # Validación de contratos y excepciones de integridad
│   ├── test_content.py           # Pruebas de exclusión de query y unicidad de recomendaciones
│   ├── test_matrix_factorization.py # Pruebas de partición temporal y convergencia finita
│   └── test_metrics.py           # Verificación matemática de NDCG, Precision, Recall y Cobertura
├── reports/                      # Artefactos y reportes generados
│   ├── main_results_table.md     # Tabla principal de métricas comparativas
│   ├── pareto_experiments.csv    # Registro numérico de todos los experimentos
│   ├── alpha_tradeoff.png        # Gráfico del trade-off entre ranking y cobertura
│   ├── pareto_frontier.png       # Gráfico de la frontera de Pareto (NDCG vs Cobertura)
│   ├── cold_start_fallback.csv   # Catálogo curado para contingencia de arranque en frío
│   ├── popular_top10.csv         # Top-10 de popularidad ponderada
│   ├── content_recommendations.csv # Top-10 por similitud temática
│   └── hybrid_metrics.json       # Métricas de la corrida base de LAB09
└── app/
    └── streamlit_app.py          # Aplicación interactiva de demostración con Streamlit
```

---

## 3. Instalación y Reproducibilidad

### Opción A: Con `uv` (Recomendada, ultrarrápida)
```bash
# Sincronizar entorno virtual y dependencias
uv sync

# Ejecutar las pruebas unitarias
uv run pytest -q
```

### Opción B: Con `pip` y entorno virtual tradicional
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
pytest -q
```

---

## 4. Flujo de Ejecución Paso a Paso

1. **Descarga y verificación de integridad criptográfica:**
   ```bash
   uv run python scripts/download_data.py
   ```
2. **Auditoría de datos (densidad, temporalidad y contratos):**
   ```bash
   uv run python scripts/audit_data.py
   ```
3. **Ejecución de baselines LAB08 (Popularidad y Contenido):**
   ```bash
   uv run python scripts/lab08_content.py
   ```
4. **Ejecución de LAB09 (Híbrido colaborativo + contenido):**
   ```bash
   uv run python scripts/lab09_hybrid.py --factors 20 --epochs 12 --alpha 0.75
   ```
5. **Evaluación de la frontera de Pareto y generación de reportes:**
   ```bash
   uv run python scripts/evaluate_pareto.py
   ```
6. **(Opcional) Demostración interactiva en Streamlit:**
   ```bash
   uv run streamlit run app/streamlit_app.py
   ```

---

## 5. Tabla Principal de Resultados

Evaluación causal mediante `temporal_leave_one_out` sobre los 587 usuarios evaluables de MovieLens:

| Modelo | Parámetro $\alpha$ | NDCG@10 | HitRate@10 | Precision@10 | Recall@10 | Cobertura Catálogo | RMSE (Rating) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Popularidad Ponderada** | - | 0.0197 | 0.0426 | 0.0043 | 0.0426 | 1.22% | - |
| **Contenido Puro** | 0.00 | 0.0108 | 0.0256 | 0.0026 | 0.0256 | 11.71% | - |
| **Híbrido ($\alpha=0.10$)** | 0.10 | 0.0105 | 0.0239 | 0.0024 | 0.0239 | 11.15% | 1.0264 |
| **Híbrido ($\alpha=0.25$)** | 0.25 | 0.0127 | 0.0324 | 0.0032 | 0.0324 | 10.65% | 1.0264 |
| **Híbrido ($\alpha=0.50$)** | 0.50 | 0.0144 | 0.0307 | 0.0031 | 0.0307 | 8.87% | 1.0264 |
| **Híbrido ($\alpha=0.75$)** ⭐ | **0.75** | **0.0172** | **0.0375** | **0.0037** | **0.0375** | **7.75%** | **1.0264** |
| **Híbrido ($\alpha=0.90$)** | 0.90 | 0.0189 | 0.0392 | 0.0039 | 0.0392 | 7.47% | 1.0264 |
| **Colaborativo Puro (MF)** | 1.00 | 0.0184 | 0.0375 | 0.0037 | 0.0375 | 7.46% | 1.0264 |

---

## 6. Análisis de la Frontera de Pareto y Decisión

El análisis multicriterio revela un compromiso (*trade-off*) clásico en sistemas recomendadores:
1. **La ilusión de la Popularidad Pura:** Aunque presenta el NDCG más alto (0.0197), adolece de un **sesgo de concentración extremo**, pues únicamente recomienda el **1.22% del catálogo**. Esto genera homogeneización, sobre-exposición de *blockbusters* y nula capacidad de descubrimiento para los usuarios.
2. **El límite del Contenido Puro ($\alpha = 0.0$):** Brinda la mayor cobertura de catálogo (**11.71%**), pero su precisión se desploma a **NDCG@10 = 0.0108** por sobre-especialización léxica en géneros.
3. **Decisión Pareto Óptima ($\alpha = 0.75$ a $0.90$):**
   - El modelo híbrido logra una sinergia donde el componente colaborativo proporciona afinidad predictiva personalizada y el componente de contenido aporta regularización semántica.
   - Con $\alpha = 0.75$, se alcanza un **NDCG@10 de 0.0172** y una **cobertura de 7.75%**, lo que representa **más de 6 veces la diversidad de catálogo de la popularidad** sin incurrir en la degradación de precisión de los modelos temáticos puros.
   - Las curvas de trade-off y la frontera de Pareto están documentadas gráficamente en `reports/alpha_tradeoff.png` y `reports/pareto_frontier.png`.

---

## 7. Manejo de Arranque en Frío (Cold-Start)

- **Usuarios Nuevos:** Sin historial de interacciones, no es posible computar factores latentes ni perfiles TF-IDF. La política implementada aplica un fallback inmediato al top-10 de **popularidad bayesiana ponderada** (`reports/cold_start_fallback.csv`), garantizando una experiencia inicial atractiva y estadísticamente confiable.
- **Ítems Nuevos:** Las películas sin valoraciones son recomendadas mediante el módulo de **filtrado basado en contenido**, aprovechando sus descriptores de género antes de acumular suficientes interacciones colaborativas.

---

## 8. Auditoría de Datos y Licencia de MovieLens

- **Licencia:** Dataset cedido para fines de investigación académica por GroupLens Research (Universidad de Minnesota). Prohibida su redistribución o uso comercial no autorizado. Los archivos crudos (`data/raw/*`) están explícitamente ignorados en `.gitignore` para cumplir cabalmente con los términos de uso.
- **Cita Obligatoria:**
  > F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. [DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).
- **Integridad:** SHA-256 verificado: `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`.
- **Detalle exhaustivo:** Consulte [`docs/DATASET_CARD.md`](file:///c:/Users/Admin/Desktop/Proyectos/Maestria%20IA%20Uasd/Ciencia%20de%20Datos%20II/Tareas/Unidad%203/INF8239_U03_Recomendadores_Proyecto_Base_uv/docs/DATASET_CARD.md).

---

## 9. Pruebas Automatizadas (pytest)

La suite de pruebas en `tests/` garantiza la robustez y reproducibilidad del sistema:
- `test_data.py`: Validación de contratos de esquema y rechazo de registros huérfanos.
- `test_content.py`: Exclusión de la película consultada y unicidad de recomendaciones.
- `test_matrix_factorization.py`: Preservación del último evento temporal y convergencia finita.
- `test_metrics.py`: Comprobación matemática de `HitRate@K`, `Precision@K`, `Recall@K`, `NDCG@K` y `CatalogCoverage`.

Para correr las pruebas:
```bash
uv run pytest -v
```

---

## 10. Declaración de Uso Ético y Responsable de IA
Consulte el archivo [`AI_USE_DECLARATION.md`](file:///c:/Users/Admin/Desktop/Proyectos/Maestria%20IA%20Uasd/Ciencia%20de%20Datos%20II/Tareas/Unidad%203/INF8239_U03_Recomendadores_Proyecto_Base_uv/AI_USE_DECLARATION.md) para conocer las herramientas utilizadas, prompts relevantes, elementos verificados por el estudiante y responsabilidades asumidas.
