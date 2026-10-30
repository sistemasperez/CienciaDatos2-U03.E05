# INF-8239 · Ejercicio 05: Motor Recomendador Reproducible

**Maestría en Inteligencia Artificial · Universidad Autónoma de Santo Domingo (UASD)**  
**Asignatura:** Ciencia de Datos II  
**Docente:** Edwin Ramón José Nolasco  
**Estudiante:** Juan Aníbal Pérez  
**Repositorio GitHub:** [https://github.com/sistemasperez/CienciaDatos2-U03.E05](https://github.com/sistemasperez/CienciaDatos2-U03.E05)  

---

## 1. Tabla Principal de Resultados Comparativos

Para comparar de forma justa cada enfoque, evalué los modelos bajo el protocolo de corte temporal (`temporal_leave_one_out`) sobre los 587 usuarios evaluables de MovieLens (`ml-latest-small`). El objetivo fue predecir si la última película que el usuario realmente vio en el tiempo aparece dentro de sus 10 recomendaciones principales (Top-10):

| Modelo / Enfoque | Parámetro $\alpha$ | NDCG@10 | HitRate@10 | Precision@10 | Recall@10 | Cobertura Catálogo | RMSE (Rating) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Popularidad Ponderada (Baseline)** | - | 0.0197 | 0.0426 | 0.0043 | 0.0426 | 1.22% | - |
| **Contenido Puro (TF-IDF de Géneros)** | 0.00 | 0.0108 | 0.0256 | 0.0026 | 0.0256 | 11.71% | - |
| **Híbrido ($\alpha=0.10$)** | 0.10 | 0.0105 | 0.0239 | 0.0024 | 0.0239 | 11.15% | 1.0264 |
| **Híbrido ($\alpha=0.25$)** | 0.25 | 0.0127 | 0.0324 | 0.0032 | 0.0324 | 10.65% | 1.0264 |
| **Híbrido ($\alpha=0.50$)** | 0.50 | 0.0144 | 0.0307 | 0.0031 | 0.0307 | 8.87% | 1.0264 |
| **Híbrido ($\alpha=0.75$) [Mi Decisión Pareto]** ⭐ | **0.75** | **0.0172** | **0.0375** | **0.0037** | **0.0375** | **7.75%** | **1.0264** |
| **Híbrido ($\alpha=0.90$)** | 0.90 | 0.0189 | 0.0392 | 0.0039 | 0.0392 | 7.47% | 1.0264 |
| **Colaborativo Puro (MF - SVD)** | 1.00 | 0.0184 | 0.0375 | 0.0037 | 0.0375 | 7.46% | 1.0264 |

---

## 2. Datos, Licencia, Auditoría y Partición Temporal

### 2.1 Conjunto de Datos y Licencia
- Utilicé el dataset oficial de MovieLens (`ml-latest-small`), el cual cuenta con 100,836 calificaciones, 610 usuarios y 9,724 películas registradas.
- **Verificación de integridad:** Validé que el archivo descargado coincida con el hash SHA-256 esperado (`696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`).
- **Respeto a los términos de uso:** Para cumplir con la licencia de GroupLens (Universidad de Minnesota), los datos crudos están excluidos en `.gitignore` (`data/raw/*`) y no se redistribuyen en el repositorio de GitHub. El script `scripts/download_data.py` los descarga directamente desde la fuente oficial.
- **Auditoría:** La matriz usuario-ítem tiene una densidad de apenas **1.697%** (es decir, una esparcidad del **98.303%**), con una calificación promedio global de 3.50 estrellas.

### 2.2 ¿Por qué utilicé Corte Temporal (`temporal_leave_one_out`)?
En sistemas recomendadores, si hacemos una división aleatoria tradicional (*random train/test split*), cometemos un error grave de **fuga de datos (data leakage temporal)**: el modelo usaría calificaciones que el usuario hizo en el futuro para intentar predecir lo que vio en el pasado.
- Para evitar esto, ordené cronológicamente las interacciones de cada usuario por su marca de tiempo (`timestamp`).
- Dejé en `test` exactamente la última interacción en el tiempo de cada usuario, y usé todo su historial previo para entrenar (`train`: 100,226 filas).
- De esta forma, evalué al sistema simulando la vida real: ¿es capaz el recomendador de predecir la próxima película que verá el usuario a continuación?

---

## 3. Explicación de Baselines, Errores y Límites Encontrados

Al probar cada modelo por separado, identifiqué claramente sus fortalezas y limitaciones:

1. **Popularidad Ponderada (Fórmula tipo IMDb):**
   - **Qué hace:** Ordena los ítems combinando el número de votos y la calificación promedio para evitar que películas con 1 solo voto de 5 estrellas lideren el ranking.
   - **Lo que encontré:** Obtiene el NDCG más alto de la tabla (0.0197) porque casi todos los usuarios tienden a ver los grandes éxitos (*blockbusters*).
   - **Su límite crítico:** Es un espejismo de precisión. **Solo recomienda el 1.22% de todo el catálogo**. Le muestra exactamente las mismas 10 películas a todo el mundo, ignorando los gustos individuales y matando por completo la personalización y el descubrimiento de películas menos conocidas (*long tail*).

2. **Filtrado Basado en Contenido (TF-IDF de Géneros + Similitud Coseno):**
   - **Qué hace:** Crea un vector temático por película a partir de sus géneros y calcula la similitud con las películas que al usuario le gustaron previamente.
   - **Lo que encontré:** Logra la mayor cobertura de catálogo de todos los modelos (**11.71%**), permitiendo recomendar una variedad mucho más amplia.
   - **Su límite crítico:** Tiene la menor precisión (NDCG = 0.0108). Sufre de **sobre-especialización temática**: si a un usuario le gustó una película animada, el modelo tiende a recomendarle solo películas idénticas de animación, perdiendo relevancia y capacidad de sorpresa.

3. **Factorización Matricial (SVD Latente con Descenso de Gradiente):**
   - **Qué hace:** Descompone la matriz esparsa en 20 factores latentes para usuarios e ítems, entrenado durante 12 épocas con regularización $L_2 = 0.05$.
   - **Error numérico:** Alcanza un error cuadrático medio de **RMSE = 1.0264 estrellas** al predecir calificaciones en test.
   - **Su límite:** Depende totalmente de las interacciones previas. Si llega un usuario o una película sin calificaciones, el modelo no tiene factores calculados y no puede generar recomendaciones (*problema de arranque en frío*).

---

## 4. Hibridación y Decisión en la Frontera de Pareto

Para unir lo mejor de ambos mundos, construí un recomendador híbrido que combina linealmente los puntajes normalizados del modelo colaborativo y del modelo de contenido:

$$\text{Score}_{\text{híbrido}} = \alpha \cdot \text{Score}_{\text{colaborativo}} + (1 - \alpha) \cdot \text{Score}_{\text{contenido}}$$

Donde $\alpha \in [0.0, 1.0]$ me permite calibrar el peso que le doy a cada señal.

### Mi Decisión Final: $\alpha = 0.75$ (Punto Óptimo de Pareto)
Al analizar la curva de compromiso (*trade-off*) entre precisión de ranking y cobertura de catálogo, seleccioné **$\alpha = 0.75$** como la mejor solución de ingeniería:
- **NDCG@10 = 0.0172:** Mantiene una precisión de ranking muy cercana a la del colaborativo puro (0.0184) y a la popularidad.
- **Cobertura de Catálogo = 7.75%:** Ofrece **más de 6 veces la diversidad de catálogo de la popularidad pura** ($1.22\%$).
- **Conclusión de la decisión:** Esta configuración protege al sistema contra el aburrimiento del usuario (evita recomendar siempre los mismos títulos obvios), inyecta diversidad guiada por contenido y mantiene un alto nivel de acierto en las recomendaciones.

*(Los gráficos generados de la frontera de Pareto y trade-off se encuentran disponibles en `reports/pareto_frontier.png` y `reports/alpha_tradeoff.png`).*

---

## 5. Estrategia para Arranque en Frío (Cold-Start)

Diseñé e implementé una estrategia clara para cuando el sistema no cuenta con datos suficientes:

- **Para Usuarios Nuevos (Cold-Start de Usuario):** Como un usuario nuevo no tiene historial de calificaciones, no es posible calcular sus factores latentes ni su perfil de contenido. En este caso, el sistema activa un *fallback* automático que le entrega el Top-10 de **popularidad bayesiana ponderada** (`reports/cold_start_fallback.csv`). Esto garantiza que el nuevo usuario reciba títulos reconocidos y de alta calidad mientras construye su historial.
- **Para Ítems Nuevos (Cold-Start de Ítem):** Cuando se agrega una película recién estrenada que aún no tiene valoraciones, el modelo colaborativo no puede recomendarla. El sistema utiliza entonces su perfil de contenido (géneros y TF-IDF) para asociarla a usuarios con afinidad a esa temática, resolviendo el problema antes de que reciba sus primeras notas.

---

## 6. Declaración de Uso Ético y Responsable de Inteligencia Artificial

Cumpliendo con las directrices académicas de la asignatura, detallo el uso de herramientas de asistencia:

- **Herramienta utilizada:** Gemini 3.8 Flash (High) dentro del entorno Antigravity IDE.
- **En qué me apoyé:**
  1. Revisión de sintaxis en el código de vectorización de scikit-learn y compatibilidad con matrices NumPy 2D.
  2. Apoyo en la estructuración de la suite de pruebas unitarias (`pytest`) para validar los contratos de datos y la cobertura de catálogo.
  3. Formulación vectorizada de las funciones de ranking en `metrics.py` (`Precision@K`, `Recall@K`, `NDCG@K`).
- **Verificaciones y correcciones que realicé:**
  - Verifiqué a mano la fórmula del descuento logarítmico en `NDCG@K` para asegurarme de que los descuentos en el denominador usaran $\log_2(\text{rank} + 1)$ y manejaran correctamente el caso binario con leave-one-out.
  - Comprobé que el ordenamiento temporal en `temporal_leave_one_out` no tuviera fugas de información y que $\max(\text{timestamp}_{\text{train}}) \le \text{timestamp}_{\text{test}}$.
  - Corrí la suite completa de pruebas unitarias (`uv run pytest`), asegurando que las **9 pruebas pasaran al 100%**.
  - Verifiqué que el archivo `.gitignore` excluyera los datos de MovieLens para respetar la licencia de GroupLens.
- **Responsabilidad:** Como estudiante, asumo la autoría, verificación y defensa completa del código, los experimentos y las conclusiones presentadas en este informe.
