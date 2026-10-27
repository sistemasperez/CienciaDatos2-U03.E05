# System Card · Sistema Recomendador Cinematográfico Híbrido

## Usuarios y propósito
- **Audiencia objetivo:** Usuarios de plataformas audiovisuales y streaming que buscan descubrir películas alineadas con sus preferencias históricas y atributos temáticos de su interés.
- **Propósito del sistema:** Proveer recomendaciones personalizadas de ítems del catálogo ordenadas por relevancia, mitigando la sobrecarga de información y balanceando precisión con exploración de catálogo.

## Catálogo y candidatos
- **Catálogo base:** 9,742 películas indexadas en MovieLens (`ml-latest-small`), de las cuales 9,724 cuentan con interacción registrada en entrenamiento.
- **Generación de candidatos:** Para cada usuario, se excluyen los ítems previamente consumidos/calificados (`seen`). A partir del modelo de Factorización Matricial, se preseleccionan los mejores 100 ítems candidatos (`top_n=100`) para ser reordenados por el modelo híbrido.

## Señales utilizadas
- **Explícitas:** Calificaciones en escala discreta de 0.5 a 5.0 estrellas emitidas por 610 usuarios.
- **Temporales:** Marcas de tiempo (timestamps UNIX) utilizadas estrictamente para asegurar un protocolo de evaluación causal (`temporal_leave_one_out`), evitando fugas de información futura (*temporal data leakage*).
- **Semánticas / Contenido:** Metadatos de género por película vectorizados mediante TF-IDF con ponderación bayesiana centrada en las calificaciones positivas del usuario ($\text{rating} - 2.5$).

## Modelos y fallback
1. **Popularidad Ponderada (Baseline & Cold-Start Fallback):**
   - Utiliza la fórmula tipo IMDb / Bayesiana:
     $$\text{Score} = \frac{v}{v + m} \cdot R + \frac{m}{v + m} \cdot C$$
     donde $v$ es el conteo de votos del ítem, $m$ es el umbral de cuantil mínimo (percentil 80), $R$ es el promedio del ítem y $C$ es la media global del dataset ($3.50$).
2. **Filtrado Basado en Contenido (TF-IDF + Coseno):**
   - Perfil de usuario generado como combinación convexa ponderada de los vectores TF-IDF de las películas consumidas. Puntuación mediante similitud coseno en $[0, 1]$.
3. **Factorización Matricial (SVD / MF mediante SGD):**
   - Descomposición en $K=20$ factores latentes optimizados mediante descenso de gradiente estocástico regularizado ($L_2$ con $\lambda=0.05$, $\eta=0.01$, 12 épocas). Predicción del rating centrada en la media global.
4. **Ensamble Híbrido Lineal:**
   - Normalización min-max de puntuaciones colaborativas y de contenido:
     $$\text{Score}_{\text{híbrido}} = \alpha \cdot \text{Score}_{\text{colab\_norm}} + (1 - \alpha) \cdot \text{Score}_{\text{contenido\_norm}}$$
   - Decisión óptima en frontera de Pareto: $\alpha = 0.75$ a $0.90$.
5. **Estrategia de Fallback:** Ante usuarios fríos (sin historial) o ítems fríos (sin interacciones previas), el sistema transiciona a la lista curada de popularidad ponderada (`reports/cold_start_fallback.csv`).

## Métricas offline
Evaluación mediante partición temporal *leave-one-out* sobre 587 usuarios evaluables:
- **Calidad de Predicción de Rating:** $\text{RMSE} = 1.0264$ estrellas en test.
- **Calidad de Ranking:** $\text{NDCG@10} = 0.0172$, $\text{HitRate@10} = 0.0375$, $\text{Precision@10} = 0.0037$, $\text{Recall@10} = 0.0375$ (para $\alpha=0.75$).
- **Cobertura de Catálogo:** $7.75\%$ del catálogo activo recomendado al menos una vez en los top-10 de los usuarios.

## Cold start, cobertura y diversidad
- **Usuarios Fríos:** Imposibilidad de construir perfiles latentes o vectores TF-IDF. Se suministra el top-10 de popularidad ponderada con garantía de calidad y fiabilidad estadística.
- **Ítems Fríos:** Nuevos estrenos sin calificaciones. El recomendador de contenido basado exclusivamente en géneros permite sugerirlos inmediatamente sin requerir interacciones colaborativas previas.
- **Compromiso de Pareto:** El modelo puramente colaborativo o de popularidad colapsa en un conjunto reducido de ítems superventas (cobertura $<1.5\%$). El híbrido eleva la cobertura a $>7.5\%$ sin degradar el NDCG, diversificando la experiencia del usuario.

## Riesgos y monitoreo
- **Burbujas de filtro / Sobre-especialización:** El filtrado por contenido puro tiende a recomendar títulos con géneros idénticos; se monitorea mediante la métrica de cobertura y entropía de catálogo.
- **Sesgo de popularidad ("Rich-get-richer"):** La popularidad pura margina el catálogo *long-tail*. El componente de factorización matricial e hibridación redistribuye recomendaciones a través del catálogo.
- **Deriva temporal (*Concept Drift*):** Los gustos del usuario y las tendencias cinematográficas evolucionan; se requiere reentrenamiento periódico de los factores latentes y recalibración de la partición temporal.
