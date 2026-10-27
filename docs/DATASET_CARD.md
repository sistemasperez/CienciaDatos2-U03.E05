# Dataset Card · MovieLens (ml-latest-small)

## Fuente, fecha, versión y hash
- **Fuente oficial:** GroupLens Research, Department of Computer Science and Engineering, University of Minnesota ([grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/)).
- **URL de descarga:** `https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`
- **Fecha de generación del dataset:** 26 de septiembre de 2018.
- **Rango temporal de las interacciones:** 29 de marzo de 1996 a 24 de septiembre de 2018 (`828124615` a `1537799250` en UNIX timestamp).
- **Versión:** `ml-latest-small` (versión de desarrollo).
- **Checksum SHA-256:** `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`

## Licencia y cita
- **Términos de uso:** La Universidad de Minnesota y GroupLens autorizan el uso de estos datos exclusivamente para propósitos de investigación académica y educación bajo las siguientes condiciones:
  1. No declarar ni sugerir respaldo de la Universidad de Minnesota o GroupLens.
  2. Citar formalmente la publicación de referencia en cualquier publicación derivada.
  3. No redistribuir el dataset sin incluir los mismos términos de licencia (en este repositorio los datos crudos están en `.gitignore` para cumplir estrictamente con las políticas de no redistribución indebida).
  4. Prohibido su uso comercial o lucrativo sin autorización previa expresa de GroupLens.
- **Cita formal obligatoria:**
  > F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. [DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

## Archivos y diccionario
El archivo comprimido contiene los siguientes ficheros CSV codificados en UTF-8:

1. **`ratings.csv` (100,836 registros):**
   - `userId` (int): Identificador anónimo del usuario (610 usuarios únicos).
   - `movieId` (int): Identificador de la película (9,724 películas valoradas).
   - `rating` (float): Calificación en escala de 0.5 a 5.0 estrellas (incrementos de 0.5).
   - `timestamp` (int): Segundos transcurridos desde epoch UTC (1 de enero de 1970).

2. **`movies.csv` (9,742 registros):**
   - `movieId` (int): Identificador de la película correspondiente al catálogo de MovieLens.
   - `title` (str): Título de la película, incluyendo frecuentemente el año de lanzamiento entre paréntesis.
   - `genres` (str): Lista de géneros asociados separados por pleca (`|`) tales como *Action*, *Adventure*, *Animation*, *Children*, *Comedy*, *Crime*, *Documentary*, *Drama*, *Fantasy*, *Film-Noir*, *Horror*, *Musical*, *Mystery*, *Romance*, *Sci-Fi*, *Thriller*, *War*, *Western*, o `(no genres listed)`.

3. **`links.csv` y `tags.csv`:**
   - Enlaces a bases de datos externas (IMDb e TMDb) y etiquetas de texto libre generadas por usuarios.

## Procedimiento de descarga
La descarga y validación se realiza mediante el script automatizado reproducible:
```bash
uv run python scripts/download_data.py
```
El script descarga en streaming el archivo `.zip`, calcula su resumen criptográfico SHA-256 en memoria, valida rutas seguras para evitar path traversal y extrae los archivos en el directorio local `data/raw/ml-latest-small/`.

## Población observada y exclusiones
- **Usuarios:** 610 usuarios seleccionados aleatoriamente que cumplieron el criterio de inclusión de haber calificado al menos 20 películas.
- **Anonimización:** No se incluye información demográfica, geográfica ni nombres de usuario reales.
- **Esparcidad:** Con 100,836 calificaciones entre 610 usuarios y 9,724 ítems valorados (de 9,742 totales en catálogo), la densidad de la matriz de interacción es de **1.697%** (esparcidad de 98.303%).

## Calidad, sesgos y usos prohibidos
- **Sesgo de selección / popularidad:** Un grupo reducido de películas populares concentra una fracción sustancial de los votos ("long-tail distribution"), mientras que muchas películas de nicho tienen muy pocas valoraciones.
- **Sesgo de positividad:** La media de calificación es 3.50 (mediana 3.50, percentil 75 en 4.0), lo que refleja que los usuarios tienden a calificar películas que eligen ver y les agradan.
- **Data Leakage y corte temporal:** Debido a la naturaleza dinámica de los gustos, la evaluación requiere división cronológica (`temporal_leave_one_out`), evitando particiones aleatorias que filtren información del futuro.
- **Usos prohibidos:** Re-identificación de usuarios, entrenamiento de modelos con fines comerciales directos sin licencia corporativa, o uso en sistemas de producción médica/legal sin garantías de fiabilidad.
