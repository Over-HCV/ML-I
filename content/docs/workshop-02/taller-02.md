# Taller 2 — Modelos de aprendizaje supervisado
**Aprendizaje Automático 1 — 2026-2**
**Profesor:** Daniel Rambaut · **Modalidad:** Grupal
**Fecha límite:** 24 de octubre

**Estudiante(s):** _____________________________

*Regresión lineal simple y múltiple, regresión logística y árboles de decisión.*

---

**Indicaciones generales.** Justifique todas las respuestas. En la teoría, muestre las operaciones; puede usar una calculadora para comprobar resultados. En la práctica, escriba el código y explique las decisiones: una salida de Python por sí sola no constituye una respuesta. Redondee los resultados reportados a cuatro decimales, pero use su precisión completa en los cálculos.

**Guía de lectura.** Los enunciados son originales y toman como guía conceptual a Müller y Guido, *Introduction to Machine Learning with Python* (2017), especialmente los capítulos 2, 3, 5 y 6. No corresponden a ejercicios copiados del libro. Las referencias y las lecturas específicas se encuentran al final del taller.

---

## Parte I. Componente teórico-matemático

### Ejercicio 1 — Regresión lineal simple: ajuste e influencia de una observación

Una empresa estudia la relación entre horas de funcionamiento de un equipo ($x$) y consumo de energía en kWh ($y$). Para este ejercicio se usan los siguientes datos didácticos:

| $x$ | 1 | 2 | 3 | 4 | 5 |
|:---:|:-:|:-:|:-:|:-:|:-:|
| $y$ | 2 | 4 | 5 | 4 | 5 |

Se propone $\hat{y}_i = \beta_0 + \beta_1 x_i$ y se minimiza

$$S(\beta_0, \beta_1) = \sum_{i=1}^{n} \left(y_i - \beta_0 - \beta_1 x_i\right)^2.$$

**a.** Derive $S$ respecto a ambos parámetros e iguale a cero. Obtenga las ecuaciones normales y, a partir de ellas, deduzca

$$\hat{\beta}_1 = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}.$$

Indique qué condición sobre los valores de $x$ garantiza una pendiente única.

**b.** Calcule los coeficientes, las cinco predicciones y los residuos $e_i = y_i - \hat{y}_i$. Compruebe que $\sum_i e_i = 0$ y explique por qué esta propiedad depende de incluir un intercepto en mínimos cuadrados.

**c.** Calcule $MSE = n^{-1}\sum_i e_i^2$ y $R^2 = 1 - \sum_i e_i^2 / \sum_i (y_i - \bar{y})^2$. Compare el error con el predictor constante $\hat{y} = \bar{y}$ e interprete la pendiente con sus unidades.

**d.** Sustituya únicamente el último valor de $y$ por $12$ y vuelva a ajustar el modelo. Compare las pendientes y las predicciones para $x = 6$. Explique cómo la pérdida cuadrática responde a este cambio y por qué predecir para $x = 20$ exigiría una cautela adicional.

---

### Ejercicio 2 — Regresión lineal múltiple: solución matricial y colinealidad

Considere un modelo de demanda $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$. $x_1$ representa inversión publicitaria en unidades de presupuesto y $x_2 \in \{0,1\}$ indica si se realiza una promoción. La demanda $y$ se mide en centenas de unidades. Use los siguientes datos didácticos:

| Observación | $x_1$ | $x_2$ | $y$ |
|:-----------:|:-----:|:-----:|:---:|
| 1 | 0 | 0 | 2 |
| 2 | 1 | 0 | 4 |
| 3 | 2 | 0 | 5 |
| 4 | 0 | 1 | 5 |
| 5 | 1 | 1 | 6 |
| 6 | 2 | 1 | 8 |

**a.** Construya la matriz $X$ incluyendo la columna del intercepto y el vector $y$. Indique sus dimensiones y calcule $X^\top X$ y $X^\top y$.

**b.** Partiendo de $S(\beta) = \lVert y - X\beta \rVert_2^2$, derive $X^\top X \hat{\beta} = X^\top y$. Resuelva el sistema; no es necesario calcular explícitamente la inversa.

**c.** Interprete $\hat{\beta}_1$ y $\hat{\beta}_2$ manteniendo fija la otra variable. Prediga la demanda para $(x_1, x_2) = (1.5, 1)$ y calcule el MSE de entrenamiento. Explique por qué los coeficientes no prueban un efecto causal.

**d.** Se agrega $x_3 = 2x_1$. Demuestre que las nuevas columnas son linealmente dependientes y explique por qué los coeficientes dejan de ser únicos aunque puedan conservarse las predicciones. Proponga una forma de ajustar el modelo sin invertir una matriz singular.

**e.** Justifique por qué añadir una variable a una regresión por mínimos cuadrados no puede aumentar su error mínimo de entrenamiento, pero sí puede aumentar el error en datos nuevos.

---

### Ejercicio 3 — Regresión logística: probabilidad, odds y umbral

Un modelo didáctico estima la probabilidad de falla de una máquina en la próxima semana ($Y = 1$). Sus variables son $x_1$, años de uso, y $x_2$, número de alertas registradas durante el último mes:

$$z = -3 + 0.6x_1 + 1.2x_2, \qquad p = P(Y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}.$$

**a.** Deduzca $\log\left(p/(1 - p)\right) = z$. Calcule el factor por el que se multiplican los odds al aumentar un año de uso, manteniendo constante $x_2$. Distinga ese factor de un cambio en puntos porcentuales.

**b.** Para las máquinas $A = (2,1)$, $B = (5,0)$ y $C = (4,2)$, calcule $z$, $p$ y la clase predicha con umbrales $t = 0.5$ y $t = 0.8$. Use siempre $\hat{y} = 1$ cuando $p \ge t$.

**c.** Deduzca la frontera de decisión en el plano $(x_1, x_2)$ para un umbral general $t \in (0,1)$ y particularice para los dos umbrales anteriores. Explique qué cambia en la frontera y qué permanece igual en las probabilidades al modificar solamente el umbral.

**d.** Suponga que un falso positivo cuesta 2 unidades, un falso negativo cuesta 8 y un acierto cuesta 0. Suponiendo probabilidades bien calibradas, compare los costos esperados de predecir 0 y 1 y deduzca el umbral que minimiza el costo. Aplique ese umbral a las tres máquinas.

**e.** Explique por qué un modelo lineal $\hat{y} = \beta_0 + \beta^\top x$ ajustado por mínimos cuadrados no garantiza predicciones en $[0,1]$, y por qué la función sigmoide sí lo hace.

---

### Ejercicio 4 — Regresión logística: pérdida, gradiente y regularización

Para un único predictor, considere tres observaciones $(x_i, y_i)$: $(-1,0)$, $(0,0)$ y $(1,1)$. Defina

$$p_i = \sigma(\beta_0 + \beta_1 x_i), \qquad J = -\frac{1}{3}\sum_{i=1}^{3}\left[\,y_i \log p_i + (1 - y_i)\log(1 - p_i)\,\right].$$

Use logaritmos naturales y los valores iniciales $\beta_0 = \beta_1 = 0$.

**a.** Calcule las probabilidades y el valor inicial de $J$.

**b.** Utilizando $\sigma'(z) = \sigma(z)\left(1 - \sigma(z)\right)$, deduzca

$$\frac{\partial J}{\partial \beta_0} = \frac{1}{3}\sum_i (p_i - y_i), \qquad \frac{\partial J}{\partial \beta_1} = \frac{1}{3}\sum_i (p_i - y_i)\,x_i.$$

**c.** Realice una actualización simultánea por descenso de gradiente con tasa $\eta = 0.5$: $\beta_j^{\text{nuevo}} = \beta_j - \eta\,\partial J / \partial \beta_j$. Calcule las nuevas probabilidades y la nueva pérdida. ¿Disminuyó $J$?

**d.** Se incorpora $J_\lambda = J + \lambda \beta_1^2 / 2$, sin penalizar el intercepto. Deduzca el nuevo gradiente y evalúelo en el punto obtenido en el literal anterior para $\lambda = 1$. Explique por qué esta penalización no altera el primer paso si se parte de $\beta_1 = 0$.

**e.** Explique la relación cualitativa entre $C$ y la intensidad de regularización en `LogisticRegression`. Discuta por qué cambiar las escalas de entrada puede cambiar un ajuste regularizado. No suponga que $C = 1/\lambda$ reproduce exactamente la normalización de $J_\lambda$.

---

### Ejercicio 5 — Árboles de decisión: división, predicción y complejidad

Un nodo contiene 12 observaciones: 6 de clase 0 y 6 de clase 1. Se consideran dos divisiones candidatas; cada par indica los conteos $(n_0, n_1)$:

| División | Hijo izquierdo | Hijo derecho |
|:--------:|:--------------:|:------------:|
| A | $(5,1)$ | $(1,5)$ |
| B | $(4,0)$ | $(2,6)$ |

**a.** Calcule la impureza Gini del padre y de cada hijo, $G = 1 - p_0^2 - p_1^2$. Compare las reducciones

$$\Delta G = G_{\text{padre}} - \frac{n_L}{n}G_L - \frac{n_R}{n}G_R.$$

Seleccione la mejor división y explique por qué se pondera por el tamaño de los hijos, en lugar de promediar sus impurezas sin pesos.

**b.** Para la división seleccionada, indique la clase mayoritaria y la probabilidad estimada de clase 1 en cada hoja. ¿Cómo cambiarían las clases asignadas a las hojas si se usara el umbral $t = 0.8$?

**c.** Para regresión, considere $x = (1,2,3,4,5,6)$ y $y = (2,3,7,8,9,12)$. Compare las divisiones $x \le 2.5$ y $x \le 4.5$, usando como predicción de cada hoja su media. Calcule la suma de errores cuadrados total de cada división y elija la mejor. Prediga para $x = 7$.

**d.** Demuestre que la media minimiza $\sum_{i \in R}(y_i - c)^2$ en una hoja $R$. Explique por qué el árbol anterior no extrapola una tendencia lineal fuera del intervalo observado.

**e.** Explique cómo `max_depth` y `min_samples_leaf` controlan el sobreajuste y describa qué patrón esperaría en los errores de entrenamiento y validación al aumentar excesivamente la profundidad.

---

## Parte II. Componente práctico-computacional

### Problemas y conjuntos de datos

Se trabajará con dos conjuntos reales incluidos en scikit-learn. Una vez instaladas las dependencias, su carga no requiere cuentas ni descargas externas.

**Caso A: estimar progresión de la diabetes.** Use `load_diabetes`: 442 observaciones, 10 predictores y una medida cuantitativa de progresión un año después de la medición inicial. La variable objetivo es continua, no una etiqueta de diagnóstico. Cargue `scaled=False` para que cualquier transformación aprendida se ajuste exclusivamente con entrenamiento. Se usará `bmi` como único predictor en el modelo simple y los diez predictores en el múltiple.

**Caso B: clasificar muestras de tejido.** Use `load_breast_cancer`: 569 observaciones y 30 características numéricas de imágenes de núcleos celulares. La codificación original es 0 para maligno y 1 para benigno; en este taller se recodifica como 1 = maligno y 0 = benigno. Las métricas de la clase positiva deben corresponder a muestras malignas. El análisis es académico; no constituye una validación para uso clínico.

### Protocolo experimental obligatorio

Cree, para cada caso, una partición fija de aproximadamente 60 % entrenamiento, 20 % validación y 20 % prueba: primero reserve el 20 % de prueba y luego separe el 25 % del 80 % restante para validación. Use `random_state=42` en ambas divisiones. En clasificación, estratifique ambas divisiones con su respectivo vector `y`.

Mantenga exactamente los mismos índices dentro de cada caso para todos los modelos. La exploración que oriente decisiones se hace sobre entrenamiento. Puede reportar los tamaños de las tres particiones.

Ajuste escaladores e imputadores, si fueran necesarios, solamente con entrenamiento y aplique las transformaciones aprendidas a validación y prueba. No incluya la variable objetivo entre los predictores.

> **Nota.** En los ejercicios 6 a 9, ajuste con entrenamiento y tome decisiones con validación. Reserve prueba para el ejercicio 10. En este taller conserve los modelos ya ajustados: no los reentrene con entrenamiento más validación, para mantener fijos los modelos y umbrales evaluados.
>
> Registre las versiones de Python y de las bibliotecas. Use `random_state=42` en los árboles, conserve los modelos seleccionados y documente los desempates antes de consultar prueba.

### Código inicial de carga y partición

Este bloque solo prepara los datos; los ajustes, búsquedas y análisis deben ser implementados por el estudiante. Requiere scikit-learn 1.1 o posterior por la opción `scaled=False`.

```python
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split

def separar(X, y, clasificacion=False):
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42,
        stratify=y if clasificacion else None,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42,
        stratify=y_dev if clasificacion else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

diabetes = load_diabetes(as_frame=True, scaled=False)
reg = separar(diabetes.data, diabetes.target)
Xr_train, Xr_val, Xr_test, yr_train, yr_val, yr_test = reg

cancer = load_breast_cancer(as_frame=True)
y_maligno = (cancer.target == 0).astype(int)
clf = separar(cancer.data, y_maligno, clasificacion=True)
Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test = clf

def rmse(y, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y, y_pred) ** 0.5
```

### Criterios comunes

En regresión reporte MAE, RMSE y $R^2$. En clasificación reporte precisión, recall, F1, ROC-AUC y matriz de confusión con orden de etiquetas $[0,1]$. Para ROC-AUC use probabilidades de la clase 1, no etiquetas. Para precisión y F1 use `zero_division=0` si es necesario y explique el caso. Las curvas, las tablas y las interpretaciones deben indicar si corresponden a entrenamiento, validación o prueba.

---

### Ejercicio 6 — Caso A: regresión lineal simple y modelo de referencia

**a.** Cargue y separe los datos con el protocolo anterior. Sobre entrenamiento, revise tipos, faltantes y rangos; presente un gráfico de `bmi` contra la variable objetivo. Describa si una recta parece una aproximación razonable. No elimine observaciones únicamente porque parezcan atípicas.

**b.** Usando solo `bmi`, implemente con NumPy las expresiones de $\hat{\beta}_0$ y $\hat{\beta}_1$ deducidas en el ejercicio 1. Compare coeficientes y predicciones de validación con `LinearRegression`; reporte la diferencia absoluta máxima entre las predicciones y compruebe concordancia numérica con `np.allclose`.

**c.** Ajuste `DummyRegressor(strategy="mean")` con entrenamiento. Compare este predictor y la regresión simple mediante MAE, RMSE y $R^2$ en entrenamiento y validación. Interprete la mejora frente a predecir una constante, incluyendo el caso en que alguna métrica no mejore.

**d.** Grafique los residuos de validación frente a los valores predichos, con una línea horizontal en cero. Identifique patrones de curvatura, dispersión variable o errores extremos y explique sus implicaciones para el modelo.

**e.** Reporte el intercepto, la pendiente y su interpretación en las unidades de los datos cargados. Explique por qué una asociación con `bmi` no identifica por sí sola el efecto de intervenir sobre esa variable.

---

### Ejercicio 7 — Caso A: regresión múltiple y árbol de regresión

**a.** Con los diez predictores, construya un `Pipeline` con `StandardScaler` y `LinearRegression`. Compare sus métricas de entrenamiento y validación con el modelo simple y el predictor constante. No interprete una mejora de entrenamiento como evidencia suficiente de mejora en generalización.

**b.** Presente los coeficientes del modelo múltiple ordenados por magnitud absoluta. Interprete dos de ellos en términos de un aumento de una desviación estándar de entrenamiento. Examine la matriz de correlaciones de entrenamiento y explique cómo los predictores correlacionados dificultan interpretar los coeficientes como importancias independientes.

**c.** Ajuste nueve `DecisionTreeRegressor` sobre los predictores originales, usando `criterion="squared_error"`, `max_depth` $\in \{2,4,6\}$ y `min_samples_leaf` $\in \{5,15,30\}$. Registre RMSE de entrenamiento y validación, profundidad real y número de hojas. Seleccione el árbol con menor RMSE de validación; en empate, prefiera menor profundidad máxima y después mayor mínimo de muestras por hoja.

**d.** Compare regresión simple, regresión múltiple y árbol seleccionado. Elija un candidato final de regresión según RMSE de validación y registre la decisión. En empate numérico, prefiera el modelo simple, luego el múltiple y después el árbol. Mantenga el predictor constante como referencia.

**e.** Explique por qué el árbol puede representar relaciones no lineales sin estandarización y por qué su flexibilidad no garantiza superar al modelo múltiple. Sustente la respuesta con la brecha entre entrenamiento y validación.

---

### Ejercicio 8 — Caso B: regresión logística y elección del umbral

**a.** Prepare las particiones estratificadas y compruebe la recodificación de la clase positiva. Reporte la proporción de malignos en entrenamiento y validación. Ajuste un `DummyClassifier(strategy="prior")` como referencia; use sus probabilidades para calcular ROC-AUC.

**b.** Construya pipelines con `StandardScaler` y `LogisticRegression` con `solver="lbfgs"` y `max_iter=5000`, conservando la regularización L2 predeterminada. Explore $C \in \{0.01, 0.1, 1, 10\}$. Registre ROC-AUC y F1 de entrenamiento y validación con umbral $0.5$. Compruebe la convergencia y seleccione el $C$ con mayor ROC-AUC de validación; en empate, prefiera el menor $C$.

**c.** Conserve fijo ese modelo y evalúe en validación $t \in \{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8\}$, usando $\hat{y} = \mathbb{1}[p \ge t]$. Tabule precisión, recall, F1, FP, FN y costo $L = FP + 5FN$. Este costo es un supuesto didáctico, no una estimación clínica. Seleccione el umbral con menor $L$; en empate, mayor recall y después menor umbral. Compare con $t = 0.5$.

**d.** Grafique precisión y recall en función del umbral y explique por qué ROC-AUC del modelo permanece igual al cambiar únicamente $t$. Distinga el uso de ROC-AUC para elegir $C$ del uso del costo para fijar el punto de operación.

**e.** Interprete el signo de tres coeficientes estandarizados respecto a la clase maligna. Para uno calcule $e^{\beta_j}$ e interprete el cambio en odds manteniendo fijas las demás variables, sin atribuir causalidad. Conserve el pipeline y el umbral elegidos.

---

### Ejercicio 9 — Caso B: árbol de clasificación e interpretación

**a.** Sobre los mismos predictores originales y particiones, entrene `DecisionTreeClassifier` con `criterion="gini"` y `random_state=42` para `max_depth` $\in \{2,3,5,\text{None}\}$ y `min_samples_leaf` $\in \{1,5,15\}$: doce combinaciones. No estandarice los predictores.

**b.** Registre ROC-AUC y F1 en entrenamiento y validación, profundidad real y número de hojas. Para F1, use $\mathbb{1}[p \ge 0.5]$; esta regla explícita evita ambigüedades de desempate de `predict`. Seleccione por mayor ROC-AUC de validación; en empate, menor profundidad real, menos hojas y, finalmente, el primer modelo en el orden de la cuadrícula indicada.

**c.** Para el árbol seleccionado, explore los mismos siete umbrales del ejercicio 8 y seleccione por $L = FP + 5FN$, con los mismos desempates. Compare su costo de validación con el de la regresión logística seleccionada. Elija y registre el candidato final de clasificación por menor costo; en empate, mayor ROC-AUC y después regresión logística.

**d.** Dibuje los primeros tres niveles del árbol con `plot_tree`, mostrando nombres de variables y clases; aclare que la visualización está truncada si el árbol es más profundo. Extraiga una ruta completa hasta una hoja, escriba su regla con condiciones `if`/`else` e indique su número de muestras y proporción de malignos de entrenamiento.

**e.** Grafique las cinco mayores `feature_importances_`. Compare con los coeficientes del modelo logístico y explique por qué sus escalas no son directamente comparables ni implican efectos causales. Use las métricas registradas para discutir si hay evidencia de sobreajuste en algún árbol.

---

### Ejercicio 10 — Evaluación final y recomendación sustentada

Antes de abrir prueba, registre los dos candidatos finales, sus predictores, hiperparámetros y, para clasificación, sus umbrales. Conserve esos ajustes. Realice una única evaluación final de los modelos predefinidos a continuación.

**a. Regresión.** Evalúe en prueba el predictor constante, la regresión simple, la regresión múltiple y el árbol seleccionado. Construya una tabla con MAE, RMSE y $R^2$ e identifique el candidato que había elegido con validación. Compare su RMSE de validación y prueba sin cambiar de candidato por los resultados observados.

**b. Clasificación.** Evalúe en prueba el clasificador de referencia, la regresión logística y el árbol seleccionado. Use los umbrales congelados para los dos últimos y $t = 0.5$ para la referencia. Reporte umbral, precisión, recall, F1, ROC-AUC, FP, FN y $L = FP + 5FN$; puede dividir esta información en dos tablas para facilitar la lectura.

**c.** Dibuje las matrices de confusión de los dos modelos de clasificación, con filas reales y columnas predichas en orden $[0,1]$. Para el candidato elegido, explique con palabras qué significan FP y FN en este conjunto y verifique a mano el costo a partir de la matriz.

**d.** Redacte una recomendación de 250 a 350 palabras que cubra ambos casos. Relacione desempeño, interpretabilidad y complejidad. Explique qué haría si un candidato rindiera peor en prueba que en validación, sin reutilizar prueba para ajustarlo. Incluya dos limitaciones de los datos o del diseño experimental y una mejora futura, por ejemplo validación cruzada dentro del conjunto de desarrollo o evaluación externa.

---

## Formato de entrega y evaluación

Un PDF con el desarrollo de los ejercicios teóricos 1 a 5.

Un `notebook.ipynb` con secciones 6 a 10, código ejecutado, tablas, gráficas y respuestas interpretativas. Incluya su nombre y las versiones de las dependencias; el notebook debe ejecutarse de principio a fin.

Identifique las particiones en todas las comparaciones. Las gráficas deben tener título, ejes y leyenda cuando corresponda. Reporte las decisiones previas a la evaluación final; no entregue solamente el mejor resultado.

Cada ejercicio vale 10 puntos, para un total de 100. En los ejercicios teóricos se evalúan procedimiento (6), resultados (2) e interpretación (2). En los prácticos se evalúan implementación y protocolo experimental (4), evidencia en tablas o gráficas (3) y análisis (3). Fecha límite de entrega: por definir.

---

## Referencias y lecturas sugeridas

Müller, A. C. y Guido, S. (2017). *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media. PDF de referencia compartido.

- **Capítulo 2:** modelos lineales, pp. 45–67; árboles de decisión, pp. 70–82. Apoyan los ejercicios 1–5 y 6–9.
- **Capítulo 3:** preprocesamiento y escalamiento, pp. 132–139. **Capítulo 5:** selección de modelos y métricas, pp. 251–302. **Capítulo 6:** cadenas de algoritmos y pipelines, pp. 305–320. Apoyan el protocolo de los ejercicios 6–10.

Documentación oficial de scikit-learn: `load_diabetes`, `load_breast_cancer` y descripción de los conjuntos de datos.

Repositorio de los autores: material complementario del libro. Consulte la documentación de su versión de scikit-learn cuando una llamada del libro use una API antigua.
