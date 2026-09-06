# Taller 1 — KNN y Métricas de Clasificación
**Aprendizaje Automático 1 — 2026-2**
**Profesor:** Daniel Rambaut · **Modalidad:** Individual
**Fecha límite:** Sábado 12 de septiembre de 2026, 11:59 p. m.

**Estudiante:** _____________________________

---

## Parte I — Componente teórico-matemático

### Ejercicio 1 — Fundamento probabilístico de KNN

Sea un conjunto de entrenamiento $D = \{(x_i, y_i)\}_{i=1}^{n}$, con $x_i \in \mathbb{R}^d$, $y_i \in \{0,1\}$.

Para una nueva observación $x$, sea $N_k(x)$ el conjunto de sus $k$ vecinos más cercanos. Se define

$$\hat{p}_k(x) = \frac{1}{k}\sum_{x_i \in N_k(x)} y_i$$

**a.** Demuestre que $\hat{p}_k(x)$ corresponde a la proporción de vecinos pertenecientes a la clase positiva.

> _Respuesta:_
>
>

**b.** Demuestre que, para un problema binario, KNN asigna la clase 1 cuando $\hat{p}_k(x) > \tfrac{1}{2}$.

> _Respuesta:_
>
>

**c.** Suponga $k=5$ y etiquetas de los vecinos $\{1,0,1,1,0\}$. Calcule $\hat{p}_k(x)$ y determine la predicción.

> _Respuesta:_
>
>

**d.** Repita el procedimiento para $k=7$ y etiquetas $\{0,0,1,0,1,1,0\}$.

> _Respuesta:_
>
>

**e.** Muestre matemáticamente por qué un valor impar de $k$ evita empates en un problema de clasificación binaria.

> _Respuesta:_
>
>

---

### Ejercicio 2 — Distancia euclidiana vs. Manhattan

Sean $A=(a_1,a_2)$ y $B=(b_1,b_2)$:

$$d_E(A,B) = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2}, \qquad d_M(A,B) = |a_1-b_1| + |a_2-b_2|$$

Considere $A=(2,3)$, $B=(7,7)$, $C=(5,2)$.

**a.** Calcule $d_E(A,B)$.

> _Respuesta:_
>
>

**b.** Calcule $d_M(A,B)$.

> _Respuesta:_
>
>

**c.** Calcule $d_E(A,C)$ y $d_M(A,C)$.

> _Respuesta:_
>
>

**d.** Determine cuál de los puntos $B$ o $C$ es el vecino más cercano de $A$ según cada métrica.

> _Respuesta:_
>
>

**e.** Demuestre que $d_E(A,B) \le d_M(A,B)$, partiendo de

$$(|a_1-b_1| + |a_2-b_2|)^2 \ge (a_1-b_1)^2 + (a_2-b_2)^2$$

> _Respuesta:_
>
>

---

### Ejercicio 3 — Efecto de la escala sobre la distancia

Considere $A=(20,1000)$, $B=(25,1200)$, $C=(40,1010)$.

**a.** Calcule las distancias euclidianas $d(A,B)$ y $d(A,C)$.

> _Respuesta:_
>
>

**b.** Determine cuál punto es el vecino más cercano a $A$.

> _Respuesta:_
>
>

**c.** Ahora la segunda característica se divide entre 100: $A'=(20,10)$, $B'=(25,12)$, $C'=(40,10.1)$. Calcule $d(A',B')$ y $d(A',C')$.

> _Respuesta:_
>
>

**d.** Determine si cambió el vecino más cercano.

> _Respuesta:_
>
>

**e.** A partir de $d(x,z) = \sqrt{\sum_{j=1}^{d}(x_j - z_j)^2}$, muestre por qué una variable con escala considerablemente mayor puede dominar la distancia total.

> _Respuesta:_
>
>

---

### Ejercicio 4 — Matriz de confusión y métricas básicas

|            | $\hat{Y}=0$ | $\hat{Y}=1$ |
|------------|:-----------:|:-----------:|
| **$Y=0$**  |     850     |     100     |
| **$Y=1$**  |      20     |      30     |

**a.** Identifique $TP, TN, FP, FN$.

> _Respuesta:_
>
>

**b.** Calcule el Accuracy: $\text{Accuracy} = \dfrac{TP+TN}{TP+TN+FP+FN}$

> _Respuesta:_
>
>

**c.** Calcule el Precision: $\text{Precision} = \dfrac{TP}{TP+FP}$

> _Respuesta:_
>
>

**d.** Calcule el Recall: $\text{Recall} = \dfrac{TP}{TP+FN}$

> _Respuesta:_
>
>

**e.** Calcule el F1: $F1 = 2\cdot\dfrac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$

> _Respuesta:_
>
>

**f.** Demuestre algebraicamente que $F1 = \dfrac{2TP}{2TP+FP+FN}$.

> _Respuesta:_
>
>

---

### Ejercicio 5 — Comparación de dos modelos y costo de clasificación

**Modelo A**

|            | $\hat{Y}=0$ | $\hat{Y}=1$ |
|------------|:-----------:|:-----------:|
| **$Y=0$**  |     900     |      50     |
| **$Y=1$**  |      30     |      20     |

**Modelo B**

|            | $\hat{Y}=0$ | $\hat{Y}=1$ |
|------------|:-----------:|:-----------:|
| **$Y=0$**  |     850     |     100     |
| **$Y=1$**  |      10     |      40     |

**a.** Calcule, para cada modelo: Accuracy, Precision, Recall, F1.

> _Respuesta:_
>
>

**b.** Cada FP cuesta 1 unidad y cada FN cuesta 5 unidades. Calcule $L = FP + 5FN$ para cada modelo.

> _Respuesta:_
>
>

**c.** Determine cuál modelo tiene mayor Accuracy.

> _Respuesta:_
>
>

**d.** Determine cuál modelo tiene mayor Recall.

> _Respuesta:_
>
>

**e.** Determine cuál modelo tiene menor costo $L$.

> _Respuesta:_
>
>

**f.** Muestre que el modelo con mayor Accuracy no necesariamente coincide con el que minimiza el costo.

> _Respuesta:_
>
>

---

## Parte II — Componente práctico-computacional
*(Python · dataset **Give Me Some Credit**)*

### Ejercicio 6 — Exploración inicial y preparación de datos

**a.** Cargar el dataset con pandas.
**b.** Identificar la variable objetivo `SeriousDlqin2yrs`.
**c.** Determinar número de observaciones, número de variables, tipos de datos, valores faltantes por variable.
**d.** Calcular cantidad y porcentaje de observaciones por clase de la variable objetivo.
**e.** Graficar la distribución de la variable objetivo.
**f.** Tratar valores faltantes (por ejemplo, con la mediana).
**g.** Separar en 80% entrenamiento / 20% prueba, partición estratificada, `random_state=42`. Presentar tabla con la distribución de clases en train y test.

> _Código y respuesta:_
>
> ```python
>
> ```

---

### Ejercicio 7 — KNN con k=5: sin escalar vs. estandarizado

Entrenar KNN con $k=5$ sin escalamiento y luego con `StandardScaler`. Calcular Accuracy, Precision, Recall, F1 y graficar la matriz de confusión de cada modelo. Construir tabla comparativa:

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Sin escalamiento | | | | |
| Estandarizado | | | | |

**a.** ¿Cambió el desempeño tras estandarizar?
**b.** ¿Qué métrica presentó el cambio más importante?
**c.** Elija un modelo y justifique.

> _Código y respuesta:_
>
> ```python
>
> ```

---

### Ejercicio 8 — Efecto del número de vecinos $k$

Entrenar KNN (datos estandarizados) para $k \in \{1,3,5,7,9,11,15,21\}$, calculando Accuracy, Precision, Recall, F1 para cada uno; almacenar en un DataFrame y graficar $k$ vs. F1.

**a.** ¿Qué $k$ produce el mayor F1?
**b.** ¿Qué $k$ produce el mayor Recall?
**c.** ¿Qué $k$ produce el mayor Accuracy?
**d.** ¿Coinciden los tres en el mismo modelo?

> _Código y respuesta:_
>
> ```python
>
> ```

---

### Ejercicio 9 — Métrica de distancia: Euclidiana vs. Manhattan

Tomar los tres mejores $k$ del ejercicio 8 y comparar distancia Euclidiana vs. Manhattan para cada uno.

| k | Distancia | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| | Euclidean | | | | |
| | Manhattan | | | | |
| | Euclidean | | | | |
| | Manhattan | | | | |
| | Euclidean | | | | |
| | Manhattan | | | | |

**a.** ¿La métrica de distancia modifica el desempeño?
**b.** ¿Qué combinación de $k$ y distancia da el mayor F1?
**c.** ¿Cuál da el mayor Recall?
**d.** ¿Coinciden la mejor configuración según F1 y según Recall?

> _Código y respuesta:_
>
> ```python
>
> ```

---

### Ejercicio 10 — Análisis final y costo de negocio

Seleccionar el mejor modelo de los ejercicios previos. Identificar $TP, TN, FP, FN$; recalcular Accuracy, Precision, Recall, F1. Con costo $FP=1$, $FN=5$, calcular $L = FP + 5FN$. Comparar contra otros dos modelos:

| Modelo | k | FP | FN | Recall | F1 | Costo |
|---|---|---|---|---|---|---|
| Modelo 1 | | | | | | |
| Modelo 2 | | | | | | |
| Modelo 3 | | | | | | |

**a.** ¿El modelo de mayor Accuracy es también el de menor costo?
**b.** ¿Qué modelo tiene el menor número de falsos negativos?
**c.** ¿Qué modelo recomendaría si el objetivo es identificar clientes de alto riesgo de incumplimiento?
**d.** Justifique la selección final con la matriz de confusión y las métricas.

> _Código y respuesta:_
>
> ```python
>
> ```
