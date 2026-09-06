# Taller 1 — KNN y Métricas de Clasificación · Guía de Respuestas
**Aprendizaje Automático 1 — 2026-2**

> **Estado:** los 10 ejercicios están desarrollados con procedimiento
> completo y, en la Parte II, con resultados reales obtenidos ejecutando
> Python sobre el dataset *Give Me Some Credit* (150.000 observaciones).
> Falta revisar/pulir la redacción final y trasladar el contenido a LaTeX.

---

## Parte I — Componente teórico-matemático

### Ejercicio 1 — Fundamento probabilístico de KNN

**Planteamiento:** formalizar por qué $\hat{p}_k(x)=\frac{1}{k}\sum_{x_i\in N_k(x)} y_i$ es una proporción, por qué el umbral $\tfrac12$ define la regla de decisión, y por qué $k$ impar evita empates.

**a. $\hat{p}_k(x)$ es una proporción**

Dado que $y_i \in \{0,1\}$, cada etiqueta funciona como una variable indicadora, de forma que $y_i=1$ aporta exactamente una unidad a la suma cuando el vecino pertenece a la clase positiva, mientras que $y_i=0$ no aporta nada; por lo tanto,

$$\sum_{x_i \in N_k(x)} y_i = \#\{x_i \in N_k(x): y_i=1\} = m,$$

es decir, la sumatoria cuenta literalmente cuántos de los $k$ vecinos son positivos, sin que los negativos alteren el resultado. Al dividir entre $k$ se obtiene $\hat p_k(x)=m/k$, que corresponde exactamente a la definición frecuentista de una proporción (casos favorables sobre casos totales), quedando demostrado que $\hat p_k(x)$ es la proporción de vecinos de clase positiva.

**b. El umbral $\tfrac12$ como regla de mayoría**

KNN asigna a $x$ la clase mayoritaria entre sus $k$ vecinos. Si $m$ es el número de vecinos positivos, $k-m$ es el de negativos, y la clase 1 es mayoritaria cuando $m>k-m$, lo cual equivale a $2m>k$, es decir a $m/k>\tfrac12$. Como $\hat p_k(x)=m/k$, esta condición es exactamente $\hat p_k(x)>\tfrac12$: el umbral no se impone arbitrariamente, es la traducción algebraica de "la clase con más votos gana".

**c. Caso $k=5$, etiquetas $\{1,0,1,1,0\}$**

$m=1+0+1+1+0=3$, de forma que $\hat p_k(x)=3/5=0.6>0.5$ → KNN predice la **clase 1**.

**d. Caso $k=7$, etiquetas $\{0,0,1,0,1,1,0\}$**

$m=0+0+1+0+1+1+0=3$, de forma que $\hat p_k(x)=3/7\approx0.4286<0.5$ → KNN predice la **clase 0**.

**e. Por qué $k$ impar evita empates**

Un empate ocurre cuando $m=k-m$, es decir $k=2m$. Esta igualdad exige que $k$ sea par, porque $2m$ es siempre par para cualquier entero $m$; en consecuencia, si $k$ es impar no existe ningún entero $m$ que satisfaga $k=2m$, y el empate resulta matemáticamente imposible. Esto es justo lo que se observó en c y d: al ser $k$ impar en ambos casos, $\hat p_k(x)$ nunca cae exactamente en 0.5, y siempre hay una clase estrictamente mayoritaria.

---

### Ejercicio 2 — Distancia euclidiana vs. Manhattan

**Planteamiento:** comparar $d_E$ y $d_M$ sobre $A=(2,3)$, $B=(7,7)$, $C=(5,2)$, y demostrar la desigualdad general $d_E \le d_M$.

**a.** $d_E(A,B) = \sqrt{(2-7)^2+(3-7)^2} = \sqrt{25+16} = \sqrt{41} \approx 6.4031$

**b.** $d_M(A,B) = |2-7|+|3-7| = 5+4 = 9$

**c.** $d_E(A,C) = \sqrt{(2-5)^2+(3-2)^2} = \sqrt{10} \approx 3.1623$, $\quad d_M(A,C) = |2-5|+|3-2| = 4$

**d. Vecino más cercano de A**

Comparando: $d_E(A,B)\approx6.40$ vs. $d_E(A,C)\approx3.16$, y $d_M(A,B)=9$ vs. $d_M(A,C)=4$; en ambos casos **C** resulta el vecino más cercano — aquí las dos métricas coinciden en su conclusión (esto no siempre ocurre, como se ve en el ejercicio 3).

**e. Demostración $d_E(A,B)\le d_M(A,B)$**

Expandiendo el cuadrado sugerido:

$$(|a_1-b_1|+|a_2-b_2|)^2 = (a_1-b_1)^2 + 2|a_1-b_1||a_2-b_2| + (a_2-b_2)^2$$

Como el término cruzado $2|a_1-b_1||a_2-b_2|$ es siempre no negativo (producto de valores absolutos), se cumple $(|a_1-b_1|+|a_2-b_2|)^2 \ge (a_1-b_1)^2+(a_2-b_2)^2$, es decir $d_M(A,B)^2 \ge d_E(A,B)^2$. Como ambas distancias son no negativas, al tomar raíz cuadrada la desigualdad se preserva: $d_M(A,B)\ge d_E(A,B)$. La igualdad solo se alcanza cuando $|a_1-b_1||a_2-b_2|=0$, es decir cuando los puntos difieren en una única coordenada.

---

### Ejercicio 3 — Efecto de la escala sobre la distancia

**Planteamiento:** contrastar $A,B,C$ en escala original vs. escala reducida ($A',B',C'$) para mostrar cómo una variable de rango mayor domina la distancia euclidiana — la base conceptual de por qué se estandariza antes de aplicar KNN.

**a. Distancias en escala original**

$$d(A,B)=\sqrt{(20-25)^2+(1000-1200)^2}=\sqrt{40025}\approx200.06 \qquad d(A,C)=\sqrt{(20-40)^2+(1000-1010)^2}=\sqrt{500}\approx22.36$$

**b. Vecino más cercano:** como $d(A,C)\approx22.36 \ll d(A,B)\approx200.06$, el punto **C** es el vecino más cercano a $A$.

**c. Distancias tras dividir la segunda variable entre 100**

Con $A'=(20,10)$, $B'=(25,12)$, $C'=(40,10.1)$:

$$d(A',B')=\sqrt{5^2+2^2}=\sqrt{29}\approx5.385 \qquad d(A',C')=\sqrt{20^2+0.1^2}=\sqrt{400.01}\approx20.0003$$

**d. ¿Cambió el vecino más cercano?**

Sí: ahora $d(A',B')\approx5.39 < d(A',C')\approx20.00$, así que el vecino más cercano pasó a ser **B** — el punto opuesto al de antes de reescalar. En la escala original, la segunda variable (diferencias de cientos) dominaba por completo el cálculo; al dividirla entre 100, su diferencia se vuelve comparable a la de la primera variable, y esta última —antes irrelevante— pasa a determinar la distancia.

**e. Por qué una escala mayor domina la distancia**

En $d(x,z)=\sqrt{\sum_{j=1}^d(x_j-z_j)^2}$, cada variable contribuye a través del cuadrado de su diferencia; si la variable $j$ tiene un rango considerablemente mayor al de las demás (como la segunda coordenada aquí, del orden de miles frente a decenas), $(x_j-z_j)^2$ será, en términos absolutos, mucho mayor que los demás términos, de forma que la suma —y por lo tanto la distancia— queda determinada casi exclusivamente por esa variable, sin importar si conceptualmente es la más relevante para el problema. Este es precisamente el motivo por el cual, antes de aplicar KNN, se recomienda estandarizar o normalizar las variables.

---

### Ejercicio 4 — Matriz de confusión y métricas básicas

**Planteamiento:** a partir de la matriz $\begin{smallmatrix}850&100\\20&30\end{smallmatrix}$, extraer $TP,TN,FP,FN$ y derivar Accuracy, Precision, Recall y F1, cerrando con la identidad algebraica $F1=\frac{2TP}{2TP+FP+FN}$.

**a.** Leyendo filas = clase real, columnas = clase predicha: $TN=850$, $FP=100$, $FN=20$, $TP=30$.

**b.** $\text{Accuracy}=\dfrac{30+850}{1000}=0.88$

**c.** $\text{Precision}=\dfrac{30}{130}=\dfrac{3}{13}\approx0.2308$

**d.** $\text{Recall}=\dfrac{30}{50}=0.6$

**e.** $F1=2\cdot\dfrac{(3/13)(3/5)}{3/13+3/5}=2\cdot\dfrac{9/65}{54/65}=\dfrac{18}{54}=\dfrac13\approx0.3333$

**f. Demostración algebraica de $F1=\dfrac{2TP}{2TP+FP+FN}$**

Con $P=\dfrac{TP}{TP+FP}$ y $R=\dfrac{TP}{TP+FN}$:

$$P\cdot R = \frac{TP^2}{(TP+FP)(TP+FN)}, \qquad P+R = \frac{TP(TP+FN)+TP(TP+FP)}{(TP+FP)(TP+FN)} = \frac{TP(2TP+FP+FN)}{(TP+FP)(TP+FN)}$$

Sustituyendo en $F1=2\cdot\dfrac{P\cdot R}{P+R}$, el denominador común $(TP+FP)(TP+FN)$ se cancela:

$$F1 = 2\cdot\frac{TP^2}{TP(2TP+FP+FN)} = \frac{2TP}{2TP+FP+FN}$$

Verificación numérica: $\dfrac{2(30)}{2(30)+100+20}=\dfrac{60}{180}=\dfrac13$, idéntico al resultado del inciso e.

---

### Ejercicio 5 — Comparación de dos modelos y costo de clasificación

**Planteamiento:** contrastar Modelo A y Modelo B en las cuatro métricas estándar y en un costo asimétrico $L=FP+5FN$, para mostrar que "mejor Accuracy" y "menor costo" no son sinónimos cuando los errores no valen lo mismo.

**a. Métricas de cada modelo**

Modelo A ($TN=900,FP=50,FN=30,TP=20$): $\text{Acc}_A=0.92$, $\text{Prec}_A=\frac{20}{70}\approx0.2857$, $\text{Rec}_A=0.4$, $F1_A=\frac13\approx0.3333$

Modelo B ($TN=850,FP=100,FN=10,TP=40$): $\text{Acc}_B=0.89$, $\text{Prec}_B=\frac{40}{140}\approx0.2857$, $\text{Rec}_B=0.8$, $F1_B=\frac{8}{19}\approx0.4211$

**b. Costo $L=FP+5FN$**

$$L_A=50+5(30)=200 \qquad L_B=100+5(10)=150$$

**c. Mayor Accuracy:** Modelo A (0.92 > 0.89).

**d. Mayor Recall:** Modelo B (0.8 > 0.4), por amplio margen.

**e. Menor costo:** Modelo B (150 < 200).

**f. El modelo de mayor Accuracy no minimiza el costo**

El Modelo A tiene mayor Accuracy pero mayor costo que el Modelo B. Esto sucede porque el Accuracy pondera todos los errores por igual, mientras que $L$ pondera cada falso negativo cinco veces más que cada falso positivo: el Modelo A minimiza el número total de errores (80 vs. 110), pero comete más falsos negativos (30 vs. 10) —los más costosos—, mientras que el Modelo B compensa con más falsos positivos (100 vs. 50) —los baratos. Cuando los dos tipos de error tienen consecuencias distintas, elegir el modelo solo por Accuracy puede ser una decisión subóptima desde el costo real de negocio.

---

## Parte II — Componente práctico-computacional
*(Python · dataset **Give Me Some Credit**)*

### Fuente de datos

Se ubicó el dataset original de la competencia de Kaggle *Give Me Some
Credit* (2011, predicción de incumplimiento crediticio a 2 años) replicado
en un repositorio público de GitHub, y se cargó con pandas: 150.000
observaciones × 11 variables, coincidiendo con la dimensión documentada
oficialmente para este dataset.

### Ejercicio 6 — Exploración inicial y preparación de datos

**a–c. Carga y diagnóstico:** 150.000 observaciones × 11 variables (10 predictoras + `SeriousDlqin2yrs`); 6 variables `int64` y 4 `float64`. Solo dos variables tienen faltantes: `MonthlyIncome` (29.731, 19,82 %) y `NumberOfDependents` (3.924, 2,62 %).

**d. Balance de clases**

| Clase | n | % |
|---|---|---|
| 0 (no incumple) | 139.974 | 93,32 % |
| 1 (incumple) | 10.026 | 6,68 % |

Casi 14 casos negativos por cada positivo: un modelo trivial que siempre prediga 0 ya alcanzaría ~93 % de Accuracy sin ser útil, de modo que Precision/Recall/F1 sobre la clase 1 serán las métricas que realmente informen del desempeño.

**e. Gráfica**

![Distribución de la variable objetivo](figuras/ex6_distribucion_target.png)

**f. Imputación de faltantes**

Se imputó con la **mediana** en lugar de la media porque ambas variables con faltantes tienen colas largas hacia la derecha (unos pocos ingresos o números de dependientes extremadamente altos que arrastrarían la media), y la mediana es robusta a esos atípicos. Medianas usadas: `MonthlyIncome`=5.400, `NumberOfDependents`=0.

**g. Partición 80/20 estratificada**

| Conjunto | Clase 0 (n / %) | Clase 1 (n / %) |
|---|---|---|
| Entrenamiento (120.000) | 111.979 / 93,32 % | 8.021 / 6,68 % |
| Prueba (30.000) | 27.995 / 93,32 % | 2.005 / 6,68 % |

Se usó `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`; la estratificación mantiene la proporción de clases prácticamente idéntica entre train y test, evitando que el azar produzca una partición con una proporción de incumplimientos notablemente distinta a la del conjunto completo (lo cual sesgaría la evaluación).

---

### Ejercicio 7 — KNN k=5: sin escalar vs. estandarizado

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Sin escalamiento | 0,9326 | 0,4224 | 0,0244 | 0,0462 |
| Estandarizado | 0,9329 | 0,4936 | 0,1357 | 0,2128 |

![Matrices de confusión k=5](figuras/ex7_matrices_confusion.png)

**a. ¿Cambió el desempeño tras estandarizar?** Sí, y de forma sustancial en la capacidad de detectar la clase positiva: el Accuracy apenas se mueve (0,9326→0,9329), pero el Recall se multiplica por ~5,5 veces (0,0244→0,1357) y el F1 por >4,5 veces (0,0462→0,2128).

**b. Métrica con el cambio más importante:** el **Recall**. Consistente con el ejercicio 3e: sin estandarizar, variables como `MonthlyIncome` o `DebtRatio` (rangos de miles) dominan el cálculo de distancias frente a `age` o `NumberOfDependents` (rangos de decenas/unidades), y el modelo termina casi ignorando estas últimas; al estandarizar, todas aportan en escala comparable y el modelo pasa de detectar 49 de 2.005 incumplimientos reales a detectar 272 de 2.005.

**c. Modelo elegido:** el **estandarizado** — Accuracy prácticamente igual, pero Recall y F1 muy superiores; en un problema de riesgo crediticio, donde no detectar un incumplimiento suele costar más que una alerta equivocada (ver ejercicio 10), esa mejora es preferible casi en cualquier escenario de negocio razonable.

---

### Ejercicio 8 — Efecto del número de vecinos $k$

| k | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 1 | 0,9044 | 0,2496 | 0,2144 | **0,2307** |
| 3 | 0,9276 | 0,3978 | 0,1621 | 0,2303 |
| 5 | 0,9329 | 0,4936 | 0,1357 | 0,2128 |
| 7 | 0,9345 | 0,5412 | 0,1277 | 0,2066 |
| 9 | **0,9350** | 0,5652 | 0,1167 | 0,1935 |
| 11 | 0,9346 | 0,5583 | 0,1027 | 0,1735 |
| 15 | 0,9343 | 0,5549 | 0,0883 | 0,1523 |
| 21 | 0,9348 | 0,5863 | 0,0813 | 0,1428 |

![k vs métricas](figuras/ex8_k_vs_metricas.png)

**a. Mejor F1:** $k=1$ (0,2307). **b. Mejor Recall:** $k=1$ (0,2144). **c. Mejor Accuracy:** $k=9$ (0,9350).

**d. ¿Coinciden los tres?** No: F1 y Recall se maximizan en $k=1$, mientras que Accuracy se maximiza en $k=9$. Al aumentar $k$, el modelo promedia sobre vecindarios cada vez más grandes que, dado el fuerte desbalance (93 %/7 %), terminan dominados por vecinos de la clase mayoritaria, empujando las predicciones hacia la clase 0 y sacrificando Recall a cambio de un Accuracy apenas mayor; $k=1$, al fijarse solo en el vecino más próximo, es más sensible a los (pocos) vecinos positivos cercanos, favoreciendo el Recall aunque a costa de mayor varianza (más falsos positivos, como se ve en el ejercicio 10).

---

### Ejercicio 9 — Distancia Euclidiana vs. Manhattan

Se comparan los tres mejores $k$ según F1 del ejercicio 8: $k=1,3,5$.

| k | Distancia | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | Euclidean | 0,9044 | 0,2496 | 0,2144 | **0,2307** |
| 1 | Manhattan | 0,9054 | 0,2518 | 0,2105 | 0,2293 |
| 3 | Euclidean | 0,9276 | 0,3978 | 0,1621 | 0,2303 |
| 3 | Manhattan | 0,9283 | 0,4000 | 0,1466 | 0,2146 |
| 5 | Euclidean | 0,9329 | 0,4936 | 0,1357 | 0,2128 |
| 5 | Manhattan | 0,9329 | 0,4921 | 0,1237 | 0,1977 |

**a. ¿La métrica modifica el desempeño?** Sí, pero de forma mucho más discreta que la estandarización o $k$: para cada $k$, Manhattan da Accuracy/Precision levemente superiores pero Recall/F1 levemente inferiores a Euclidiana, con diferencias que no superan 1,5 puntos porcentuales.

**b. Mayor F1:** $k=1$, Euclidiana (0,2307). **c. Mayor Recall:** $k=1$, Euclidiana (0,2144).

**d. ¿Coinciden F1 y Recall en la misma configuración?** Sí, ambas apuntan a $k=1$-Euclidiana. Tiene sentido: a diferencia del trade-off marcado entre $k=1$ y $k=9$ del ejercicio 8, aquí cambiar la métrica de distancia no altera lo suficiente cuál vecino es "más cercano" como para invertir el orden de los modelos — con los datos ya estandarizados, la diferencia entre penalizar cuadráticamente (Euclidiana) o linealmente (Manhattan) las diferencias por variable resulta menor que la que introduce cambiar $k$.

---

### Ejercicio 10 — Análisis final y costo de negocio

Se comparan tres modelos representativos: el mejor F1/Recall (k=1, Euclidiana, estandarizado), el mejor Accuracy (k=9, Euclidiana, estandarizado), y el modelo base sin estandarizar (k=5, ejercicio 7).

| Modelo | k | FP | FN | Recall | F1 | Costo $L=FP+5FN$ |
|---|---|---|---|---|---|---|
| Modelo 1 (mejor F1/Recall) | 1 | 1.293 | 1.575 | 0,2144 | 0,2307 | **9.168** |
| Modelo 2 (mejor Accuracy) | 9 | 180 | 1.771 | 0,1167 | 0,1935 | **9.035** |
| Modelo 3 (base, sin estandarizar) | 5 | 67 | 1.956 | 0,0244 | 0,0462 | **9.847** |

**a. ¿El modelo de mayor Accuracy es también el de menor costo?** En este caso concreto, **sí**: el Modelo 2 tiene simultáneamente el mayor Accuracy (0,9350) y el menor costo (9.035) de los tres. Vale la pena anotarlo con honestidad: esto **no contradice** lo demostrado teóricamente en el ejercicio 5f, donde se construyó un ejemplo específico en el que Accuracy y costo mínimo sí divergían; simplemente muestra que la relación entre ambas métricas depende de la forma concreta de los errores de cada modelo y no es una ley universal — aquí, el modelo con más vecinos (k=9) resulta lo bastante conservador (muy pocos FP) como para también minimizar el costo, pero eso podría no sostenerse con otra función de costo u otro dataset.

**b. ¿Menor número de falsos negativos?** El **Modelo 1** (k=1), con 1.575 FN, frente a 1.771 del Modelo 2 y 1.956 del Modelo 3.

**c. ¿Qué modelo recomendaría para identificar clientes de alto riesgo?** El **Modelo 1**, a pesar de no tener ni el mayor Accuracy ni el menor costo $L$. Si el objetivo explícito es maximizar la detección de incumplimientos reales (priorizar Recall), el Modelo 1 detecta 430 de los 2.005 incumplimientos del conjunto de prueba (21,44 %), casi el doble que el Modelo 2 (234 de 2.005, 11,67 %); el costo $L$ tal como está definido pondera los falsos positivos con peso 1, lo cual puede subestimar el valor real de detectar más incumplimientos si, en la práctica, el costo de no detectar a un mal pagador es incluso mayor a 5 veces el de una alerta equivocada.

**d. Justificación final:** la comparación deja en evidencia una tensión real entre tres criterios de selección —Accuracy, un costo explícito ponderado, y Recall puro— que no siempre apuntan al mismo modelo: el Modelo 2 es la opción más conservadora y "eficiente en costo" bajo la $L$ definida aquí, mientras que el Modelo 1 es la más agresiva en detección de riesgo. La elección final no es un ejercicio puramente estadístico sino una decisión de negocio que depende de cuánto le cuesta realmente a la institución cada tipo de error — si ese costo relativo real fuera mayor a 5:1, el balance se inclinaría aún más hacia el Modelo 1.

---

## Próximos pasos
1. Releer y ajustar la redacción a tu gusto (este documento prioriza que
   el razonamiento quede explícito; se puede recortar donde ya te
   convenza).
2. Armar el notebook `.ipynb` real con el código que generó estos
   resultados (está corrido y verificado, falta trasladarlo a celdas
   documentadas como exige el taller).
3. Trasladar el contenido final a LaTeX.
