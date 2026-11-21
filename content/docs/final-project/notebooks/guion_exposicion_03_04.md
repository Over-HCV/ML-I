# Guion de la exposición: Notebooks 03 y 04

Tiempo total: **5 minutos**. Cada tramo de los notebooks tiene un recuadro **Lo que digo** con las cifras y debajo
la evidencia. Este guion es la versión hablada y un poco más larga. No hay que decirlo palabra por palabra: Son las
ideas en el orden en que salen en pantalla.

| Notebook | Tramo | Pregunta | Tiempo | Acumulado |
|---|---|---|---|---|
| 03 | A | Qué vale la pena modelar a fondo | 0:45 | 0:45 |
| 03 | B | Cuántas muestras llegan mañana | 1:15 | 2:00 |
| 04 | A | Qué muestras vigilar al recibirlas | 1:45 | 3:45 |
| 04 | B | Cuánto creerle a un positivo | 1:15 | 5:00 |

**Antes de empezar:** Abrir los dos notebooks ya ejecutados, plegar la sección "Código del proyecto" y dejar el 04
listo en el video de Bayes. No ejecutar nada en vivo (los paneles del anexo necesitan el núcleo y es un riesgo).

**Frase de enlace con el 02:** "Lo que sigue lo hicimos sobre una base simulada con la misma estructura que la real,
porque así podemos usar cosas que la exportación real no trae, como los controles de calidad de cada ensayo."

---

## Notebook 03, tramo A: Qué vale la pena modelar (0:45)

**A.1 (contador de 16).** "Con la base del laboratorio se podían plantear 16 preguntas distintas: Cuántas muestras
llegan, cuáles salen mal, cuánto tarda el informe, qué clientes dejan de enviar. No nos daba el tiempo para
estudiarlas todas a fondo así que tuvimos que escoger."

**A.2 (tarjetas de la regla).** "Para escoger sin hacernos trampa escribimos una regla antes de ver resultados. Cada
pregunta pasa por una prueba rápida, con modelos sin afinar, y se compara contra una referencia que no usa modelo.
Para seguir tiene que ganarle a esa referencia en los cuatro periodos de validación, no solo en promedio. Tiene que
necesitar de verdad un modelo, o sea que no baste una sola variable. Y tiene que tener el mejor puntaje de su tipo,
que combina qué tan bien predice con cuánto le sirve al laboratorio."

**A.3 (embudo).** "De 16 quedan 3: La carga diaria, que vemos ahora, la muestra no conforme vista desde la recepción,
que es el 04, y los perfiles de muestra, que es el 05."

**A.4 (barras naranjas).** "Lo curioso es que la pregunta que mejor predecía, si un no conforme es falso, quedó
fuera. Con una sola variable, cuánto pasa el recuento del límite, se lograba el 95 % de lo que lograba el modelo. Ahí
sobra el modelo y basta una regla, que retomamos al final."

## Notebook 03, tramo B: Cuántas muestras llegan mañana (1:15)

**B.1 (planteamiento).** "La primera pregunta elegida es cuántas muestras van a llegar mañana. La respondemos la
víspera, solo con lo que ya se sabe: Los días anteriores, el calendario y lo agendado. Sirve para cuadrar turnos,
medios de cultivo e incubadoras. La vara es lo que el laboratorio ya puede hacer sin modelo, promediar los últimos
cinco días."

**B.2 (tabla didáctica y figura del L16).** "Lo modelamos en tres niveles. Primero los modelos del curso tal como
vienen: Ya ahí gana la regresión lineal múltiple y la red neuronal no la mejora. Segundo, las decisiones de la
receta: El modelo, el escalado, qué variables entran, si usamos logaritmo... Todas las combinaciones eran 128 corridas
y con un diseño de Taguchi las redujimos a 16. Para comprobar que el atajo servía corrimos también las 128 y el orden
de importancia salió casi igual, con una correlación de 0.94. Tercero, el ajuste fino y una sola evaluación en
validación."

**B.3 (tarjetas y serie).** "Gana la regresión lineal, el modelo más sencillo. Promediar cinco días se equivoca en
unas 15 muestras por día y el modelo en 11, en días de unas 54 muestras. Los días mal planificados bajan del 56 % al
42 %. Y no es suerte: El intervalo de la mejora no toca el cero. Sirve para la dotación base del día siguiente, con
margen."

**Enlace al 04:** "La segunda pregunta elegida no es cuántas llegan sino cuáles van a salir mal."

## Notebook 04, tramo A: Qué muestras vigilar al recibirlas (1:45)

**A.1 (planteamiento y 8.9 %).** "Cerca del 9 % de las muestras salen no conformes. Queremos saberlo al recibirlas,
antes de sembrarlas, para priorizarlas o avisarle al cliente. Antes de mirar fijamos cuánto cuesta cada error: Dejar
pasar una no conforme cuesta 5 porque el aviso llega tarde, y una falsa alarma cuesta 1 porque es solo tiempo de
revisión. Es un supuesto nuestro. De ahí sale el umbral teórico: Alertar cuando el riesgo pasa de un sexto, 0.167."

**A.2 (tabla, curvas ROC, tarjeta de la red).** "Repetimos los mismos tres niveles. Gana la regresión logística con un
ROC-AUC de 0.72: Ordena mejor que el azar, que sería 0.5, pero no perfecto. La red neuronal tiene 32 veces más
parámetros y no la mejora. No medimos con exactitud porque decir siempre 'conforme' ya acierta el 91 %."

**A.3 (tarjetas de calibración y figura).** "Aquí tuvimos un tropiezo útil. Para equilibrar las clases copiamos las no
conformes y eso infló las probabilidades: El modelo decía 46 % de riesgo promedio cuando la tasa real es 9 %. Lo
calibramos y bajó a 10 %. Sin ese paso ni el umbral ni lo de Bayes tendrían sentido."

**A.4 (tres tarjetas y curva de costo).** "Recorrimos los umbrales y el de menor costo fue 0.167, el mismo de la
teoría, lo que confirma la calibración. Con él salen 31 alertas por semana, acierta el 28 % de ellas y se detecta el
40 % de las no conformes. El costo baja un 19 % frente a no alertar nunca."

## Notebook 04, tramo B: Cuánto creerle a un positivo (1:15)

**B.1 (video, 32 s).** "Esta es la pregunta con que empezó el laboratorio: Cuando un ensayo da no conforme, ¿qué tan
probable es que sea falso? Cada ensayo tiene una sensibilidad y una especificidad que salen de sus controles. Aunque
el ensayo sea muy bueno, si la contaminación real es rara buena parte de las alarmas son falsas. Es el teorema de
Bayes y el video lo muestra con mil muestras." (Dejar correr el video y narrar encima.)

**B.2 (tabla por ensayo).** "En el laboratorio unos 22 de cada 100 no conformes serían falsos. El peor ensayo es
esterilidad, con 37 de cada 100, y el mejor Salmonella, con 10."

**B.3 (dos tarjetas).** "Y aquí se unen las dos partes. El clasificador da el riesgo previo de cada muestra. Estas dos
muestras dieron el mismo positivo en el mismo ensayo: En la de riesgo bajo la probabilidad de que sea falso es 38 % y
en la de riesgo alto es 3 %. El mismo resultado se lee distinto según la muestra."

**B.4 (reensayo y regla).** "Si el reensayo vuelve a dar positivo el resultado queda prácticamente confirmado, 99 %, y
las cuentas cuadran con lo observado: Esperábamos 25 % de reensayos conformes y salió 27 %. La regla que quedó
pendiente en el 03 es reensayar antes de informar cuando el recuento pasa el límite por menos de 1.66 veces, porque
ahí el 78 % de los reensayos sale conforme."

**Cierre:** "El modelo no reemplaza al analista: Le dice qué mirar primero y cuánto confiar en lo que ve."

---

## Si me paso de tiempo, qué recorto

1. El 03 A.4 (lo curioso) se puede decir en una frase y retomarlo solo en el cierre del 04.
2. En el 03 B.2 basta con "128 combinaciones reducidas a 16 con Taguchi" sin la comprobación.
3. En el 04 A.2 se salta lo de la exactitud.
4. El video de Bayes se puede cortar a la mitad y seguir con la tabla.

## Preguntas que pueden hacer

| Pregunta | Respuesta corta | Dónde está |
|---|---|---|
| ¿Por qué no usar exactitud? | Con 9 % de positivos, decir siempre "conforme" acierta el 91 % y no sirve | 04, A.2 |
| ¿De dónde sale el costo 5 a 1? | Es un supuesto académico declarado antes de mirar. El panel del umbral deja moverlo | 04, anexo (panel del umbral) |
| ¿Por qué no ganó la red neuronal? | La relación es casi aditiva: Lo que suma la logística ya explica el riesgo y la red no encuentra más | 04, anexo (red neuronal) |
| ¿Qué es Taguchi? | Un diseño de experimentos que estima el efecto de cada decisión con 16 corridas en vez de 128 | 03, anexo (nivel 2) |
| ¿Cómo evitan hacer trampa con el tiempo? | Validamos siempre con el pasado prediciendo el tramo que sigue, y la prueba solo la abre la etapa 06 | 03, anexo (auditoría de fugas) |
| ¿Esto está con datos reales? | El 03 y el 04 con la simulada de la misma estructura. La real está en el 02 | 01 y 02 |
| ¿Qué tan seguro es el 0.784? | Es con los controles, que describen el ensayo en su mejor día. La auditoría de la simulación da 0.756 real | 04, anexo (auditoría) |

## Glosario

| Término | Qué es por dentro | Por qué importa |
|---|---|---|
| Referencia o línea base | Predicción sin modelo: Promedio de los últimos días, la tasa general o un Dummy que siempre dice lo mismo | Sin ella un error de "11 muestras" no significa nada. El modelo solo vale si le gana |
| Fuga de información | Usar para predecir un dato que todavía no existe en ese momento | Da modelos que parecen excelentes y fallan en la operación. La auditoría del 03 la descarta |
| Validación de ventana creciente | Se entrena con todo el pasado y se prueba con el tramo siguiente, 4 veces, avanzando en el tiempo | Imita el uso real, donde se predice el futuro con el pasado |
| Entrenamiento, validación y prueba | Tres bloques por fecha: Uno para ajustar, otro para decidir y el último cerrado hasta el 06 | La prueba sin tocar es la única medida honesta con datos nuevos |
| RMSE | Raíz del promedio de los errores al cuadrado, en muestras por día | Tamaño típico del error. Castiga más los errores grandes, que son los que desarman un turno |
| Habilidad | 1 − (error del modelo / error de la referencia). En clasificación, 2 AUC − 1 | Pone en una sola escala preguntas muy distintas para compararlas en el cribado |
| Regresión lineal | Intercepto + suma de pesos por variables | Simple y explicable. Ganó en el 03 |
| Regresión logística | La misma suma pasada por una curva en S (sigmoide) que la vuelve probabilidad | Ganó en el 04 y da probabilidades que se pueden calibrar y usar en Bayes |
| Árbol de decisión y poda | Reglas "si X > a y Y ≤ b". La poda cobra un precio por hoja y corta las que no aportan | Las reglas se leen en voz alta. Sin poda el árbol memoriza ruido |
| KNN | Predice con el promedio de los k casos más parecidos | Referencia intuitiva. Aquí fue de los más débiles |
| Perceptrón multicapa | Capas de sumas ponderadas con activaciones. Aprende moviendo pesos en contra del gradiente del error | No mejoró a los lineales: La relación es casi aditiva y su costo no se justifica |
| Tubería | Cadena fija: Imputar, escalar, codificar y ajustar, aprendida solo con los datos de ajuste | Evita que información de validación se cuele en el preprocesamiento |
| Hiperparámetro y afinamiento | Ajustes que no aprende el modelo (vecinos k, poda α, C). La rejilla prueba combinaciones | Último ajuste fino. Aquí movió poco |
| Arreglo ortogonal L16 de Taguchi | 16 combinaciones balanceadas donde cada nivel aparece lo mismo frente a los demás | Estima el efecto de 6 decisiones con 16 corridas en vez de 128 |
| Razón señal/ruido (S/N) | −10 log₁₀ del error cuadrático medio entre bloques. Más alta es mejor | Premia un error bajo y estable entre periodos |
| Bootstrap | Repetir el cálculo 1000 veces sobre muestras sacadas al azar con reemplazo y tomar el 95 % central | Dice si la mejora es real o suerte de qué días cayeron |
| Congelar el modelo | Fijar receta, ajustes y umbral y guardarlo antes de abrir la prueba | Impide retocar el modelo mirando la prueba |
| Desbalance y sobremuestreo | Hay pocos positivos (9 %) y se copian al azar hasta igualar las clases | Puede ayudar a aprender pero infla las probabilidades |
| Calibración | Reajuste (sigmoide de Platt) para que un 20 % de riesgo signifique 20 de cada 100 | Sin ella el umbral por costo y Bayes darían cifras falsas |
| ROC-AUC | Probabilidad de que una no conforme al azar reciba más riesgo que una conforme al azar | Mide qué tan bien ordena sin dejarse engañar por el desbalance |
| Exactitud | Fracción de aciertos | Aquí engaña: Decir siempre "conforme" acierta el 91 % |
| Brier | Error cuadrático entre la probabilidad dada y lo que pasó (0 o 1) | Mide si las probabilidades son creíbles |
| Falso positivo y falso negativo | Alarma sobre una muestra que sale bien y muestra mala sin alarma | Cuestan distinto (1 y 5) y de ahí sale el umbral |
| Umbral por costo | Corte de probabilidad que minimiza 1 FP + 5 FN. En teoría vale 1/(1 + 5) | Convierte la probabilidad en una decisión: 31 alertas por semana |
| Precisión y recall | De las alertas, cuántas aciertan. De las no conformes, cuántas se alertan | Traducen el modelo a carga de trabajo y cobertura |
| Sensibilidad y especificidad | Fracción de contaminadas que dan positivo y de limpias que dan negativo, medidas con controles | Describen el ensayo en sí, sin importar la muestra |
| Prevalencia | Fracción real de contaminadas. No se observa y se estima con Rogan y Gladen | Si es baja, incluso un buen ensayo da muchas falsas alarmas en proporción |
| Bayes, previa y posterior | La previa es la creencia antes del resultado y la posterior después de combinarla con el ensayo | Permite encadenar: El clasificador da la previa, el ensayo la actualiza, el reensayo otra vez |
| VPP | S p / (S p + (1 − E)(1 − p)): Fracción de positivos que son verdaderos | Responde la pregunta del laboratorio |
| Razón de odds | e^β: Cuánto se multiplican las chances de no conformidad cuando la variable sube una unidad | "Un lácteo tiene 3.6 veces las chances", aunque es asociación y no causa |
