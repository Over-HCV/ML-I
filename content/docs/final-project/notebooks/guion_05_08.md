# Guion: Diapositivas 05 a 08

Tiempo objetivo: cerca de 4 minutos, unas 600 palabras dichas con calma.

Cada diapositiva tiene tres partes. La idea es lo que debe quedar claro aunque se olvide el texto. Qué señalar indica a dónde mirar o apuntar. El guion es lo que se dice tal cual.

---

## 05. Cribar, decidir y probar una vez

### Idea
Antes de modelar decidimos qué vale la pena modelar y cómo evaluarlo sin hacer trampa.

### Qué señalar
- El embudo de la izquierda: 16, 13, 10 y 3.
- Las tres franjas de tiempo.
- El 16 verde de Taguchi.

### Guion
Con esta base había muchas preguntas posibles y no nos daba el tiempo para estudiarlas todas. Por eso primero escribimos una regla para escoger.

Partimos de 16 análisis candidatos. Los de Bayes no entran al cribado porque son cálculos y no modelos. A los 13 que quedan les hicimos una prueba rápida. Siguen solo los que le ganan a su línea base en los cuatro bloques de tiempo y no dependen de una sola variable. Eso deja 10. De esos escogimos el de mayor puntaje de cada tipo, que combina qué tan bien predice con cuánto le sirve al laboratorio. Así llegamos a 3, uno por tipo.

A la derecha está la regla más importante del proyecto: Nunca entrenamos con datos del futuro. Partimos la base por fecha. Con 2023 hasta mitad de 2025 ajustamos y comparamos modelos. Con el segundo semestre de 2025 escogimos el modelo y el umbral. Y 2026 quedó cerrado hasta el final y lo abrimos una sola vez.

Por último, para ajustar cada modelo había seis decisiones, como el escalado o qué variables entran. Probar todas las combinaciones eran 128 corridas y con un diseño de Taguchi lo hicimos en 16. Para confiar en el atajo corrimos también las 128 y el orden de importancia salió casi igual, con una correlación de 0.94.

---

## 06. Cuántas muestras llegarán mañana

### Idea
La regresión lineal le gana a la referencia y el error se traduce en días bien o mal planificados.

### Qué señalar
- La barra azul de lineal múltiple arriba y las dos referencias abajo.
- Los dos números grandes: 14.6 contra 9.8.

### Guion
La primera pregunta es cuántas muestras van a llegar mañana. La predecimos en la programación, cuando ya se conoce la agenda de muestreos.

A la izquierda comparamos los modelos del curso con validación cruzada por bloques. Ganó la regresión lineal múltiple con un RMSE de 13.3 muestras por día. La red neuronal quedó segunda y no la mejoró. Abajo están las dos referencias sin modelo: Repetir la media de los últimos cinco días y el Dummy que siempre dice el promedio.

Congelamos la lineal y la evaluamos en prueba. La referencia se equivoca en promedio en 14.6 muestras por día y el modelo en 9.8.

Para leer ese número hay que saber que un día hábil trae unas 44 muestras. Si aceptamos un margen de 11 muestras, que es el 20 % de la media, la referencia falla el 57 % de los días y el modelo el 38 %. Además el error en prueba fue 15 % menor que en validación, así que el modelo no estaba sobreajustado.

Sirve para fijar el personal base, pero con margen.

---

## 07. Qué muestras son de riesgo

### Idea
La logística con un umbral fijado por costo da una lista corta y útil de muestras para vigilar.

### Qué señalar
- La barra naranja de regresión logística.
- El punto del umbral 0.167 en la curva de costo.
- La tabla de 2 por 2 y los tres números de abajo.

### Guion
La segunda pregunta es qué muestras son de riesgo en el momento de recibirlas, antes de sembrarlas. Cerca del 9 % sale no conforme. Por eso la exactitud no sirve: Decir siempre conforme acertaría el 91 % y no detectaría nada. Medimos con ROC-AUC y con costo.

Ganó la regresión logística con un ROC-AUC de 0.70 en validación cruzada. Otra vez la red neuronal queda cerca pero no la supera.

Para convertir la probabilidad en una alerta fijamos los costos antes de mirar resultados: Perder una no conforme cuesta 5 y una falsa alarma cuesta 1. Con eso la teoría da un umbral de 1 sobre 6, o sea 0.167. La curva muestra que en validación el costo mínimo cae justo ahí. Eso nos dice que las probabilidades quedaron bien calibradas.

En prueba dejamos el umbral congelado durante 44 semanas. Salen unas 21 alertas por semana y más o menos una de cada cuatro es una no conforme real. Detectamos un tercio de las no conformes y el costo baja 15 % frente a la mejor política sin modelo.

El ROC-AUC se sostuvo en 0.72 de validación a prueba. Lo que bajó fue la cobertura, del 40 % al 33 %, y lo reportamos tal cual.

---

## 08. Redes neuronales: Del perceptrón al MLP

### Idea
Una red con 30 veces más parámetros no supera a la logística, porque el riesgo se explica con efectos que se suman.

### Qué señalar
- Los dos números grandes: 0.702 contra 0.692.
- Los parámetros: 136 contra 4385.
- Las tres cifras de la red a la derecha.

### Guion
Como el perceptrón multicapa es tema del curso, le dimos una pasada didáctica. Tomamos la misma pregunta de riesgo con los mismos bloques y las mismas variables, y pusimos una capa oculta entre las variables y la probabilidad.

La red tiene 32 neuronas ocultas con activación ReLU. Entrenamos hasta 32 épocas con parada temprana y la mejor fue la 23. Le pusimos una regularización alpha de 0.001 para castigar los pesos grandes.

El resultado es que la logística tiene 136 parámetros y la red 4385, más de 30 veces más. Aun así la logística ordena un poco mejor, 0.702 frente a 0.692, con la misma variación entre bloques. Y el Brier, que mide si las probabilidades son creíbles, es prácticamente el mismo.

¿Por qué pasa esto? El riesgo se explica sobre todo con efectos que se suman, como un transporte más cálido o más largo. Eso la logística ya lo captura. La capa oculta sirve para encontrar interacciones entre variables y aquí no encontró ninguna que pague su costo.

Por eso el modelo que congelamos es la logística: Es más simple, se puede explicar en recepción y funciona igual o mejor. Y con esas probabilidades pasamos a la tercera pregunta.

---

## Si preguntan

**¿Por qué RMSE en validación y error absoluto medio en prueba?**
El RMSE castiga más los errores grandes y lo usamos para escoger. El error absoluto medio se lee directo en muestras por día y lo usamos para contar el resultado.

**¿Por qué no usar exactitud en la clasificación?**
Con 9 % de positivos un modelo que nunca alerta acierta el 91 %. El ROC-AUC mide qué tan bien ordena y el costo mide la decisión.

**¿De dónde salen los costos 5 y 1?**
Son un supuesto académico que fijamos antes de mirar. El siguiente paso es revisarlos con el laboratorio.

**¿Por qué bajó el recall en prueba?**
El umbral quedó congelado y en 2026 el modelo alerta menos (21 por semana en vez de 31). El ordenamiento se mantuvo igual, así que es un tema del corte y no del modelo.

**¿Por qué Taguchi y no búsqueda en rejilla?**
Usamos los dos. Taguchi escoge la receta completa (modelo, escalado, variables, logaritmo) con 16 corridas. La rejilla afina después los ajustes finos de los dos mejores modelos.
