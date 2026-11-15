# %% [markdown]
# # Etapa 04: Prever la carga de trabajo del laboratorio
#
# El cribado eligió un análisis de regresión y aquí lo modelamos a fondo en tres niveles. Primero pasamos por los
# modelos del curso con su configuración de partida, después cribamos los factores de la tubería con un arreglo
# ortogonal de Taguchi y al final afinamos los hiperparámetros de lo que sobrevivió. El ganador lo evaluamos una
# sola vez en validación antes de congelarlo.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import regresion

iniciar_etapa("04_regresion")
c = regresion.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# ## 1. El problema y las referencias ingenuas
#
# Decir que el modelo se equivoca en tantas muestras por día no dice nada si no lo ponemos al lado de lo que el
# laboratorio ya consigue sin modelo. Por eso empezamos por las referencias ingenuas (repetir lo de ayer, promediar
# la última semana) y las evaluamos en los mismos tramos de validación de los bloques temporales que después usan
# los modelos.

# %%
regresion.planteamiento(c).mostrar()

# %% [markdown]
# Para medir el error usamos el RMSE, la raíz del promedio de los errores al cuadrado, que se lee en muestras por
# día y castiga más los errores grandes. La **habilidad** compara el RMSE del modelo con el de la mejor referencia:
# Vale cero si el modelo empata con ella, es negativa si pierde y se acerca a uno cuando casi no se equivoca.
#
# $$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}\big(y_t - {\color{#3B64A8}{\hat y_t}}\big)^2}
# \qquad\qquad
# \text{habilidad} = 1 - \frac{{\color{#B8423F}{\mathrm{RMSE}_{\text{modelo}}}}}{\mathrm{RMSE}_{\text{referencia}}}$$

# %%
regresion.referencias(c).mostrar()

# %% [markdown]
# ## 2. Pasada didáctica: Los modelos del curso
#
# Cada modelo usa el grupo ampliado de variables y el preprocesamiento por defecto (imputación por la mediana y
# estandarización), sin afinar nada.
#
# ### 2.1 Regresión lineal simple
#
# Empezamos por la variable con mayor correlación absoluta con el objetivo y la dejamos en sus unidades originales,
# para que la pendiente se lea en muestras por día.

# %%
regresion.recta_simple(c).mostrar()

# %% [markdown]
# ### 2.2 Árbol de regresión con poda por costo-complejidad
#
# Un árbol que crece sin límite termina memorizando el entrenamiento. La poda por costo-complejidad le cobra un
# precio `ccp_alpha` a cada hoja y corta las ramas que no reducen el error lo suficiente para pagarlo. Calculamos la
# ruta de poda del árbol completo y nos quedamos con el `ccp_alpha` de menor RMSE medio en los bloques así que la
# complejidad la decide la validación y no una profundidad que pongamos a mano.

# %%
regresion.poda(c).mostrar()

# %% [markdown]
# ### 2.3 Regresión múltiple, KNN, árbol y perceptrón en los mismos bloques
#
# La regresión lineal múltiple predice la carga del día siguiente como una suma ponderada de lo que ya sabemos la
# víspera (rezagos, medias móviles, calendario y agenda), con un peso ${\color{#3B64A8}{\beta_j}}$ por variable:
#
# $$\hat y_{t+1} = {\color{#3B64A8}{\beta_0}} + \sum_{j=1}^{p} {\color{#3B64A8}{\beta_j}}\, x_{t,j}$$
#
# KNN promedia la carga de los días más parecidos, el árbol parte los días en grupos con reglas sucesivas y el
# perceptrón multicapa es una red neuronal pequeña que combina las variables en capas antes de dar la predicción. Los
# cuatro van dentro de una tubería de scikit-learn que imputa, escala y codifica con lo aprendido solo en el tramo
# de ajuste de cada bloque. Así se arma la de la regresión múltiple:

# %%
from proyecto.modelos import tuberias

tuberia = tuberias.construir_tuberia("regresion", "lineal", c.numericas, c.categoricas)
mostrar(tuberia)   # imputar y escalar las numéricas, codificar las categóricas y ajustar el modelo

# %%
regresion.didactica(c).mostrar()

# %% [markdown]
# ### 2.4 Qué dicen los coeficientes de la regresión múltiple
#
# Como las variables están estandarizadas, cada coeficiente es el cambio esperado en la carga cuando la variable
# sube una desviación estándar y las demás se quedan quietas.

# %%
regresion.coeficientes(c).mostrar()

# %% [markdown]
# ## 3. Nivel 2: Cribado de la tubería con un arreglo L16 de Taguchi
#
# Además del modelo, la tubería decide el escalado, el grupo de variables, si el objetivo va en logaritmo, si se
# seleccionan variables y cómo se codifican las discretas. Probar todas las combinaciones exige 4 × 2⁵ = 128
# corridas. Un **arreglo ortogonal L16** estima el efecto de cada uno de los seis factores con solo 16, porque cada
# nivel de un factor aparece el mismo número de veces frente a cada nivel de los demás. El precio es que algunas
# interacciones (cuando el efecto de un factor depende del nivel de otro) quedan mezcladas con los efectos
# principales, y la tabla de confusiones dice exactamente cuáles.
#
# El arreglo sale de `arreglos.arreglo_l16()` y antes de usarlo comprobamos que de verdad sea ortogonal:

# %%
from proyecto.cribado import arreglos
from proyecto.reporte.html import tabla_html

l16 = arreglos.arreglo_l16()                        # 16 corridas: A con 4 niveles y doce columnas de 2 niveles
ortogonal = arreglos.verificar_ortogonalidad(l16)   # cada par de columnas aparece balanceado
usadas = l16[list(regresion.COLUMNAS_L16.values())].T.set_axis(range(1, 17), axis=1)   # una fila por columna usada
titulo = f"Columnas del L16 que usamos, una por fila (ortogonal: {'Sí' if ortogonal else 'No'})"
mostrar(tabla_html(usadas, titulo, decimales=0, con_indice=True))

# %% [markdown]
# A cada corrida le medimos el RMSE en cada bloque y lo resumimos con la razón señal/ruido de Taguchi en la versión
# "menor es mejor". Como lleva un menos adelante, una S/N más alta es mejor. Además castiga a la vez un error alto y
# un error que cambia mucho entre bloques:
#
# $$\frac{S}{N} = -10\,\log_{10}\Big(\frac{1}{B}\sum_{b=1}^{B}{\color{#B8423F}{\mathrm{RMSE}_b^2}}\Big)$$

# %%
regresion.diseno(c).mostrar()

# %% [markdown]
# Con las 16 corridas calculamos los efectos principales (la S/N media de las corridas que usan cada nivel), la
# tabla de respuesta (el delta, o sea la diferencia entre el mejor y el peor nivel de cada factor) y el óptimo que
# predice el modelo aditivo. Ese modelo supone que los factores no interactúan y suma lo que aporta cada nivel sobre
# la media de las 16 corridas:
#
# $$\widehat{S/N} = {\color{#6B6B6B}{\overline{S/N}}} + \sum_{f}\Big({\color{#2F8A4E}{\overline{S/N}_{f,\,\text{nivel}}}} - {\color{#6B6B6B}{\overline{S/N}}}\Big)$$

# %%
regresion.corridas_l16(c).mostrar()

# %% [markdown]
# ### Corrida de confirmación y factorial completo
#
# El óptimo predicho es una combinación que el L16 no tiene por qué haber probado así que la corremos aparte y
# comparamos la S/N observada con la predicha, con una tolerancia de dos veces el error residual del modelo aditivo.
# Después, si el tiempo estimado (el del L16 multiplicado por 128 / 16) no pasa de un minuto, corremos también el
# factorial completo para comprobar el atajo de Taguchi.

# %%
regresion.confirmacion(c).mostrar()

# %% [markdown]
# ## 4. Nivel 3: Afinamiento de hiperparámetros
#
# Al afinamiento pasa la configuración de mayor S/N observada (la confirmación o la mejor del L16) y, para no
# apostarle a una sola familia, la segunda mejor familia de modelos con los demás factores iguales. Las rejillas las
# recorremos con `GridSearchCV` sobre los mismos bloques temporales.

# %%
regresion.afinamiento(c).mostrar()

# %% [markdown]
# ### Evaluación única en validación
#
# La regla la fijamos antes de mirar: Gana el menor RMSE de validación y, si la diferencia no llega al uno por
# ciento, el modelo más sencillo (lineal, árbol, KNN, perceptrón). Para saber qué tanto se puede mover la habilidad
# con otros días usamos un **bootstrap**, que repite el cálculo mil veces sobre días sacados al azar con reemplazo y
# se queda con el rango donde cae el 95 % central de los resultados.

# %%
regresion.evaluacion_validacion(c).mostrar()

# %%
regresion.diagnostico(c).mostrar()

# %% [markdown]
# ## 5. Congelamiento del modelo
#
# Los hiperparámetros y la configuración quedan fijos y el modelo se reajusta con entrenamiento y validación, porque
# los días recientes describen mejor el nivel actual de la serie. La etapa 07 lo evalúa una sola vez en la prueba.

# %%
regresion.congelar(c).mostrar()

# %% [markdown]
# El modelo congelado queda en disco y desde ahí lo usa la etapa 07. Así se carga y se le piden predicciones. Eso
# sí, estos días de validación ya los vio el reajuste así que la tabla solo muestra la llamada y no mide nada: El
# error de verdad se mide en la prueba.

# %%
import joblib
from proyecto.base import rutas

modelo = joblib.load(rutas.DIR_ARTEFACTOS / "modelos" / "regresion.joblib")   # el que acabamos de congelar
dias = c.validacion.tail(5)
tabla = dias[[c.fecha, c.objetivo]].assign(predicción=modelo.predict(dias[c.numericas + c.categoricas]))
mostrar(tabla_html(tabla, "Modelo congelado: Muestras por día en los últimos días de validación", decimales=1))

# %% [markdown]
# Antes de cerrar volvimos al L16 para ver qué habría pasado con otra combinación de factores. El panel lo responde
# sin volver a correr nada, con lo que guardamos de las 16 corridas. Si la combinación
# que se elige está en el L16 se puede comparar lo que predice el modelo aditivo con lo que de verdad se midió.

# %%
from proyecto.etapas.comun import legible
from proyecto.reporte import interactivo as ui

congelada = {f: legible(v) for f, v in c.configuracion_final.items()}   # niveles de la configuración congelada
mostrar(ui.panel_taguchi(c.taguchi, congelada))

# %% [markdown]
# ## 6. Qué significa en la operación
#
# Al final traducimos el error a muestras por día y a cuántos días la predicción se aleja más de una tolerancia
# práctica, que fijamos en el 20 % de la carga diaria media de validación.

# %%
regresion.operacion(c).mostrar()

# %%
regresion.terminar(c)
