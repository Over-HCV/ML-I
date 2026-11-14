# %% [markdown]
# # 05 · Clasificación y Bayes: qué muestras vigilar y cuánto creerle a un positivo
#
# Dos partes que se apoyan una en la otra. En la parte A se modela, con los mismos tres niveles de la
# etapa 04, la probabilidad de que una muestra aceptada termine no conforme, vista desde la recepción,
# y se fija un umbral de alerta por costo. En la parte B se responde la pregunta original del
# laboratorio, la probabilidad de que un positivo sea falso, combinando la fiabilidad de cada ensayo
# (sus controles de calidad) con el riesgo previo de cada muestra que entrega el clasificador.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import clasificacion

iniciar_etapa("05_clasificacion_bayes")
c = clasificacion.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# ## Parte A · Clasificación
#
# ## 1. El problema, la decisión y sus costos
#
# El costo de cada error se fija antes de ver resultados, porque de él depende el umbral y no al revés.
# Con una clase positiva tan escasa la exactitud engaña, de forma que se usan el ROC-AUC (probabilidad
# de que una no conforme reciba más riesgo que una conforme), la precisión media (área bajo la curva
# precisión-recall, cuya referencia es la prevalencia) y el Brier (error cuadrático de la probabilidad,
# que premia la calibración), frente a un `Dummy` que predice la proporción del entrenamiento.

# %%
clasificacion.planteamiento(c).mostrar()

# %% [markdown]
# ## 2. Pasada didáctica: los modelos del curso
#
# Cada modelo usa el grupo ampliado y el preprocesamiento por defecto (imputación por la mediana,
# estandarización y one-hot), sin afinar, en los mismos bloques temporales.
#
# ### 2.1 Árbol de clasificación con poda por costo-complejidad

# %%
clasificacion.poda(c).mostrar()

# %% [markdown]
# ### 2.2 KNN, regresión logística, árbol podado y perceptrón, con sus curvas ROC

# %%
clasificacion.didactica(c).mostrar()

# %% [markdown]
# ### 2.3 Qué dice la regresión logística
#
# Con las numéricas estandarizadas, cada coeficiente es el cambio en el logaritmo de las odds de no
# conformidad cuando la variable sube una desviación estándar, y su exponencial es la razón de odds
# (1.5 significa odds un 50 % mayores).

# %%
clasificacion.logistica(c).mostrar()

# %% [markdown]
# ### 2.4 Las reglas del árbol podado
#
# Cada hoja es una conjunción de condiciones con la fracción de no conformes de las muestras de
# entrenamiento que caen en ella; se muestran las de mayor riesgo.

# %%
clasificacion.reglas_arbol(c).mostrar()

# %% [markdown]
# ## 3. Nivel 2: cribado de la tubería con un arreglo L16 de Taguchi
#
# Los factores son el modelo, el escalado, el grupo de variables, el tratamiento del desbalance
# (sobremuestreo de la clase minoritaria, solo al ajustar), la selección de variables y la
# codificación; las 128 combinaciones se reducen a 16 con la misma asignación de columnas de la etapa
# 04. La respuesta es el ROC-AUC de cada bloque, resumido con la S/N «mayor es mejor»,
# S/N = −10 · log10(media de 1 / AUC²).

# %%
clasificacion.diseno(c).mostrar()

# %%
clasificacion.corridas_l16(c).mostrar()

# %% [markdown]
# ### Comprobación dirigida: modelo × sobremuestreo
#
# El sobremuestreo no afecta igual a todos los modelos: en el perceptrón con parada temprana, copias
# de un mismo no conforme caen a la vez en su entrenamiento y en su validación interna; y en todos, al
# igualar las clases, las probabilidades dejan de estar calibradas. Para separar esto de las
# confusiones del L16, cada modelo didáctico se repite con sobremuestreo.

# %%
clasificacion.sobremuestreo(c).mostrar()

# %% [markdown]
# ### Corrida de confirmación y factorial completo

# %%
clasificacion.confirmacion(c).mostrar()

# %% [markdown]
# ## 4. Nivel 3: afinamiento de hiperparámetros
#
# Como en la etapa 04: la configuración de mayor S/N observada y la segunda mejor familia de modelos,
# con `GridSearchCV` en los mismos bloques y el ROC-AUC como métrica.

# %%
clasificacion.afinamiento(c).mostrar()

# %% [markdown]
# ### Evaluación única en validación
#
# Regla fijada antes de mirar: gana el mayor ROC-AUC de validación y, si la diferencia es menor que
# 0.005, el más sencillo (regresión logística, árbol, KNN, perceptrón).

# %%
clasificacion.evaluacion_validacion(c).mostrar()

# %% [markdown]
# ## 5. Calibración
#
# El umbral teórico y la parte B suponen que una probabilidad de 0.2 significa que unas veinte de
# cada cien muestras así salen no conformes. La versión calibrada usa `CalibratedClassifierCV` con el
# método sigmoide (Platt, dos parámetros, adecuado con pocos no conformes por tramo) y los mismos
# bloques temporales como `cv`. Regla: se usa la calibrada si la tubería sobremuestrea o si su Brier de
# validación es menor.

# %%
clasificacion.calibracion(c).mostrar()

# %% [markdown]
# ## 6. Umbral por costo en validación
#
# Se recorren umbrales de 0.02 a 0.50 (más el teórico) y se elige el de menor costo, desempatando
# hacia el más alto. Al lado del costo van siempre la precisión, el recall y las alertas por semana,
# que es la carga real que recibe el laboratorio.

# %%
clasificacion.umbral_por_costo(c).mostrar()

# %% [markdown]
# ## 7. Congelamiento del modelo
#
# Hiperparámetros, configuración, calibración y umbral quedan fijos; el modelo se reajusta con
# entrenamiento y validación (si va calibrado, con bloques temporales de todo el desarrollo).

# %%
clasificacion.congelar(c).mostrar()

# %% [markdown]
# ## Parte B · Bayes: ¿qué tan probable es que un positivo sea un falso positivo?
#
# ## 8. La idea en palabras y en fórmulas
#
# Un ensayo se describe con su **sensibilidad** s (fracción de contaminadas que da positivo, medida
# con controles positivos) y su **especificidad** e (fracción de limpias que da negativo, medida con
# controles negativos). Con la **prevalencia** π, el teorema de Bayes da el valor predictivo positivo:
#
# $$\text{VPP} = P(\text{contaminada} \mid +) = \frac{s\,\pi}{s\,\pi + (1-e)(1-\pi)}, \qquad
# P(\text{falso positivo} \mid +) = 1 - \text{VPP}.$$
#
# Con prevalencia baja, el término (1 − e)(1 − π) pesa mucho aunque 1 − e sea pequeño: con e = 0.98,
# π = 0.05 y s ≈ 0.95, más de una de cada cuatro alarmas es falsa. La prevalencia no se observa; se
# observa la tasa de positivos p, y despejando π de p = s π + (1 − e)(1 − π) se obtiene la corrección
# de Rogan–Gladen. Si una muestra se reensaya, el posterior del primer resultado hace de previa del
# segundo.

# %% [markdown]
# ## 9. Sensibilidad y especificidad de cada ensayo
#
# Solo controles anteriores al periodo de prueba; intervalos de Wilson al 95 %.

# %%
clasificacion.sensibilidad_especificidad(c).mostrar()

# %% [markdown]
# ## 10. VPP por ensayo
#
# La tasa de positivos sale de las muestras aceptadas del desarrollo. La incertidumbre se propaga
# con sorteos de las posteriores Beta de sensibilidad, especificidad y tasa de positivos (previa
# uniforme y sus conteos), con la semilla del proyecto.

# %%
clasificacion.vpp_por_ensayo(c).mostrar()

# %% [markdown]
# ## 11. VPP por muestra: la previa que aporta el clasificador
#
# El clasificador estima p(x), la tasa de positivos esperada para muestras como x. Aplicando
# Rogan–Gladen muestra a muestra se obtiene la previa π(x) y con ella VPP(x) = s · π(x) / p(x). Se
# usa el modelo de la parte A ajustado solo con entrenamiento, sobre las muestras de validación.
# π(x) = 0 significa que el positivo se explica entero por el error del ensayo.

# %%
clasificacion.vpp_por_muestra(c).mostrar()

# %% [markdown]
# ## 12. Posterior tras el reensayo
#
# Con dos positivos la confianza sube; con un positivo y un negativo, baja (suponiendo resultados
# independientes dada la condición real). El modelo predice además la fracción de reensayos
# conformes, P(− | + inicial) = VPP · (1 − s) + (1 − VPP) · e, que se compara con lo observado.

# %%
clasificacion.reensayo(c).mostrar()

# %% [markdown]
# ## 13. Auditoría de la simulación
#
# Solo porque los datos son simulados existe una verdad oculta (qué muestra estaba de verdad
# contaminada); con datos reales el reensayo es la única referencia, imperfecta.

# %%
clasificacion.auditoria(c).mostrar()

# %% [markdown]
# ## 14. Una regla simple para el reensayo
#
# En la etapa 03, «falso positivo entre los no conformes» quedó fuera porque una sola variable, la
# razón logarítmica entre el recuento y el límite, explicaba casi todo. Aquí se lee esa relación por
# tramos con los no conformes reensayados del desarrollo.

# %%
clasificacion.regla_reensayo(c).mostrar()

# %%
clasificacion.terminar(c)
