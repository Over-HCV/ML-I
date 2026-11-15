# %% [markdown]
# # Etapa 02: Limpieza, integración y exploración
#
# Aquí leemos las tablas crudas, las limpiamos con reglas fijas dejando constancia de cada corrección, las juntamos
# en una sola tabla por muestra y exploramos el periodo de desarrollo (entrenamiento y validación) sin mirar el
# bloque de prueba.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import exploracion

iniciar_etapa("02_exploracion")
c = exploracion.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# ## 1. Limpieza con auditoría
#
# La limpieza convierte tipos y fechas, unifica cómo se escriben las etiquetas, quita los duplicados exactos,
# vuelve faltantes los valores imposibles, interpreta el recuento reportado (un número, una cota como "<10" o
# presencia y ausencia) y corrige los resultados que contradicen su recuento. Cada problema queda en una tabla de
# auditoría con las filas afectadas y lo que hicimos con ellas.
#
# La corrimos sobre la tabla completa (antes de partirla por tiempo) porque sus reglas están escritas en el código
# y no aprenden nada de los datos. Lo que sí se aprende de los datos (imputación, escalado, codificación) lo dejamos
# dentro de las tuberías de cada modelo, ajustado solo con entrenamiento.

# %%
exploracion.limpieza_auditada(c).mostrar()

# %% [markdown]
# ## 2. Partición temporal
#
# Cada muestra cae en un bloque según su fecha de recepción. De aquí en adelante **toda la exploración usa solo el
# periodo de desarrollo** y del bloque de prueba únicamente contamos filas.

# %%
exploracion.particion_temporal(c).mostrar()

# %% [markdown]
# Dentro del entrenamiento tampoco podíamos validar mezclando fechas al azar porque eso deja que el modelo aprenda
# de semanas posteriores a las que luego predice. Por eso usamos una validación cruzada de **ventana creciente**:
# En cada bloque el modelo se ajusta con todo el pasado y se valida con el tramo que sigue y en el siguiente bloque
# ese tramo pasa a ser parte del pasado. La función `bloques_temporales` arma esos cortes y la animación los
# muestra uno por uno:

# %%
from proyecto.base import config
from proyecto.evaluacion import validacion
from proyecto.reporte.graficos_presentacion import animacion_bloques

entrenamiento = c.desarrollo[c.desarrollo["fecha_recepcion"] < config.CORTE_VALIDACION]
fechas = entrenamiento["fecha_recepcion"].reset_index(drop=True)
bloques = validacion.bloques_temporales(fechas)   # pares (ajustar, validar) que scikit-learn recibe como cv=
mostrar(animacion_bloques(fechas, bloques))

# %% [markdown]
# ## 3. Faltantes
#
# Separamos dos clases de faltantes. Los **estructurales** son datos que no pueden existir para esa fila (una
# muestra rechazada no tiene resultado) y los **no estructurales** son datos que debieron registrarse y no se
# registraron. Los primeros no los imputamos como si fueran olvidos y los segundos se imputan dentro de la tubería
# de cada modelo.

# %%
exploracion.faltantes(c).mostrar()

# %% [markdown]
# ## 4. Volumen y estacionalidad
#
# La primera pregunta es cuántas muestras van a llegar así que empezamos por el volumen semanal de cada línea de
# negocio y por el patrón dentro del año.

# %%
exploracion.volumen(c).mostrar()

# %% [markdown]
# ## 5. Resultado no conforme
#
# Una muestra es **no conforme** cuando su recuento supera el límite que exige la norma para ese producto. Miramos
# qué tan frecuente es y cómo cambia por sector, matriz (el tipo de producto) y ensayo. Después miramos cómo se
# relaciona con las variables numéricas que ya se conocen cuando la muestra llega.

# %%
exploracion.tasa_no_conforme(c).mostrar()

# %%
exploracion.distribuciones(c).mostrar()

# %% [markdown]
# ## 6. Tiempos de entrega
#
# El tiempo entre la recepción y el informe depende sobre todo de cuántos días incuba cada ensayo.

# %%
exploracion.entrega(c).mostrar()

# %% [markdown]
# ## 7. Estabilidad en el tiempo
#
# Un modelo ajustado con el pasado solo sirve si el futuro se le parece así que nos preguntamos si la tasa de no
# conformes se mantiene mes a mes o si deriva (cambia de nivel con el tiempo).

# %%
exploracion.estabilidad(c).mostrar()

# %% [markdown]
# ## 8. Correlaciones entre variables numéricas
#
# Usamos la correlación de Spearman, que mide si dos variables suben y bajan juntas aunque no sea en línea recta
# (trabaja con el orden de los valores y no con los valores mismos). Va de menos uno a uno y la escribimos ρ. Con
# ella buscamos variables redundantes y variables asociadas con los resultados.

# %%
exploracion.correlaciones(c).mostrar()

# %% [markdown]
# ## 9. Concentración del volumen por cliente
#
# De cómo se distribuye el volumen entre clientes depende cuánta historia tiene cada uno y qué tan confiables son
# las variables que construimos con su pasado.

# %%
exploracion.concentracion(c).mostrar()

# %% [markdown]
# ## 10. Lo que esto implica para el modelado

# %%
exploracion.implicaciones(c).mostrar()

# %%
exploracion.terminar(c)
