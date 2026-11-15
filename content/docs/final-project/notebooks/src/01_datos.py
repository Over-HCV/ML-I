# %% [markdown]
# # Etapa 01: El problema y la base de muestras
#
# En esta etapa contamos qué pregunta queríamos responder, qué columnas esperamos de la base del laboratorio
# (y en qué momento del proceso se conoce cada una) y dejamos guardadas las tablas crudas tal como llegan, sin
# limpiar nada todavía.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import datos

iniciar_etapa("01_datos")
c = datos.iniciar()
mostrar(c.encabezado, c.autores)

# %% [markdown]
# ## 1. El problema
#
# Un laboratorio de microbiología recibe todos los días muestras de producto terminado, materia prima, agua y
# muestreos en planta de clientes cosméticos, farmacéuticos, de alimentos, veterinarios y de centrales de mezclas.
# De cada muestra guarda quién la envió, cómo llegó, qué ensayo se le hizo, quién la procesó y qué resultado dio. El
# contexto es el de Rodam Análisis SAS en Bogotá, aunque lo que estudiamos es la base de muestras y no la empresa.
# Con esa base nos propusimos tres preguntas:
#
# a. **Demanda.** Cuántas muestras van a llegar, para programar analistas, medios de cultivo e incubadoras.
# b. **Riesgo.** Qué tan probable es que una muestra salga no conforme (fuera del límite que exige la norma) y, si
#    sale positiva, qué tan probable es que sea un falso positivo.
# c. **Grupos.** Qué clientes o muestras se parecen entre sí.
#
# De cada pregunta salían varios análisis posibles y no nos daba el tiempo para estudiarlos todos a fondo. Por eso
# armamos un catálogo de candidatos, los pasamos por la misma prueba rápida en la etapa 03 y solo a los ganadores
# los modelamos en serio.

# %% [markdown]
# ## 2. El contrato de datos
#
# Antes de tener datos escribimos un contrato (`proyecto/datos/esquema.py`), que es el único lugar del código
# donde aparecen los nombres de las columnas. De cada columna dice su tipo, su papel, si es obligatoria y el
# **momento** del proceso en que el dato pasa a conocerse (la toma, la llegada, la recepción, la lectura...). Si
# la exportación real no trae una columna opcional, los análisis que la usan simplemente se omiten.

# %%
datos.contrato(c).mostrar()

# %% [markdown]
# El momento es lo que más adelante nos protege de la fuga de información, que es usar para predecir un dato que
# todavía no existe en el instante de la predicción. Cada análisis declara en qué momento predice y solo puede
# usar las columnas conocidas hasta ahí. Así se lo preguntamos al contrato para una predicción en la recepción:

# %%
from proyecto.datos import esquema
from proyecto.reporte.html import fichas

al_recibir = esquema.columnas("muestras", hasta="recepcion")   # lo que ya se sabe cuando la muestra llega
despues = [x for x in esquema.columnas("muestras") if x not in al_recibir]
mostrar(fichas(al_recibir, f"{len(al_recibir)} columnas de muestras que se conocen al recibirla"),
        fichas(despues, f"{len(despues)} que llegan después y quedan prohibidas para esa predicción", color="rojo"))

# %% [markdown]
# ## 3. Las tablas crudas
#
# Obtenemos las tablas, las validamos contra el contrato (si falta una columna obligatoria la etapa se detiene y
# dice cuáles) y las guardamos tal como llegan, en CSV. Así la limpieza de la etapa 02 queda como un paso aparte
# que se puede auditar.

# %%
datos.tablas_crudas(c).mostrar()

# %% [markdown]
# Bastó con mirar las tablas para ver que hacía falta una etapa de limpieza: La misma categoría escrita de varias
# formas, recuentos reportados como texto, valores imposibles y filas repetidas.

# %%
datos.como_llegan(c).mostrar()

# %% [markdown]
# ## 4. Los datos simulados y la exportación real
#
# Daniel, del equipo, va a conseguir la exportación real del laboratorio, pero no la teníamos a tiempo para
# construir el proyecto. Mientras tanto trabajamos con una base simulada que reproduce el recorrido de cada muestra
# por el laboratorio, con la misma suciedad que esperamos encontrar. Las variables ocultas que usa el simulador (si
# la muestra estaba de verdad contaminada, qué tan higiénico es cada cliente) quedan aparte en `datos/auditoria/` y
# ningún modelo las lee. Cuando llegue la exportación bastaría copiarla en `datos/reales/`, describir en
# `configuracion/mapeo_columnas.json` cómo se llama cada columna y cada etiqueta y fijar `PROYECTO_ORIGEN=real`, sin
# tocar el código. Eso sí, todas las cifras de estos notebooks describen la base simulada y el método, no todavía
# al laboratorio.

# %%
datos.exportacion_real(c).mostrar()

# %% [markdown]
# ## 5. Protocolo de evaluación y mapa del proyecto
#
# Partimos la base por tiempo con cortes que fijamos en `config` antes de ver un solo resultado. El
# **entrenamiento** sirve para ajustar y comparar modelos (con validación cruzada de ventana creciente, que
# explicamos en la etapa 02), la **validación** para decidir (el umbral, el modelo que se congela) y la **prueba**
# se guarda aparte y solo se abre una vez, en la etapa 07. Las etapas se pasan la información únicamente por
# archivos en disco.

# %%
datos.protocolo(c).mostrar()

# %%
datos.terminar(c)
