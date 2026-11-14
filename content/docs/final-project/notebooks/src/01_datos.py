# %% [markdown]
# # 01 · Datos: el problema y el conjunto de datos
#
# Esta etapa plantea el problema, presenta el conjunto de datos (qué tablas y columnas se esperan y en
# qué momento del proceso se conoce cada una) y entrega las tablas crudas con las que trabaja el resto
# del proyecto, sin limpiar nada todavía.

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
# Un laboratorio de microbiología recibe a diario muestras de producto terminado, materia prima, agua
# y muestreos en planta de clientes de los sectores cosmético, farmacéutico, de alimentos, veterinario
# y de centrales de mezclas, y de cada muestra guarda quién la envió, en qué condiciones llegó, qué
# ensayo se le hizo, quién la procesó y cuál fue el resultado (contexto de empresa: Rodam Análisis
# SAS, Bogotá; el objeto de estudio es la base de muestras, no la empresa). Con esa base se plantean
# tres preguntas:
#
# - **Anticipar la demanda:** cuántas muestras van a llegar, para programar analistas, medios e
#   incubadoras.
# - **Estimar el riesgo de cada resultado:** qué tan probable es que una muestra resulte no conforme
#   y, si sale positiva, que se trate de un falso positivo.
# - **Entender los grupos:** qué clientes o muestras se parecen entre sí.
#
# De cada pregunta salen varios análisis posibles, de forma que el proyecto declara un catálogo de
# candidatos, los somete a la misma prueba rápida (etapa 03) y solo estudia a fondo a los ganadores.

# %% [markdown]
# ## 2. El contrato de datos
#
# El contrato (`proyecto/datos/esquema.py`) es la única fuente de verdad sobre los nombres de
# columna: de cada una declara su tipo, su papel, si es obligatoria y el **momento** del proceso en que
# el dato pasa a conocerse. Si la exportación real no trae una columna opcional, los análisis que la
# usan simplemente se omiten.
#
# El momento es lo que más adelante impide la fuga de información: cada análisis declara en qué
# momento se hace la predicción, y sus variables solo pueden ser columnas conocidas hasta ese momento.

# %%
datos.contrato(c).mostrar()

# %% [markdown]
# ## 3. Las tablas crudas
#
# Se obtienen las tablas, se validan contra el contrato (si falta una columna obligatoria la etapa se
# detiene con la lista completa) y se guardan tal como llegan, en CSV, de forma que la limpieza de la
# etapa 02 quede como un paso aparte y auditable.

# %%
datos.tablas_crudas(c).mostrar()

# %% [markdown]
# ### Cómo llegan los datos
#
# Basta mirar las tablas para ver por qué hace falta una etapa de limpieza: la misma categoría
# escrita de varias formas, recuentos reportados como texto, valores imposibles y filas repetidas.

# %%
datos.como_llegan(c).mostrar()

# %% [markdown]
# ## 4. De dónde salen los datos y cómo se conecta la exportación real
#
# La exportación real todavía no está disponible, de forma que mientras tanto las tablas salen de un
# simulador que reproduce el recorrido de cada muestra por el laboratorio (con la misma «suciedad»);
# las variables latentes que usa (si la muestra estaba de verdad contaminada, la higiene del cliente)
# se guardan aparte en `datos/auditoria/` y ningún modelo las lee. Para usar los datos reales basta
# copiar los archivos en `datos/reales/`, describir en `configuracion/mapeo_columnas.json` cómo se
# llama cada columna y cada etiqueta, y fijar `PROYECTO_ORIGEN=real`, sin tocar el código.

# %%
datos.exportacion_real(c).mostrar()

# %% [markdown]
# ## 5. Protocolo de evaluación y mapa del proyecto
#
# La partición es por tiempo, con cortes fijados en `config` antes de mirar resultados:
# **entrenamiento** para ajustar y comparar (validación cruzada de ventana creciente),
# **validación** para decidir (umbral, modelo que se congela) y **prueba**, que se guarda aparte y se
# abre una sola vez en la etapa 07. Las etapas solo se comunican por archivos en disco.

# %%
datos.protocolo(c).mostrar()

# %%
datos.terminar(c)
