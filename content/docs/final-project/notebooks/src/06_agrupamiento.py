# %% [markdown]
# # 06 · Agrupamiento: perfiles de muestra y segmentos de clientes
#
# Aquí no hay una respuesta correcta que predecir: se busca si las muestras (y, en un segundo
# análisis, los clientes) se reparten en grupos con un perfil reconocible y útil para la operación.
# Se usan los dos métodos del curso, K-means y DBSCAN, sobre tablas construidas solo con fechas de
# desarrollo.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import agrupamiento

iniciar_etapa("06_agrupamiento")
c = agrupamiento.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# ## 1. Qué se agrupa y con qué reglas
#
# El cribado eligió un análisis con su regla fija; además se conserva uno complementario, fuera de la
# regla, porque al laboratorio le interesan sus clientes. Las reglas de decisión se fijan antes de ver
# resultados, para que ningún número de grupos se acomode a lo que se quiere ver. Los datos son
# simulados: los grupos describen el mecanismo de la simulación y sus lecturas son un ejercicio de
# método.

# %%
agrupamiento.reglas(c).mostrar()

# %% [markdown]
# ## 2. Análisis seleccionado: perfiles de muestra
#
# ### 2.1 Preparación: por qué escalar
#
# K-means y DBSCAN deciden con distancias euclidianas, que suman diferencias en unidades distintas
# (mililitros, horas, días, UFC): sin escalar, la variable de mayor varianza decide casi sola quién
# está cerca de quién. El estadístico de Hopkins compara la distancia entre vecinos reales con la de
# puntos al azar (por encima de 0.5 hay tendencia a agruparse, sin decir cuántos grupos).

# %%
agrupamiento.preparacion(c, "seleccionado").mostrar()

# %% [markdown]
# ### 2.2 K-means: número de grupos
#
# La inercia siempre baja al aumentar k, así que el codo solo orienta; la elección se apoya en la
# silueta, que sí penaliza partir un grupo que no debía partirse.

# %%
agrupamiento.kmeans(c, "seleccionado").mostrar()

# %% [markdown]
# ### 2.3 DBSCAN: grupos por densidad
#
# DBSCAN recibe un radio (eps) y un mínimo de vecinos (min_samples) y deja como ruido los puntos sin
# suficientes vecinos cerca. La curva de k-distancias orienta la franja de eps. Los puntos de ruido no
# son errores: son los casos que vale la pena revisar a mano.

# %%
agrupamiento.dbscan(c, "seleccionado").mostrar()

# %% [markdown]
# ### 2.4 K-means frente a DBSCAN
#
# El índice de Rand ajustado (ARI) vale 1 si dos particiones coinciden y cerca de 0 si se parecen
# como dos particiones al azar. La proyección en dos componentes principales solo sirve para dibujar.

# %%
agrupamiento.comparar_particiones(c, "seleccionado").mostrar()

# %% [markdown]
# ### 2.5 Perfiles: de grupos numerados a perfiles con nombre
#
# El perfil se calcula sobre las variables originales, y cada media se expresa en desviaciones
# estándar respecto a la general, lo que permite nombrar el grupo por las variables en que más se
# aparta. Los nombres salen del código, no de una lectura a ojo.

# %%
agrupamiento.perfiles(c, "seleccionado").mostrar()

# %% [markdown]
# ## 3. Análisis complementario: segmentos de clientes
#
# No ganó la regla de cribado, pero responde la pregunta de negocio más cercana a una decisión
# comercial (qué tipos de cliente atiende el laboratorio). Se siguen exactamente las mismas reglas.
# Con pocas filas y muchas variables la densidad se diluye, y es esperable que DBSCAN no logre separar
# grupos que K-means sí reparte.

# %%
agrupamiento.preparacion(c, "complementario").mostrar()

# %%
agrupamiento.kmeans(c, "complementario").mostrar()

# %%
agrupamiento.dbscan(c, "complementario").mostrar()

# %%
agrupamiento.comparar_particiones(c, "complementario").mostrar()

# %% [markdown]
# ### Segmentos de clientes y su lectura comercial

# %%
agrupamiento.perfiles(c, "complementario").mostrar()

# %% [markdown]
# ### Auditoría con la simulación
#
# La simulación asignó a cada cliente un segmento latente que el laboratorio real no conocería; solo
# por eso se puede revisar si los grupos encontrados lo recuperan.

# %%
agrupamiento.auditoria(c).mostrar()

# %% [markdown]
# ## 4. Silueta del cribado frente a la silueta de esta etapa
#
# Las dos cuentas difieren en las filas (el cribado usa solo entrenamiento; aquí, todo el desarrollo)
# y en el preprocesamiento (el cribado solo escala; aquí se aplica antes log1p). Para separar los dos
# efectos se recalcula la mejor silueta con las filas de esta etapa pero sin log1p.

# %%
agrupamiento.silueta_cribado(c).mostrar()

# %% [markdown]
# ## 5. Síntesis

# %%
agrupamiento.sintesis(c).mostrar()

# %%
agrupamiento.terminar(c)
