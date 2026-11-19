# %% [markdown]
# # Etapa 05: Perfiles de muestra y segmentos de clientes
#
# Aquí no hay una respuesta correcta que predecir. Lo que buscamos es si las muestras (y en un segundo análisis los
# clientes) se dividen en grupos con un perfil que se pueda reconocer y que le sirva a la operación. Usamos los dos
# métodos del curso, K-means y DBSCAN, sobre tablas armadas solo con fechas de desarrollo.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import agrupamiento

iniciar_etapa("05_agrupamiento")
c = agrupamiento.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# ## 1. Qué agrupamos y con qué reglas
#
# El cribado eligió un análisis con su regla fija. Además conservamos uno complementario por fuera de la regla porque
# al laboratorio le interesan sus clientes. Las reglas de decisión las fijamos antes de ver resultados para que
# ningún número de grupos se acomode a lo que queríamos ver.

# %%
agrupamiento.reglas(c).mostrar()

# %% [markdown]
# ## 2. Análisis seleccionado: Perfiles de muestra
#
# ### 2.1 Preparación: Por qué escalar
#
# K-means y DBSCAN deciden con distancias euclidianas, que suman diferencias en unidades distintas (mililitros,
# horas, días, UFC). Sin escalar, la variable de mayor varianza decide casi sola quién está cerca de quién. El
# estadístico de **Hopkins** compara la distancia entre vecinos reales con la de puntos puestos al azar: Por encima
# de 0.5 hay tendencia a formar grupos, aunque no dice cuántos.

# %%
agrupamiento.preparacion(c, "seleccionado").mostrar()

# %% [markdown]
# ### 2.2 K-means: Número de grupos
#
# K-means pone ${\color{#C46A2B}{k}}$ centros, le asigna cada muestra al centro más cercano, mueve cada centro al
# promedio de sus muestras y repite hasta que nada cambia. La **inercia** es lo que busca bajar, la suma de las
# distancias al cuadrado de cada muestra a su centro ${\color{#C46A2B}{\mu_j}}$:
#
# $$W = \sum_{j=1}^{k}\;\sum_{i \in C_j} \big\lVert x_i - {\color{#C46A2B}{\mu_j}} \big\rVert^2$$
#
# La inercia siempre baja al aumentar k así que el codo solo orienta. La elección la hacemos con la **silueta**, que
# compara para cada muestra la distancia media a su propio grupo ${\color{#B8423F}{a(i)}}$ con la distancia media al
# grupo vecino más cercano ${\color{#2F8A4E}{b(i)}}$. Se acerca a uno si la muestra está bien metida en su grupo, vale
# cero si queda en la frontera y es negativa si estaría mejor en el otro. A diferencia de la inercia, sí castiga
# partir un grupo que no había que partir.
#
# $$s(i) = \frac{{\color{#2F8A4E}{b(i)}} - {\color{#B8423F}{a(i)}}}{\max\{{\color{#B8423F}{a(i)}},\, {\color{#2F8A4E}{b(i)}}\}}$$

# %%
agrupamiento.kmeans(c, "seleccionado").mostrar()

# %% [markdown]
# Así converge K-means con el k elegido sobre una muestra fija de los perfiles, dibujada en las dos componentes
# principales. Cada cuadro es una vuelta de asignar y promediar. Las líneas son el camino de cada centro:

# %%
from proyecto.modelos.agrupamiento import pasos_kmeans
from proyecto.reporte import graficos_presentacion as g

p = agrupamiento.datos_paneles(c)   # muestra fija de los perfiles y su proyección en dos componentes
centros, etiquetas = pasos_kmeans(p.kmeans_muestra, p.kmeans["k"])   # K-means desde cero, vuelta por vuelta
plano = (p.kmeans_muestra - p.kmeans_media) @ p.kmeans_componentes.T
mostrar(g.animacion_kmeans(plano, etiquetas, (centros - p.kmeans_media) @ p.kmeans_componentes.T))

# %% [markdown]
# Con el panel se puede probar otro k sobre la misma muestra y ver cómo cambian la nube y la silueta (necesita el
# núcleo en marcha):

# %%
from proyecto.reporte import interactivo as ui

mostrar(ui.panel_kmeans(p))   # número de grupos k: silueta y nube en dos componentes

# %% [markdown]
# ### 2.3 DBSCAN: Grupos por densidad
#
# DBSCAN recibe un radio (eps) y un mínimo de vecinos (min_samples). Un grupo es una zona donde cada punto tiene
# suficientes vecinos dentro del radio y los puntos que no los tienen quedan como ruido. La curva de k-distancias
# orienta en qué franja buscar eps. Los puntos de ruido no son errores: Son los casos que vale la pena revisar a mano.

# %%
agrupamiento.dbscan(c, "seleccionado").mostrar()

# %% [markdown]
# ### 2.4 K-means frente a DBSCAN
#
# El índice de Rand ajustado (ARI) vale 1 si dos particiones coinciden y cerca de 0 si se parecen tanto como dos
# particiones al azar. La proyección en dos componentes principales solo sirve para dibujar.

# %%
agrupamiento.comparar_particiones(c, "seleccionado").mostrar()

# %% [markdown]
# ### 2.5 Perfiles: De grupos numerados a perfiles con nombre
#
# El perfil lo calculamos sobre las variables originales y cada media la expresamos en desviaciones estándar respecto
# a la general, así que cada grupo se puede nombrar por las variables en que más se aparta. Los nombres salen del
# código y no de una lectura a ojo.

# %%
agrupamiento.perfiles(c, "seleccionado").mostrar()

# %% [markdown]
# ## 3. Análisis complementario: Segmentos de clientes
#
# No ganó la regla de cribado pero responde la pregunta de negocio más cercana a una decisión comercial, qué tipos
# de cliente atiende el laboratorio. Seguimos exactamente las mismas reglas. Con pocas filas y muchas variables la
# densidad se diluye así que esperábamos que DBSCAN no lograra separar grupos que K-means sí divide.

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
# La simulación le asignó a cada cliente un segmento latente que el laboratorio real no conocería. Solo por eso
# podemos revisar si los grupos que encontramos lo recuperan.

# %%
agrupamiento.auditoria(c).mostrar()

# %% [markdown]
# ## 4. Silueta del cribado frente a la silueta de esta etapa
#
# Las dos cuentas no usan las mismas filas (el cribado usa solo entrenamiento y aquí va todo el desarrollo) ni el
# mismo preprocesamiento (el cribado solo escala y aquí antes se aplica log1p). Para separar los dos efectos
# recalculamos la mejor silueta con las filas de esta etapa pero sin log1p.

# %%
agrupamiento.silueta_cribado(c).mostrar()

# %% [markdown]
# ## 5. Síntesis

# %%
agrupamiento.sintesis(c).mostrar()

# %%
agrupamiento.terminar(c)
