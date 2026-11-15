# %% [markdown]
# # Etapa 03: Qué vale la pena modelar a fondo
#
# Con la base del laboratorio se pueden plantear muchos análisis (prever volúmenes, anticipar resultados, agrupar
# clientes) y no nos daba el tiempo para estudiarlos todos a fondo. Así que los pasamos a todos por la misma prueba
# rápida y decidimos con una regla, escrita antes de ver resultados, cuáles pasan al modelado de las etapas 04 a 06.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import cribado

iniciar_etapa("03_cribado_analisis")
c = cribado.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# ## 1. El embudo de cribado
#
# La idea la tomamos del diseño de experimentos: Probar de forma barata y pareja muchas opciones y dejar el estudio
# costoso solo para las que pasan el filtro. Los cuatro pasos los escribimos antes de ver resultados:
#
# a. **Catálogo.** Cada candidato es una ficha con su unidad, su objetivo, el momento en que predice, su métrica,
#    su línea base, la decisión del laboratorio que apoya y un valor de negocio de 1 a 3 que le pusimos nosotros.
# b. **Prueba rápida.** Modelos sin afinar, entrenados solo con el entrenamiento y validados con los bloques de
#    ventana creciente de la etapa 02.
# c. **Regla de selección.** Unos umbrales fijos deciden quién es elegible y un puntaje decide quién gana en cada
#    tipo.
# d. **Modelado a fondo.** Solo los ganadores pasan a las etapas 04, 05 y 06.
#
# Para comparar candidatos tan distintos necesitábamos una sola escala, que llamamos **habilidad**: Qué parte del
# camino entre la línea base y la perfección recorre el modelo. En regresión es la fracción del error de la
# referencia que el modelo elimina (cero si empata con la referencia, uno si no se equivoca nunca) y en
# clasificación es el coeficiente de Gini, que sale del área bajo la curva ROC (AUC, la probabilidad de que el
# modelo le dé más riesgo a una no conforme que a una conforme tomadas al azar):
#
# $$\text{habilidad}_{\text{regresión}} = 1 - \frac{{\color{#B8423F}{\mathrm{RMSE}_{\text{modelo}}}}}{{\color{#3B64A8}{\mathrm{RMSE}_{\text{referencia}}}}}
# \qquad\qquad
# \text{habilidad}_{\text{clasificación}} = 2\,{\color{#2F8A4E}{\mathrm{AUC}}} - 1$$
#
# El RMSE es la raíz del error cuadrático medio, o sea el tamaño típico del error en las unidades del objetivo. En
# las series de tiempo la referencia es un pronóstico ingenuo (el promedio de los últimos periodos) porque ganarle a
# la media global sería demasiado fácil. El puntaje multiplica la habilidad por el valor de negocio
# ${\color{#C46A2B}{v}}$, de modo que un análisis muy útil para el laboratorio puede ganar con algo menos de
# habilidad:
#
# $$\text{puntaje} = \frac{\text{habilidad}\;{\color{#C46A2B}{v}}}{3}$$
#
# Las fichas de tipo Bayes no entrenan modelos sino que aplican el teorema de Bayes a los controles de calidad, así
# que no se criban y pasan directo a la etapa 05.

# %%
cribado.catalogo_y_regla(c).mostrar()

# %% [markdown]
# ## 2. Tablas de modelado y auditoría de fugas
#
# Cada ficha arma su tabla a partir de las tablas limpias, con una fila por unidad de análisis (un día, una
# muestra, un cliente), la fecha, el objetivo y las variables. La auditoría revisa que cada variable se conozca
# antes o en el momento de la predicción y que ningún identificador ni objetivo entre como variable. Después
# partimos cada tabla por fecha y la guardamos en dos archivos (desarrollo y prueba) junto con un JSON que dice el
# papel de cada columna. El bloque de prueba se guarda sin abrirlo y solo la etapa 07 puede leerlo.

# %%
cribado.tablas_modelado(c).mostrar()

# %% [markdown]
# ## 3. La prueba rápida
#
# A cada candidato supervisado lo probamos con su línea base, un modelo lineal o logístico y un árbol poco profundo
# sobre el grupo ampliado de variables. A cada candidato de agrupamiento, con el estadístico de Hopkins y la mejor
# silueta de K-means (las dos se explican en la sección 5). Todo con las filas del entrenamiento.

# %%
cribado.prueba_rapida(c).mostrar()

# %% [markdown]
# ## 4. ¿Basta una regla simple?
#
# Un modelo solo se justifica si combina información. Para comprobarlo entrenamos en cada bloque un árbol que ve una
# sola variable (la que más aporta) y, si su habilidad llega al umbral frente al mejor modelo rápido, lo que
# recomendaríamos es una regla operativa y no un modelo.

# %%
cribado.regla_simple(c).mostrar()

# %% [markdown]
# ## 5. Candidatos de agrupamiento
#
# Aquí no hay línea base que superar así que la prueba hace dos preguntas. La primera es si los datos tienden a
# formar grupos, y la responde el estadístico de **Hopkins**: Compara qué tan cerca quedan los puntos reales de sus
# vecinos frente a puntos puestos al azar y da cerca de 0.5 cuando no hay grupos y se acerca a uno cuando sí. La
# segunda es qué tan separados quedan los grupos de K-means y la responde la **silueta**, que va de menos uno a uno
# y es alta cuando cada punto está más cerca de los de su grupo que de los del grupo vecino.

# %%
cribado.candidatos_agrupamiento(c).mostrar()

# %% [markdown]
# ## 6. Selección
#
# El embudo resume cuántos candidatos quedan en cada paso. Las comparaciones muestran los mejores puntajes de cada
# tipo, incluidos los no elegibles, para que se vea qué dejamos de lado y por qué.

# %%
cribado.seleccion(c).mostrar()

# %% [markdown]
# Para ver con qué llega el clasificador elegido a la etapa 05 le preguntamos a su tabla qué momento declara, qué
# variables usa y si la auditoría de fugas encontró algo:

# %%
from proyecto.reporte.html import fichas
from proyecto.variables.comunes import auditar_momentos

tabla, papeles = c.tablas_modelado[c.seleccion["clasificacion"]]
hallazgos = auditar_momentos(tabla, papeles)   # lista vacía: ninguna variable se conoce después del momento
mostrar(fichas(papeles["numericas"] + papeles["categoricas"],
               f"Predice en \"{papeles['momento']}\" con estas variables ({len(hallazgos)} hallazgos de fuga)"))

# %% [markdown]
# ## 7. Lo que sigue
#
# La selección no se toca después: Las etapas 04 a 06 leen lo que quedó en `artefactos/03_cribado.json`. La única
# excepción, que dejamos declarada aquí, es un análisis de agrupamiento complementario sobre los clientes.

# %%
cribado.consecuencias(c).mostrar()

# %%
cribado.terminar(c)
