# %% [markdown]
# # 03 · Cribado de análisis: qué vale la pena modelar a fondo
#
# Con la base del laboratorio se pueden plantear muchos análisis (prever volúmenes, anticipar
# resultados, agrupar clientes), y estudiarlos todos a fondo no es viable. En esta etapa se los somete
# a la misma prueba rápida y se decide, con una regla fijada de antemano, cuáles pasan al modelado
# detallado de las etapas 04 a 06.

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
# La idea viene del diseño de experimentos: probar de forma barata y uniforme muchas opciones y dar el
# estudio costoso solo a las que pasan el filtro. Los cuatro pasos se escriben antes de ver resultados:
#
# - **Catálogo:** cada candidato es una ficha con su unidad, objetivo, momento de predicción, métrica,
#   línea base, decisión de negocio que apoya y un valor de negocio de 1 a 3.
# - **Prueba rápida:** modelos sin afinar, solo con el entrenamiento y bloques temporales de ventana
#   creciente.
# - **Regla de selección:** umbrales fijos deciden quién es elegible y un puntaje (habilidad × valor
#   de negocio) decide quién gana en cada tipo.
# - **Modelado a fondo:** solo los ganadores pasan a las etapas 04, 05 y 06.
#
# La **habilidad** compara candidatos de naturaleza distinta: en regresión es 1 − RMSE / RMSE de la
# referencia (la fracción del error de la línea base que el modelo elimina), en clasificación es
# 2 · AUC − 1 (el coeficiente de Gini). En las series de tiempo la referencia es un pronóstico ingenuo
# (el promedio de los últimos periodos), porque superar a la media global sería demasiado fácil. Las
# fichas de tipo «bayes» no entrenan modelos: aplican el teorema de Bayes a los controles de calidad,
# así que no se criban y pasan directo a la etapa 05.

# %%
cribado.catalogo_y_regla(c).mostrar()

# %% [markdown]
# ## 2. Tablas de modelado y auditoría de fugas
#
# Cada ficha construye su tabla (una fila por unidad de análisis, con fecha, objetivo y variables) a
# partir de las tablas limpias. La auditoría comprueba que cada variable se conozca antes o en el
# momento de la predicción y que ningún identificador u objetivo entre como variable. Cada tabla se
# parte por fecha y se guarda en dos archivos separados (desarrollo y prueba) junto con un JSON de
# papeles de columnas; el bloque de prueba se guarda sin abrirlo y solo la etapa 07 puede leerlo.

# %%
cribado.tablas_modelado(c).mostrar()

# %% [markdown]
# ## 3. La prueba rápida
#
# Cada candidato supervisado se evalúa con la línea base, un modelo lineal o logístico y un árbol poco
# profundo sobre el grupo de variables ampliado; cada candidato de agrupamiento, con el estadístico de
# Hopkins y la mejor silueta de K-means. Todo con las filas del entrenamiento.

# %%
cribado.prueba_rapida(c).mostrar()

# %% [markdown]
# ## 4. ¿Basta una regla simple?
#
# Un modelo solo se justifica si combina información: en cada bloque se entrena un árbol que ve una
# sola variable (la que más aporta) y, si su habilidad llega al umbral frente al mejor modelo rápido,
# la recomendación es una regla operativa y no un modelo.

# %%
cribado.regla_simple(c).mostrar()

# %% [markdown]
# ## 5. Candidatos de agrupamiento
#
# Sin línea base que superar, la prueba pregunta si hay tendencia a formar grupos (Hopkins: cerca de
# 0.5 sin estructura, hacia 1 con grupos) y qué tan separados quedan los grupos de K-means (silueta).

# %%
cribado.candidatos_agrupamiento(c).mostrar()

# %% [markdown]
# ## 6. Selección
#
# El embudo resume cuántos candidatos quedan en cada paso; las comparaciones muestran los mejores
# puntajes de cada tipo, incluidos los no elegibles, para que se vea qué se dejó de lado y por qué.

# %%
cribado.seleccion(c).mostrar()

# %% [markdown]
# ## 7. Consecuencias para las etapas siguientes
#
# La selección no se toca después: las etapas 04 a 06 leen `artefactos/03_cribado.json`. La única
# excepción, declarada aquí, es un análisis de agrupamiento complementario sobre los clientes.

# %%
cribado.consecuencias(c).mostrar()

# %%
cribado.terminar(c)
