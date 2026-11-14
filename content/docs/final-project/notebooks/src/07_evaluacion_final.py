# %% [markdown]
# # 07 · Evaluación final: abrir la prueba una sola vez
#
# Las etapas 04, 05 y 06 dejaron modelos congelados (variables, hiperparámetros, calibración y
# umbral) sin mirar nunca el periodo de prueba. Aquí se abre ese periodo una sola vez, con
# `leer_prueba`, que deja constancia de cada lectura, y se mide lo que los modelos hacen con datos que
# nadie usó para decidir. Primero se muestra lo congelado, después se abre la prueba y al final se
# comparan las dos cosas, sin reajustar ni reelegir nada.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import evaluacion_final as final

iniciar_etapa("07_evaluacion_final")
c = final.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# ## 1. Lo que quedó congelado
#
# Desde aquí no se cambia nada: ni el modelo, ni las variables, ni los hiperparámetros, ni la
# calibración, ni el umbral.

# %%
final.congelado(c).mostrar()

# %% [markdown]
# ## 2. Apertura de la prueba
#
# Se leen tres bloques: la carga diaria (regresión), las muestras en recepción (clasificación y
# Bayes) y los no conformes reensayados (VPP y regla de reensayo). Cada lectura queda registrada en
# `artefactos/_etapas/lecturas_prueba.log`.

# %%
final.abrir_prueba(c).mostrar()

# %% [markdown]
# ## 3. Regresión: ¿cuántas muestras llegarán mañana?
#
# El modelo congelado predice cada día con la información de la víspera y se compara con las mismas
# referencias ingenuas de la etapa 04; la habilidad se mide frente a la misma referencia de entonces.

# %%
final.regresion(c).mostrar()

# %% [markdown]
# ### 3.1 Residuos, fin de año y operación
#
# La tolerancia práctica es la misma de la etapa 04 (en muestras por día), de forma que la prueba no
# mueve la vara con que se juzga.

# %%
final.regresion_diagnostico(c).mostrar()

# %% [markdown]
# ## 4. Clasificación: ¿qué muestras están en riesgo?
#
# Cada muestra de prueba recibe una probabilidad y la alerta se levanta con el umbral de validación.
# Al lado del costo van la precisión y las alertas por semana, porque un costo bajo con demasiadas
# falsas alarmas no sirve en recepción.

# %%
final.clasificacion(c).mostrar()

# %% [markdown]
# ### 4.1 Retrospectiva: el umbral que habría convenido
#
# Solo por transparencia: un umbral elegido en la prueba estaría ajustado a ella y su costo sería
# optimista. Esta celda no decide nada.

# %%
final.retrospectiva(c).mostrar()

# %% [markdown]
# ## 5. Bayes en la prueba: ¿qué tan probable es que un positivo sea falso?
#
# Con la sensibilidad y la especificidad del desarrollo y la probabilidad del clasificador como previa
# de cada muestra se calcula el VPP(x) de cada positivo; la referencia observable es el reensayo.

# %%
final.bayes_prueba(c).mostrar()

# %% [markdown]
# ### 5.1 Auditoría de la simulación
#
# Solo existe porque el generador guardó la verdad oculta; con datos reales no aplica.

# %%
final.auditoria(c).mostrar()

# %% [markdown]
# ### 5.2 La regla de reensayo en la prueba
#
# Se aplica el mismo borde fijado en la etapa 05, leído del artefacto, a los reensayos de prueba.

# %%
final.regla_reensayo(c).mostrar()

# %% [markdown]
# ## 6. Agrupamiento
#
# El agrupamiento no se evalúa en la prueba por diseño: no hay respuesta correcta con que comparar.
# Se resume lo que dejó la etapa 06.

# %%
final.agrupamiento(c).mostrar()

# %% [markdown]
# ## 7. Resumen para la presentación
#
# Las cifras son las de la prueba, salvo donde se indica otra cosa.

# %%
final.resumen(c).mostrar()

# %%
final.terminar(c)
