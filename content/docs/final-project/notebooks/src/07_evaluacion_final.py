# %% [markdown]
# # Etapa 07: Abrir la prueba una sola vez
#
# Las etapas 04, 05 y 06 dejaron modelos congelados (variables, hiperparámetros, calibración y umbral) sin mirar
# nunca el periodo de prueba. Aquí abrimos ese periodo una sola vez, con `leer_prueba`, que deja constancia de cada
# lectura, y medimos lo que hacen los modelos con datos que nadie usó para decidir. Primero mostramos lo congelado,
# después abrimos la prueba y al final comparamos las dos cosas sin reajustar ni reelegir nada.

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
# Desde aquí no cambiamos nada: Ni el modelo, ni las variables, ni los hiperparámetros, ni la calibración, ni el
# umbral.

# %%
final.congelado(c).mostrar()

# %% [markdown]
# ## 2. Apertura de la prueba
#
# Leemos tres bloques: La carga diaria (regresión), las muestras en recepción (clasificación y Bayes) y los no
# conformes reensayados (VPP y regla de reensayo). Esta es la única celda del proyecto que toca la prueba y cada
# lectura queda escrita en `artefactos/_etapas/lecturas_prueba.log`:

# %%
from proyecto.base.artefactos import leer_prueba

c.prueba_reg = leer_prueba(c.nombre_reg)                 # carga diaria de la prueba
c.prueba_clf = leer_prueba(c.nombre_clf)                 # muestras recibidas en la prueba
c.prueba_reensayo = leer_prueba(final.NOMBRE_REENSAYO)   # no conformes reensayados en la prueba
final.abrir_prueba(c).mostrar()

# %% [markdown]
# Los modelos congelados se usan tal cual salieron de las etapas 04 y 05. El de la carga recibe las variables de la
# víspera y el clasificador da la probabilidad de no conformidad, que se compara con el umbral congelado. Así se ven
# los primeros días y las primeras muestras de la prueba:

# %%
from proyecto.reporte.html import tabla_html

dias = c.prueba_reg.head(5)                                                 # primeros días de la prueba
carga = c.modelo_reg.predict(dias[c.reg["variables_entrada"]])              # modelo congelado en la etapa 04
muestras = c.prueba_clf.head(5)                                             # primeras muestras de la prueba
riesgo = c.modelo_clf.predict_proba(muestras[c.clf["variables_entrada"]])[:, 1]   # congelado en la etapa 05
tabla_dias = dias[[c.papeles_reg["fecha"], c.papeles_reg["objetivo"]]].assign(predicción=carga)
tabla_muestras = muestras[["id_muestra", "ensayo"]].assign(riesgo=riesgo, alerta=riesgo >= c.umbral)
mostrar(tabla_html(tabla_dias, "Carga predicha", decimales=1), tabla_html(tabla_muestras, "Riesgo y alerta", decimales=3))

# %% [markdown]
# ## 3. Regresión: ¿Cuántas muestras van a llegar mañana?
#
# El modelo congelado predice cada día con la información de la víspera y lo comparamos con las mismas referencias
# ingenuas de la etapa 04. La habilidad se mide frente a la misma referencia de entonces.

# %%
final.regresion(c).mostrar()

# %% [markdown]
# ### 3.1 Residuos, fin de año y operación
#
# La tolerancia práctica es la misma de la etapa 04 (en muestras por día) para que la prueba no mueva la vara con que
# la juzgamos.

# %%
final.regresion_diagnostico(c).mostrar()

# %% [markdown]
# ## 4. Clasificación: ¿Qué muestras están en riesgo?
#
# Cada muestra de prueba recibe una probabilidad y la alerta se levanta con el umbral de validación. Al lado del costo
# van la precisión y las alertas por semana porque un costo bajo con demasiadas falsas alarmas no le sirve a la
# recepción.

# %%
final.clasificacion(c).mostrar()

# %% [markdown]
# ### 4.1 Retrospectiva: El umbral que habría convenido
#
# Esto va solo por transparencia. Un umbral elegido en la prueba estaría ajustado a ella y su costo sería optimista,
# así que esta celda no decide nada.

# %%
final.retrospectiva(c).mostrar()

# %% [markdown]
# ## 5. Bayes en la prueba: ¿Qué tan probable es que un positivo sea falso?
#
# Con la sensibilidad y la especificidad del desarrollo y la probabilidad del clasificador como previa de cada
# muestra calculamos el VPP(x) de cada positivo. La referencia que se puede observar es el reensayo.

# %%
final.bayes_prueba(c).mostrar()

# %% [markdown]
# ### 5.1 Auditoría de la simulación
#
# Solo existe porque el generador guardó la verdad oculta. Con datos reales no aplica.

# %%
final.auditoria(c).mostrar()

# %% [markdown]
# ### 5.2 La regla de reensayo en la prueba
#
# Aplicamos a los reensayos de la prueba el mismo borde que fijamos en la etapa 05, leído de su artefacto.

# %%
final.regla_reensayo(c).mostrar()

# %% [markdown]
# ## 6. Agrupamiento
#
# El agrupamiento no se evalúa en la prueba porque no hay una respuesta correcta con que compararlo. Resumimos lo
# que dejó la etapa 06.

# %%
final.agrupamiento(c).mostrar()

# %% [markdown]
# ## 7. Qué respondimos, qué tan bien y qué haríamos con los datos reales
#
# Cerramos con las cuatro preguntas del laboratorio. Para cada una va la respuesta, la cifra de la prueba que la
# respalda y lo que la limita. Después va lo que recomendaríamos, lo que no alcanzamos a resolver y los pasos para
# correr todo con la exportación real. Las cifras son de la prueba salvo donde se dice otra cosa.

# %%
final.resumen(c).mostrar()

# %%
final.terminar(c)
