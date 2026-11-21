# %% [markdown]
# # Etapa 03: Qué vale la pena modelar a fondo y cuántas muestras van a llegar
#
# Este notebook está ordenado para la exposición. Tiene dos tramos y en cada uno van primero las ideas que decimos
# (en los recuadros) y debajo la evidencia técnica de cada afirmación. El detalle completo de la etapa (todas las
# tablas, el diseño de Taguchi por dentro y los paneles) queda en el **anexo** al final y no se expone.
#
# | Tramo | Pregunta | Tiempo |
# |---|---|---|
# | A | Qué vale la pena modelar a fondo | 0:45 |
# | B | Cuántas muestras llegan mañana | 1:15 |
#
# Esta etapa y la siguiente trabajan sobre la base simulada, que tiene la misma estructura que la base real del
# notebook 02. Así podemos aplicar técnicas que la exportación real no permite, por ejemplo con los controles de
# calidad que usa la etapa 04.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import cribado

iniciar_etapa("03_cribado_regresion")
c = cribado.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# # Tramo A: Qué vale la pena modelar a fondo
#
# ## A.1 El problema
#
# > **Lo que digo.** Con la base del laboratorio se podían plantear **16 preguntas**: Cuántas muestras llegan,
# > cuáles salen mal, cuánto tarda el informe, qué clientes dejan de enviar... No nos daba el tiempo para estudiarlas
# > todas a fondo así que tuvimos que escoger.
# >
# > **Muestro:** El contador de candidatos (5 de regresión, 5 de clasificación, 3 de agrupamiento y 3 de Bayes).

# %%
catalogo = cribado.catalogo_y_regla(c)
tablas = cribado.tablas_modelado(c)   # una tabla por pregunta, con su auditoría de fugas (anexo)
catalogo.mostrar(1)

# %% [markdown]
# ## A.2 Qué hicimos: Una regla escrita antes de mirar
#
# > **Lo que digo.** Antes de ver un solo resultado escribimos una regla. Cada pregunta pasa por una **prueba rápida**
# > con modelos sin afinar y se compara contra una **referencia** que no usa modelo. Para seguir tiene que cumplir tres
# > cosas:
# >
# > 1. Ganarle a la referencia en **los 4 periodos** de validación y no solo en promedio.
# > 2. Que **no baste una sola variable**: Si un árbol con una variable logra el 90 % de lo que logra el modelo, sobra
# >    el modelo.
# > 3. Tener el **mayor puntaje** de su tipo, que combina qué tan bien predice con cuánto le sirve al laboratorio
# >    (un valor de 1 a 3 que pusimos nosotros).
# >
# > **Si preguntan cómo se comparan preguntas tan distintas:** Con la **habilidad**, la fracción del error de la
# > referencia que el modelo elimina (0 si empata con ella, 1 si no se equivoca nunca). En clasificación es 2 AUC − 1.
#
# $$\text{habilidad}_{\text{regresión}} = 1 - \frac{{\color{#B8423F}{\mathrm{RMSE}_{\text{modelo}}}}}{{\color{#3B64A8}{\mathrm{RMSE}_{\text{referencia}}}}}
# \qquad
# \text{habilidad}_{\text{clasificación}} = 2\,{\color{#2F8A4E}{\mathrm{AUC}}} - 1
# \qquad
# \text{puntaje} = \frac{\text{habilidad}\;{\color{#C46A2B}{v}}}{3}$$

# %%
prueba = cribado.prueba_rapida(c)
regla = cribado.regla_simple(c)
agrupamiento = cribado.candidatos_agrupamiento(c)
catalogo.mostrar(2)   # la regla tal como quedó escrita en cribado/analisis.py

# %% [markdown]
# ## A.3 Resultado: De 16 preguntas a 3
#
# > **Lo que digo.** De 16 candidatos quedan **3**: La **carga diaria** (regresión, la vemos ahora en el tramo B), la
# > **muestra no conforme vista desde la recepción** (clasificación, etapa 04) y los **perfiles de muestra**
# > (agrupamiento, etapa 05). Los 3 de Bayes pasan directo porque son cálculos y no modelos.
# >
# > **Muestro:** El embudo 16 → 13 → 10 → 3.

# %%
seleccion = cribado.seleccion(c)
seleccion.mostrar(0)

# %% [markdown]
# ## A.4 Lo curioso: La mayor habilidad no siempre gana
#
# > **Lo que digo.** La pregunta que mejor predecía era "¿este no conforme es falso?" (habilidad **0.689**, un AUC
# > de **0.845**) y **quedó fuera**. Una sola variable, **cuánto pasa el recuento del límite**, lograba el **95 %** de lo
# > que lograba el modelo. Ahí sobra el modelo y basta una regla, que retomamos al final de la etapa 04.
# >
# > **Muestro:** Las tres barras naranjas pasan la línea roja del 90 % y por eso salen.

# %%
regla.mostrar(0, 2)

# %%
consecuencias = cribado.consecuencias(c)
contexto_cribado = c   # el anexo lo vuelve a usar
cribado.terminar(c)    # cierra la parte A; la etapa sigue abierta para la parte B

# %% [markdown]
# # Tramo B: Cuántas muestras llegan mañana
#
# ## B.1 El problema
#
# > **Lo que digo.** Queremos saber **la víspera** cuántas muestras van a llegar al día siguiente, para ajustar los
# > turnos de recepción y siembra y preparar medios de cultivo e incubadoras. Solo usamos lo que se sabe la víspera:
# > Los días anteriores, el calendario y lo que ya está agendado.
# >
# > La vara para medirnos es lo que el laboratorio ya puede hacer sin modelo: **Promediar los últimos 5 días**.
# >
# > **Muestro:** La ficha del planteamiento y los días de entrenamiento (589) y de validación (125).

# %%
from proyecto.etapas import regresion

c = regresion.iniciar()   # la misma etapa: sus cifras se suman a las de la parte A
planteamiento = regresion.planteamiento(c)
referencias = regresion.referencias(c)
planteamiento.mostrar()

# %% [markdown]
# ## B.2 Qué hicimos: Tres niveles
#
# > **Lo que digo.**
# >
# > 1. **Los modelos del curso tal como vienen:** Recta simple, regresión múltiple, KNN, árbol y red neuronal. Ya aquí
# >    gana la regresión lineal múltiple (elimina el **29 %** del error de la referencia) y la red neuronal no la
# >    mejora.
# > 2. **Las decisiones de la receta:** Probamos seis decisiones (el modelo, el escalado, qué variables entran, si se
# >    usa logaritmo, si se seleccionan variables y cómo se codifican). Todas las combinaciones eran **128 corridas** y
# >    un diseño de **Taguchi L16** las redujo a **16**. Para comprobar el atajo corrimos también las 128: El orden de
# >    importancia de los factores salió casi igual (correlación **0.94**) con solo **16 de las 128 corridas**.
# > 3. **El ajuste fino** de los dos mejores modelos y una sola evaluación en validación.
# >
# > **Muestro:** La tabla de la pasada didáctica (columna Habilidad) y después la figura del L16 contra las 128.

# %%
recta = regresion.recta_simple(c)
poda = regresion.poda(c)
didactica = regresion.didactica(c)
coeficientes = regresion.coeficientes(c)
didactica.mostrar(0)

# %%
diseno = regresion.diseno(c)
corridas = regresion.corridas_l16(c)
confirmacion = regresion.confirmacion(c)
afinamiento = regresion.afinamiento(c)
confirmacion.mostrar(2, 3)

# %% [markdown]
# ## B.3 Resultado
#
# > **Lo que digo.** Gana la **regresión lineal**, el modelo más sencillo. Promediar los últimos 5 días se equivoca en
# > unas **15 muestras por día** y el modelo baja ese error a **11**, en días que traen unas **54 muestras**. Los días
# > mal planificados (error mayor que el 20 % de la carga) bajan del **56 %** al **42 %**. La mejora no es suerte: Su
# > intervalo bootstrap (habilidad de 0.154 a 0.343) no toca el cero. Sirve para fijar la dotación base del día
# > siguiente, pero con margen.
# >
# > **Muestro:** Las tarjetas de días fuera de tolerancia y la serie de validación, donde la línea azul (el modelo)
# > sigue los picos mejor que la naranja (el promedio de 5 días).

# %%
evaluacion = regresion.evaluacion_validacion(c)
diagnostico = regresion.diagnostico(c)
congelado = regresion.congelar(c)
operacion = regresion.operacion(c)
operacion.mostrar(0)
diagnostico.mostrar(0)

# %% [markdown]
# ---
#
# # Anexo: Detalle técnico de la etapa (no se expone)
#
# Aquí queda todo lo que la etapa calculó y no mostramos en los tramos: Las tablas completas, el diseño de Taguchi por
# dentro y los paneles. Nada se vuelve a calcular, son los mismos resultados de arriba.
#
# ## Tramo A
#
# ### Catálogo, líneas base y auditoría de fugas
#
# Cada candidato es una ficha con su unidad, su objetivo, el momento en que predice, su métrica, su línea base y la
# decisión que apoya. La auditoría revisa que cada variable se conozca antes o en el momento de la predicción. El
# bloque de prueba se guarda sin abrirlo y solo la etapa 06 lo lee.

# %%
catalogo.mostrar_sin(1, 2)
tablas.mostrar()

# %% [markdown]
# ### Prueba rápida, regla simple y agrupamiento
#
# En agrupamiento no hay referencia que superar. El estadístico de **Hopkins** dice si los datos tienden a formar
# grupos (0.5 sin grupos, cerca de 1 con grupos) y la **silueta** dice qué tan separados quedan los grupos de
# K-means (de −1 a 1).

# %%
prueba.mostrar()
regla.mostrar(1)
agrupamiento.mostrar()

# %% [markdown]
# ### Selección completa y lo que sigue

# %%
seleccion.mostrar_sin(0)
consecuencias.mostrar()

# %% [markdown]
# Las variables con que llega el clasificador elegido a la etapa 04:

# %%
from proyecto.reporte.html import fichas
from proyecto.variables.comunes import auditar_momentos

tabla, papeles = contexto_cribado.tablas_modelado[contexto_cribado.seleccion["clasificacion"]]
hallazgos = auditar_momentos(tabla, papeles)   # lista vacía: ninguna variable se conoce después del momento
mostrar(fichas(papeles["numericas"] + papeles["categoricas"],
               f"Predice en \"{papeles['momento']}\" con estas variables ({len(hallazgos)} hallazgos de fuga)"))

# %% [markdown]
# ## Tramo B
#
# ### Referencias ingenuas y RMSE
#
# El RMSE es la raíz del promedio de los errores al cuadrado, se lee en muestras por día y castiga más los errores
# grandes:
#
# $$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}\big(y_t - {\color{#3B64A8}{\hat y_t}}\big)^2}$$

# %%
referencias.mostrar()

# %% [markdown]
# ### Nivel 1: Los modelos del curso
#
# La recta simple usa la variable más correlacionada con el objetivo. El árbol se poda por costo-complejidad
# (`ccp_alpha` cobra un precio por hoja). La regresión múltiple predice la carga como una suma ponderada:
#
# $$\hat y_{t+1} = {\color{#3B64A8}{\beta_0}} + \sum_{j=1}^{p} {\color{#3B64A8}{\beta_j}}\, x_{t,j}$$
#
# Todos van dentro de una tubería que imputa, escala y codifica con lo aprendido solo en el tramo de ajuste de cada
# bloque:

# %%
from proyecto.modelos import tuberias

tuberia = tuberias.construir_tuberia("regresion", "lineal", c.numericas, c.categoricas)
mostrar(tuberia)   # imputar y escalar las numéricas, codificar las categóricas y ajustar el modelo

# %%
recta.mostrar()
poda.mostrar()
didactica.mostrar_sin(0)
coeficientes.mostrar()

# %% [markdown]
# ### Nivel 2: El arreglo L16 de Taguchi
#
# Un arreglo ortogonal estima el efecto de cada factor con 16 corridas porque cada nivel de un factor aparece el
# mismo número de veces frente a cada nivel de los demás. El precio es que algunas interacciones quedan mezcladas con
# los efectos principales. Cada corrida se resume con la razón señal/ruido "menor es mejor", que castiga a la vez un
# error alto y uno que cambia entre bloques, y el óptimo sale del modelo aditivo:
#
# $$\frac{S}{N} = -10\,\log_{10}\Big(\frac{1}{B}\sum_{b=1}^{B}{\color{#B8423F}{\mathrm{RMSE}_b^2}}\Big)
# \qquad
# \widehat{S/N} = {\color{#6B6B6B}{\overline{S/N}}} + \sum_{f}\Big({\color{#2F8A4E}{\overline{S/N}_{f,\,\text{nivel}}}} - {\color{#6B6B6B}{\overline{S/N}}}\Big)$$

# %%
from proyecto.cribado import arreglos
from proyecto.reporte.html import tabla_html

l16 = arreglos.arreglo_l16()                        # 16 corridas: A con 4 niveles y doce columnas de 2 niveles
ortogonal = arreglos.verificar_ortogonalidad(l16)   # cada par de columnas aparece balanceado
usadas = l16[list(regresion.COLUMNAS_L16.values())].T.set_axis(range(1, 17), axis=1)   # una fila por columna usada
titulo = f"Columnas del L16 que usamos, una por fila (ortogonal: {'Sí' if ortogonal else 'No'})"
mostrar(tabla_html(usadas, titulo, decimales=0, con_indice=True))

# %%
diseno.mostrar()
corridas.mostrar()
confirmacion.mostrar_sin(2, 3)

# %% [markdown]
# ### Nivel 3: Afinamiento y evaluación única en validación
#
# Gana el menor RMSE de validación y, si la diferencia no llega al uno por ciento, el modelo más sencillo. El
# **bootstrap** repite el cálculo mil veces sobre días sacados al azar con reemplazo y se queda con el 95 % central.

# %%
afinamiento.mostrar()
evaluacion.mostrar()
diagnostico.mostrar_sin(0)

# %% [markdown]
# ### Modelo congelado y operación
#
# El modelo se reajusta con entrenamiento y validación y queda en disco para la etapa 06, que lo evalúa una sola vez
# en la prueba. Así se carga y se le piden predicciones (estos días ya los vio el reajuste así que la tabla no mide
# nada):

# %%
congelado.mostrar()
operacion.mostrar(1)

# %%
import joblib
from proyecto.base import rutas

modelo = joblib.load(rutas.DIR_ARTEFACTOS / "modelos" / "regresion.joblib")   # el que acabamos de congelar
dias = c.validacion.tail(5)
tabla = dias[[c.fecha, c.objetivo]].assign(predicción=modelo.predict(dias[c.numericas + c.categoricas]))
mostrar(tabla_html(tabla, "Modelo congelado: Muestras por día en los últimos días de validación", decimales=1))

# %% [markdown]
# El panel responde qué habría pasado con otra combinación de factores del L16, con lo que guardamos de las 16
# corridas y sin volver a correr nada:

# %%
from proyecto.etapas.comun import legible
from proyecto.reporte import interactivo as ui

congelada = {f: legible(v) for f, v in c.configuracion_final.items()}   # niveles de la configuración congelada
mostrar(ui.panel_taguchi(c.taguchi, congelada))

# %%
regresion.terminar(c)
