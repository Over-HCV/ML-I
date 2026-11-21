# %% [markdown]
# # Etapa 04: Qué muestras vigilar y cuánto creerle a un positivo
#
# Este notebook está ordenado para la exposición. Tiene dos tramos que se apoyan uno en el otro: En el A estimamos
# al recibir cada muestra qué tan probable es que salga no conforme y en el B usamos ese riesgo para saber cuánto
# creerle a un resultado positivo. En cada tramo van primero las ideas que decimos (en los recuadros) y debajo la
# evidencia técnica. El detalle completo de la etapa (la red neuronal por dentro, el diseño de Taguchi, los paneles y
# la auditoría de la simulación) queda en el **anexo** al final y no se expone.
#
# | Tramo | Pregunta | Tiempo |
# |---|---|---|
# | A | Qué muestras vigilar al recibirlas | 1:45 |
# | B | Cuánto creerle a un positivo | 1:15 |

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import clasificacion
from proyecto.reporte import graficos_presentacion as g

iniciar_etapa("04_clasificacion_bayes")
c = clasificacion.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# # Tramo A: Qué muestras vigilar al recibirlas
#
# ## A.1 El problema y la decisión clave
#
# > **Lo que digo.** Cerca del **9 %** de las muestras salen no conformes. Queremos saber **al recibir** una muestra,
# > antes de sembrarla, si tiene riesgo alto para priorizarla o avisarle al cliente.
# >
# > Antes de mirar resultados fijamos cuánto cuesta cada error: **Dejar pasar una no conforme cuesta 5** (el aviso al
# > cliente llega tarde) y **una falsa alarma cuesta 1** (tiempo extra de revisión). Es un supuesto académico. De ahí
# > sale un umbral teórico: **Alertar si el riesgo pasa de 1/6 = 0.167**, el punto donde alertar y no alertar
# > cuestan lo mismo en promedio.
# >
# > **Muestro:** La ficha del planteamiento y las tarjetas de 8.5 % y 8.9 % de no conformes.
#
# $$\text{Costo}({\color{#C46A2B}{t}}) = c_{FP}\,{\color{#6B6B6B}{FP(t)}} + c_{FN}\,{\color{#B8423F}{FN(t)}}
# \qquad\qquad
# {\color{#C46A2B}{t^{*}}} = \frac{c_{FP}}{c_{FP} + c_{FN}} = \frac{1}{1 + 5}$$

# %%
planteamiento = clasificacion.planteamiento(c)
planteamiento.mostrar(0, 1)

# %% [markdown]
# ## A.2 Qué hicimos: Los mismos tres niveles de la etapa 03
#
# > **Lo que digo.** Repetimos los tres niveles: Los modelos del curso tal como vienen, el diseño de Taguchi para las
# > decisiones de la receta (aquí una de ellas es qué hacer con el desbalance) y el ajuste fino. Gana la **regresión
# > logística**, con un **ROC-AUC de 0.724** en validación: Ordena mejor que el azar (0.5) pero no de forma perfecta (1).
# > La **red neuronal** tiene **32 veces más parámetros** (4385 frente a 136) y no la mejora.
# >
# > **Si preguntan por qué no la exactitud:** Decir siempre "conforme" acierta el 91 % y no sirve para nada. Por eso
# > medimos con el ROC-AUC, que mide qué tan bien ordena el modelo a las muestras por riesgo.
# >
# > **Muestro:** La tabla de la pasada didáctica (columna ROC-AUC), las curvas ROC (todas por encima de la diagonal)
# > y la tarjeta de la red frente a la logística.

# %%
poda = clasificacion.poda(c)
didactica = clasificacion.didactica(c)
logistica = clasificacion.logistica(c)
reglas = clasificacion.reglas_arbol(c)
red = clasificacion.red_neuronal(c)
didactica.mostrar(0, 2)
red.mostrar(2)

# %%
diseno = clasificacion.diseno(c)
corridas = clasificacion.corridas_l16(c)
sobremuestreo = clasificacion.sobremuestreo(c)
confirmacion = clasificacion.confirmacion(c)
afinamiento = clasificacion.afinamiento(c)
evaluacion = clasificacion.evaluacion_validacion(c)
evaluacion.mostrar(0)

# %% [markdown]
# ## A.3 Un tropiezo útil: Calibrar
#
# > **Lo que digo.** Copiar las no conformes para equilibrar las clases (**sobremuestreo**) casi no cambia cómo ordena
# > el modelo pero **infla las probabilidades**: El modelo decía **46 %** de riesgo promedio cuando la tasa real es
# > **9 %**. Por eso lo **calibramos** y el promedio bajó a **10 %**. Sin ese paso el umbral de 0.167 y toda la parte
# > de Bayes no tendrían sentido, porque necesitan que un 20 % de riesgo signifique de verdad unas 20 de cada 100.
# >
# > **Muestro:** Las tarjetas sin calibrar frente a calibrado y la figura, donde la línea naranja (calibrada) se pega
# > a la diagonal y la azul queda muy por debajo.

# %%
calibracion = clasificacion.calibracion(c)
calibracion.mostrar(1, 0)

# %% [markdown]
# ## A.4 Resultado: El umbral
#
# > **Lo que digo.** Recorrimos los umbrales de 0.02 a 0.50 y el de **menor costo es 0.167**, el mismo que daba la
# > teoría. Eso confirma que la calibración funcionó. Con él salen **31 alertas por semana**: El **28 %** de las
# > alertas acierta y se detecta el **40 %** de las no conformes. El costo baja de **2925** (no alertar nunca) a
# > **2376**, un **19 % menos**. Alertar todas costaría 5964.
# >
# > **Muestro:** Las tres tarjetas (no alertar nunca, el modelo, alertar todas) y la figura del costo con su mínimo
# > en la línea negra.

# %%
umbral = clasificacion.umbral_por_costo(c)
congelado = clasificacion.congelar(c)
umbral.mostrar(1, 0)

# %% [markdown]
# # Tramo B: Cuánto creerle a un positivo
#
# ## B.1 La idea
#
# > **Lo que digo.** Esta es la pregunta con que empezó el laboratorio: Cuando un ensayo da "no conforme", ¿qué tan
# > probable es que sea falso? Cada ensayo tiene una **sensibilidad** (de las contaminadas, cuántas detecta) y una
# > **especificidad** (de las limpias, cuántas deja pasar como limpias), que salen de sus controles de calidad. Aunque
# > el ensayo sea muy bueno, si la contaminación real es rara **buena parte de las alarmas son falsas**. Eso lo dice
# > el teorema de Bayes y el video lo muestra con 1000 muestras.
# >
# > **Muestro:** El video (32 s). La fórmula queda de apoyo: Arriba los verdaderos positivos y abajo se les suman los
# > falsos.
#
# $$\mathrm{VPP} = P(\text{contaminada} \mid +) = \frac{{\color{#2F8A4E}{S\,{\color{#3B64A8}{p}}}}}{{\color{#2F8A4E}{S\,{\color{#3B64A8}{p}}}}
# + {\color{#B8423F}{(1-E)(1-{\color{#3B64A8}{p}})}}}
# \qquad\qquad P(\text{falso positivo} \mid +) = 1 - \mathrm{VPP}$$

# %%
mostrar(g.video("bayes"))   # Manim: el árbol de 1000 muestras se vuelve la fórmula

# %% [markdown]
# ## B.2 Resultado 1: Cuánto creerle a cada ensayo
#
# > **Lo que digo.** En el laboratorio unos **22 de cada 100** no conformes serían falsos (VPP de **0.784**). El ensayo
# > menos fiable es **esterilidad**, con **37 falsos de cada 100**, y el más fiable **Salmonella** (10 de cada 100).
# > La diferencia no viene solo de la especificidad: Un ensayo donde la contaminación es rara da más falsos en
# > proporción.
# >
# > **Muestro:** La columna P(falso positivo) de la tabla y la figura por ensayo.

# %%
fiabilidad = clasificacion.sensibilidad_especificidad(c)
vpp_ensayo = clasificacion.vpp_por_ensayo(c)
vpp_ensayo.mostrar(0, 1)

# %% [markdown]
# ## B.3 Resultado 2: El mismo positivo se lee distinto según la muestra
#
# > **Lo que digo.** Aquí se unen las dos partes. El clasificador del tramo A da el riesgo previo de cada muestra.
# > Tomamos **dos muestras con el mismo positivo en el mismo ensayo** (mesófilos aerobios): En la de riesgo bajo la
# > probabilidad de que el positivo sea falso es **38 %** y en la de riesgo alto es **3 %**. Mismo resultado, distinta
# > confianza.
# >
# > **Muestro:** Las dos tarjetas lado a lado (maquillaje en polvo frente a gel).

# %%
vpp_muestra = clasificacion.vpp_por_muestra(c)
vpp_muestra.mostrar(2)

# %% [markdown]
# ## B.4 Cierre: El reensayo y la regla simple
#
# > **Lo que digo.** Si el reensayo vuelve a dar positivo el resultado queda prácticamente confirmado (**99 %**). Las
# > cuentas cuadran con lo observado: Esperábamos **25 %** de reensayos conformes y salió **27 %**. Y la regla simple
# > que dejamos pendiente en la etapa 03 queda así: **Reensayar antes de informar cuando el recuento pasa el límite
# > por menos de 1.66 veces**, porque en ese grupo el **78 %** de los reensayos sale conforme (frente al 15 % cuando lo
# > pasa por más).
# >
# > **Frase de cierre:** El modelo no reemplaza al analista: Le dice qué mirar primero y cuánto confiar en lo que ve.
# >
# > **Muestro:** El hallazgo del reensayo y la tarjeta de la regla.

# %%
reensayo = clasificacion.reensayo(c)
auditoria = clasificacion.auditoria(c)
regla = clasificacion.regla_reensayo(c)
reensayo.mostrar(1)
regla.mostrar(1)

# %% [markdown]
# ---
#
# # Anexo: Detalle técnico de la etapa (no se expone)
#
# Aquí queda todo lo que la etapa calculó y no mostramos en los tramos. Nada se vuelve a calcular salvo los paneles,
# que reentrenan en vivo con el núcleo en marcha.
#
# ## Tramo A
#
# ### Métricas y referencia
#
# Como la clase positiva es escasa medimos con tres cosas frente a un `Dummy` que predice siempre la proporción del
# entrenamiento. El **ROC-AUC** es la probabilidad de que una no conforme al azar reciba más riesgo que una conforme al
# azar. La **precisión media** es el área bajo la curva de precisión contra recall y su referencia es la prevalencia.
# El **Brier** es el error cuadrático de la probabilidad y premia que las probabilidades sean creíbles.

# %%
planteamiento.mostrar_sin(0, 1)

# %% [markdown]
# ### Nivel 1: Los modelos del curso
#
# Cada modelo usa el grupo ampliado de variables y el preprocesamiento por defecto, dentro de una tubería que aprende
# la imputación, la escala y las categorías solo con el tramo de ajuste de cada bloque. El árbol se poda con
# `ccp_alpha` y va sin escalar porque escalar no cambia sus cortes.

# %%
from proyecto.modelos import tuberias

tuberia = tuberias.construir_tuberia("clasificacion", "logistica", c.numericas, c.categoricas)
mostrar(tuberia)   # imputar y escalar las numéricas, codificar las categóricas y ajustar el modelo

# %%
poda.mostrar()
didactica.mostrar_sin(0, 2)

# %% [markdown]
# ### Qué dice la regresión logística y las reglas del árbol
#
# La logística combina las variables con un peso cada una y pasa la suma por la sigmoide. La exponencial de cada
# coeficiente es la razón de odds: Un 1.5 quiere decir odds de no conformidad un 50 % mayores.
#
# $$p(x) = \sigma\big({\color{#6E5BA8}{\beta_0 + \beta^\top x}}\big)
# = \frac{1}{1 + e^{-({\color{#6E5BA8}{\beta_0 + \beta^\top x}})}}$$

# %%
logistica.mostrar()
reglas.mostrar()

# %% [markdown]
# ### La red neuronal por dentro
#
# Un **perceptrón** multiplica cada entrada por un peso, suma un sesgo y pasa el resultado por una activación. El
# **perceptrón multicapa** pone una capa oculta de esas unidades entre las variables y la salida:
#
# $$h = \mathrm{ReLU}\big({\color{#6E5BA8}{W_1}}\, x + {\color{#6E5BA8}{b_1}}\big), \quad
# \hat p = {\color{#B8423F}{\sigma}}\big({\color{#6E5BA8}{w_2}}^\top h + {\color{#6E5BA8}{b_2}}\big)$$
#
# La activación es lo que deja a la red doblar la frontera. Sin ella las dos capas se reducirían a la misma
# logística. La red aprende con **retropropagación**: Mide la pérdida logarítmica más una penalización α sobre los
# pesos, calcula con la regla de la cadena cuánto cambia la pérdida al mover cada peso y lo mueve un paso η en contra
# de ese gradiente. Una pasada por el entrenamiento es una **época** y la **parada temprana** detiene el entrenamiento
# cuando lleva varias épocas sin mejorar.
#
# $${\color{#6E5BA8}{W_1}} \leftarrow {\color{#6E5BA8}{W_1}} - \eta\,{\color{#B8423F}{\frac{\partial L}{\partial W_1}}},
# \qquad
# {\color{#B8423F}{\frac{\partial L}{\partial W_1}}} = \frac{\partial L}{\partial \hat p}\;
# \frac{\partial \hat p}{\partial h}\;\frac{\partial h}{\partial W_1} + 2\,{\color{#3B64A8}{\alpha}}\,{\color{#6E5BA8}{W_1}}$$

# %%
from proyecto.reporte.figuras import mostrar_figura

mostrar_figura(g.figura_activaciones())   # ReLU, sigmoide y tangente hiperbólica
mostrar(g.video("red"))                   # Manim: adelante, pérdida, gradientes hacia atrás y actualización

# %%
red.mostrar_sin(2)
p = c.paneles   # arreglos de la red que calculó la sección
mostrar(g.animacion_red(p.red_perdida, p.red_auc, p.red_histogramas, p.red_bordes))

# %% [markdown]
# El panel reentrena la red en vivo con otro tamaño de capa oculta o con otro α:

# %%
from proyecto.reporte import interactivo as ui

mostrar(ui.panel_red(c.paneles))   # capas ocultas y α: pérdida, ROC-AUC y la logística de referencia

# %% [markdown]
# ### Nivel 2: El arreglo L16 de Taguchi
#
# Los factores son el modelo, el escalado, el grupo de variables, qué hacer con el desbalance (sobremuestrear la clase
# minoritaria, solo al ajustar), la selección de variables y la codificación. La respuesta es el ROC-AUC de cada
# bloque, resumido con la razón señal/ruido "mayor es mejor":
#
# $$\frac{S}{N} = -10\,\log_{10}\Big(\frac{1}{B}\sum_{b=1}^{B}\frac{1}{{\color{#3B64A8}{\mathrm{AUC}_b^2}}}\Big)$$
#
# Aparte comprobamos la interacción entre el modelo y el sobremuestreo, que en el L16 queda mezclada, y corrimos el
# óptimo aditivo para confirmarlo.

# %%
diseno.mostrar()
corridas.mostrar()
sobremuestreo.mostrar()
confirmacion.mostrar()

# %% [markdown]
# ### Nivel 3: Afinamiento y evaluación única en validación
#
# Las rejillas se recorren con `GridSearchCV` en los mismos bloques. Gana el mayor ROC-AUC de validación y, si la
# diferencia es menor que 0.005, el modelo más sencillo. Los dos paneles muestran qué hace cada hiperparámetro: En
# KNN el número de vecinos decide cuánto se promedia y en el árbol α decide cuánto se poda.

# %%
afinamiento.mostrar()
evaluacion.mostrar_sin(0)

# %%
mostrar(ui.panel_knn(c.paneles))     # vecinos k: sobreajuste con k pequeño, subajuste con k grande

# %%
mostrar(ui.panel_arbol(c.paneles))   # poda ccp_alpha: hojas, ROC-AUC y el árbol que queda

# %% [markdown]
# ### Calibración y umbral por dentro
#
# La versión calibrada usa `CalibratedClassifierCV` con el método sigmoide (el de Platt) y los mismos bloques
# temporales como `cv`. Con el umbral teórico la regla de decisión cabe en una línea:

# %%
calibracion.mostrar(2)

# %%
from proyecto.base import config
from proyecto.reporte.html import tarjetas

alerta = c.p_val >= c.umbral_teorico            # alertar cuando el riesgo pasa el umbral teórico
fp = int((alerta & (c.y_val == 0)).sum())       # falsas alarmas: conformes con alerta
fn = int((~alerta & (c.y_val == 1)).sum())      # no conformes sin alerta
cifras = [{"titulo": "Falsas alarmas (FP)", "valor": fp}, {"titulo": "Sin alerta (FN)", "valor": fn, "color": "rojo"},
          {"titulo": "Costo", "valor": config.COSTO_FP * fp + config.COSTO_FN * fn, "color": "naranja"}]
mostrar(tarjetas(cifras, titulo=f"Con el umbral teórico t = {c.umbral_teorico:.3f} en validación"))

# %%
umbral.mostrar_sin(0, 1)
umbrales = list(c.tabla_umbral["Umbral"])   # el mismo recorrido de la tabla
mostrar(g.animacion_umbral(c.p_val, c.y_val, umbrales, config.COSTO_FP, config.COSTO_FN, c.umbral))

# %%
mostrar(ui.panel_umbral(c.paneles))   # umbral y razón de costos: precisión, recall, alertas por semana y costo

# %% [markdown]
# ### Modelo congelado
#
# Los hiperparámetros, la configuración, la calibración y el umbral quedan fijos y el modelo se reajusta con
# entrenamiento y validación. El panel responde qué habría pasado con otra combinación del L16:

# %%
from proyecto.etapas.comun import legible

congelado.mostrar()
congelada = {f: legible(v) for f, v in c.configuracion_final.items()}   # niveles de la configuración congelada
mostrar(ui.panel_taguchi(c.taguchi, congelada, rotulo="Clasificación", etapa="04"))

# %% [markdown]
# ## Tramo B
#
# ### Sensibilidad y especificidad de cada ensayo
#
# Solo usamos controles anteriores al periodo de prueba y los intervalos son de Wilson al 95 %. La prevalencia no se
# observa: Se despeja de la tasa de positivos con la corrección de Rogan y Gladen. Para la incertidumbre del VPP
# sorteamos miles de veces la sensibilidad, la especificidad y la tasa de positivos de sus distribuciones Beta.

# %%
fiabilidad.mostrar()
vpp_ensayo.mostrar(2)

# %%
import numpy as np
from proyecto.evaluacion import bayes

t = c.tabla_vpp                                     # sensibilidad, especificidad y VPP de cada ensayo
previas = np.geomspace(1e-3, 0.6, 200)[:, None]     # rejilla de previas para dibujar
vpp = bayes.valor_predictivo_positivo(previas, t["sensibilidad"].to_numpy(), t["especificidad"].to_numpy())
marcas = dict(zip(t["ensayo"], zip(t["prevalencia_corregida"], t["vpp"])))
mostrar_figura(g.figura_vpp(previas[:, 0], vpp, t["ensayo"].tolist(), marcas))

# %%
mostrar(ui.panel_bayes(c.paneles))   # previa, sensibilidad y especificidad: VPP y falsos positivos de cada mil

# %% [markdown]
# ### VPP por muestra
#
# El clasificador estima p(x), la tasa de positivos que se espera para muestras como x. Con Rogan y Gladen muestra a
# muestra sale la previa π(x) y con ella VPP(x) = S π(x) / p(x).

# %%
vpp_muestra.mostrar_sin(2)

# %% [markdown]
# ### Reensayo
#
# Con dos positivos la confianza sube y con un positivo y un negativo baja, si los dos resultados son independientes
# dada la condición real de la muestra. El modelo también predice qué fracción de los reensayos debería salir
# conforme, P(− | + inicial) = VPP (1 − S) + (1 − VPP) E, que sí podemos comparar con lo observado.

# %%
reensayo.mostrar_sin(1)

# %% [markdown]
# ### Auditoría de la simulación y regla del reensayo
#
# Solo porque los datos son simulados existe una verdad oculta (qué muestra estaba de verdad contaminada). Con datos
# reales la única referencia es el reensayo, que también falla.

# %%
auditoria.mostrar()
regla.mostrar_sin(1)

# %%
clasificacion.terminar(c)
