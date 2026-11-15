# %% [markdown]
# # Etapa 05: Qué muestras vigilar y cuánto creerle a un positivo
#
# Esta etapa tiene dos partes que se apoyan una en la otra. En la parte A modelamos, con los mismos tres niveles de
# la etapa 04, la probabilidad de que una muestra aceptada termine no conforme vista desde la recepción, y fijamos
# un umbral de alerta por costo. En la parte B respondemos la pregunta con que empezó el laboratorio, qué tan
# probable es que un positivo sea falso. Para eso juntamos la fiabilidad de cada ensayo (sus controles de calidad)
# con el riesgo previo de cada muestra que da el clasificador de la parte A.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import clasificacion

iniciar_etapa("05_clasificacion_bayes")
c = clasificacion.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# ## Parte A: Clasificación
#
# ## 1. El problema, la decisión y sus costos
#
# El costo de cada error lo fijamos antes de ver resultados porque de él depende el umbral y no al revés. Si se
# alerta cuando la probabilidad pasa un umbral ${\color{#C46A2B}{t}}$, cada umbral deja unas falsas alarmas
# ${\color{#6B6B6B}{FP(t)}}$ y unas no conformes sin alerta ${\color{#B8423F}{FN(t)}}$, y el costo es
#
# $$\text{Costo}({\color{#C46A2B}{t}}) = c_{FP}\,{\color{#6B6B6B}{FP(t)}} + c_{FN}\,{\color{#B8423F}{FN(t)}}
# \qquad\qquad
# {\color{#C46A2B}{t^{*}}} = \frac{c_{FP}}{c_{FP} + c_{FN}}$$
#
# El segundo es el umbral teórico, el que sale si las probabilidades están bien calibradas. Como la clase positiva es
# tan escasa la exactitud engaña así que medimos con tres cosas, frente a un `Dummy` que predice siempre la
# proporción del entrenamiento:
#
# a. **ROC-AUC.** La probabilidad de que una no conforme tomada al azar reciba más riesgo que una conforme. Vale 0.5
#    si el modelo no ordena nada y 1 si ordena perfecto.
#
# b. **Precisión media.** El área bajo la curva de precisión contra recall. Su referencia es la prevalencia.
#
# c. **Brier.** El error cuadrático de la probabilidad, que premia que un 20 % de riesgo signifique de verdad unas
#    veinte no conformes de cada cien.

# %%
clasificacion.planteamiento(c).mostrar()

# %% [markdown]
# ## 2. Pasada didáctica: Los modelos del curso
#
# Cada modelo usa el grupo ampliado de variables y el preprocesamiento por defecto (imputación por la mediana,
# estandarización y one-hot), sin afinar nada y en los mismos bloques temporales de la etapa 04. Todo va dentro de
# una tubería de scikit-learn que aprende la imputación, la escala y las categorías solo con el tramo de ajuste de
# cada bloque. Así se arma la de la regresión logística:

# %%
from proyecto.modelos import tuberias

tuberia = tuberias.construir_tuberia("clasificacion", "logistica", c.numericas, c.categoricas)
mostrar(tuberia)   # imputar y escalar las numéricas, codificar las categóricas y ajustar el modelo

# %% [markdown]
# ### 2.1 Árbol de clasificación con poda por costo-complejidad
#
# Como en la etapa 04 el árbol crece completo y después lo podamos con `ccp_alpha`, el precio que se le cobra a cada
# hoja. Nos quedamos con el alfa de mayor ROC-AUC medio en los bloques. El árbol va sin escalar porque escalar no
# cambia sus cortes y así los umbrales quedan en las unidades de cada variable.

# %%
clasificacion.poda(c).mostrar()

# %% [markdown]
# ### 2.2 KNN, regresión logística, árbol podado y perceptrón con sus curvas ROC
#
# KNN da como riesgo la fracción de no conformes entre las muestras más parecidas, la logística suma señales en una
# escala que después convierte en probabilidad, el árbol parte las muestras con reglas sucesivas y el perceptrón
# multicapa es una red neuronal pequeña (más abajo la abrimos por dentro). La curva ROC pone en un eje la fracción
# de no conformes que se alertan y en el otro la de conformes que se alertan por error, para todos los umbrales a la
# vez. Si el modelo no ordenara nada sería la diagonal.

# %%
clasificacion.didactica(c).mostrar()

# %% [markdown]
# ### 2.3 Qué dice la regresión logística
#
# La regresión logística combina las variables con un peso ${\color{#6E5BA8}{\beta_j}}$ cada una y pasa la suma por
# la función sigmoide $\sigma$, que la lleva al intervalo entre cero y uno:
#
# $$p(x) = \sigma\big({\color{#6E5BA8}{\beta_0 + \beta^\top x}}\big)
# = \frac{1}{1 + e^{-({\color{#6E5BA8}{\beta_0 + \beta^\top x}})}}$$
#
# Con las numéricas estandarizadas cada coeficiente es el cambio en el logaritmo de las odds de no conformidad
# cuando la variable sube una desviación estándar. Las odds son la probabilidad de no conformidad dividida entre la
# de conformidad, y la exponencial del coeficiente es la razón de odds: Un 1.5 quiere decir odds un 50 % mayores.

# %%
clasificacion.logistica(c).mostrar()

# %% [markdown]
# ### 2.4 Las reglas del árbol podado
#
# Cada hoja es un grupo de condiciones con la fracción de no conformes de las muestras de entrenamiento que caen en
# ella. Mostramos las de mayor riesgo.

# %%
clasificacion.reglas_arbol(c).mostrar()

# %% [markdown]
# ### 2.5 La red neuronal por dentro
#
# Un **perceptrón** es la unidad más simple de una red: Multiplica cada entrada por un peso
# ${\color{#6E5BA8}{w}}$, suma un sesgo ${\color{#6E5BA8}{b}}$ y pasa el resultado por una función de activación
# $\varphi$. El **perceptrón multicapa** (MLP) pone una capa oculta de esas unidades entre las variables y la salida,
# y la salida es otra vez una logística sobre lo que dejó la capa oculta:
#
# $$\hat y = \varphi\big({\color{#6E5BA8}{w}}^\top x + {\color{#6E5BA8}{b}}\big)
# \qquad\qquad
# h = \mathrm{ReLU}\big({\color{#6E5BA8}{W_1}}\, x + {\color{#6E5BA8}{b_1}}\big), \quad
# \hat p = {\color{#B8423F}{\sigma}}\big({\color{#6E5BA8}{w_2}}^\top h + {\color{#6E5BA8}{b_2}}\big)$$
#
# La activación es lo que deja a la red doblar la frontera. Sin ella las dos capas se reducirían a una sola suma
# ponderada, o sea a la misma logística de la sección 2.3. Estas son las que más se usan:

# %%
from proyecto.reporte import graficos_presentacion as g
from proyecto.reporte.figuras import mostrar_figura

mostrar_figura(g.figura_activaciones())   # ReLU, sigmoide y tangente hiperbólica

# %% [markdown]
# La red aprende con **retropropagación**. Primero mide la pérdida logarítmica, que castiga dar poca probabilidad a
# lo que pasó, más una penalización ${\color{#3B64A8}{\alpha}}$ sobre el tamaño de los pesos. Después calcula con la
# regla de la cadena cuánto cambia la pérdida al mover cada peso, capa por capa desde la salida hacia atrás, y mueve
# cada peso un paso $\eta$ en contra de ese gradiente:
#
# $${\color{#6E5BA8}{W_1}} \leftarrow {\color{#6E5BA8}{W_1}} - \eta\,{\color{#B8423F}{\frac{\partial L}{\partial W_1}}},
# \qquad
# {\color{#B8423F}{\frac{\partial L}{\partial W_1}}} = \frac{\partial L}{\partial \hat p}\;
# \frac{\partial \hat p}{\partial h}\;\frac{\partial h}{\partial W_1} + 2\,{\color{#3B64A8}{\alpha}}\,{\color{#6E5BA8}{W_1}}$$
#
# El video muestra una pasada completa con la arquitectura que usamos: La señal va hacia adelante, se mide la
# pérdida, los gradientes vuelven hacia atrás y los pesos se actualizan.

# %%
mostrar(g.video("red"))   # Manim: adelante, pérdida, gradientes hacia atrás y actualización

# %% [markdown]
# Una pasada por todo el entrenamiento se llama **época**. Para no sobreajustar usamos **parada temprana**:
# scikit-learn aparta una fracción del entrenamiento, mide la exactitud ahí después de cada época y detiene el
# entrenamiento cuando lleva varias épocas sin mejorar.

# %%
clasificacion.red_neuronal(c).mostrar()

# %% [markdown]
# La misma red entrenada época por época con `partial_fit`, midiendo en validación después de cada época. A la
# derecha se ve cómo se separan (o no) las probabilidades de las conformes y de las no conformes:

# %%
p = c.paneles   # arreglos de la red que calculó la sección anterior
mostrar(g.animacion_red(p.red_perdida, p.red_auc, p.red_histogramas, p.red_bordes))

# %% [markdown]
# Le dimos varias vueltas a si una red más grande o con menos penalización ordenaría mejor. El panel reentrena la red
# en vivo con otro tamaño de capa oculta o con otro ${\color{#3B64A8}{\alpha}}$ (necesita el núcleo en marcha):

# %%
from proyecto.reporte import interactivo as ui

mostrar(ui.panel_red(c.paneles))   # capas ocultas y α: pérdida, ROC-AUC y la logística de referencia

# %% [markdown]
# ## 3. Nivel 2: Cribado de la tubería con un arreglo L16 de Taguchi
#
# Los factores son el modelo, el escalado, el grupo de variables, qué hacer con el desbalance (sobremuestrear la
# clase minoritaria, solo al ajustar), la selección de variables y la codificación. Las 128 combinaciones se reducen
# a 16 con la misma asignación de columnas de la etapa 04. La respuesta es el ROC-AUC de cada bloque, resumido con la
# razón señal/ruido de Taguchi en la versión "mayor es mejor", que castiga a la vez un ROC-AUC bajo y uno que cambia
# mucho entre bloques:
#
# $$\frac{S}{N} = -10\,\log_{10}\Big(\frac{1}{B}\sum_{b=1}^{B}\frac{1}{{\color{#3B64A8}{\mathrm{AUC}_b^2}}}\Big)$$

# %%
clasificacion.diseno(c).mostrar()

# %%
clasificacion.corridas_l16(c).mostrar()

# %% [markdown]
# ### Comprobación aparte: Modelo y sobremuestreo
#
# Sospechábamos que el sobremuestreo no le pega igual a todos los modelos. En el perceptrón con parada temprana las
# copias de un mismo no conforme pueden caer a la vez en su entrenamiento y en su validación interna, y en todos los
# modelos igualar las clases hace que las probabilidades dejen de estar calibradas. Como en el L16 esta interacción
# queda mezclada, repetimos cada modelo didáctico con sobremuestreo.

# %%
clasificacion.sobremuestreo(c).mostrar()

# %% [markdown]
# ### Corrida de confirmación
#
# El óptimo aditivo no tiene por qué estar entre las 16 corridas así que lo corremos aparte y comparamos su S/N con
# la predicha.

# %%
clasificacion.confirmacion(c).mostrar()

# %% [markdown]
# ## 4. Nivel 3: Afinamiento de hiperparámetros
#
# Igual que en la etapa 04 pasan la configuración de mayor S/N observada y la segunda mejor familia de modelos. Las
# rejillas las recorremos con `GridSearchCV` en los mismos bloques y con el ROC-AUC como métrica.

# %%
clasificacion.afinamiento(c).mostrar()

# %% [markdown]
# ### Los hiperparámetros a mano
#
# Antes de seguir quisimos ver qué hace cada hiperparámetro de los modelos del curso. En KNN el número de vecinos
# ${\color{#6E5BA8}{k}}$ decide cuánto se promedia y en el árbol ${\color{#6E5BA8}{\alpha}}$ decide cuánto se poda.
# Los dos paneles reentrenan en vivo sobre una submuestra fija.

# %%
mostrar(ui.panel_knn(c.paneles))     # vecinos k: sobreajuste con k pequeño, subajuste con k grande

# %%
mostrar(ui.panel_arbol(c.paneles))   # poda ccp_alpha: hojas, ROC-AUC y el árbol que queda

# %% [markdown]
# ### Evaluación única en validación
#
# La regla la fijamos antes de mirar: Gana el mayor ROC-AUC de validación y, si la diferencia es menor que 0.005,
# el más sencillo (regresión logística, árbol, KNN, perceptrón).

# %%
clasificacion.evaluacion_validacion(c).mostrar()

# %% [markdown]
# ## 5. Calibración
#
# El umbral teórico y la parte B suponen que una probabilidad de 0.2 quiere decir que unas veinte de cada cien
# muestras así salen no conformes. La versión calibrada usa `CalibratedClassifierCV` con el método sigmoide (el de
# Platt, que solo ajusta dos parámetros y por eso aguanta pocos no conformes por tramo) y los mismos bloques
# temporales como `cv`. La regla es usar la calibrada si la tubería sobremuestrea o si su Brier de validación es
# menor.

# %%
clasificacion.calibracion(c).mostrar()

# %% [markdown]
# ## 6. Umbral por costo en validación
#
# La regla de decisión cabe en una línea: Se alerta si el riesgo pasa el umbral y cada umbral deja sus falsas
# alarmas, sus omisiones y su costo. Así se ve con el umbral teórico:

# %%
from proyecto.base import config
from proyecto.reporte.html import tarjetas

alerta = c.p_val >= c.umbral_teorico            # alertar cuando el riesgo pasa el umbral teórico
fp = int((alerta & (c.y_val == 0)).sum())       # falsas alarmas: conformes con alerta
fn = int((~alerta & (c.y_val == 1)).sum())      # no conformes sin alerta
cifras = [{"titulo": "Falsas alarmas (FP)", "valor": fp}, {"titulo": "Sin alerta (FN)", "valor": fn, "color": "rojo"},
          {"titulo": "Costo", "valor": config.COSTO_FP * fp + config.COSTO_FN * fn, "color": "naranja"}]
mostrar(tarjetas(cifras, titulo=f"Con el umbral teórico t = {c.umbral_teorico:.3f} en validación"))

# %% [markdown]
# Después recorremos umbrales de 0.02 a 0.50 (más el teórico) y elegimos el de menor costo. Si hay empate gana el más
# alto. Al lado del costo van siempre la precisión, el recall y las alertas por semana, que es la carga que de verdad
# recibe el laboratorio.

# %%
clasificacion.umbral_por_costo(c).mostrar()

# %% [markdown]
# La animación recorre los mismos umbrales. A la izquierda la línea naranja corta las probabilidades de validación y a
# la derecha se ve cómo se mueve el costo:

# %%
umbrales = list(c.tabla_umbral["Umbral"])   # el mismo recorrido de la tabla
mostrar(g.animacion_umbral(c.p_val, c.y_val, umbrales, config.COSTO_FP, config.COSTO_FN, c.umbral))

# %% [markdown]
# En el panel se puede mover el umbral a mano o cambiar cuánto más cara es una omisión que una falsa alarma, para ver
# dónde quedaría el óptimo:

# %%
mostrar(ui.panel_umbral(c.paneles))   # umbral y razón de costos: precisión, recall, alertas por semana y costo

# %% [markdown]
# ## 7. Congelamiento del modelo
#
# Los hiperparámetros, la configuración, la calibración y el umbral quedan fijos y el modelo se reajusta con
# entrenamiento y validación. Si va calibrado, la calibración usa los bloques temporales de todo el desarrollo.

# %%
clasificacion.congelar(c).mostrar()

# %% [markdown]
# Antes de cerrar la parte A volvimos al L16 para ver qué habría pasado con otra combinación de factores. El panel lo
# responde con lo que guardamos de las 16 corridas, sin volver a correr nada:

# %%
from proyecto.etapas.comun import legible

congelada = {f: legible(v) for f, v in c.configuracion_final.items()}   # niveles de la configuración congelada
mostrar(ui.panel_taguchi(c.taguchi, congelada, rotulo="Clasificación", etapa="05"))

# %% [markdown]
# ## Parte B: Bayes, o qué tan probable es que un positivo sea falso
#
# ## 8. La idea en palabras y en fórmulas
#
# Un ensayo se describe con dos números que salen de sus controles de calidad:
#
# a. **Sensibilidad** ${\color{#2F8A4E}{S}}$. La fracción de muestras contaminadas que dan positivo. Se mide con
#    controles positivos.
#
# b. **Especificidad** $E$. La fracción de muestras limpias que dan negativo. Se mide con controles negativos.
#
# Con la probabilidad previa de contaminación ${\color{#3B64A8}{p}}$ el teorema de Bayes da el **valor predictivo
# positivo** (VPP), la fracción de positivos que son verdaderos. Arriba va la masa de
# ${\color{#2F8A4E}{\text{verdaderos positivos}}}$ y abajo se le suma la de
# ${\color{#B8423F}{\text{falsos positivos}}}$:
#
# $$\mathrm{VPP} = P(\text{contaminada} \mid +) = \frac{{\color{#2F8A4E}{S\,{\color{#3B64A8}{p}}}}}{{\color{#2F8A4E}{S\,{\color{#3B64A8}{p}}}}
# + {\color{#B8423F}{(1-E)(1-{\color{#3B64A8}{p}})}}}
# \qquad\qquad P(\text{falso positivo} \mid +) = 1 - \mathrm{VPP}$$
#
# Con una prevalencia baja el término rojo pesa mucho aunque $1 - E$ sea pequeño. Por ejemplo, con E = 0.98,
# p = 0.05 y S = 0.95 más de una de cada cuatro alarmas es falsa. El video lo cuenta con un árbol de mil muestras
# del ensayo de esterilidad:

# %%
mostrar(g.video("bayes"))   # Manim: el árbol de 1000 muestras se vuelve la fórmula

# %% [markdown]
# La prevalencia no se observa. Lo que se observa es la tasa de positivos y despejando ${\color{#3B64A8}{p}}$ de
# esa tasa (que mezcla verdaderos y falsos positivos) sale la corrección de Rogan y Gladen. Si una muestra se
# reensaya, la probabilidad después del primer resultado hace de previa para el segundo.

# %% [markdown]
# ## 9. Sensibilidad y especificidad de cada ensayo
#
# Solo usamos controles anteriores al periodo de prueba y los intervalos son de Wilson al 95 %, que se portan bien
# aunque la proporción esté cerca de uno.

# %%
clasificacion.sensibilidad_especificidad(c).mostrar()

# %% [markdown]
# ## 10. VPP por ensayo
#
# La tasa de positivos sale de las muestras aceptadas del desarrollo. Para la incertidumbre sorteamos miles de veces
# la sensibilidad, la especificidad y la tasa de positivos de sus distribuciones Beta (una previa uniforme más los
# conteos de cada una) con la semilla del proyecto y calculamos el VPP en cada sorteo.

# %%
clasificacion.vpp_por_ensayo(c).mostrar()

# %% [markdown]
# La misma fórmula sirve para cualquier previa. Con `bayes.valor_predictivo_positivo` sobre una rejilla de previas
# sale la curva de cada ensayo y el punto es donde queda cada uno con su prevalencia corregida:

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
# ## 11. VPP por muestra: La previa que aporta el clasificador
#
# El clasificador estima p(x), la tasa de positivos que se espera para muestras como x. Si aplicamos Rogan y Gladen
# muestra a muestra sale la previa π(x) y con ella VPP(x) = S π(x) / p(x). Usamos el modelo de la parte A ajustado
# solo con entrenamiento, sobre las muestras de validación. Que π(x) = 0 quiere decir que el positivo se explica
# entero por el error del ensayo.

# %%
clasificacion.vpp_por_muestra(c).mostrar()

# %% [markdown]
# ## 12. Lo que pasa después del reensayo
#
# Con dos positivos la confianza sube y con un positivo y un negativo baja, si suponemos que los dos resultados son
# independientes dada la condición real de la muestra. El modelo también predice qué fracción de los reensayos
# debería salir conforme, P(− | + inicial) = VPP (1 − S) + (1 − VPP) E, que sí podemos comparar con lo observado.

# %%
clasificacion.reensayo(c).mostrar()

# %% [markdown]
# ## 13. Auditoría de la simulación
#
# Solo porque los datos son simulados existe una verdad oculta (qué muestra estaba de verdad contaminada). Con datos
# reales la única referencia es el reensayo, que también falla.

# %%
clasificacion.auditoria(c).mostrar()

# %% [markdown]
# ## 14. Una regla simple para el reensayo
#
# En la etapa 03 dejamos por fuera el análisis "falso positivo entre los no conformes" porque una sola variable, la
# razón logarítmica entre el recuento y el límite, explicaba casi todo. Aquí leemos esa relación por tramos con los
# no conformes reensayados del desarrollo.

# %%
clasificacion.regla_reensayo(c).mostrar()

# %%
clasificacion.terminar(c)
