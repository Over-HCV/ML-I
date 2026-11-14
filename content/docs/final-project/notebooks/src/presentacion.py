# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Presentación · Analítica predictiva sobre la base de muestras de un laboratorio de microbiología
#
# Versión corta para la exposición de diez minutos: once diapositivas con las cifras que ya calcularon los
# notebooks de las etapas y, entre ellas, los fragmentos de código que muestran cómo se hizo cada paso (las mismas
# llamadas al paquete `proyecto/`). Aquí no se repite el cálculo pesado: los arreglos pequeños que usan los
# fragmentos y las animaciones se prepararon al construir el notebook. El detalle completo está en
# `Proyecto_Final.ipynb`.

# %%
from proyecto.base.salida import mostrar
from proyecto.etapas import presentacion as p
from proyecto.reporte import graficos_presentacion as g
from proyecto.reporte import interactivo as ui   # paneles con controles para la exposición en vivo
from proyecto.reporte.figuras import mostrar_figura

d = p.iniciar()   # estilo editorial y arreglos pequeños calculados al construir (sin la partición de prueba)

# %%
p.diapositiva_portada().mostrar()

# %%
p.diapositiva_problema().mostrar()

# %%
p.diapositiva_datos().mostrar()

# %% [markdown]
# **Cómo se valida sin mirar el futuro.** La validación cruzada no baraja las filas: cada bloque se ajusta con el
# pasado y se valida con el tramo que sigue, y el periodo de validación y el de prueba quedan después de todos ellos.

# %%
import pandas as pd
from proyecto.evaluacion import validacion

fechas = pd.Series(d.fechas_entrenamiento)        # fecha de recepción de cada muestra de entrenamiento
bloques = validacion.bloques_temporales(fechas)   # índices (ajustar, validar) que se pasan como cv= a scikit-learn
mostrar_figura(g.figura_bloques(fechas, bloques))

# %%
p.diapositiva_exploracion().mostrar()

# %%
p.diapositiva_metodo().mostrar()

# %%
from proyecto.cribado import arreglos
from proyecto.reporte.html import tabla_html

l16 = arreglos.arreglo_l16()                        # 16 corridas: A con 4 niveles y doce columnas de 2 niveles
ortogonal = arreglos.verificar_ortogonalidad(l16)   # cada par de columnas aparece balanceado
titulo = f"Arreglo ortogonal L16 (ortogonalidad verificada: {'sí' if ortogonal else 'no'})"
mostrar(tabla_html(l16, titulo, decimales=0, maximo_filas=16))

# %%
p.diapositiva_regresion().mostrar()

# %% [markdown]
# **El modelo de la carga diaria** es una regresión lineal sobre retrasos, medias móviles, calendario y agenda; su
# habilidad compara el error con el de la mejor referencia ingenua:
#
# $$\hat y_{t+1} = {\color{#3B64A8}{\beta_0}} + \sum_j {\color{#3B64A8}{\beta_j}}\, x_{t,j}
# \qquad\qquad \text{habilidad} = 1 - \frac{{\color{#B8423F}{\mathrm{RMSE}_{\text{modelo}}}}}{\mathrm{RMSE}_{\text{referencia}}}$$
#
# El modelo congelado se guarda en disco y se usa así (los días de la tabla son de validación, que el reajuste
# final ya vio: la celda muestra la llamada, el error se midió en prueba):

# %%
import joblib
from proyecto.base import rutas

modelo = joblib.load(rutas.DIR_ARTEFACTOS / "modelos" / "regresion.joblib")   # congelado en la etapa 04
r = d.regresion
dias = r.dias.assign(predicción=modelo.predict(r.dias[r.variables]))
mostrar(tabla_html(dias[[r.fecha, r.objetivo, "predicción"]], "Modelo congelado: muestras por día", decimales=1))

# %%
p.diapositiva_clasificacion().mostrar()

# %% [markdown]
# **Del riesgo a la alerta.** La regresión logística da una probabilidad, y la alerta se dispara cuando supera el
# umbral ${\color{#C46A2B}{t}}$, elegido en validación por costo: perder una no conforme (FN) cuesta más que una falsa
# alarma (FP).
#
# $$p(x) = \sigma\big({\color{#6E5BA8}{\beta_0 + \beta^\top x}}\big)
# \qquad\qquad
# \text{Costo}({\color{#C46A2B}{t}}) = c_{FP}\,{\color{#6B6B6B}{FP(t)}} + c_{FN}\,{\color{#B8423F}{FN(t)}}$$
#
# La tubería se arma con una sola llamada; scikit-learn la dibuja paso a paso:

# %%
from proyecto.modelos import tuberias

clf = p.artefacto("05_clasificacion")
cfg = clf["configuracion"]                           # lo que eligieron el L16, la rejilla y la validación
tuberia = tuberias.construir_tuberia(
    "clasificacion", cfg["modelo"], d.numericas, d.categoricas, escalado=cfg["escalado"],
    codificacion=cfg["codificacion"], seleccion=cfg["seleccion"], desbalance=cfg["desbalance"])
mostrar(tuberia)

# %%
umbrales = [u["Umbral"] for u in clf["tabla_umbrales"]]   # el mismo recorrido de la etapa 05
costos = clf["costos"]
animacion = g.animacion_umbral(d.proba_validacion, d.y_validacion, umbrales,
                               costos["falso_positivo"], costos["falso_negativo"], clf["umbral"])
mostrar(animacion)

# %%
mostrar(ui.panel_umbral(d))   # mueva el umbral: precisión, recall, alertas por semana y costo

# %%
mostrar(ui.panel_knn(d))      # vecinos k: sobreajuste con k pequeño, subajuste con k grande

# %%
mostrar(ui.panel_arbol(d))    # poda ccp_alpha: hojas, ROC-AUC y el árbol que queda

# %%
p.diapositiva_red().mostrar()

# %% [markdown]
# **Del perceptrón al perceptrón multicapa.** Un perceptrón combina las entradas con unos pesos y aplica una
# función de activación; el MLP apila una capa oculta de esas unidades antes de la salida logística:
#
# $$\hat y = \varphi\big({\color{#6E5BA8}{w}}^\top x + {\color{#6E5BA8}{b}}\big)
# \qquad\qquad
# h = \mathrm{ReLU}\big({\color{#6E5BA8}{W_1}}\, x + {\color{#6E5BA8}{b_1}}\big), \quad
# \hat p = {\color{#B8423F}{\sigma}}\big({\color{#6E5BA8}{w_2}}^\top h + {\color{#6E5BA8}{b_2}}\big)$$
#
# **Retropropagación.** La red mide la pérdida logarítmica (con la penalización ${\color{#3B64A8}{\alpha}}$ sobre los
# pesos), calcula su gradiente capa por capa con la regla de la cadena y mueve cada peso en contra de él:
#
# $${\color{#6E5BA8}{W_1}} \leftarrow {\color{#6E5BA8}{W_1}} - \eta\,{\color{#B8423F}{\frac{\partial L}{\partial W_1}}},
# \qquad
# {\color{#B8423F}{\frac{\partial L}{\partial W_1}}} = \frac{\partial L}{\partial \hat p}\;
# \frac{\partial \hat p}{\partial h}\;\frac{\partial h}{\partial W_1} + 2\,{\color{#3B64A8}{\alpha}}\,{\color{#6E5BA8}{W_1}}$$

# %%
mostrar(g.video("red"))   # Manim: adelante, pérdida, gradientes hacia atrás y actualización (arquitectura real)

# %%
from proyecto.modelos import registro

red = registro.crear_modelo("clasificacion", "mlp")   # MLPClassifier con parada temprana
mostrar_figura(g.figura_arquitectura(d.mlp["n_entradas"], red.hidden_layer_sizes, red.activation))
mostrar_figura(g.figura_activaciones())

# %%
# Al construir: tuberia = construir_tuberia("clasificacion", "mlp", ...).fit(X_ent, y_ent); red = tuberia[-1]
curvas = g.figura_curva_perdida(d.mlp_perdida, d.mlp_validacion, d.mlp["paciencia"], d.mlp["fraccion_validacion"],
                                d.mlp["tasa_mayoritaria"])   # la exactitud de responder siempre «conforme», como referencia
mostrar_figura(curvas)

# %%
# La misma red, entrenada época por época con red.partial_fit(X_ent, y_ent) y evaluada en validación tras cada época
mostrar(g.animacion_red(d.red_perdida, d.red_auc, d.red_histogramas, d.red_bordes))

# %%
mostrar(ui.panel_red(d))      # capas ocultas y α: la red se reentrena en vivo sobre una submuestra fija

# %%
p.diapositiva_bayes().mostrar()

# %% [markdown]
# **Bayes en una línea.** De todos los positivos esperados, la fracción verdadera es la masa de
# ${\color{#2F8A4E}{\text{verdaderos positivos}}}$ sobre el total, que suma la masa de
# ${\color{#B8423F}{\text{falsos positivos}}}$; $S$ y $E$ salen de los controles de cada ensayo y
# ${\color{#3B64A8}{p}}$ es la probabilidad previa de contaminación:
#
# $$\mathrm{VPP} = \frac{{\color{#2F8A4E}{S\,{\color{#3B64A8}{p}}}}}{{\color{#2F8A4E}{S\,{\color{#3B64A8}{p}}}}
# + {\color{#B8423F}{(1-E)(1-{\color{#3B64A8}{p}})}}}$$

# %%
mostrar(g.video("bayes"))   # Manim: el árbol de 1000 muestras del ensayo de esterilidad se vuelve la fórmula

# %%
import numpy as np
from proyecto.evaluacion import bayes

ensayos = pd.DataFrame(clf["bayes"]["vpp_por_ensayo"])
previas = np.geomspace(1e-3, 0.6, 200)[:, None]   # rejilla de previas para dibujar
vpp = bayes.valor_predictivo_positivo(previas, ensayos["sensibilidad"].to_numpy(), ensayos["especificidad"].to_numpy())
observados = dict(zip(ensayos["ensayo"], zip(ensayos["prevalencia_corregida"], ensayos["vpp"])))
mostrar_figura(g.figura_vpp(previas[:, 0], vpp, ensayos["ensayo"].tolist(), observados))

# %%
mostrar(ui.panel_bayes(d))    # previa, sensibilidad y especificidad → VPP y falsos positivos de cada 1000

# %%
p.diapositiva_agrupamiento().mostrar()

# %%
from proyecto.modelos.agrupamiento import pasos_kmeans

# K-means desde cero sobre una muestra de los perfiles: asignar al centro más cercano y promediar, hasta que nada cambie
centros, etiquetas = pasos_kmeans(d.kmeans_muestra, d.kmeans["k"])
plano = (d.kmeans_muestra - d.kmeans_media) @ d.kmeans_componentes.T   # proyección en dos componentes principales
mostrar(g.animacion_kmeans(plano, etiquetas, (centros - d.kmeans_media) @ d.kmeans_componentes.T))

# %%
mostrar(ui.panel_kmeans(d))   # número de grupos k: silueta y nube en dos componentes

# %%
p.diapositiva_conclusiones().mostrar()
