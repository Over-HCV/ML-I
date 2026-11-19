"""Capa 8 · Etapa 04: clasificación y Bayes.

Responsabilidad única: armar las piezas de la etapa 04 (parte A: costos,
pasada didáctica, L16 de Taguchi, sobremuestreo, afinamiento, calibración,
umbral por costo y congelamiento; parte B: sensibilidad y especificidad por
ensayo, VPP por ensayo y por muestra, posterior tras el reensayo, auditoría
de la simulación y regla de reensayo), guardar el modelo y el artefacto de la
etapa, y registrar sus cifras para el informe.
"""
import time
import warnings
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning

from proyecto.base import config, rutas
from proyecto.base.artefactos import guardar_json, guardar_modelo, leer_json, leer_modelado
from proyecto.cribado import arreglos, catalogo, corridas
from proyecto.datos import carga
from proyecto.etapas import presentacion, taguchi
from proyecto.etapas.comun import Resultado, abrir, cerrar, enumerar, legible
from proyecto.evaluacion import bayes, metricas, umbral, validacion
from proyecto.evaluacion.remuestreo import intervalo_bootstrap
from proyecto.modelos import registro, tuberias
from proyecto.preparacion import particion
from proyecto.reporte import graficos_clasificacion as graficos_clf
from proyecto.reporte import graficos_modelado as graficos
from proyecto.reporte import graficos_presentacion as graficos_red
from proyecto.reporte import latex
from proyecto.reporte.figuras import guardar_figura
from proyecto.reporte.html import comparacion, nota, pasos, pestanas, tabla_html, tarjetas
from proyecto.variables.comunes import columnas_de_grupo

ETAPA = "04_clasificacion_bayes"
TAREA = "clasificacion"
NOMBRE_DUMMY = "Dummy (proporción)"
METRICAS_DIDACTICAS = {"ROC-AUC": metricas.roc_auc, "Precisión media": metricas.precision_media, "Brier": metricas.brier}
COLUMNAS_L16 = {"modelo": "A", "escalado": "c", "grupo": "d", "desbalance": "abd", "seleccion": "acd",
                "codificacion": "bcd"}
CLAVES_DIDACTICAS = {"KNN": "knn", "Regresión logística": "logistica", "Árbol podado": "arbol",
                     "Perceptrón multicapa": "mlp"}
ORDEN_SENCILLEZ = ["logistica", "arbol", "knn", "mlp"]
TOLERANCIA_EMPATE = 0.005
METODO_CALIBRACION = "sigmoid"
N_SORTEOS = 4000
VARIABLE_REGLA = "razon_log_recuento_limite"
UMBRAL_DUDA = 0.5
DESCRIPTORES = ("matriz", "tipo_producto", "historia_cliente_tasa", "temperatura_transporte_c")


# ----------------------------------------------------------------------
# Utilidades de la etapa
# ----------------------------------------------------------------------
def fuera_de_bloque(estimador, X, y, bloques):
    """Probabilidades de cada tramo de validación de los bloques, con una copia del estimador ajustada en el
    entrenamiento de ese bloque. Devuelve una lista de (índices, probabilidades)."""
    salida = []
    for entrenar, validar in bloques:
        ajustado = clone(estimador).fit(X.iloc[entrenar], y[entrenar])
        salida.append((validar, metricas.prediccion_de(ajustado, X.iloc[validar])))
    return salida


def resumir_bloques(predichas, y):
    return {m: np.array([f(y[i], p) for i, p in predichas]) for m, f in METRICAS_DIDACTICAS.items()}


def apilar_bloques(predichas, y):
    indices = np.concatenate([i for i, _ in predichas])
    return y[indices], np.concatenate([p for _, p in predichas])


def _fila_resumen(nombre, valores):
    fila = {"Modelo": nombre}
    for metrica, por_bloque in valores.items():
        fila[metrica] = float(np.mean(por_bloque))
        fila[f"± {metrica}"] = float(np.std(por_bloque, ddof=1))
    fila["Habilidad (2 AUC − 1)"] = 2 * fila["ROC-AUC"] - 1
    return fila


def iniciar():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    c = abrir(ETAPA, "Un clasificador con su carga de alertas y la probabilidad de un falso positivo")
    c.inicio = time.perf_counter()
    c.nombre = leer_json(rutas.DIR_ARTEFACTOS / "03_cribado.json")["seleccion"][TAREA]
    if c.nombre is None:
        raise RuntimeError("El cribado de la etapa 03 no eligió ningún análisis de clasificación: No hay nada que modelar.")
    c.ficha = catalogo.ficha_por_nombre(c.nombre)
    c.papeles = leer_json(rutas.DIR_MODELADO / f"{c.nombre}.json")
    c.objetivo, c.fecha = c.papeles["objetivo"], c.papeles["fecha"]
    c.desarrollo = leer_modelado(c.nombre).reset_index(drop=True)
    ent, val = validacion.separar_desarrollo(c.desarrollo, c.fecha)
    c.entrenamiento, c.validacion = ent.reset_index(drop=True), val.reset_index(drop=True)
    c.numericas, c.categoricas = columnas_de_grupo(c.papeles, "ampliado")
    columnas = c.numericas + c.categoricas
    c.X_ent, c.y_ent = c.entrenamiento[columnas], c.entrenamiento[c.objetivo].to_numpy(dtype=int)
    c.X_val, c.y_val = c.validacion[columnas], c.validacion[c.objetivo].to_numpy(dtype=int)
    c.bloques = validacion.bloques_temporales(c.entrenamiento[c.fecha])
    c.etiquetas_bloque = [f"Bloque {i + 1}" for i in range(len(c.bloques))]
    c.semanas_val = int(pd.to_datetime(c.validacion[c.fecha]).dt.to_period("W-SUN").nunique())
    return c


# ----------------------------------------------------------------------
# Parte A: Clasificación
# ----------------------------------------------------------------------
def planteamiento(c):
    """Costos, umbral teórico, tamaños, prevalencias y la referencia Dummy."""
    r = Resultado()
    fp, fn = config.COSTO_FP, config.COSTO_FN
    c.umbral_teorico = umbral.umbral_teorico()
    c.prev_ent, c.prev_val = float(c.y_ent.mean()), float(c.y_val.mean())
    r.agregar(pasos([
        ("Qué predecimos", f"{c.ficha.titulo}: Si una {c.papeles['unidad']} va a terminar con `{c.objetivo}` = 1, con lo "
                           f"que se sabe en el momento \"{c.papeles['momento']}\", o sea antes de sembrarla."),
        ("Qué decisión apoya", f"{c.ficha.decision} En la práctica sería marcar la muestra en recepción para leerla antes, "
                               "asignarle un segundo analista o avisarle al cliente."),
        ("Cuánto cuesta cada error", f"Una falsa alarma (alertar una muestra que sale conforme) cuesta {fp} unidad, que es "
                                     "el tiempo extra de revisión. Una omisión (no alertar una muestra que sale no "
                                     f"conforme) cuesta {fn} porque el aviso al cliente llega tarde."),
        ("Umbral teórico", f"Si las probabilidades están bien calibradas conviene alertar cuando p ≥ {fp} / ({fp} + {fn}) "
                           f"= {c.umbral_teorico:.3f}, que es el punto donde alertar y no alertar cuestan lo mismo en "
                           "promedio."),
    ], titulo="Planteamiento"))
    r.agregar(tarjetas([
        {"titulo": "Muestras de entrenamiento", "valor": len(c.entrenamiento), "decimales": 0,
         "detalle": f"antes de {config.CORTE_VALIDACION}"},
        {"titulo": "Muestras de validación", "valor": len(c.validacion), "decimales": 0, "color": "naranja",
         "detalle": f"{c.semanas_val} semanas, hasta antes de {config.CORTE_PRUEBA}"},
        {"titulo": "No conformes (entrenamiento)", "valor": 100 * c.prev_ent, "decimales": 1, "sufijo": " %",
         "color": "rojo", "detalle": f"{int(c.y_ent.sum())} de {len(c.y_ent)}"},
        {"titulo": "No conformes (validación)", "valor": 100 * c.prev_val, "decimales": 1, "sufijo": " %",
         "color": "rojo", "detalle": f"{int(c.y_val.sum())} de {len(c.y_val)}"},
    ]))
    c.predichas_dummy = fuera_de_bloque(registro.linea_base(TAREA), c.X_ent, c.y_ent, c.bloques)
    c.metricas_dummy = d = resumir_bloques(c.predichas_dummy, c.y_ent)
    r.agregar(nota(
        f"El `Dummy` tiene un ROC-AUC de {d['ROC-AUC'].mean():.3f}, o sea que no ordena nada. Su precisión media es "
        f"{d['Precisión media'].mean():.3f} (la prevalencia de cada tramo) y su Brier {d['Brier'].mean():.4f}, que es lo "
        "que cuesta predecir siempre la tasa base. Un modelo que sirva tiene que mejorar esas tres cifras en los mismos "
        "bloques.",
        "info", titulo="Referencia",
    ))
    latex.registrar_varias([
        ("ClfAnalisis", c.nombre, "análisis de clasificación modelado a fondo"),
        ("ClfObjetivo", c.objetivo, "columna objetivo de la clasificación"),
        ("ClfMomento", c.papeles["momento"], "momento de la predicción de la clasificación"),
        ("ClfUmbralTeorico", c.umbral_teorico, "umbral teórico por costo, CFP / (CFP + CFN)", 3),
        ("ClfMuestrasEntrenamiento", len(c.entrenamiento), "muestras de entrenamiento"),
        ("ClfMuestrasValidacion", len(c.validacion), "muestras de validación"),
        ("ClfSemanasValidacion", c.semanas_val, "semanas del periodo de validación"),
        ("ClfPositivosEntrenamiento", int(c.y_ent.sum()), "no conformes en entrenamiento"),
        ("ClfPositivosValidacion", int(c.y_val.sum()), "no conformes en validación"),
        ("ClfPrevalenciaEntrenamientoPct", 100 * c.prev_ent, "no conformes (%) en entrenamiento", 1),
        ("ClfPrevalenciaValidacionPct", 100 * c.prev_val, "no conformes (%) en validación", 1),
        ("ClfDummyAUC", float(d["ROC-AUC"].mean()), "ROC-AUC medio del Dummy por bloque", 3),
        ("ClfDummyPrecisionMedia", float(d["Precisión media"].mean()), "precisión media del Dummy por bloque", 3),
        ("ClfDummyBrier", float(d["Brier"].mean()), "Brier medio del Dummy por bloque", 4),
    ])
    return r


def poda(c):
    """Ruta de poda del árbol (sin escalar) y alfa con mayor ROC-AUC medio."""
    r = Resultado()
    # El árbol se ajusta sin escalar: no cambia sus particiones y deja los umbrales en unidades originales.
    c.arbol_completo = tuberias.construir_tuberia(TAREA, "arbol", c.numericas, c.categoricas, escalado="ninguno",
                                                  hiper={"max_depth": None, "min_samples_leaf": 20})
    filas = []
    for alfa in tuberias.alfas_de_poda(c.arbol_completo, c.X_ent, c.y_ent):
        tuberia = clone(c.arbol_completo).set_params(modelo__ccp_alpha=float(alfa))
        auc = validacion.puntuar_por_bloques(tuberia, c.X_ent, c.y_ent, c.bloques, "roc_auc")
        hojas = tuberia.fit(c.X_ent, c.y_ent).named_steps["modelo"].get_n_leaves()
        filas.append({"ccp_alpha": float(alfa), "Hojas": int(hojas), "ROC-AUC medio": auc.mean(),
                      "Desv. ROC-AUC": auc.std(ddof=1)})
    t = pd.DataFrame(filas)
    fila = t.loc[t["ROC-AUC medio"].idxmax()]
    c.alfa = float(fila["ccp_alpha"])
    decimales = {"ccp_alpha": 6, "ROC-AUC medio": 3, "Desv. ROC-AUC": 3}
    r.agregar(tabla_html(t, "Ruta de poda por costo-complejidad del árbol de clasificación", decimales=decimales,
                         resaltar=(t["ccp_alpha"] == c.alfa).to_numpy(), columnas_resaltadas="ROC-AUC medio"))
    r.agregar(nota(
        f"Sin poda el árbol tiene {int(t.iloc[0]['Hojas'])} hojas y un ROC-AUC medio de {t.iloc[0]['ROC-AUC medio']:.3f}. "
        f"El alfa elegido ({c.alfa:.6f}) lo deja en {int(fila['Hojas'])} hojas con ROC-AUC {fila['ROC-AUC medio']:.3f}"
        + (", o sea que podar mejora la capacidad de ordenar además de simplificar el árbol. Es la señal de que las ramas "
           "profundas estaban aprendiendo ruido de las muestras de entrenamiento." if c.alfa > 0 else
           ". Es decir que en este caso ningún nivel de poda mejora al árbol completo."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfHojasSinPoda", int(t.iloc[0]["Hojas"]), "hojas del árbol de clasificación sin poda"),
        ("ClfAUCSinPoda", float(t.iloc[0]["ROC-AUC medio"]), "ROC-AUC medio del árbol sin poda", 3),
        ("ClfAlfaPoda", c.alfa, "ccp_alpha elegido para el árbol didáctico", 6),
        ("ClfHojasPodado", int(fila["Hojas"]), "hojas del árbol podado"),
        ("ClfAUCPodado", float(fila["ROC-AUC medio"]), "ROC-AUC medio del árbol podado", 3),
    ])
    latex.exportar_tabla(t, "clf_poda", "Ruta de poda del árbol de clasificación", decimales=decimales)
    return r


def didactica(c):
    """KNN, regresión logística, árbol podado y perceptrón en los mismos bloques, con curvas ROC."""
    r = Resultado()
    c.didacticos = {
        "KNN": tuberias.construir_tuberia(TAREA, "knn", c.numericas, c.categoricas),
        "Regresión logística": tuberias.construir_tuberia(TAREA, "logistica", c.numericas, c.categoricas),
        "Árbol podado": clone(c.arbol_completo).set_params(modelo__ccp_alpha=c.alfa),
        "Perceptrón multicapa": tuberias.construir_tuberia(TAREA, "mlp", c.numericas, c.categoricas),
    }
    c.predichas = {n: fuera_de_bloque(e, c.X_ent, c.y_ent, c.bloques) for n, e in c.didacticos.items()}
    c.metricas_didacticas = {NOMBRE_DUMMY: c.metricas_dummy,
                             **{n: resumir_bloques(p, c.y_ent) for n, p in c.predichas.items()}}
    c.tabla_didactica = t = pd.DataFrame([_fila_resumen(n, v) for n, v in c.metricas_didacticas.items()])
    mejor = t.iloc[1:].sort_values("ROC-AUC", ascending=False).iloc[0]
    r.agregar(tabla_html(t, f"Pasada didáctica en los {len(c.bloques)} bloques (media y desviación entre bloques)",
                         decimales=4, resaltar=(t["Modelo"] == mejor["Modelo"]).to_numpy(), columnas_resaltadas="ROC-AUC"))
    auc = pd.DataFrame({n: v["ROC-AUC"] for n, v in c.metricas_didacticas.items() if n != NOMBRE_DUMMY},
                       index=c.etiquetas_bloque).T
    r.figura(graficos.figura_por_bloque(auc, "ROC-AUC", resaltar=[mejor["Modelo"]]), "04_didactico_bloques")
    r.figura(graficos_clf.figura_roc({n: apilar_bloques(p, c.y_ent) for n, p in c.predichas.items()},
                                     "Curvas ROC con las predicciones fuera de bloque (los 4 tramos juntos)"),
             "04_roc_didactico")
    c.por_modelo = pm = t.set_index("Modelo")
    auc_mlp, c.auc_log = pm.loc["Perceptrón multicapa", "ROC-AUC"], pm.loc["Regresión logística", "ROC-AUC"]
    peor = t.iloc[1:].sort_values("ROC-AUC").iloc[0]
    r.agregar(nota(
        f"El mejor modelo de la pasada es \"{mejor['Modelo']}\", con ROC-AUC medio **{mejor['ROC-AUC']:.3f}** "
        f"(± {mejor['± ROC-AUC']:.3f} entre bloques). Su precisión media es {mejor['Precisión media']:.3f} frente a "
        f"{pm.loc[NOMBRE_DUMMY, 'Precisión media']:.3f} del `Dummy` y su Brier {mejor['Brier']:.4f} frente a "
        f"{pm.loc[NOMBRE_DUMMY, 'Brier']:.4f}. El más débil es \"{peor['Modelo']}\" ({peor['ROC-AUC']:.3f}).\n\n"
        + (f"El perceptrón ({auc_mlp:.3f}) no le gana a la regresión logística ({c.auc_log:.3f}), lo que sugiere que el "
           "riesgo de no conformidad se explica sobre todo por efectos que se suman en la escala logit y que la red no "
           "encuentra estructura adicional que justifique su costo." if auc_mlp <= c.auc_log else
           f"El perceptrón ({auc_mlp:.3f}) le gana a la regresión logística ({c.auc_log:.3f}) así que hay algo de no "
           "linealidad o de interacción que la red aprovecha.")
        + (f" Eso sí, un ROC-AUC de {mejor['ROC-AUC']:.2f} es moderado: El modelo ordena mejor que el azar pero muchas "
           "muestras conformes y no conformes reciben riesgos parecidos. Sería lo esperable si una parte de la "
           "contaminación ocurre después de la recepción, adonde estas variables no llegan."
           if mejor["ROC-AUC"] < 0.8 else ""),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfMejorDidactico", mejor["Modelo"], "mejor modelo de la pasada didáctica"),
        ("ClfAUCMejorDidactico", float(mejor["ROC-AUC"]), "su ROC-AUC medio por bloque", 3),
        ("ClfAUCDesvMejorDidactico", float(mejor["± ROC-AUC"]), "desviación de su ROC-AUC entre bloques", 3),
        ("ClfPrecisionMediaMejorDidactico", float(mejor["Precisión media"]), "su precisión media", 3),
        ("ClfBrierMejorDidactico", float(mejor["Brier"]), "su Brier medio", 4),
        ("ClfPeorDidactico", peor["Modelo"], "modelo más débil de la pasada didáctica"),
        ("ClfAUCPeorDidactico", float(peor["ROC-AUC"]), "ROC-AUC medio del más débil", 3),
        ("ClfAUCLogisticaDidactico", float(c.auc_log), "ROC-AUC medio de la regresión logística didáctica", 3),
        ("ClfAUCMLPDidactico", float(auc_mlp), "ROC-AUC medio del perceptrón didáctico", 3),
    ])
    latex.exportar_tabla(t, "clf_didactica", "Pasada didáctica de clasificación (media y desviación entre bloques)",
                         decimales=4)
    return r


def red_neuronal(c):
    """El perceptrón didáctico por dentro: Lo ajusta una vez con todo el entrenamiento y lo vuelve a entrenar
    época por época (`presentacion.datos_red`, el mismo cálculo de la presentación), dibuja su arquitectura y sus
    curvas de parada temprana, lo compara con la logística y deja listos los datos de los paneles."""
    r = Resultado()
    arreglos, meta = presentacion.datos_red(c)
    mlp = meta["mlp"]
    partida = registro.crear_modelo(TAREA, "mlp").get_params()
    logistica = clone(c.didacticos["Regresión logística"]).fit(c.X_ent, c.y_ent).named_steps["modelo"]
    parametros, parametros_log = mlp["parametros"], int(logistica.coef_.size + logistica.intercept_.size)
    perdida, puntaje = arreglos["mlp_perdida"], arreglos["mlp_validacion"]
    mayoritaria = mlp["tasa_mayoritaria"]
    sub, variables = presentacion.submuestras_paneles(c)
    c.paneles = SimpleNamespace(**arreglos, **sub, mlp=mlp)
    c.variables_paneles = variables
    _actualizar_paneles(c)
    r.figura(graficos_red.figura_arquitectura(mlp["n_entradas"], tuple(mlp["ocultas"]), mlp["activacion"]),
             "04_red_arquitectura")
    r.figura(graficos_red.figura_curva_perdida(perdida, puntaje, mlp["paciencia"], mlp["fraccion_validacion"],
                                               mayoritaria, pie=False), "04_red_curvas")
    auc_mlp = c.por_modelo.loc["Perceptrón multicapa", "ROC-AUC"]
    r.agregar(comparacion([
        {"titulo": "Regresión logística", "valor": c.auc_log, "etiqueta": "ROC-AUC medio en los bloques", "decimales": 3,
         "puntos": [f"{parametros_log} parámetros", "un peso por variable y un intercepto"]},
        {"titulo": "Perceptrón multicapa", "valor": auc_mlp, "etiqueta": "ROC-AUC medio en los bloques", "decimales": 3,
         "puntos": [f"{parametros} parámetros", f"{mlp['epocas']} épocas hasta la parada temprana"]},
    ], titulo="La red frente a la logística en los mismos bloques",
        elegida="Perceptrón multicapa" if auc_mlp > c.auc_log else "Regresión logística", sello="Mayor ROC-AUC"))
    r.agregar(nota(
        f"La red que ajustamos con todo el entrenamiento tiene {mlp['n_entradas']} entradas, una capa oculta de "
        f"{sum(mlp['ocultas'])} neuronas ({mlp['activacion']}) y {parametros} parámetros, unas "
        f"{parametros / parametros_log:.0f} veces los {parametros_log} de la logística. La parada temprana la detuvo en "
        f"la época {mlp['epocas']} (la mejor fue la {mlp['mejor_epoca']}) y en la fracción apartada su exactitud apenas "
        f"pasa del {mayoritaria:.3f} que se logra respondiendo siempre \"conforme\". O sea que la exactitud no sirve "
        "para comparar modelos aquí y por eso usamos el ROC-AUC.",
        "info", titulo="Qué aprendió la red",
    ))
    latex.registrar_varias([
        ("ClfMLPEntradas", mlp["n_entradas"], "entradas del perceptrón (columnas tras el preprocesamiento)"),
        ("ClfMLPOcultas", ", ".join(str(n) for n in mlp["ocultas"]), "neuronas por capa oculta (hidden_layer_sizes)"),
        ("ClfMLPCapasOcultas", len(mlp["ocultas"]), "número de capas ocultas del perceptrón"),
        ("ClfMLPActivacion", mlp["activacion"], "función de activación de las capas ocultas"),
        ("ClfMLPOptimizador", partida["solver"], "optimizador del perceptrón en scikit-learn"),
        ("ClfMLPAlfa", float(mlp["alpha"]), "penalización L2 (alpha) del perceptrón", 4),
        ("ClfMLPParametros", parametros, "pesos y sesgos del perceptrón"),
        ("ClfLogisticaParametros", parametros_log, "coeficientes e intercepto de la regresión logística"),
        ("ClfMLPVecesParametros", parametros / parametros_log, "cuántas veces más parámetros tiene la red", 0),
        ("ClfMLPEpocas", int(mlp["epocas"]), "épocas hasta la parada temprana"),
        ("ClfMLPMejorEpoca", mlp["mejor_epoca"], "época con la mejor exactitud en la fracción apartada"),
        ("ClfMLPPaciencia", int(mlp["paciencia"]), "épocas sin mejora antes de detenerse (n_iter_no_change)"),
        ("ClfMLPFraccionValidacionPct", 100 * mlp["fraccion_validacion"], "porcentaje del entrenamiento apartado", 0),
        ("ClfMLPTopeEpocas", int(partida["max_iter"]), "tope de épocas (max_iter)"),
        ("ClfMLPExactitudMejor", float(np.max(puntaje)), "mejor exactitud en la fracción apartada", 3),
        ("ClfTasaMayoritaria", mayoritaria, "exactitud de responder siempre la clase mayoritaria", 3),
        ("ClfMLPPerdidaFinal", float(perdida[-1]), "pérdida logarítmica de entrenamiento en la última época", 3),
    ])
    return r


def _actualizar_paneles(c):
    """Rehace los valores congelados que marcan los paneles (`c.paneles.interactivo`) con lo que la etapa ya
    decidió: Se llama al avanzar (red, afinamiento, umbral y VPP por ensayo)."""
    busquedas = getattr(c, "busquedas", {})
    c.paneles.interactivo = presentacion.meta_paneles(
        c.variables_paneles, getattr(c, "umbral", None),
        list(c.tabla_umbral["Umbral"]) if hasattr(c, "tabla_umbral") else [],
        {"falso_positivo": config.COSTO_FP, "falso_negativo": config.COSTO_FN, "umbral_teorico": c.umbral_teorico},
        c.semanas_val, c.tabla_vpp.to_dict("records") if hasattr(c, "tabla_vpp") else [],
        taguchi.hiper_de(busquedas["arbol"]) if "arbol" in busquedas else {})


def logistica(c):
    """Coeficientes y razones de odds de la regresión logística."""
    r = Resultado()
    ajustada = clone(c.didacticos["Regresión logística"]).fit(c.X_ent, c.y_ent)
    coef = pd.Series(ajustada.named_steps["modelo"].coef_[0], index=tuberias.nombres_variables(ajustada))
    r.figura(graficos.figura_coeficientes(coef, "Coeficientes de la regresión logística (escala logit)"),
             "04_coeficientes_logistica")
    principales = coef.reindex(coef.abs().sort_values(ascending=False).index).head(8)
    odds = pd.DataFrame({"Variable": principales.index, "Coeficiente": principales.to_numpy(),
                         "Razón de odds": np.exp(principales.to_numpy())})
    r.agregar(tabla_html(odds, "Las ocho variables con mayor peso en la regresión logística", decimales=3))
    sube, baja = principales[principales > 0], principales[principales < 0]
    r.agregar(nota(
        "Entre las de mayor peso suben el riesgo "
        + (enumerar(f'"{n}" (razón de odds {np.exp(v):.2f})' for n, v in sube.head(3).items()) or "ninguna")
        + " y lo bajan " + (enumerar(f'"{n}" ({np.exp(v):.2f})' for n, v in baja.head(3).items()) or "ninguna")
        + ". Eso sí, los coeficientes describen asociaciones dentro del modelo ajustado y no efectos causales. Cuando dos "
          "variables están correlacionadas el peso se distribuye entre ellas de forma inestable.",
        "hallazgo",
    ))
    if len(sube):
        latex.registrar_varias([("ClfVariableSube", sube.index[0], "variable de mayor peso que aumenta el riesgo"),
                                ("ClfOddsSube", float(np.exp(sube.iloc[0])), "su razón de odds", 2)])
    if len(baja):
        latex.registrar_varias([("ClfVariableBaja", baja.index[0], "variable de mayor peso que reduce el riesgo"),
                                ("ClfOddsBaja", float(np.exp(baja.iloc[0])), "su razón de odds", 2)])
    latex.exportar_tabla(odds, "clf_odds", "Ocho variables con mayor peso en la regresión logística", decimales=3)
    return r


def reglas_arbol(c):
    """Hojas de mayor riesgo del árbol podado."""
    r = Resultado()
    ajustado = clone(c.didacticos["Árbol podado"]).fit(c.X_ent, c.y_ent)
    nombres = tuberias.nombres_variables(ajustado)
    reglas = graficos_clf.reglas_arbol(ajustado.named_steps["modelo"], nombres,
                                       binarias=[n for n in nombres if n not in c.numericas], maximo=6)
    hojas = ajustado.named_steps["modelo"].get_n_leaves()
    r.agregar(tabla_html(reglas, f"Hojas de mayor riesgo del árbol podado ({hojas} hojas en total)", decimales=3,
                         barras="Fracción de positivos"))
    auc_arbol = c.por_modelo.loc["Árbol podado", "ROC-AUC"]
    primera = reglas.iloc[0]
    r.agregar(nota(
        f"La hoja de mayor riesgo agrupa {int(primera['Filas'])} muestras de entrenamiento con una fracción de no conformes "
        f"de {primera['Fracción de positivos']:.3f}, o sea {primera['Fracción de positivos'] / c.prev_ent:.1f} veces la "
        f"tasa general ({c.prev_ent:.3f}). Las reglas son fáciles de explicar en recepción pero el árbol ordena peor que "
        f"la regresión logística ({auc_arbol:.3f} frente a {c.auc_log:.3f}) porque divide el riesgo en pocos escalones en "
        "vez de sumar muchas señales pequeñas." if auc_arbol < c.auc_log else
        f"La hoja de mayor riesgo agrupa {int(primera['Filas'])} muestras de entrenamiento con una fracción de no conformes "
        f"de {primera['Fracción de positivos']:.3f} y el árbol ordena al menos tan bien como la regresión logística así "
        "que unas pocas reglas bastan para describir el riesgo.",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfHojaRiesgoFilas", int(primera["Filas"]), "muestras en la hoja de mayor riesgo del árbol"),
        ("ClfHojaRiesgoFraccion", float(primera["Fracción de positivos"]), "fracción de no conformes en esa hoja", 3),
        ("ClfHojaRiesgoVeces", float(primera["Fracción de positivos"] / c.prev_ent), "veces la tasa general", 1),
    ])
    latex.exportar_tabla(reglas, "clf_reglas_arbol", "Hojas de mayor riesgo del árbol podado", decimales=3)
    return r


def diseno(c):
    """Factores del L16, diseño y confusiones (con la interacción modelo × desbalance)."""
    r = Resultado()
    c.columnas_todas = c.numericas + c.categoricas
    c.k_seleccion = max(1, round(len(c.columnas_todas) / 2))
    c.discretas = taguchi.discretas(c.entrenamiento, c.numericas)
    c.factores = {
        "modelo": list(registro.MODELOS[TAREA]), "escalado": ["estandar", "minmax"], "grupo": ["base", "ampliado"],
        "desbalance": [None, "sobremuestreo"], "seleccion": [None, c.k_seleccion], "codificacion": ["one_hot", "ordinal"],
    }
    c.nombres_factor = list(c.factores)
    c.construir = taguchi.constructor(TAREA, c.papeles, c.discretas)
    arreglo, c.diseno_l16 = taguchi.diseno_l16(c.factores, COLUMNAS_L16)
    ortogonal = arreglos.verificar_ortogonalidad(arreglo)
    r.agregar(pasos([
        ("modelo (columna A, 4 niveles)", enumerar(registro.NOMBRES_LEGIBLES[m] for m in c.factores["modelo"]).replace(
            " y ", " o ") + "."),
        ("escalado (c)", "Estandarizar (restar la media y dividir entre la desviación estándar) o llevar al intervalo "
                         "[0, 1] las variables numéricas."),
        ("grupo (d)", f"Grupo base ({len(c.papeles['grupos']['base'])} variables) o ampliado ({len(c.columnas_todas)}, con "
                      "la historia del cliente y del producto y variables derivadas)."),
        ("desbalance (abd)", "Nada o sobremuestrear al azar los no conformes hasta igualar las clases, solo al ajustar."),
        ("seleccion (acd)", f"Sin selección o conservar las {c.k_seleccion} variables con mayor estadístico F."),
        ("codificacion (bcd)", "One-hot u ordinal para las categóricas y las discretas ("
         + ", ".join(f'"{x}"' for x in c.discretas) + ")."),
    ], titulo=f"Factores del L16 (ortogonalidad verificada: {'Sí' if ortogonal else 'No'})"))
    confusiones = arreglos.confusiones(COLUMNAS_L16)
    vista = taguchi.vista_confusiones(confusiones)
    r.agregar(pestanas({
        "Diseño con niveles reales": tabla_html(taguchi.vista_niveles(c.diseno_l16, c.nombres_factor)
                                                .reset_index(names="Corrida"), "Diseño L16 con los niveles reales",
                                                decimales=0, maximo_filas=16),
        "Confusiones": tabla_html(vista, "Dónde cae cada interacción de dos factores", decimales=0,
                                  resaltar=(vista["Confundida con"] == "libre").to_numpy()),
    }, titulo="El diseño que corrimos"))
    fila_md = confusiones[((confusiones["factor_1"] == "modelo") & (confusiones["factor_2"] == "desbalance"))
                          | ((confusiones["factor_1"] == "desbalance") & (confusiones["factor_2"] == "modelo"))]
    confusores = list(fila_md["confundida_con"].iloc[0]) if len(fila_md) else []
    libres = taguchi.interacciones_libres(confusiones)
    r.agregar(nota(
        f"De las {len(confusiones)} interacciones de dos factores {len(libres)} quedan libres ("
        + ", ".join(x.replace(" × ", " con ") for x in libres)
        + "). Igual que en la etapa 03 la asignación protege la interacción entre modelo y escalado porque el escalado "
          "cambia las distancias de KNN y el entrenamiento del perceptrón pero no afecta al árbol. La interacción entre "
          "modelo y desbalance, en cambio, "
        + (f"queda mezclada con {enumerar(chr(34) + x + chr(34) for x in confusores)}. Si existe, una parte se va a leer "
           + ("como efecto principal de ese factor" if len(confusores) == 1 else "como efecto principal de esos factores")
           + " y por eso la comprobamos aparte más abajo." if confusores else "queda libre.")
        + " Las demás interacciones las suponemos pequeñas y ese supuesto lo pone a prueba la corrida de confirmación.",
        "info", titulo="Qué se confunde y por qué lo aceptamos",
    ))
    latex.registrar_varias([
        ("ClfKSeleccion", c.k_seleccion, "variables que conserva la selección por F"),
        ("ClfInteraccionesLibres", len(libres), "interacciones que quedan libres en el L16"),
    ])
    latex.exportar_tabla(taguchi.vista_niveles(c.diseno_l16, c.nombres_factor).reset_index(names="Corrida"),
                         "clf_l16_diseno", "Diseño L16 de clasificación con los niveles reales", decimales=0)
    return r


def corridas_l16(c):
    r = Resultado()
    c.resultados_l16 = res = corridas.ejecutar_diseno(c.diseno_l16, c.construir, c.X_ent, c.y_ent, c.bloques, "roc_auc")
    c.segundos_l16 = float(res["segundos"].sum())
    columnas_bloque = [x for x in res.columns if x.startswith("bloque_")]
    vista = taguchi.vista_niveles(res[c.nombres_factor + columnas_bloque + ["media", "desv", "sn", "segundos"]],
                                  c.nombres_factor)
    c.mejor_corrida = mejor = res["sn"].idxmax()
    decimales = {**{x: 3 for x in columnas_bloque}, "media": 3, "desv": 3, "sn": 2, "segundos": 1}
    tabla_corridas = tabla_html(vista.reset_index(names="Corrida"), "Resultados del L16: ROC-AUC por bloque, media, "
                                "desviación y S/N (dB)", decimales=decimales, resaltar=(res.index == mejor),
                                columnas_resaltadas="sn", maximo_filas=16)
    fallidas = res[res["error"] != ""]
    r.agregar(nota(
        f"Las 16 corridas tardaron **{c.segundos_l16:.1f} s** en total"
        + (f" y {len(fallidas)} fallaron ({'; '.join(fallidas['error'])})." if len(fallidas) else ", sin fallos.")
        + f" La mejor corrida observada es la {mejor} ("
        + ", ".join(f"{f} = {legible(res.loc[mejor, f])}" for f in c.nombres_factor)
        + f"), con S/N {res.loc[mejor, 'sn']:.2f} dB y ROC-AUC medio {res.loc[mejor, 'media']:.3f}.",
        "hallazgo" if not len(fallidas) else "advertencia",
    ))
    c.efectos = corridas.efectos_principales(res, c.nombres_factor)
    c.respuesta = corridas.tabla_respuesta(res, c.nombres_factor)
    c.niveles_optimo, c.sn_predicha = corridas.predecir_optimo(res, c.nombres_factor)
    r.figura(graficos.figura_efectos_principales(c.efectos, float(res["sn"].mean()), optimo=c.niveles_optimo),
             "04_efectos_principales")
    vr = taguchi.respuesta_legible(c.respuesta, c.nombres_factor)
    vr_vista = vr.rename(columns={"mejor_nivel": "Mejor nivel", "media_mejor": "S/N mejor", "peor_nivel": "Peor nivel",
                                  "media_peor": "S/N peor", "delta": "Delta", "rango": "Rango"})
    r.agregar(pestanas({"Tabla de respuesta": tabla_html(vr_vista, "Tabla de respuesta de Taguchi (S/N en dB)",
                                                         decimales=3, barras="Delta"),
                        "Las 16 corridas": tabla_corridas}, titulo="Detalle del L16"))
    interaccion = res.pivot_table(index="modelo", columns="escalado", values="sn", aggfunc="mean")
    efecto_escalado = (interaccion["estandar"] - interaccion["minmax"]).abs()
    primero, segundo = vr.iloc[0], vr.iloc[1]
    indiferentes = c.respuesta.loc[c.respuesta["delta"] < 0.1 * c.respuesta["delta"].max(), "delta"]
    r.agregar(nota(
        f"El factor que más pesa es \"{primero['Factor']}\" (delta de {primero['delta']:.3f} dB, mejor nivel "
        f"\"{primero['mejor_nivel']}\"), seguido de \"{segundo['Factor']}\" ({segundo['delta']:.3f} dB, mejor nivel "
        f"\"{segundo['mejor_nivel']}\"). "
        + f"Como la interacción entre modelo y escalado quedó libre podemos leerla por separado: Cambiar el escalado mueve "
          f"la S/N {efecto_escalado.max():.3f} dB en \"{efecto_escalado.idxmax()}\" y solo {efecto_escalado.min():.3f} dB "
          f"en \"{efecto_escalado.idxmin()}\". "
        + (("Los factores " if len(indiferentes) > 1 else "El factor ") + enumerar(
            f'"{f}" (delta {d:.3f} dB)' for f, d in indiferentes.items())
           + (" son prácticamente indiferentes" if len(indiferentes) > 1 else " es prácticamente indiferente")
           + " (menos del 10 % del delta mayor) así que su nivel óptimo lo deciden diferencias sin peso práctico."
           if len(indiferentes) else "")
        + "\n\nEl óptimo que predice el modelo aditivo es "
        + ", ".join(f"{f} = {legible(v)}" for f, v in c.niveles_optimo.items())
        + f", con S/N predicha de **{c.sn_predicha:.3f} dB**. Si todos los bloques dieran lo mismo eso equivaldría a un "
          f"ROC-AUC de {10 ** (c.sn_predicha / 20):.3f}.",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfSegundosLSixteen", c.segundos_l16, "segundos de las 16 corridas del L16", 1),
        ("ClfMejorCorrida", int(mejor), "número de la mejor corrida observada del L16"),
        ("ClfSNMejorCorrida", float(res.loc[mejor, "sn"]), "S/N (dB) de la mejor corrida", 2),
        ("ClfAUCMejorCorrida", float(res.loc[mejor, "media"]), "ROC-AUC medio de la mejor corrida", 3),
        ("ClfFactorPrincipal", primero["Factor"], "factor más influyente (mayor delta)"),
        ("ClfDeltaPrincipal", float(primero["delta"]), "delta (dB) del factor más influyente", 3),
        ("ClfFactorSegundo", segundo["Factor"], "segundo factor más influyente"),
        ("ClfDeltaSegundo", float(segundo["delta"]), "delta (dB) del segundo factor", 3),
        ("ClfSNPredicha", c.sn_predicha, "S/N (dB) predicha para el óptimo aditivo", 3),
        ("ClfAUCEquivalentePredicha", 10 ** (c.sn_predicha / 20), "ROC-AUC equivalente a la S/N predicha", 3),
        ("ClfOptimo", ", ".join(f"{f} = {legible(v)}" for f, v in c.niveles_optimo.items()), "óptimo aditivo del L16"),
    ])
    del_modelo = vr.set_index("Factor").loc["modelo"]
    latex.registrar_varias([
        ("ClfModeloMejorNivel", registro.NOMBRES_LEGIBLES.get(del_modelo["mejor_nivel"], del_modelo["mejor_nivel"]).lower(),
         "mejor nivel del factor modelo en el L16"),
        ("ClfModeloPeorNivel", registro.NOMBRES_LEGIBLES.get(del_modelo["peor_nivel"], del_modelo["peor_nivel"]).lower(),
         "peor nivel del factor modelo en el L16"),
        ("ClfSNModeloMejor", float(del_modelo["media_mejor"]), "S/N media (dB) del mejor modelo en el L16", 3),
        ("ClfSNModeloPeor", float(del_modelo["media_peor"]), "S/N media (dB) del peor modelo en el L16", 3),
    ])
    latex.exportar_tabla(vista.reset_index(names="Corrida"), "clf_l16_resultados", "Resultados del L16 de clasificación",
                         decimales=decimales)
    latex.exportar_tabla(vr_vista, "clf_respuesta", "Tabla de respuesta de Taguchi (clasificación, S/N en dB)",
                         decimales=3)
    return r


def sobremuestreo(c):
    """Comprobación dirigida de la interacción modelo × sobremuestreo con los modelos didácticos."""
    r = Resultado()
    filas = []
    for nombre in c.didacticos:
        tuberia = tuberias.construir_tuberia(
            TAREA, CLAVES_DIDACTICAS[nombre], c.numericas, c.categoricas,
            escalado="ninguno" if nombre == "Árbol podado" else "estandar", desbalance="sobremuestreo",
            hiper=({"max_depth": None, "min_samples_leaf": 20, "ccp_alpha": c.alfa} if nombre == "Árbol podado" else None))
        predichas = fuera_de_bloque(tuberia, c.X_ent, c.y_ent, c.bloques)
        resumen = resumir_bloques(predichas, c.y_ent)
        _, proba_sin = apilar_bloques(c.predichas[nombre], c.y_ent)
        y_apilada, proba_con = apilar_bloques(predichas, c.y_ent)
        sin = c.metricas_didacticas[nombre]
        filas.append({
            "Modelo": nombre, "AUC sin": sin["ROC-AUC"].mean(), "AUC con": resumen["ROC-AUC"].mean(),
            "Δ AUC": resumen["ROC-AUC"].mean() - sin["ROC-AUC"].mean(),
            "Brier sin": sin["Brier"].mean(), "Brier con": resumen["Brier"].mean(),
            "Prob. media sin": float(np.mean(proba_sin)), "Prob. media con": float(np.mean(proba_con)),
        })
    c.tabla_sobre = t = pd.DataFrame(filas)
    tasa = float(np.mean(y_apilada))
    r.agregar(tabla_html(t, f"Cada modelo didáctico sin y con sobremuestreo (media en los bloques, tasa real de no "
                            f"conformes en los tramos: {tasa:.3f})", decimales=3, columnas_resaltadas="Δ AUC"))
    orden = t.sort_values("Δ AUC")
    perdedor, segundo, menor = orden.iloc[0], orden.iloc[1], orden.iloc[-1]
    texto_auc = (
        f"El sobremuestreo le quita más ROC-AUC a \"{perdedor['Modelo']}\" ({perdedor['Δ AUC']:+.3f}) que a cualquier "
        f"otro modelo. Le sigue \"{segundo['Modelo']}\" ({segundo['Δ AUC']:+.3f}) y en el otro extremo "
        f"\"{menor['Modelo']}\" casi no cambia ({menor['Δ AUC']:+.3f}). O sea que sí hay una interacción entre modelo y "
        "desbalance. "
        + ("En el perceptrón encaja con que las copias se filtran hacia su validación interna y en el árbol con que las "
           "copias exactas de los no conformes le dejan aislar hojas pequeñas que no generalizan. La regresión logística "
           "apenas se mueve porque repetir filas desplaza sobre todo su intercepto."
           if perdedor["Modelo"] == "Perceptrón multicapa" else
           "No la encabeza el perceptrón así que la filtración de copias hacia su validación interna no es aquí el efecto "
           "que más pesa.")
    )
    sube = bool((t["Prob. media con"] > t["Prob. media sin"]).all())
    empeora = bool((t["Brier con"] > t["Brier sin"]).all())
    texto_cal = (
        f"En todos los modelos la probabilidad media sube con el sobremuestreo (pasa de entre "
        f"{t['Prob. media sin'].min():.3f} y {t['Prob. media sin'].max():.3f} a entre {t['Prob. media con'].min():.3f} y "
        f"{t['Prob. media con'].max():.3f}, con una tasa real de {tasa:.3f})" + (" y el Brier empeora" if empeora else "")
        + ". Es la descalibración que esperábamos: Aunque el orden de las muestras cambie poco las cifras dejan de ser "
          "probabilidades y la parte B necesita probabilidades." if sube else
        "La probabilidad media no sube en todos los modelos con el sobremuestreo así que la descalibración no es pareja y "
        "la revisamos más adelante con el diagrama de fiabilidad."
    )
    r.agregar(nota(texto_auc + "\n\n" + texto_cal, "hallazgo"))
    latex.registrar_varias([
        ("ClfSobreMasPierde", perdedor["Modelo"], "modelo al que el sobremuestreo le quita más ROC-AUC"),
        ("ClfSobreDeltaAUC", float(perdedor["Δ AUC"]), "cambio de su ROC-AUC con sobremuestreo", 3),
        ("ClfSobreMenosPierde", menor["Modelo"], "modelo que casi no cambia con sobremuestreo"),
        ("ClfSobreDeltaAUCMenor", float(menor["Δ AUC"]), "cambio de su ROC-AUC con sobremuestreo", 3),
        ("ClfTasaTramos", tasa, "tasa real de no conformes en los tramos de validación de los bloques", 3),
    ])
    latex.exportar_tabla(t, "clf_sobremuestreo", "Modelos didácticos sin y con sobremuestreo", decimales=3)
    return r


def confirmacion(c):
    r = Resultado()
    res = c.resultados_l16
    c.conf = conf = taguchi.confirmar(c.niveles_optimo, c.construir, c.X_ent, c.y_ent, c.bloques, "roc_auc", res,
                                      c.efectos, c.factores)
    obs, tol = conf["sn_observada"], conf["tolerancia"]
    diferencia = obs - c.sn_predicha
    c.coincide = abs(diferencia) <= tol
    mejor = c.mejor_corrida
    r.agregar(comparacion([
        {"titulo": "S/N predicha", "valor": c.sn_predicha, "etiqueta": "dB", "puntos": ["modelo aditivo del L16"]},
        {"titulo": "S/N observada", "valor": obs, "etiqueta": "dB",
         "puntos": [f"ROC-AUC medio {conf['fila']['media']:.3f}", f"desviación entre bloques {conf['fila']['desv']:.3f}"]},
        {"titulo": "Mejor corrida del L16", "valor": float(res.loc[mejor, "sn"]), "etiqueta": "dB",
         "puntos": [f"corrida {mejor}", f"ROC-AUC medio {res.loc[mejor, 'media']:.3f}"]},
    ], titulo="Confirmación del óptimo", decimales=3,
        elegida="S/N observada" if obs >= res["sn"].max() else "Mejor corrida del L16", sello="Mayor S/N"))
    r.agregar(nota(
        f"La S/N observada ({obs:.3f} dB) difiere de la predicha en {diferencia:+.3f} dB, "
        + (f"dentro de la tolerancia de ±{tol:.3f} dB. O sea que el modelo aditivo describe bien la tubería y las "
           "interacciones confundidas no parecen pesar en la combinación óptima." if c.coincide else
           f"fuera de la tolerancia de ±{tol:.3f} dB. Eso indica que alguna interacción confundida pesa (la de modelo y "
           "desbalance es la sospechosa natural) y que el óptimo aditivo hay que tomarlo con cautela.")
        + (" Además la confirmación le gana a la mejor de las 16 corridas." if obs > res["sn"].max()
           else " Con todo, la confirmación no le gana a la mejor corrida observada del L16."),
        "hallazgo" if c.coincide else "advertencia", titulo="Predicho frente a observado",
    ))
    c.resumen_completo, detalle = taguchi.factorial_completo(
        c.factores, c.segundos_l16, len(c.diseno_l16), c.construir, c.X_ent, c.y_ent, c.bloques, "roc_auc", c.respuesta,
        c.niveles_optimo)
    rc = c.resumen_completo
    if detalle is not None:
        r.figura(graficos.figura_deltas(pd.DataFrame({
            "L16 (16 corridas)": c.respuesta["delta"],
            f"Factorial completo ({rc['corridas']} corridas)": detalle["respuesta"].loc[c.respuesta.index, "delta"]}),
            "Influencia de cada factor según el diseño"), "04_l16_contra_completo")
        r.agregar(nota(
            f"El factorial completo tardó {rc['segundos']:.1f} s frente a {c.segundos_l16:.1f} s del L16. La correlación de "
            f"rangos entre los deltas de los dos diseños es **{rc['correlacion_rangos']:.2f}** y coinciden en el mejor nivel "
            f"de {len(rc['factores_mismo_optimo'])} de los {len(c.nombres_factor)} factores.",
            "hallazgo",
        ))
    else:
        tiempos = res.assign(desbalance=[legible(v) for v in res["desbalance"]]).groupby(
            ["modelo", "desbalance"])["segundos"].sum().sort_values(ascending=False)
        lenta = tiempos.index[0]
        r.agregar(nota(
            f"Con {c.segundos_l16:.1f} s para el L16 el factorial completo tardaría unos {rc['segundos_estimados']:.0f} s, "
            f"por encima del límite de {taguchi.LIMITE_SEGUNDOS_COMPLETO:.0f} s que fijamos para la etapa. Por eso no lo "
            f"corrimos. En el L16 la combinación más lenta es \"{lenta[0]}\" con desbalance \"{legible(lenta[1])}\", que "
            f"se lleva el {100 * tiempos.iloc[0] / c.segundos_l16:.0f} % del tiempo. La comparación entre el atajo y el "
            "diseño completo ya la hicimos en la etapa 03 y aquí es la corrida de confirmación la que pone a prueba el "
            "modelo aditivo.",
            "limite",
        ))
    latex.registrar_varias([
        ("ClfSNObservada", obs, "S/N (dB) observada en la corrida de confirmación", 3),
        ("ClfToleranciaSN", tol, "tolerancia (dB) de la confirmación", 3),
        ("ClfConfirmaAditivo", bool(c.coincide), "la confirmación cae dentro de la tolerancia"),
        ("ClfAUCConfirmacion", float(conf["fila"]["media"]), "ROC-AUC medio de la corrida de confirmación", 3),
        ("ClfCompletoEjecutado", rc["ejecutado"], "se ejecutó el factorial completo"),
        ("ClfSegundosCompletoEstimados", rc["segundos_estimados"], "segundos estimados del factorial completo", 0),
    ])
    return r


def afinamiento(c):
    r = Resultado()
    c.configuracion, origen, c.candidatos = taguchi.configuracion_y_candidatos(
        c.conf, c.niveles_optimo, c.resultados_l16, c.efectos, c.nombres_factor)
    c.busquedas, c.tablas_rejilla, c.segundos_rejilla = taguchi.buscar_rejillas(
        TAREA, c.candidatos, c.construir, c.X_ent, c.y_ent, c.bloques, "roc_auc", n_trabajos=-1)
    r.agregar(pestanas({
        registro.NOMBRES_LEGIBLES[m]: tabla_html(
            t.drop(columns=[x for x in t.columns if x.startswith("bloque_")]).head(8),
            f"Mejores combinaciones de la rejilla ({len(t)} en total, {c.segundos_rejilla[m]:.1f} s)", decimales=4)
        for m, t in c.tablas_rejilla.items()}, titulo="Búsqueda en rejilla sobre los bloques temporales (ROC-AUC)"))
    r.agregar(nota(
        f"La configuración de partida sale de la {origen} ("
        + ", ".join(f"{f} = {legible(v)}" for f, v in c.configuracion.items() if f != "modelo") + "). "
        + " ".join(
            f"En \"{registro.NOMBRES_LEGIBLES[m]}\" la rejilla elige "
            + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {legible(v)}"
                        for k, v in taguchi.hiper_de(b).items())
            + f", con ROC-AUC medio de {c.tablas_rejilla[m].iloc[0]['media']:.3f} en los bloques."
            for m, b in c.busquedas.items()),
        "hallazgo",
    ))
    latex.registrar("ClfOrigenConfiguracion", origen, "de dónde sale la configuración que pasa al afinamiento")
    _actualizar_paneles(c)
    for m in c.busquedas:
        clave = {"logistica": "Logistica", "arbol": "Arbol", "knn": "KNN", "mlp": "MLP"}[m]
        latex.registrar(f"ClfRejilla{clave}AUC", float(c.tablas_rejilla[m].iloc[0]["media"]),
                        f"ROC-AUC medio por bloque de la mejor combinación de la rejilla de {m}", 3)
    return r


def evaluacion_validacion(c):
    r = Resultado()
    c.proba_val = {m: metricas.prediccion_de(b.best_estimator_, c.X_val) for m, b in c.busquedas.items()}
    c.auc_val = {m: metricas.roc_auc(c.y_val, p) for m, p in c.proba_val.items()}
    ic = {m: intervalo_bootstrap(metricas.roc_auc, c.y_val, p) for m, p in c.proba_val.items()}
    mayor = max(c.auc_val, key=c.auc_val.get)
    empatados = [m for m in c.auc_val if c.auc_val[m] >= c.auc_val[mayor] - TOLERANCIA_EMPATE]
    c.final = final = min(empatados, key=ORDEN_SENCILLEZ.index)
    otro = next(m for m in c.auc_val if m != final)
    c.tabla_val = t = pd.DataFrame([{
        "Candidato": registro.NOMBRES_LEGIBLES[m], "ROC-AUC": c.auc_val[m], "AUC inf.": ic[m][1], "AUC sup.": ic[m][2],
        "Precisión media": metricas.precision_media(c.y_val, p), "Brier": metricas.brier(c.y_val, p)}
        for m, p in c.proba_val.items()]
        + [{"Candidato": NOMBRE_DUMMY, "ROC-AUC": 0.5, "AUC inf.": np.nan, "AUC sup.": np.nan,
            "Precisión media": c.prev_val, "Brier": metricas.brier(c.y_val, np.full(len(c.y_val), c.prev_ent))}])
    r.agregar(tabla_html(t, f"Bloque de validación ({len(c.y_val)} muestras, intervalos bootstrap al 95 %)", decimales=4,
                         resaltar=(t["Candidato"] == registro.NOMBRES_LEGIBLES[final]).to_numpy(),
                         columnas_resaltadas="ROC-AUC"))
    rejilla = c.tablas_rejilla[final].iloc[0]["media"]
    r.agregar(nota(
        f"Según la regla el candidato final es \"{registro.NOMBRES_LEGIBLES[final]}\", con ROC-AUC de validación "
        f"**{c.auc_val[final]:.3f}** (intervalo de {ic[final][1]:.3f} a {ic[final][2]:.3f}) frente a "
        f"{c.auc_val[otro]:.3f} de \"{registro.NOMBRES_LEGIBLES[otro]}\""
        + (f" (la diferencia es menor que {TOLERANCIA_EMPATE} y preferimos el más sencillo)." if final != mayor else ".")
        + (f" Al lado del ROC-AUC medio de la rejilla ({rejilla:.3f}) el de validación es "
           + ("parecido, o sea que el modelo se sostiene en un periodo que no vio."
              if abs(c.auc_val[final] - rejilla) < 0.02 else
              ("mayor" if c.auc_val[final] > rejilla else "menor")
              + ", una diferencia que viene tanto del modelo (entrenado con más filas) como del periodo de validación.")),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfModeloFinal", registro.NOMBRES_LEGIBLES[final], "modelo de clasificación congelado"),
        ("ClfModeloOtro", registro.NOMBRES_LEGIBLES[otro], "otro candidato evaluado en validación"),
        ("ClfAUCValidacionSinCalibrar", c.auc_val[final], "ROC-AUC de validación del final (antes de calibrar)", 3),
        ("ClfAUCValidacionOtro", c.auc_val[otro], "ROC-AUC de validación del otro candidato", 3),
        ("ClfAUCRejillaFinal", float(rejilla), "ROC-AUC medio por bloque del final en la rejilla", 3),
        ("ClfPorEmpate", bool(final != mayor), "el final se eligió por la regla de empate"),
    ])
    latex.exportar_tabla(t, "clf_validacion", "Candidatos de clasificación en validación (intervalos bootstrap)",
                         decimales=4)
    return r


def calibracion(c):
    """Calibración sigmoide con los bloques temporales y regla para usarla."""
    r = Resultado()
    c.modelo_base = c.busquedas[c.final].best_estimator_
    calibrado = CalibratedClassifierCV(clone(c.modelo_base), method=METODO_CALIBRACION, cv=c.bloques).fit(c.X_ent, c.y_ent)
    proba_cal = metricas.prediccion_de(calibrado, c.X_val)
    proba_sin = c.proba_val[c.final]
    c.brier_sin, c.brier_cal = metricas.brier(c.y_val, proba_sin), metricas.brier(c.y_val, proba_cal)
    c.auc_cal = metricas.roc_auc(c.y_val, proba_cal)
    sobremuestrea = c.candidatos[c.final]["desbalance"] == "sobremuestreo"
    c.usar_calibrado = bool(sobremuestrea or c.brier_cal < c.brier_sin)
    c.p_val = p = proba_cal if c.usar_calibrado else proba_sin
    auc_sin = c.auc_val[c.final]
    r.figura(graficos_clf.figura_fiabilidad(
        {"Sin calibrar": (c.y_val, proba_sin), "Calibrado (sigmoide, bloques temporales)": (c.y_val, proba_cal)},
        n_tramos=10, titulo="Fiabilidad en validación (10 tramos con el mismo número de muestras)"), "04_fiabilidad")
    r.agregar(comparacion([
        {"titulo": "Sin calibrar", "valor": c.brier_sin, "etiqueta": "Brier en validación", "decimales": 4,
         "puntos": [f"ROC-AUC {auc_sin:.3f}", f"probabilidad media {proba_sin.mean():.3f}"]},
        {"titulo": "Calibrado", "valor": c.brier_cal, "etiqueta": "Brier en validación", "decimales": 4,
         "puntos": [f"ROC-AUC {c.auc_cal:.3f}", f"probabilidad media {proba_cal.mean():.3f}"]},
    ], titulo=f"Calibración (tasa real de no conformes en validación: {c.prev_val:.3f})",
        elegida="Calibrado" if c.usar_calibrado else "Sin calibrar", sello="Se usa"))
    r.agregar(nota(
        (f"La tubería elegida sobremuestrea así que por la regla usamos la versión calibrada: Su probabilidad media pasa "
         f"de {proba_sin.mean():.3f} a {proba_cal.mean():.3f} y el Brier de {c.brier_sin:.4f} a {c.brier_cal:.4f}."
         if sobremuestrea else
         "La tubería elegida no sobremuestrea y la calibración "
         + (f"baja el Brier de {c.brier_sin:.4f} a {c.brier_cal:.4f} así que usamos la versión calibrada."
            if c.usar_calibrado else
            f"no mejora el Brier ({c.brier_cal:.4f} frente a {c.brier_sin:.4f}) así que nos quedamos con el modelo sin "
            "calibrar."))
        + f" El ROC-AUC pasa de {auc_sin:.3f} a {c.auc_cal:.3f}"
        + (", un cambio despreciable porque la calibración sigmoide no altera el orden y promediar los cuatro modelos de "
           "bloque apenas lo mueve." if abs(c.auc_cal - auc_sin) < 0.005 else
           " porque la versión calibrada promedia cuatro modelos ajustados con ventanas más cortas del entrenamiento.")
        + f" La probabilidad media en validación queda en {p.mean():.3f} frente a una tasa real de {c.prev_val:.3f}"
        + (f" (en el entrenamiento era {c.prev_ent:.3f}), "
           + ("o sea que el modelo calibrado sobrestima un poco el riesgo medio del periodo nuevo. Para el umbral eso pesa "
              "poco porque lo fijamos con los costos observados en la propia validación."
              if p.mean() > c.prev_val * 1.05 else
              "o sea que el modelo calibrado subestima un poco el riesgo medio del periodo nuevo. Para el umbral eso pesa "
              "poco porque lo fijamos con los costos observados en la propia validación."
              if p.mean() < c.prev_val * 0.95 else
              "prácticamente la misma cifra. O sea que la calibración se sostiene en un periodo que no vio.")),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfSobremuestrea", bool(sobremuestrea), "la tubería elegida sobremuestrea"),
        ("ClfUsaCalibrado", c.usar_calibrado, "se usa la versión calibrada"),
        ("ClfBrierSinCalibrar", c.brier_sin, "Brier de validación sin calibrar", 4),
        ("ClfBrierCalibrado", c.brier_cal, "Brier de validación calibrado", 4),
        ("ClfAUCCalibrado", c.auc_cal, "ROC-AUC de validación calibrado", 3),
        ("ClfProbMediaSinCalibrar", float(proba_sin.mean()), "probabilidad media en validación sin calibrar", 3),
        ("ClfProbMediaCalibrado", float(proba_cal.mean()), "probabilidad media en validación calibrada", 3),
    ])
    return r


def umbral_por_costo(c):
    """Recorrido de umbrales en validación, elección por costo y comparación con políticas sin modelo."""
    r = Resultado()
    fp, fn = config.COSTO_FP, config.COSTO_FN
    umbrales = sorted(set(np.round(np.arange(0.02, 0.501, 0.02), 2).tolist() + [round(c.umbral_teorico, 3)]))
    c.tabla_umbral = t = umbral.tabla_umbrales(c.y_val, c.p_val, umbrales)
    t["Alertas por semana"] = t["Alertas"] / c.semanas_val
    c.fila_umbral = f = umbral.elegir_umbral(t)
    c.umbral = u = float(f["Umbral"])
    teorico = t.loc[np.isclose(t["Umbral"], round(c.umbral_teorico, 3))].iloc[0]
    vista = t[["Umbral", "Alertas", "Alertas por semana", "Precisión", "Recall", "FP", "FN", "Costo"]]
    decimales = {"Umbral": 3, "Alertas por semana": 1, "Precisión": 3, "Recall": 3}
    tabla_umbrales = tabla_html(vista, f"Recorrido de umbrales en validación ({c.semanas_val} semanas, costo = {fp} FP + "
                                f"{fn} FN)", decimales=decimales, resaltar=np.isclose(t["Umbral"], u),
                                columnas_resaltadas="Costo", maximo_filas=30)
    r.figura(graficos_clf.figura_umbrales(t, u, c.umbral_teorico), "04_umbrales")
    c.paneles.proba_validacion, c.paneles.y_validacion = c.p_val, c.y_val
    _actualizar_paneles(c)

    positivos, negativos = int(c.y_val.sum()), int(len(c.y_val) - c.y_val.sum())
    c.costo_nada, c.costo_todo = fn * positivos, fp * negativos
    c.auc_final = metricas.roc_auc(c.y_val, c.p_val)
    c.ic_auc = intervalo_bootstrap(metricas.roc_auc, c.y_val, c.p_val)
    c.ic_costo = ic = intervalo_bootstrap(lambda y, p: metricas.costo(y, p, u), c.y_val, c.p_val)
    r.agregar(comparacion([
        {"titulo": "No alertar nunca", "valor": c.costo_nada, "etiqueta": "costo en validación", "decimales": 0,
         "puntos": ["0 alertas por semana", f"{positivos} no conformes sin aviso"]},
        {"titulo": f"Modelo con umbral {u:.3f}", "valor": float(f["Costo"]), "etiqueta": "costo en validación",
         "decimales": 0, "puntos": [f"{f['Alertas por semana']:.1f} alertas por semana",
                                    f"precisión {f['Precisión']:.3f}, recall {f['Recall']:.3f}",
                                    f"intervalo del costo de {ic[1]:.0f} a {ic[2]:.0f}"]},
        {"titulo": "Alertar todas", "valor": c.costo_todo, "etiqueta": "costo en validación", "decimales": 0,
         "puntos": [f"{len(c.y_val) / c.semanas_val:.1f} alertas por semana", f"precisión {c.prev_val:.3f}"]},
    ], titulo="El umbral elegido frente a las políticas sin modelo", elegida=1, sello="Menor costo"))
    confusion = graficos_clf.figura_confusion(int(f["TN"]), int(f["FP"]), int(f["FN"]), int(f["TP"]),
                                              f"Matriz de confusión en validación (umbral {u:.3f})")
    guardar_figura(confusion, "04_confusion")
    plt.close(confusion)
    r.agregar(pestanas({"Matriz de confusión": confusion, "Recorrido de umbrales": tabla_umbrales},
                       titulo="El umbral elegido por dentro"))
    trivial = min(c.costo_nada, c.costo_todo)
    r.agregar(nota(
        f"El umbral de menor costo es **{u:.3f}** "
        + (f"y coincide con el teórico ({fp} / ({fp} + {fn})), lo que confirma en la práctica que las probabilidades "
           "calibradas se pueden usar tal cual para decidir" if np.isclose(u, round(c.umbral_teorico, 3)) else
           f"(el teórico, {c.umbral_teorico:.3f}, cuesta {teorico['Costo']:.0f} frente a {f['Costo']:.0f}, una diferencia "
           "pequeña que confirma que las probabilidades están razonablemente calibradas)"
           if abs(teorico["Costo"] - f["Costo"]) <= 0.03 * f["Costo"] else
           f"(el teórico, {c.umbral_teorico:.3f}, cuesta {teorico['Costo']:.0f} frente a {f['Costo']:.0f} así que la "
           "calibración no es perfecta y conviene fijar el umbral con datos)")
        + f". Con él se levantan **{f['Alertas por semana']:.1f} alertas por semana** y acierta el "
          f"{100 * f['Precisión']:.0f} % de ellas (precisión {f['Precisión']:.3f}). De los no conformes alerta el "
          f"{100 * f['Recall']:.0f} %. Dicho de otra forma, por cada no conforme que detecta se revisan "
          f"{f['FP'] / max(f['TP'], 1):.1f} muestras que salen conformes.\n\n"
        + (f"El costo baja de {trivial:.0f} (la mejor política sin modelo) a {f['Costo']:.0f}, un ahorro del "
           f"{100 * (1 - f['Costo'] / trivial):.0f} %. El intervalo bootstrap del costo va de {ic[1]:.0f} a {ic[2]:.0f}"
           + (", todo por debajo de esa política." if ic[2] < trivial else
              " así que llega a tocar el costo de esa política y el ahorro no es seguro.")
           if f["Costo"] < trivial else
           f"El modelo no mejora a la mejor política sin modelo ({trivial:.0f}) así que con estos costos no conviene "
           "usarlo para alertar.")
        + " Eso sí, como elegimos el umbral en estos mismos datos el costo de validación es algo optimista. La etapa 06 lo "
          "mide en la prueba con el umbral ya fijo.",
        "hallazgo" if f["Costo"] < trivial else "advertencia",
    ))
    latex.registrar_varias([
        ("ClfUmbral", u, "umbral de alerta elegido por costo en validación", 3),
        ("ClfUmbralCosto", float(f["Costo"]), "costo en validación con el umbral elegido", 0),
        ("ClfUmbralCostoInf", float(ic[1]), "límite inferior del IC 95 % del costo", 0),
        ("ClfUmbralCostoSup", float(ic[2]), "límite superior del IC 95 % del costo", 0),
        ("ClfUmbralTeoricoCosto", float(teorico["Costo"]), "costo en validación con el umbral teórico", 0),
        ("ClfAlertasSemana", float(f["Alertas por semana"]), "alertas por semana con el umbral elegido", 1),
        ("ClfAlertasValidacion", int(f["Alertas"]), "alertas totales en validación"),
        ("ClfPrecisionUmbral", float(f["Precisión"]), "precisión con el umbral elegido", 3),
        ("ClfRecallUmbral", float(f["Recall"]), "recall con el umbral elegido", 3),
        ("ClfFUnoUmbral", float(f["F1"]), "F1 con el umbral elegido", 3),
        ("ClfTP", int(f["TP"]), "verdaderos positivos en validación"),
        ("ClfFP", int(f["FP"]), "falsos positivos en validación"),
        ("ClfFN", int(f["FN"]), "falsos negativos en validación"),
        ("ClfTN", int(f["TN"]), "verdaderos negativos en validación"),
        ("ClfRevisadasPorDetectado", float(f["FP"] / max(f["TP"], 1)), "conformes revisadas por no conforme detectado", 1),
        ("ClfCostoNoAlertar", c.costo_nada, "costo en validación de no alertar nunca"),
        ("ClfCostoAlertarTodo", c.costo_todo, "costo en validación de alertar todas"),
        ("ClfAhorroPct", 100 * (1 - float(f["Costo"]) / trivial), "ahorro (%) frente a la mejor política sin modelo", 0),
    ])
    latex.exportar_tabla(vista, "clf_umbrales", "Recorrido de umbrales en validación", decimales=decimales)
    return r


def congelar(c):
    r = Resultado()
    X_des = c.desarrollo[c.numericas + c.categoricas]
    y_des = c.desarrollo[c.objetivo].to_numpy(dtype=int)
    base = clone(c.modelo_base)
    if c.usar_calibrado:
        c.modelo_final = CalibratedClassifierCV(base, method=METODO_CALIBRACION,
                                                cv=validacion.bloques_temporales(c.desarrollo[c.fecha])).fit(X_des, y_des)
    else:
        c.modelo_final = base.fit(X_des, y_des)
    guardar_modelo(c.modelo_final, rutas.DIR_ARTEFACTOS / "modelos" / "clasificacion.joblib")
    c.configuracion_final = {**c.candidatos[c.final]}
    f = c.fila_umbral
    r.agregar(tarjetas([
        {"titulo": "Modelo congelado", "valor": c.auc_final, "decimales": 3, "prefijo": "AUC ",
         "detalle": registro.NOMBRES_LEGIBLES[c.final] + (" calibrado" if c.usar_calibrado else "")},
        {"titulo": "Intervalo del AUC", "valor": c.ic_auc[1], "decimales": 3, "color": "verde",
         "detalle": f"hasta {c.ic_auc[2]:.3f} (bootstrap en validación)"},
        {"titulo": "Umbral de alerta", "valor": c.umbral, "decimales": 3, "color": "naranja",
         "detalle": f"{f['Alertas por semana']:.1f} alertas por semana, precisión {f['Precisión']:.2f}"},
        {"titulo": "Filas del reajuste", "valor": len(c.desarrollo), "decimales": 0, "color": "morado",
         "detalle": "entrenamiento + validación"},
    ]))
    hiper = taguchi.hiper_de(c.busquedas[c.final])
    c.taguchi = taguchi.taguchi_para_artefacto(c.factores, COLUMNAS_L16, c.respuesta, c.niveles_optimo, c.sn_predicha,
                                               c.conf, c.coincide, c.segundos_l16, c.resumen_completo,
                                               efectos=c.efectos, resultados=c.resultados_l16)
    latex.registrar_varias([
        ("ClfAUCValidacion", c.auc_final, "ROC-AUC de validación del modelo que se congela", 3),
        ("ClfAUCValidacionInf", c.ic_auc[1], "límite inferior del IC 95 % del ROC-AUC en validación", 3),
        ("ClfAUCValidacionSup", c.ic_auc[2], "límite superior del IC 95 % del ROC-AUC en validación", 3),
        ("ClfPrecisionMediaValidacion", metricas.precision_media(c.y_val, c.p_val), "precisión media en validación", 3),
        ("ClfFilasReajuste", len(c.desarrollo), "filas del reajuste final (entrenamiento + validación)"),
        ("ClfConfiguracionFinal", ", ".join(f"{k} = {legible(v)}" for k, v in c.configuracion_final.items()),
         "configuración congelada"),
        ("ClfHiperparametros", ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {legible(v)}"
                                         for k, v in hiper.items()) or "ninguno", "hiperparámetros congelados"),
    ])
    return r


# ----------------------------------------------------------------------
# Parte B: Bayes
# ----------------------------------------------------------------------
def sensibilidad_especificidad(c):
    r = Resultado()
    controles = particion.solo_desarrollo(pd.read_parquet(rutas.DIR_LIMPIOS / "controles.parquet"), "fecha")
    c.sens_espec = se = bayes.estimar_sens_espec(controles).set_index("ensayo")
    vista = se.reset_index().rename(columns={
        "ensayo": "Ensayo", "n_positivos": "Controles +", "sensibilidad": "Sensibilidad", "sens_inf": "Sens. inf.",
        "sens_sup": "Sens. sup.", "n_negativos": "Controles −", "especificidad": "Especificidad",
        "espec_inf": "Espec. inf.", "espec_sup": "Espec. sup."})
    r.agregar(tabla_html(vista, f"Sensibilidad y especificidad por ensayo ({len(controles)} controles del desarrollo)",
                         decimales=3))
    menos_esp, menos_sens = se["especificidad"].idxmin(), se["sensibilidad"].idxmin()
    r.agregar(nota(
        f"La especificidad va de {se['especificidad'].min():.3f} (\"{menos_esp}\") a {se['especificidad'].max():.3f} y "
        f"la sensibilidad de {se['sensibilidad'].min():.3f} (\"{menos_sens}\") a {se['sensibilidad'].max():.3f}. Eso sí, "
        "un control positivo lleva una carga alta y conocida y un control negativo se manipula en condiciones ideales. O "
        "sea que los dos números describen el ensayo en su mejor día: Con muestras reales la sensibilidad puede ser menor "
        "(cargas bajas) y la especificidad también (la muestra se puede contaminar al manipularla, algo que el control "
        "estéril no sufre).",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfControlesDesarrollo", len(controles), "controles de calidad del desarrollo"),
        ("ClfEnsayos", len(se), "ensayos con sensibilidad y especificidad estimadas"),
        ("ClfEspecMin", float(se["especificidad"].min()), "especificidad mínima entre ensayos", 3),
        ("ClfEspecMinEnsayo", menos_esp, "ensayo con la menor especificidad"),
        ("ClfEspecMax", float(se["especificidad"].max()), "especificidad máxima entre ensayos", 3),
        ("ClfSensMin", float(se["sensibilidad"].min()), "sensibilidad mínima entre ensayos", 3),
        ("ClfSensMinEnsayo", menos_sens, "ensayo con la menor sensibilidad"),
        ("ClfSensMax", float(se["sensibilidad"].max()), "sensibilidad máxima entre ensayos", 3),
    ])
    latex.exportar_tabla(vista, "bayes_sens_espec", "Sensibilidad y especificidad por ensayo (intervalos de Wilson)",
                         decimales=3)
    return r


def vpp_por_ensayo(c):
    """VPP por ensayo con incertidumbre por sorteos de las posteriores Beta."""
    r = Resultado()
    generador = np.random.default_rng(config.SEMILLA)
    conteo = c.desarrollo.groupby("ensayo", observed=True)[c.objetivo].agg(["size", "sum"])
    filas = []
    for ensayo, (n, positivos) in conteo.iterrows():
        if ensayo not in c.sens_espec.index:
            continue
        fse = c.sens_espec.loc[ensayo]
        detectados = round(fse["sensibilidad"] * fse["n_positivos"])
        limpios = round(fse["especificidad"] * fse["n_negativos"])
        s_sorteo = generador.beta(detectados + 1, fse["n_positivos"] - detectados + 1, N_SORTEOS)
        e_sorteo = generador.beta(limpios + 1, fse["n_negativos"] - limpios + 1, N_SORTEOS)
        p_sorteo = generador.beta(positivos + 1, n - positivos + 1, N_SORTEOS)
        previa_sorteo = bayes.previa_corregida(p_sorteo, s_sorteo, e_sorteo)
        vpp_sorteo = bayes.valor_predictivo_positivo(previa_sorteo, s_sorteo, e_sorteo)
        p_obs = positivos / n
        previa = float(bayes.previa_corregida(p_obs, fse["sensibilidad"], fse["especificidad"]))
        vpp = float(bayes.valor_predictivo_positivo(previa, fse["sensibilidad"], fse["especificidad"]))
        filas.append({
            "ensayo": ensayo, "muestras": int(n), "positivos": int(positivos), "tasa_positivos": p_obs,
            "sensibilidad": float(fse["sensibilidad"]), "especificidad": float(fse["especificidad"]),
            "prevalencia_corregida": previa, "vpp": vpp,
            "vpp_inf": float(np.nanpercentile(vpp_sorteo, 2.5)), "vpp_sup": float(np.nanpercentile(vpp_sorteo, 97.5)),
            "prob_falso_positivo": 1 - vpp, "sorteos_previa_cero": float(np.mean(previa_sorteo == 0)),
        })
    c.tabla_vpp = t = pd.DataFrame(filas)
    c.vpp_global = g = float(np.sum(t["vpp"] * t["positivos"]) / t["positivos"].sum())
    vista = t.drop(columns=["sorteos_previa_cero"]).rename(columns={
        "ensayo": "Ensayo", "muestras": "Muestras", "positivos": "Positivos", "tasa_positivos": "Tasa de positivos",
        "sensibilidad": "Sens.", "especificidad": "Espec.", "prevalencia_corregida": "Prevalencia corregida",
        "vpp": "VPP", "vpp_inf": "VPP inf.", "vpp_sup": "VPP sup.", "prob_falso_positivo": "P(falso positivo)"})
    r.agregar(tabla_html(vista, "VPP por ensayo en el desarrollo (intervalo del 95 % por sorteos de las posteriores Beta)",
                         decimales=3, barras="P(falso positivo)"))
    r.figura(graficos_clf.figura_vpp_ensayo(t), "04_vpp_ensayo")
    peor, mejor = t.sort_values("vpp").iloc[0], t.sort_values("vpp").iloc[-1]
    r.agregar(nota(
        f"Si ponderamos por el número de positivos el VPP estimado del laboratorio es **{g:.3f}**: De cada cien "
        f"resultados no conformes unos {100 * (1 - g):.0f} serían falsos positivos según los controles. El ensayo menos "
        f"fiable es \"{peor['ensayo']}\", con VPP {peor['vpp']:.3f} (intervalo de {peor['vpp_inf']:.3f} a "
        f"{peor['vpp_sup']:.3f}) y una prevalencia corregida de {peor['prevalencia_corregida']:.3f}. El más fiable es "
        f"\"{mejor['ensayo']}\" ({mejor['vpp']:.3f}).\n\n"
        "La diferencia entre ensayos no viene solo de la especificidad: Un ensayo con prevalencia baja da muchos falsos "
        "positivos en proporción aunque su especificidad sea alta, que es justo el efecto de la sección 8."
        + (" En algunos sorteos la tasa observada queda por debajo de la de falsos positivos y la prevalencia corregida se "
           "recorta a cero (" + ", ".join(f'"{e}" en el {100 * f:.0f} % de los sorteos' for e, f in
                                          t.set_index("ensayo")["sorteos_previa_cero"].items() if f > 0.01)
           + "), lo que ensancha el intervalo hacia VPP = 0." if (t["sorteos_previa_cero"] > 0.01).any() else ""),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfSorteos", N_SORTEOS, "sorteos de las posteriores Beta para el VPP"),
        ("ClfVPPGlobal", g, "VPP del laboratorio ponderado por positivos (desarrollo)", 3),
        ("ClfFalsosPorCien", 100 * (1 - g), "falsos positivos por cada cien no conformes (controles)", 0),
        ("ClfVPPPeorEnsayo", peor["ensayo"], "ensayo con el menor VPP"),
        ("ClfVPPPeor", float(peor["vpp"]), "VPP del ensayo menos fiable", 3),
        ("ClfVPPPeorInf", float(peor["vpp_inf"]), "límite inferior del intervalo del VPP menos fiable", 3),
        ("ClfVPPPeorSup", float(peor["vpp_sup"]), "límite superior del intervalo del VPP menos fiable", 3),
        ("ClfVPPMejorEnsayo", mejor["ensayo"], "ensayo con el mayor VPP"),
        ("ClfVPPMejor", float(mejor["vpp"]), "VPP del ensayo más fiable", 3),
    ])
    latex.exportar_tabla(vista, "bayes_vpp_ensayo", "VPP por ensayo en el desarrollo", decimales=3)
    _actualizar_paneles(c)
    return r


def vpp_por_muestra(c):
    """VPP(x) con la previa por muestra que aporta el clasificador, y dos ejemplos."""
    r = Resultado()
    v, se = c.validacion, c.sens_espec
    s_x = v["ensayo"].map(se["sensibilidad"]).to_numpy(dtype=float)
    e_x = v["ensayo"].map(se["especificidad"]).to_numpy(dtype=float)
    previa_x = np.asarray(bayes.previa_corregida(c.p_val, s_x, e_x), dtype=float)
    vpp_x = np.asarray(bayes.valor_predictivo_positivo(previa_x, s_x, e_x), dtype=float)
    c.positivas = pos = v.assign(p_x=c.p_val, previa_x=previa_x, vpp_x=vpp_x)[c.y_val == 1]
    c.recorte_cero, c.recorte_uno = float(np.mean(pos["previa_x"] == 0)), float(np.mean(pos["previa_x"] == 1))
    r.figura(graficos_clf.figura_vpp_muestra(pos.rename(columns={"vpp_x": "vpp"}),
                                             c.tabla_vpp.set_index("ensayo")["vpp"].to_dict()), "04_vpp_muestra")
    c.resumen_muestra = res = pos.groupby("ensayo", observed=True)["vpp_x"].describe(percentiles=[0.1, 0.5, 0.9])
    vista = res[["count", "mean", "10%", "50%", "90%"]].reset_index().rename(columns={
        "ensayo": "Ensayo", "count": "Positivos", "mean": "VPP(x) medio", "10%": "Percentil 10", "50%": "Mediana",
        "90%": "Percentil 90"})
    r.agregar(tabla_html(vista, "VPP(x) entre los positivos de validación, por ensayo", decimales=3))

    ejemplo = pos["ensayo"].value_counts().index[0]
    del_ensayo = pos[pos["ensayo"] == ejemplo]
    c.riesgo_bajo = bajo = del_ensayo.iloc[(del_ensayo["previa_x"] - del_ensayo["previa_x"].quantile(0.1)).abs().argmin()]
    c.riesgo_alto = alto = del_ensayo.iloc[(del_ensayo["previa_x"] - del_ensayo["previa_x"].quantile(0.9)).abs().argmin()]
    descriptores = [x for x in DESCRIPTORES if x in pos.columns]

    def describir(fila):
        return [f"{x}: {legible(round(fila[x], 3)) if isinstance(fila[x], (float, np.floating)) else fila[x]}"
                for x in descriptores]

    r.agregar(comparacion([
        {"titulo": f"Muestra de riesgo {nombre} ({fila['id_muestra']})", "valor": fila["vpp_x"], "etiqueta": "VPP(x)",
         "decimales": 3, "puntos": [f"p(x) = {fila['p_x']:.3f}, π(x) = {fila['previa_x']:.3f}",
                                    f"P(falso positivo) = {1 - fila['vpp_x']:.3f}"] + describir(fila)}
        for nombre, fila in (("bajo", bajo), ("alto", alto))
    ], titulo=f"El mismo positivo de \"{ejemplo}\" en dos muestras distintas (percentiles 10 y 90 de π(x))"))
    r.agregar(nota(
        f"Las dos muestras dieron positivo en el mismo ensayo (misma sensibilidad y especificidad) y aun así la "
        f"probabilidad de que el positivo sea falso es {1 - bajo['vpp_x']:.3f} en la de riesgo bajo y "
        f"{1 - alto['vpp_x']:.3f} en la de riesgo alto. O sea que la previa que aporta el clasificador cambia cómo leemos "
        "el resultado. "
        + f"Entre los positivos de validación el VPP(x) va de {pos['vpp_x'].min():.3f} a {pos['vpp_x'].max():.3f} "
          f"(percentiles 10 y 90 de {pos['vpp_x'].quantile(0.1):.3f} y {pos['vpp_x'].quantile(0.9):.3f})"
        + (f". En el {100 * c.recorte_cero:.1f} % la previa se recorta a cero (VPP = 0)" if c.recorte_cero > 0 else
           ". Ninguna previa se recorta a cero")
        + (f" y en el {100 * c.recorte_uno:.1f} % a uno." if c.recorte_uno > 0 else " y ninguna a uno.")
        + " Con un ROC-AUC moderado la previa por muestra separa los casos solo en parte así que el VPP(x) sirve para "
          "ordenar qué positivos revisar primero y no para descartar un positivo sin reensayo.",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfPositivosValidacionBayes", len(pos), "positivos de validación con VPP(x)"),
        ("ClfVPPxMin", float(pos["vpp_x"].min()), "VPP(x) mínimo entre los positivos de validación", 3),
        ("ClfVPPxMax", float(pos["vpp_x"].max()), "VPP(x) máximo entre los positivos de validación", 3),
        ("ClfVPPxPDiez", float(pos["vpp_x"].quantile(0.1)), "percentil 10 del VPP(x)", 3),
        ("ClfVPPxPNoventa", float(pos["vpp_x"].quantile(0.9)), "percentil 90 del VPP(x)", 3),
        ("ClfRecorteCeroPct", 100 * c.recorte_cero, "positivos con previa recortada a cero (%)", 1),
        ("ClfEnsayoEjemplo", ejemplo, "ensayo de los dos ejemplos de VPP(x)"),
        ("ClfFalsoRiesgoBajo", 1 - float(bajo["vpp_x"]), "P(falso positivo) en la muestra de riesgo bajo", 3),
        ("ClfFalsoRiesgoAlto", 1 - float(alto["vpp_x"]), "P(falso positivo) en la muestra de riesgo alto", 3),
    ])
    latex.exportar_tabla(vista, "bayes_vpp_muestra", "VPP(x) entre los positivos de validación, por ensayo", decimales=3)
    return r


def reensayo(c):
    """Posterior tras el reensayo y reensayos conformes esperados frente a observados."""
    r = Resultado()
    muestras = particion.solo_desarrollo(pd.read_parquet(rutas.DIR_LIMPIOS / "muestras.parquet"), "fecha_recepcion")
    c.reensayadas = re = muestras[muestras["falso_positivo"].notna()]
    positivos = muestras[muestras["no_conforme"] == 1]
    observado = re.groupby("ensayo", observed=True)["falso_positivo"].agg(["size", "mean"])
    filas = []
    for fila in c.tabla_vpp.itertuples(index=False):
        s, e, previa = fila.sensibilidad, fila.especificidad, fila.prevalencia_corregida
        n_re, tasa_re = observado.loc[fila.ensayo] if fila.ensayo in observado.index else (0, np.nan)
        filas.append({
            "ensayo": fila.ensayo, "VPP (un +)": fila.vpp,
            "Posterior (+, +)": float(bayes.posterior_secuencial(previa, s, e, [1, 1])),
            "Posterior (+, −)": float(bayes.posterior_secuencial(previa, s, e, [1, 0])),
            "Reensayados": int(n_re),
            "Fracción reensayada": n_re / max(int((positivos["ensayo"] == fila.ensayo).sum()), 1),
            "Reensayo conforme esperado": fila.vpp * (1 - s) + (1 - fila.vpp) * e,
            "Reensayo conforme observado": float(tasa_re),
        })
    c.tabla_reensayo = t = pd.DataFrame(filas)
    vista = t.rename(columns={"ensayo": "Ensayo"})
    r.agregar(tabla_html(vista, "Posterior tras el reensayo y reensayos conformes esperados frente a observados "
                                "(desarrollo)", decimales=3))
    c.reensayos_total = int(t["Reensayados"].sum())
    c.esperado_global = esp = float(np.average(t["Reensayo conforme esperado"], weights=t["Reensayados"].clip(lower=1e-9)))
    c.observado_global = obs = float(re["falso_positivo"].mean())
    r.agregar(nota(
        f"Tras dos positivos la posterior queda entre {t['Posterior (+, +)'].min():.3f} y {t['Posterior (+, +)'].max():.3f} "
        "según el ensayo, o sea que un reensayo positivo prácticamente confirma el resultado. Tras un positivo y un "
        f"negativo baja a entre {t['Posterior (+, −)'].min():.3f} y {t['Posterior (+, −)'].max():.3f}. Es decir que un "
        "reensayo negativo hace más probable la falsa alarma pero no la demuestra.\n\n"
        f"En el desarrollo se reensayaron {c.reensayos_total} positivos y salió conforme el {100 * obs:.1f} %, frente al "
        f"{100 * esp:.1f} % que predicen los controles"
        + (". O sea que el laboratorio ve más reensayos conformes de los que explica la especificidad de los controles: O "
           "los positivos reales tienen más falsos de lo que dicen los controles (la especificidad efectiva es menor que "
           "la nominal) o los reensayos se piden sobre todo en los positivos dudosos." if obs > esp * 1.15 else
           ". O sea que el laboratorio ve menos reensayos conformes de los que predicen los controles, lo que apunta a "
           "que los reensayos se piden en positivos de por sí más sólidos o a que los controles subestiman la especificidad."
           if obs < esp * 0.85 else
           " así que los controles y la experiencia del laboratorio cuadran.")
        + "\n\nEso sí, \"reensayo conforme\" no es lo mismo que \"falso positivo\": Una muestra contaminada puede dar "
          "negativo en el reensayo (con probabilidad 1 − s, más si la contaminación es baja o no está pareja) y un falso "
          "positivo puede repetirse si su causa es sistemática (un lote de medio contaminado), que es justo el caso en que "
          "falla el supuesto de independencia.",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ClfPosteriorMasMasMin", float(t["Posterior (+, +)"].min()), "posterior mínima tras dos positivos", 3),
        ("ClfPosteriorMasMasMax", float(t["Posterior (+, +)"].max()), "posterior máxima tras dos positivos", 3),
        ("ClfPosteriorMasMenosMin", float(t["Posterior (+, −)"].min()), "posterior mínima tras positivo y negativo", 3),
        ("ClfPosteriorMasMenosMax", float(t["Posterior (+, −)"].max()), "posterior máxima tras positivo y negativo", 3),
        ("ClfReensayados", c.reensayos_total, "positivos reensayados en el desarrollo"),
        ("ClfReensayoConformeObservadoPct", 100 * obs, "reensayos conformes observados (%) en el desarrollo", 1),
        ("ClfReensayoConformeEsperadoPct", 100 * esp, "reensayos conformes esperados (%) según los controles", 1),
    ])
    latex.exportar_tabla(vista, "bayes_reensayo", "Posterior tras el reensayo y reensayos conformes (desarrollo)",
                         decimales=3)
    return r


def auditoria(c):
    """Con datos simulados, compara lo estimado con la verdad oculta."""
    r = Resultado()
    verdad = carga.leer_verdad()
    c.resumen_auditoria = None
    if verdad is None:
        r.agregar(nota("No hay verdad oculta (datos reales): La auditoría no aplica y la única referencia es el reensayo, "
                       "con las limitaciones de la sección anterior.", "simulacion"))
        return r
    verdad_m = verdad["muestras"][["id_muestra", "contaminada_real"]]
    verdad_e = verdad["ensayos"].set_index("ensayo")
    auditadas = c.desarrollo[["id_muestra", "ensayo", c.objetivo]].merge(verdad_m, on="id_muestra", how="inner")
    filas = []
    for fila in c.tabla_vpp.itertuples(index=False):
        del_ensayo = auditadas[auditadas["ensayo"] == fila.ensayo]
        positivos = del_ensayo[c.objetivo] == 1
        conocido = fila.ensayo in verdad_e.index
        filas.append({
            "ensayo": fila.ensayo, "Sens. estimada": fila.sensibilidad,
            "Sens. real": float(verdad_e.loc[fila.ensayo, "sensibilidad"]) if conocido else np.nan,
            "Espec. estimada": fila.especificidad,
            "Espec. efectiva real": float(verdad_e.loc[fila.ensayo, "especificidad_efectiva"]) if conocido else np.nan,
            "Prevalencia corregida": fila.prevalencia_corregida,
            "Prevalencia real": float((del_ensayo["contaminada_real"] == 1).mean()),
            "VPP estimado": fila.vpp, "vpp_inf": fila.vpp_inf, "vpp_sup": fila.vpp_sup,
            "VPP real": float(del_ensayo.loc[positivos, "contaminada_real"].mean()),
        })
    t = pd.DataFrame(filas)
    t["Dentro del IC"] = ((t["VPP real"] >= t["vpp_inf"]) & (t["VPP real"] <= t["vpp_sup"])).map({True: "Sí", False: "No"})
    real_global = float(auditadas.loc[auditadas[c.objetivo] == 1, "contaminada_real"].mean())
    vista = t.drop(columns=["vpp_inf", "vpp_sup"]).rename(columns={"ensayo": "Ensayo"})
    r.agregar(tabla_html(vista, "Auditoría: Estimado con controles frente a la verdad de la simulación (desarrollo)",
                         decimales=3))
    r.figura(graficos_clf.figura_vpp_ensayo(t.rename(columns={"VPP estimado": "vpp", "VPP real": "vpp_real"}),
                                            columna_real="vpp_real",
                                            titulo="VPP por ensayo: estimado con controles frente al verdadero"),
             "04_auditoria_vpp")
    val = c.positivas[["id_muestra", "ensayo", "vpp_x"]].merge(verdad_m, on="id_muestra")
    vpp_ensayo = val["ensayo"].map(c.tabla_vpp.set_index("ensayo")["vpp"]).to_numpy(dtype=float)
    brier_x = metricas.brier(val["contaminada_real"], val["vpp_x"])
    brier_e = metricas.brier(val["contaminada_real"], vpp_ensayo)
    auc_x = metricas.roc_auc(val["contaminada_real"], val["vpp_x"])
    auc_e = metricas.roc_auc(val["contaminada_real"], vpp_ensayo)
    r.figura(graficos_clf.figura_fiabilidad(
        {"VPP(x), previa por muestra": (val["contaminada_real"], val["vpp_x"])}, n_tramos=5,
        titulo="Calibración del VPP(x) en los positivos de validación", etiqueta_x="VPP(x) estimado (media del tramo)",
        etiqueta_y="Fracción realmente contaminada"), "04_auditoria_vpp_muestra")
    fp_real = float(1 - c.reensayadas.merge(verdad_m, on="id_muestra")["contaminada_real"].mean())
    sesgo = (t["Espec. estimada"] - t["Espec. efectiva real"]).mean()
    peor = t.sort_values("VPP real").iloc[0]
    dentro = int((t["Dentro del IC"] == "Sí").sum())
    r.agregar(nota(
        f"El VPP real del desarrollo es **{real_global:.3f}** frente a {c.vpp_global:.3f} estimado con los controles"
        + (f". La especificidad de los controles supera en promedio a la efectiva en {sesgo:.4f} porque en la simulación "
           "las muestras también se contaminan al manipularlas, algo que el control estéril no sufre. Como la prevalencia "
           "es baja esa diferencia pequeña de especificidad termina en un VPP sobreestimado."
           if sesgo > 0.002 and c.vpp_global > real_global else ".")
        + f" El ensayo con menor VPP real es \"{peor['ensayo']}\" ({peor['VPP real']:.3f}, estimado "
          f"{peor['VPP estimado']:.3f}) y el VPP real cae dentro del intervalo estimado en {dentro} de los {len(t)} "
          "ensayos.\n\n"
        + f"Entre los positivos de validación el VPP(x) separa contaminadas de falsas alarmas con un ROC-AUC de "
          f"{auc_x:.3f} (el del VPP por ensayo es {auc_e:.3f}) y su Brier es {brier_x:.4f} frente a {brier_e:.4f} del "
          "VPP por ensayo"
        + (", o sea que la previa por muestra sí aporta información sobre cuáles positivos son reales."
           if brier_x < brier_e else
           ", o sea que en estos datos la previa por muestra no mejora la probabilidad del VPP por ensayo.")
        + f"\n\nEntre los positivos reensayados la fracción que de verdad no estaba contaminada es {fp_real:.3f}, frente a "
          f"un {c.observado_global:.3f} de reensayos conformes. La diferencia es el error del propio reensayo.",
        "simulacion", titulo="Lo que dice la verdad oculta",
    ))
    c.resumen_auditoria = {
        "vpp_real_global": real_global, "vpp_estimado_global": c.vpp_global, "por_ensayo": t.to_dict("records"),
        "vpp_muestra": {"brier": brier_x, "brier_vpp_ensayo": brier_e, "auc": auc_x, "auc_vpp_ensayo": auc_e},
        "falso_positivo_real_reensayados": fp_real,
    }
    latex.registrar_varias([
        ("ClfAudVPPReal", real_global, "VPP real del desarrollo (verdad de la simulación)", 3),
        ("ClfAudSesgoEspec", float(sesgo), "exceso medio de la especificidad de controles sobre la efectiva", 4),
        ("ClfAudEnsayosDentroIC", dentro, "ensayos cuyo VPP real cae en el intervalo estimado"),
        ("ClfAudAUCVPPx", auc_x, "ROC-AUC del VPP(x) frente a la contaminación real (validación)", 3),
        ("ClfAudAUCVPPEnsayo", auc_e, "ROC-AUC del VPP por ensayo frente a la contaminación real", 3),
        ("ClfAudBrierVPPx", brier_x, "Brier del VPP(x) frente a la contaminación real", 4),
        ("ClfAudBrierVPPEnsayo", brier_e, "Brier del VPP por ensayo frente a la contaminación real", 4),
        ("ClfAudFalsoRealReensayados", fp_real, "fracción realmente no contaminada entre los reensayados", 3),
    ])
    latex.exportar_tabla(vista, "bayes_auditoria", "Auditoría del VPP frente a la verdad de la simulación", decimales=3)
    return r


def regla_reensayo(c):
    """Fracción de reensayos conformes por tramo de la razón recuento/límite y regla que se desprende."""
    r = Resultado()
    tabla_fp = leer_modelado("falso_positivo_lectura")
    con_razon = tabla_fp[tabla_fp[VARIABLE_REGLA].notna()]
    tramos = pd.qcut(con_razon[VARIABLE_REGLA], 6, duplicates="drop")
    por_tramo = con_razon.groupby(tramos, observed=True)["falso_positivo"].agg(["size", "sum"])
    inf_w, sup_w = bayes.intervalo_wilson(por_tramo["sum"].to_numpy(), por_tramo["size"].to_numpy())
    c.tabla_tramos = t = pd.DataFrame({
        "tramo": [f"{i.left:.2f} a {i.right:.2f}" for i in por_tramo.index], "borde": [i.right for i in por_tramo.index],
        "n": por_tramo["size"].to_numpy(), "tasa": (por_tramo["sum"] / por_tramo["size"]).to_numpy(),
        "inf": np.asarray(inf_w), "sup": np.asarray(sup_w)})
    sin_razon = tabla_fp[tabla_fp[VARIABLE_REGLA].isna()]
    r.figura(graficos_clf.figura_tasa_por_tramo(t, "log10(recuento / límite), por sextiles",
                                                "Fracción de reensayos conformes",
                                                "Reensayos conformes según cuánto supera el recuento al límite",
                                                referencia=UMBRAL_DUDA), "04_regla_reensayo")
    dudosos = t[t["tasa"] >= UMBRAL_DUDA]
    if len(dudosos):
        c.borde = borde = float(dudosos["borde"].max())
        bajo = con_razon[VARIABLE_REGLA] <= borde
        c.tasa_bajo = float(con_razon.loc[bajo, "falso_positivo"].mean())
        c.tasa_alto = float(con_razon.loc[~bajo, "falso_positivo"].mean())
        c.captura = float(con_razon.loc[bajo, "falso_positivo"].sum() / con_razon["falso_positivo"].sum())
        texto = (
            f"Cuando el recuento pasa el límite por poco (log10(recuento / límite) ≤ {borde:.2f}, o sea menos de "
            f"{10 ** borde:.2f} veces el límite) sale conforme el {100 * c.tasa_bajo:.0f} % de los reensayos, frente al "
            f"{100 * c.tasa_alto:.0f} % cuando lo pasa por más. La regla que sale de ahí es simple: Reensayar antes de "
            f"informar todo no conforme cuantitativo con recuento por debajo de {10 ** borde:.2f} veces el límite. Ese "
            f"grupo es el {100 * bajo.mean():.0f} % de los reensayados con recuento y concentra el {100 * c.captura:.0f} % "
            "de los reensayos conformes.")
        latex.registrar_varias([
            ("ClfReglaBorde", borde, "borde de log10(recuento / límite) de la regla de reensayo", 2),
            ("ClfReglaVecesLimite", 10 ** borde, "veces el límite bajo las que se reensaya", 2),
            ("ClfReglaTasaBajoPct", 100 * c.tasa_bajo, "reensayos conformes (%) bajo el borde (desarrollo)", 0),
            ("ClfReglaTasaAltoPct", 100 * c.tasa_alto, "reensayos conformes (%) sobre el borde (desarrollo)", 0),
            ("ClfReglaGrupoPct", 100 * float(bajo.mean()), "reensayados con recuento que caen bajo el borde (%)", 0),
            ("ClfReglaCapturaPct", 100 * c.captura, "reensayos conformes que concentra el grupo (%)", 0),
        ])
    else:
        c.borde, c.tasa_bajo, c.tasa_alto, c.captura = None, np.nan, np.nan, np.nan
        texto = (f"Ningún tramo de la razón llega a una fracción de reensayos conformes de {UMBRAL_DUDA:.0%} así que la "
                 "razón ordena el riesgo pero no define por sí sola un grupo donde el falso positivo sea lo más probable.")
    r.agregar(nota(
        texto + f" En los ensayos sin recuento (presencia o ausencia, {len(sin_razon)} reensayados) la fracción de "
                f"reensayos conformes es {sin_razon['falso_positivo'].mean():.3f} y para ellos la referencia es el VPP de "
                "su ensayo. La explicación es física: Un recuento apenas por encima del límite puede venir de una "
                "contaminación puntual durante el análisis, mientras que un recuento muy alto difícilmente sale de un "
                "error de manipulación.",
        "hallazgo", titulo="Regla para el reensayo",
    ))
    latex.registrar_varias([
        ("ClfSinRazonReensayados", len(sin_razon), "reensayados de ensayos sin recuento"),
        ("ClfSinRazonConformes", float(sin_razon["falso_positivo"].mean()), "fracción conforme en ensayos sin recuento", 3),
    ])
    latex.exportar_tabla(t, "bayes_tramos_reensayo", "Reensayos conformes por sextil de log10(recuento / límite)",
                         decimales={"borde": 2, "tasa": 3, "inf": 3, "sup": 3})
    _guardar_artefacto(c)
    return r


def _guardar_artefacto(c):
    c.segundos = time.perf_counter() - c.inicio
    cfg = c.configuracion_final
    entrada = columnas_de_grupo(c.papeles, cfg["grupo"])
    columnas_ejemplo = ["id_muestra", "ensayo", "p_x", "previa_x", "vpp_x"]
    guardar_json({
        "analisis": c.nombre, "objetivo": c.objetivo, "momento": c.papeles["momento"],
        "costos": {"falso_positivo": config.COSTO_FP, "falso_negativo": config.COSTO_FN,
                   "umbral_teorico": c.umbral_teorico},
        "variables_entrada": [*entrada[0], *entrada[1]],
        "variables_modelo": tuberias.nombres_variables(c.modelo_base),
        "discretas": c.discretas, "modelo": c.final,
        "configuracion": {f: (None if v is None else v) for f, v in cfg.items()},
        "hiperparametros": taguchi.hiper_de(c.busquedas[c.final]),
        "calibracion": {"usada": c.usar_calibrado, "metodo": METODO_CALIBRACION if c.usar_calibrado else None,
                        "brier_sin": c.brier_sin, "brier_calibrado": c.brier_cal, "auc_sin": c.auc_val[c.final],
                        "auc_calibrado": c.auc_cal},
        "regla_eleccion": f"mayor ROC-AUC de validación; empate (< {TOLERANCIA_EMPATE}) → más sencillo "
                          f"({' < '.join(ORDEN_SENCILLEZ)})",
        "reajuste": "entrenamiento + validación con hiperparámetros, calibración y umbral congelados",
        "umbral": c.umbral, "semanas_validacion": c.semanas_val,
        "umbral_elegido": {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                           for k, v in c.fila_umbral.to_dict().items()},
        "tabla_umbrales": c.tabla_umbral.to_dict("records"),
        "costo_politicas": {"no_alertar": c.costo_nada, "alertar_todo": c.costo_todo},
        "validacion": c.tabla_val.to_dict("records"),
        "auc_validacion": {"estimado": c.ic_auc[0], "inferior": c.ic_auc[1], "superior": c.ic_auc[2]},
        "precision_media_validacion": metricas.precision_media(c.y_val, c.p_val),
        "costo_validacion": {"estimado": c.ic_costo[0], "inferior": c.ic_costo[1], "superior": c.ic_costo[2]},
        "didactica": c.tabla_didactica.to_dict("records"),
        "sobremuestreo": c.tabla_sobre.to_dict("records"),
        "taguchi": c.taguchi,
        "rejilla": {m: {"hiper": taguchi.hiper_de(b), "auc_bloques": float(c.tablas_rejilla[m].iloc[0]["media"]),
                        "segundos": c.segundos_rejilla[m]} for m, b in c.busquedas.items()},
        "bayes": {
            "sens_espec": c.sens_espec.reset_index().to_dict("records"),
            "vpp_por_ensayo": c.tabla_vpp.to_dict("records"),
            "vpp_global": c.vpp_global,
            "vpp_por_muestra": {"resumen": c.resumen_muestra.reset_index().to_dict("records"),
                                "minimo": float(c.positivas["vpp_x"].min()), "maximo": float(c.positivas["vpp_x"].max()),
                                "recorte_cero": c.recorte_cero, "recorte_uno": c.recorte_uno,
                                "ejemplos": {"bajo": c.riesgo_bajo[columnas_ejemplo].to_dict(),
                                             "alto": c.riesgo_alto[columnas_ejemplo].to_dict()}},
            "posterior_reensayo": c.tabla_reensayo.to_dict("records"),
            "reensayo_conforme": {"observado": c.observado_global, "esperado": c.esperado_global,
                                  "reensayados": c.reensayos_total},
            "regla_reensayo": {"variable": VARIABLE_REGLA, "borde": c.borde, "tasa_bajo": c.tasa_bajo,
                               "tasa_alto": c.tasa_alto, "captura": c.captura, "tramos": c.tabla_tramos.to_dict("records")},
        },
        "auditoria": c.resumen_auditoria,
        "segundos_etapa": c.segundos,
    }, rutas.DIR_ARTEFACTOS / "04_clasificacion.json")


def terminar(c):
    cerrar({"analisis": c.nombre, "modelo": c.final, "calibrado": c.usar_calibrado, "auc_validacion": c.auc_final,
            "umbral": c.umbral, "vpp_global": c.vpp_global, "segundos": c.segundos})
