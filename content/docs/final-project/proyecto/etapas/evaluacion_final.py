"""Capa 8 · Etapa 07: evaluación final en la prueba.

Responsabilidad única: armar las piezas de la etapa 07 (lo que quedó
congelado, apertura única de la prueba, regresión, clasificación, Bayes y
regla de reensayo en la prueba, resumen del agrupamiento y resumen final),
guardar el artefacto de la etapa y registrar sus cifras para el informe. Es
el único módulo que lee la partición de prueba, siempre con `leer_prueba`.
"""
import time

import numpy as np
import pandas as pd

from proyecto.base import config, rutas
from proyecto.base.artefactos import guardar_json, leer_json, leer_modelo, leer_prueba
from proyecto.datos import carga
from proyecto.etapas.comun import Resultado, abrir, cerrar
from proyecto.evaluacion import bayes, metricas, umbral
from proyecto.evaluacion.remuestreo import intervalo_bootstrap
from proyecto.modelos import registro
from proyecto.reporte import graficos_clasificacion as graficos_clf
from proyecto.reporte import graficos_modelado as graficos
from proyecto.reporte import latex
from proyecto.reporte.html import comparacion, linea_tiempo, nota, pasos, tabla_html, tarjetas

ETAPA = "07_evaluacion_final"
N_REPLICAS = 1000
NOMBRE_REENSAYO = "falso_positivo_lectura"
MINIMO_POSITIVOS = 30


def _hiper_legibles(hiper):
    if not hiper:
        return "ninguno (mínimos cuadrados)"
    return ", ".join(f"{k} = {v}" for k, v in hiper.items())


def _con_intervalo(funcion, *arreglos):
    """(estimado, inferior, superior) por bootstrap de filas, con la semilla del proyecto."""
    return intervalo_bootstrap(funcion, *arreglos, n=N_REPLICAS)


def _texto_ic(trio, decimales=3):
    return f"{trio[0]:.{decimales}f} (IC 95 %: {trio[1]:.{decimales}f} a {trio[2]:.{decimales}f})"


def iniciar():
    c = abrir(ETAPA, "Los modelos congelados frente a un periodo que nadie miró")
    c.inicio = time.perf_counter()
    leer = lambda nombre: leer_json(rutas.DIR_ARTEFACTOS / nombre)  # noqa: E731
    c.cribado, c.reg, c.clf, c.grp = (leer("03_cribado.json"), leer("04_regresion.json"),
                                      leer("05_clasificacion.json"), leer("06_agrupamiento.json"))
    c.nombre_reg, c.nombre_clf = c.cribado["seleccion"]["regresion"], c.cribado["seleccion"]["clasificacion"]
    c.papeles_reg = leer_json(rutas.DIR_MODELADO / f"{c.nombre_reg}.json")
    c.papeles_clf = leer_json(rutas.DIR_MODELADO / f"{c.nombre_clf}.json")
    c.modelo_reg = leer_modelo(rutas.DIR_ARTEFACTOS / "modelos" / "regresion.joblib")
    c.modelo_clf = leer_modelo(rutas.DIR_ARTEFACTOS / "modelos" / "clasificacion.joblib")
    c.umbral = float(c.clf["umbral"])
    c.nombre_modelo_reg = registro.NOMBRES_LEGIBLES[c.reg["modelo"]]
    return c


def congelado(c):
    r = Resultado()
    reg, clf = c.reg, c.clf
    c.fila_val_reg = f = next(x for x in reg["validacion"] if x["Predictor"] == c.nombre_modelo_reg)
    c.umbral_val = u = clf["umbral_elegido"]
    tabla = pd.DataFrame([
        {"Análisis": c.nombre_reg, "Tipo": "regresión", "Modelo": c.nombre_modelo_reg,
         "Variables": f"grupo {reg['configuracion']['grupo']} ({len(reg['variables_entrada'])})",
         "Hiperparámetros": _hiper_legibles(reg["hiperparametros"]), "Calibración y umbral": "no aplica",
         "Validación": f"RMSE {f['RMSE']:.2f}, habilidad {f['Habilidad']:.3f}"},
        {"Análisis": c.nombre_clf, "Tipo": "clasificación", "Modelo": registro.NOMBRES_LEGIBLES[clf["modelo"]],
         "Variables": f"grupo {clf['configuracion']['grupo']} ({len(clf['variables_entrada'])})",
         "Hiperparámetros": _hiper_legibles(clf["hiperparametros"]),
         "Calibración y umbral": f"{'sigmoide' if clf['calibracion']['usada'] else 'sin calibrar'}, umbral {c.umbral:.3f}",
         "Validación": (f"ROC-AUC {clf['auc_validacion']['estimado']:.3f}, costo {u['Costo']:.0f}, "
                        f"precisión {u['Precisión']:.3f}")},
    ])
    r.agregar(tabla_html(tabla, "Modelos congelados antes de abrir la prueba"))
    r.agregar(linea_tiempo([
        (config.FECHA_INICIO, "Entrenamiento", "ajuste y validación cruzada por bloques temporales"),
        (config.CORTE_VALIDACION, "Validación", "elección del modelo, la calibración y el umbral"),
        (config.CORTE_PRUEBA, "Prueba", "se abre una sola vez, en esta etapa"),
        (config.FECHA_FIN, "Fin de los datos", "último día del periodo de prueba"),
    ], titulo="Protocolo temporal"))
    r.agregar(pasos([
        ("Ajustar", "solo con el entrenamiento, comparando por bloques temporales"),
        ("Decidir", "con la validación: modelo, calibración y umbral"),
        ("Reajustar", "el modelo congelado con entrenamiento + validación, sin tocar sus decisiones"),
        ("Evaluar", "una vez, con la prueba; nada de lo que sigue cambia una decisión"),
    ], titulo="Ajustar, decidir, evaluar", actual=4))
    r.agregar(nota(
        f"La regresión elige con la regla «{reg['regla_eleccion']}» y la clasificación con «{clf['regla_eleccion']}». Las "
        "dos reglas y sus resultados quedaron escritos en los artefactos de las etapas 04 y 05 antes de esta celda, de "
        "forma que lo que se mida en la prueba no puede retroalimentar la elección.",
        "info", titulo="Nada se reelige después de este punto",
    ))
    latex.exportar_tabla(tabla, "final_congelado", "Modelos congelados antes de abrir la prueba")
    return r


def abrir_prueba(c):
    """Lee los tres bloques de prueba (cada lectura queda registrada)."""
    r = Resultado()
    c.prueba_reg = leer_prueba(c.nombre_reg).sort_values(c.papeles_reg["fecha"]).reset_index(drop=True)
    prueba_clf = leer_prueba(c.nombre_clf).reset_index(drop=True)
    c.prueba_clf = prueba_clf[prueba_clf[c.papeles_clf["objetivo"]].notna()].reset_index(drop=True)
    c.prueba_reensayo = leer_prueba(NOMBRE_REENSAYO).reset_index(drop=True)
    c.semanas = int(pd.to_datetime(c.prueba_clf[c.papeles_clf["fecha"]]).dt.to_period("W-SUN").nunique())
    fechas = pd.to_datetime(c.prueba_reg[c.papeles_reg["fecha"]])
    r.agregar(tarjetas([
        {"titulo": "Días hábiles de prueba", "valor": len(c.prueba_reg), "decimales": 0,
         "detalle": f"{fechas.min():%Y-%m-%d} a {fechas.max():%Y-%m-%d}"},
        {"titulo": "Muestras de prueba", "valor": len(c.prueba_clf), "decimales": 0, "color": "naranja",
         "detalle": f"{c.semanas} semanas"},
        {"titulo": "Reensayos de prueba", "valor": len(c.prueba_reensayo), "decimales": 0, "color": "morado",
         "detalle": "no conformes con resultado de reensayo"},
    ], titulo="Lo que contiene la prueba"))
    latex.registrar_varias([
        ("FinDiasPrueba", len(c.prueba_reg), "días hábiles de prueba"),
        ("FinPruebaDesde", f"{fechas.min():%Y-%m-%d}", "primer día de prueba"),
        ("FinPruebaHasta", f"{fechas.max():%Y-%m-%d}", "último día de prueba"),
        ("FinMuestrasPrueba", len(c.prueba_clf), "muestras de prueba con resultado"),
        ("FinSemanasPrueba", c.semanas, "semanas del periodo de prueba"),
        ("FinReensayosPrueba", len(c.prueba_reensayo), "no conformes reensayados en la prueba"),
    ])
    return r


def regresion(c):
    """Métricas del modelo congelado y de las referencias en la prueba, frente a validación."""
    r = Resultado()
    p, reg, f = c.papeles_reg, c.reg, c.fila_val_reg
    c.referencia = reg["referencia_habilidad"]
    c.y_reg = y = c.prueba_reg[p["objetivo"]].to_numpy(dtype=float)
    c.pred_reg = pred = np.asarray(c.modelo_reg.predict(c.prueba_reg[reg["variables_entrada"]]), dtype=float)
    c.fechas_reg = pd.to_datetime(c.prueba_reg[p["fecha"]])
    c.base_reg = base = c.prueba_reg[c.referencia].to_numpy(dtype=float)
    c.completo = completo = np.isfinite(base)
    filas = []
    for nombre, pr in [(c.nombre_modelo_reg, pred)] + [(ref, c.prueba_reg[ref].to_numpy(dtype=float))
                                                       for ref in p["referencias"]]:
        validos = np.isfinite(pr) & completo
        fila = {"Predictor": nombre, **metricas.metricas_reg(y[validos], pr[validos])}
        fila["Habilidad"] = metricas.habilidad("regresion", y[validos], pr[validos], base[validos])
        filas.append(fila)
    c.tabla_reg = pd.DataFrame(filas)
    c.ic_reg = ic = {
        "RMSE": _con_intervalo(metricas.rmse, y[completo], pred[completo]),
        "MAE": _con_intervalo(metricas.mae, y[completo], pred[completo]),
        "R2": _con_intervalo(metricas.r2, y[completo], pred[completo]),
        "Habilidad": _con_intervalo(lambda a, b, cc: metricas.habilidad("regresion", a, b, cc),
                                    y[completo], pred[completo], base[completo]),
    }
    decimales = {"MAE": 2, "RMSE": 2, "R2": 3, "Habilidad": 3}
    r.agregar(tabla_html(c.tabla_reg, f"Regresión en la prueba (muestras por día; habilidad frente a «{c.referencia}»)",
                         decimales=decimales, resaltar=[0]))
    metr = ("RMSE", "MAE", "R2", "Habilidad")
    lado = pd.DataFrame({"Métrica": ["RMSE", "MAE", "R²", "Habilidad"], "Validación": [f[m] for m in metr],
                         "Prueba": [ic[m][0] for m in metr], "Prueba inf.": [ic[m][1] for m in metr],
                         "Prueba sup.": [ic[m][2] for m in metr]})
    r.agregar(tabla_html(lado, "Validación frente a prueba, modelo de regresión congelado (IC 95 % bootstrap)",
                         decimales=3, columnas_resaltadas="Prueba"))
    c.rmse_val = rv = f["RMSE"]
    inf, sup = f["RMSE inf."], f["RMSE sup."]
    c.rmse_prueba = rp = ic["RMSE"][0]
    c.habilidad_prueba = ic["Habilidad"][0]
    r.agregar(nota(
        f"En la prueba el RMSE es **{rp:.2f}** muestras por día, frente a {rv:.2f} en validación "
        f"({100 * (rp / rv - 1):+.0f} %). "
        + (f"Queda por encima del intervalo de validación ({inf:.2f} a {sup:.2f}), de forma que el periodo nuevo resultó "
           "más difícil de lo que la validación anticipaba." if rp > sup else
           f"Queda por debajo del intervalo de validación ({inf:.2f} a {sup:.2f}), de forma que el periodo de prueba "
           "resultó más fácil de predecir que el de validación." if rp < inf else
           f"Queda dentro del intervalo de validación ({inf:.2f} a {sup:.2f}), de forma que la validación anticipó bien el "
           "error del periodo nuevo.")
        + f"\n\nLa habilidad frente a «{c.referencia}» es {_texto_ic(ic['Habilidad'])}, contra {f['Habilidad']:.3f} en "
          "validación"
        + (", así que el modelo sigue mejorando a la referencia ingenua con un margen que el intervalo no alcanza a borrar."
           if ic["Habilidad"][1] > 0 else
           ", y el intervalo toca el cero, así que en la prueba no puede afirmarse que el modelo supere a la referencia."
           if c.habilidad_prueba > 0 else
           ", es decir, en la prueba el modelo no mejora a la referencia ingenua."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("FinRegModelo", c.nombre_modelo_reg, "modelo de regresión evaluado en la prueba"),
        ("FinRegReferencia", c.referencia, "referencia de la habilidad en la prueba"),
        ("FinRMSEPrueba", rp, "RMSE del modelo de regresión en la prueba", 2),
        ("FinRMSEPruebaInf", ic["RMSE"][1], "límite inferior del IC 95 % del RMSE en la prueba", 2),
        ("FinRMSEPruebaSup", ic["RMSE"][2], "límite superior del IC 95 % del RMSE en la prueba", 2),
        ("FinMAEPrueba", ic["MAE"][0], "MAE del modelo de regresión en la prueba", 2),
        ("FinRDosPrueba", ic["R2"][0], "R cuadrado del modelo de regresión en la prueba", 3),
        ("FinHabilidadPrueba", ic["Habilidad"][0], "habilidad del modelo en la prueba", 3),
        ("FinHabilidadPruebaInf", ic["Habilidad"][1], "límite inferior del IC 95 % de la habilidad en la prueba", 3),
        ("FinHabilidadPruebaSup", ic["Habilidad"][2], "límite superior del IC 95 % de la habilidad en la prueba", 3),
        ("FinCambioRMSEPct", 100 * (rp / rv - 1), "cambio (%) del RMSE de validación a prueba", 0),
    ])
    latex.exportar_tabla(c.tabla_reg, "final_regresion", "Regresión en la prueba (modelo y referencias)",
                         decimales=decimales)
    latex.exportar_tabla(lado, "final_regresion_validacion", "Validación frente a prueba, regresión", decimales=3)
    return r


def regresion_diagnostico(c):
    """Serie y residuos en la prueba, ventana de fin de año y lectura operativa."""
    r = Resultado()
    y, pred, base, fechas, completo = c.y_reg, c.pred_reg, c.base_reg, c.fechas_reg, c.completo
    r.figura(graficos.figura_serie_prediccion(fechas, y, {c.nombre_modelo_reg: pred, c.referencia: base},
                                              unidad="muestras por día"), "07_serie_prueba")
    r.figura(graficos.figura_residuos(fechas, y, pred, unidad="muestras por día"), "07_residuos_prueba")
    residuo = y - pred
    mes, dia = fechas.dt.month.to_numpy(), fechas.dt.day.to_numpy()
    ventana = ((mes == 12) & (dia >= 20)) | ((mes == 1) & (dia <= 8))
    c.n_ventana = n = int(ventana.sum())
    c.residuo_ventana = rv = float(residuo[ventana].mean()) if n else np.nan
    c.residuo_resto = rr = float(residuo[~ventana].mean())
    c.mae_prueba = mae = c.ic_reg["MAE"][0]
    if n == 0:
        texto = ("La prueba no incluye días entre el 20 de diciembre y el 8 de enero, de forma que el efecto de las fiestas "
                 "de fin de año no puede revisarse en ella.")
    else:
        dias = fechas[ventana]
        antes_de_diciembre = fechas.max().month < 12
        texto = (
            "La ventana de fin de año (20 de diciembre a 8 de enero) cae en la prueba "
            f"{'solo ' if antes_de_diciembre else ''}entre {dias.min():%Y-%m-%d} y {dias.max():%Y-%m-%d} ({n} días hábiles)"
            + (f", porque el periodo empieza el {config.CORTE_PRUEBA} y termina el {fechas.max():%Y-%m-%d}, antes de "
               "diciembre." if antes_de_diciembre else ".")
            + f" En esos días el residuo medio (observado − predicho) es {rv:+.1f} muestras, frente a {rr:+.1f} en el resto. "
            + ("El modelo sobrestima la carga de esos días, es decir, tras las fiestas llegan menos muestras de las que "
               "sugieren sus variables, y ninguna de ellas marca específicamente el periodo de fin de año."
               if rv < rr - 0.5 * mae else
               "El modelo subestima la carga de esos días, es decir, tras las fiestas llegan más muestras de las que "
               "sugieren sus variables, y ninguna de ellas marca específicamente el periodo de fin de año."
               if rv > rr + 0.5 * mae else
               "La diferencia es menor que la mitad del error absoluto medio, así que con estos días no se distingue un "
               "efecto propio de las fiestas.")
            + (f" Con solo {n} días la cifra es orientativa." if n < 10 else ""))
    r.agregar(nota(texto, "hallazgo", titulo="Fin de año"))

    op = c.reg["operacion"]
    c.tolerancia = tol = float(op["tolerancia"])
    fraccion = tol / float(op["media_validacion"])
    c.media_prueba = media = float(np.mean(y))
    c.fuera_modelo = fm = float(np.mean(np.abs(residuo[completo]) > tol))
    c.fuera_ref = fr = float(np.mean(np.abs(y[completo] - base[completo]) > tol))
    r.agregar(comparacion([
        {"titulo": c.referencia, "valor": 100 * fr, "etiqueta": "% de días fuera de tolerancia", "decimales": 1,
         "sufijo": " %", "puntos": [f"en validación {100 * op['dias_fuera_referencia']:.1f} %"]},
        {"titulo": c.nombre_modelo_reg, "valor": 100 * fm, "decimales": 1, "sufijo": " %",
         "etiqueta": "% de días fuera de tolerancia",
         "puntos": [f"en validación {100 * op['dias_fuera']:.1f} %", f"error absoluto medio {mae:.1f}"]},
    ], titulo=f"Días de prueba con error mayor que {tol:.1f} muestras ({100 * fraccion:.0f} % de la media diaria de "
              "validación)", elegida=c.nombre_modelo_reg if fm <= fr else c.referencia, sello="Menos días fuera"))
    r.agregar(nota(
        f"En la prueba llegaron en promedio {media:.1f} muestras por día hábil (en validación {op['media_validacion']:.1f}), "
        f"y el error en un día típico es de {mae:.1f} muestras, el {100 * mae / media:.0f} % de la media. "
        + (f"El {100 * fm:.0f} % de los días queda fuera de la tolerancia, contra el {100 * fr:.0f} % de la referencia, de "
           "forma que el modelo reduce los días mal planificados sin eliminarlos." if fm < fr else
           f"El {100 * fm:.0f} % de los días queda fuera de la tolerancia, sin mejorar el {100 * fr:.0f} % de la "
           "referencia, de forma que la ganancia en RMSE no se traduce en menos días mal planificados."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("FinDiasFinDeAnio", n, "días hábiles de prueba en la ventana de fin de año"),
        ("FinResiduoFinDeAnio", rv, "residuo medio en la ventana de fin de año", 1),
        ("FinResiduoResto", rr, "residuo medio fuera de la ventana de fin de año", 1),
        ("FinMediaPrueba", media, "muestras por día hábil en promedio (prueba)", 1),
        ("FinDiasFueraPct", 100 * fm, "días de prueba fuera de tolerancia (%), modelo", 1),
        ("FinDiasFueraReferenciaPct", 100 * fr, "días de prueba fuera de tolerancia (%), referencia", 1),
    ])
    return r


def clasificacion(c):
    """Clasificador congelado en la prueba: métricas con intervalos, costo, confusión y fiabilidad."""
    r = Resultado()
    clf, u, um = c.clf, c.umbral_val, c.umbral
    c.y_clf = y = c.prueba_clf[c.papeles_clf["objetivo"]].to_numpy(dtype=int)
    c.p_clf = p = metricas.prediccion_de(c.modelo_clf, c.prueba_clf[clf["variables_entrada"]])
    c.res_clf = res = metricas.metricas_clf(y, p, um)
    alertas = res["TP"] + res["FP"]

    def precision(a, b):
        alerta = b >= um
        return float(a[alerta].mean()) if alerta.any() else np.nan

    def recall(a, b):
        return float((b[a == 1] >= um).mean()) if (a == 1).any() else np.nan

    c.ic_clf = ic = {
        "ROC-AUC": _con_intervalo(metricas.roc_auc, y, p),
        "Precisión media": _con_intervalo(metricas.precision_media, y, p),
        "Brier": _con_intervalo(metricas.brier, y, p),
        "Precisión": _con_intervalo(precision, y, p),
        "Recall": _con_intervalo(recall, y, p),
        "F1": _con_intervalo(lambda a, b: metricas.f1(a, b, um), y, p),
        "Costo por muestra": _con_intervalo(lambda a, b: metricas.costo(a, b, um) / len(a), y, p),
    }
    n_val = u["TN"] + u["FP"] + u["FN"] + u["TP"]
    c.valores_validacion = vv = {
        "ROC-AUC": clf["auc_validacion"]["estimado"], "Precisión media": clf["precision_media_validacion"],
        "Brier": clf["calibracion"]["brier_calibrado"] if clf["calibracion"]["usada"] else clf["calibracion"]["brier_sin"],
        "Precisión": u["Precisión"], "Recall": u["Recall"], "F1": u["F1"], "Costo por muestra": u["Costo"] / n_val,
    }
    lado = pd.DataFrame({"Métrica": list(ic), "Validación": [vv[m] for m in ic], "Prueba": [ic[m][0] for m in ic],
                         "Prueba inf.": [ic[m][1] for m in ic], "Prueba sup.": [ic[m][2] for m in ic]})
    r.agregar(tabla_html(lado, f"Validación frente a prueba, clasificador congelado (umbral {um:.3f}; IC 95 % bootstrap)",
                         decimales=3, columnas_resaltadas="Prueba",
                         pie="El costo se expresa por muestra porque los dos periodos tienen tamaños distintos."))
    c.costo_nada = int(config.COSTO_FN * int(y.sum()))
    c.costo_todo = int(config.COSTO_FP * int((y == 0).sum()))
    c.alertas_semana = asem = alertas / c.semanas
    r.agregar(comparacion([
        {"titulo": "No alertar", "valor": c.costo_nada, "etiqueta": "costo en prueba", "decimales": 0,
         "puntos": [f"{int(y.sum())} no conformes sin aviso"]},
        {"titulo": f"Modelo con umbral {um:.3f}", "valor": res["Costo"], "etiqueta": "costo en prueba", "decimales": 0,
         "puntos": [f"{asem:.1f} alertas por semana", f"precisión {res['Precisión']:.3f}, recall {res['Recall']:.3f}"]},
        {"titulo": "Alertar todo", "valor": c.costo_todo, "etiqueta": "costo en prueba", "decimales": 0,
         "puntos": [f"{len(y) / c.semanas:.1f} alertas por semana"]},
    ], titulo=f"El umbral congelado frente a las políticas sin modelo ({c.semanas} semanas de prueba)",
        elegida=int(np.argmin([c.costo_nada, res["Costo"], c.costo_todo])), sello="Menor costo"))
    r.figura(graficos_clf.figura_confusion(res["TN"], res["FP"], res["FN"], res["TP"],
                                           f"Matriz de confusión en prueba (umbral {um:.3f})"), "07_confusion")
    r.figura(graficos_clf.figura_fiabilidad({"Prueba": (y, p)}, n_tramos=10, titulo="Fiabilidad en prueba (10 tramos "
                                            "iguales)"), "07_fiabilidad")
    c.auc_val, c.auc_prueba = av, ap = vv["ROC-AUC"], ic["ROC-AUC"]
    c.ahorro = ahorro = 1 - res["Costo"] / min(c.costo_nada, c.costo_todo)
    prev = float(y.mean())
    r.agregar(nota(
        f"El ROC-AUC en prueba es {_texto_ic(ap)}, frente a {av:.3f} en validación"
        + (", dentro del intervalo de prueba, de forma que la capacidad de ordenar se sostiene en el periodo nuevo."
           if ap[1] <= av <= ap[2] else
           ", fuera del intervalo de prueba y por encima, de forma que el modelo ordena peor en el periodo nuevo."
           if av > ap[2] else
           ", fuera del intervalo de prueba y por debajo, de forma que el modelo ordena algo mejor en el periodo nuevo.")
        + f"\n\nCon el umbral congelado se levantan {asem:.1f} alertas por semana (en validación "
          f"{u['Alertas por semana']:.1f}); acierta el {100 * res['Precisión']:.0f} % de ellas y se captura el "
          f"{100 * res['Recall']:.0f} % de los no conformes. "
        + (f"El costo, {res['Costo']:.0f}, es un {100 * ahorro:.0f} % menor que el de la mejor política sin modelo, así que "
           "la alerta paga sus falsas alarmas." if ahorro > 0 else
           f"El costo, {res['Costo']:.0f}, no mejora a la mejor política sin modelo, así que en la prueba la alerta no paga "
           "sus falsas alarmas.")
        + f"\n\nLa probabilidad media predicha es {p.mean():.3f} y la tasa real de no conformes {prev:.3f}"
        + (": la calibración se sostiene." if abs(p.mean() - prev) <= 0.1 * prev else
           ", de forma que el modelo sobrestima el riesgo medio del periodo nuevo." if p.mean() > prev else
           ", de forma que el modelo subestima el riesgo medio del periodo nuevo."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("FinClfModelo", registro.NOMBRES_LEGIBLES[clf["modelo"]], "modelo de clasificación evaluado en la prueba"),
        ("FinUmbral", um, "umbral congelado de la clasificación", 3),
        ("FinAUCPrueba", ap[0], "ROC-AUC en la prueba", 3),
        ("FinAUCPruebaInf", ap[1], "límite inferior del IC 95 % del ROC-AUC en la prueba", 3),
        ("FinAUCPruebaSup", ap[2], "límite superior del IC 95 % del ROC-AUC en la prueba", 3),
        ("FinPrecisionMediaPrueba", ic["Precisión media"][0], "precisión media en la prueba", 3),
        ("FinBrierPrueba", ic["Brier"][0], "Brier en la prueba", 4),
        ("FinPrecisionPrueba", res["Precisión"], "precisión con el umbral en la prueba", 3),
        ("FinPrecisionPruebaInf", ic["Precisión"][1], "límite inferior del IC 95 % de la precisión", 3),
        ("FinPrecisionPruebaSup", ic["Precisión"][2], "límite superior del IC 95 % de la precisión", 3),
        ("FinRecallPrueba", res["Recall"], "recall con el umbral en la prueba", 3),
        ("FinRecallPruebaInf", ic["Recall"][1], "límite inferior del IC 95 % del recall", 3),
        ("FinRecallPruebaSup", ic["Recall"][2], "límite superior del IC 95 % del recall", 3),
        ("FinFUnoPrueba", ic["F1"][0], "F1 con el umbral en la prueba", 3),
        ("FinCostoPrueba", res["Costo"], "costo con el umbral en la prueba", 0),
        ("FinCostoPorMuestraPrueba", ic["Costo por muestra"][0], "costo por muestra en la prueba", 3),
        ("FinCostoNoAlertar", c.costo_nada, "costo en la prueba de no alertar"),
        ("FinCostoAlertarTodo", c.costo_todo, "costo en la prueba de alertar todo"),
        ("FinAhorroPct", 100 * ahorro, "ahorro (%) frente a la mejor política sin modelo en la prueba", 0),
        ("FinAlertasSemana", asem, "alertas por semana en la prueba", 1),
        ("FinTP", int(res["TP"]), "verdaderos positivos en la prueba"),
        ("FinFP", int(res["FP"]), "falsos positivos en la prueba"),
        ("FinFN", int(res["FN"]), "falsos negativos en la prueba"),
        ("FinTN", int(res["TN"]), "verdaderos negativos en la prueba"),
        ("FinProbMediaPrueba", float(p.mean()), "probabilidad media predicha en la prueba", 3),
        ("FinPrevalenciaPrueba", prev, "tasa real de no conformes en la prueba", 3),
    ])
    latex.exportar_tabla(lado, "final_clasificacion_validacion", "Validación frente a prueba, clasificación", decimales=3)
    return r


def retrospectiva(c):
    """Umbral que habría convenido en la prueba (no se usa para decidir)."""
    r = Resultado()
    rejilla = [fila["Umbral"] for fila in c.clf["tabla_umbrales"]]
    fila = umbral.elegir_umbral(umbral.tabla_umbrales(c.y_clf, c.p_clf, rejilla))
    c.umbral_retro, c.costo_retro = ur, cr = float(fila["Umbral"]), float(fila["Costo"])
    costo = c.res_clf["Costo"]
    r.agregar(nota(
        (f"En la prueba el umbral de menor costo habría sido {ur:.3f}, con costo {cr:.0f} frente a {costo:.0f} del umbral "
         f"congelado ({c.umbral:.3f}), una diferencia del {100 * (costo - cr) / costo:.1f} %. "
         if not np.isclose(ur, c.umbral) else
         f"En la prueba el umbral de menor costo coincide con el congelado ({c.umbral:.3f}). ")
        + "Esta cifra es retrospectiva y no se usa para ninguna decisión.",
        "advertencia", titulo="Retrospectiva, no se usa",
    ))
    latex.registrar_varias([("FinUmbralRetro", ur, "umbral retrospectivo de menor costo en la prueba (no se usa)", 3),
                            ("FinCostoRetro", cr, "costo con el umbral retrospectivo (no se usa)", 0)])
    return r


def bayes_prueba(c):
    """VPP(x) de los positivos de prueba y reensayos conformes esperados frente a observados."""
    r = Resultado()
    se = pd.DataFrame(c.clf["bayes"]["sens_espec"]).set_index("ensayo")
    c.vpp_desarrollo = vd = pd.DataFrame(c.clf["bayes"]["vpp_por_ensayo"]).set_index("ensayo")["vpp"]
    s_x = c.prueba_clf["ensayo"].map(se["sensibilidad"]).to_numpy(dtype=float)
    e_x = c.prueba_clf["ensayo"].map(se["especificidad"]).to_numpy(dtype=float)
    previa_x = np.asarray(bayes.previa_corregida(c.p_clf, s_x, e_x), dtype=float)
    vpp_x = np.asarray(bayes.valor_predictivo_positivo(previa_x, s_x, e_x), dtype=float)
    pos = c.prueba_clf.assign(s=s_x, e=e_x, vpp_x=vpp_x)[c.y_clf == 1]
    pos = pos.merge(c.prueba_reensayo[["id_muestra", "falso_positivo"]], on="id_muestra", how="left")
    pos["conforme_esperado"] = pos["vpp_x"] * (1 - pos["s"]) + (1 - pos["vpp_x"]) * pos["e"]
    re = pos[pos["falso_positivo"].notna()]
    c.positivas, c.reensayadas = pos, re
    por = pos.groupby("ensayo", observed=True).agg(positivos=("vpp_x", "size"), vpp_x_medio=("vpp_x", "mean")).join(
        re.groupby("ensayo", observed=True).agg(reensayados=("falso_positivo", "size"),
                                                conforme_observado=("falso_positivo", "mean"),
                                                conforme_esperado=("conforme_esperado", "mean")), how="left")
    por["vpp_desarrollo"] = vd.reindex(por.index)
    por["reensayados"] = por["reensayados"].fillna(0).astype(int)
    c.por_ensayo = por
    c.tabla_bayes = t = por.reset_index()[["ensayo", "positivos", "vpp_desarrollo", "vpp_x_medio", "reensayados",
                                           "conforme_esperado", "conforme_observado"]]
    vista = t.rename(columns={
        "ensayo": "Ensayo", "positivos": "Positivos", "vpp_desarrollo": "VPP del ensayo (desarrollo)",
        "vpp_x_medio": "VPP(x) medio", "reensayados": "Reensayados", "conforme_esperado": "Reensayo conforme esperado",
        "conforme_observado": "Reensayo conforme observado"})
    r.agregar(tabla_html(vista, "VPP en los positivos de prueba y reensayos conformes esperados frente a observados",
                         decimales=3))
    c.vpp_x_global = g = float(pos["vpp_x"].mean())
    c.esperado_global = esp = float(re["conforme_esperado"].mean()) if len(re) else np.nan
    c.observado_global = obs = float(re["falso_positivo"].mean()) if len(re) else np.nan
    ic = bayes.intervalo_wilson(int(re["falso_positivo"].sum()), len(re)) if len(re) else (np.nan, np.nan)
    c.ic_observado = ic = (float(np.asarray(ic[0])), float(np.asarray(ic[1])))
    r.agregar(nota(
        f"Entre los {len(pos)} positivos de prueba el VPP(x) medio es **{g:.3f}**, es decir, según los controles y el "
        f"clasificador uno de cada {1 / max(1 - g, 1e-9):.0f} positivos sería falso. Se reensayaron {len(re)} y salió "
        f"conforme el {100 * obs:.1f} % (IC 95 % de Wilson: {100 * ic[0]:.1f} a {100 * ic[1]:.1f} %), frente al "
        f"{100 * esp:.1f} % que predicen el VPP(x) y la sensibilidad del ensayo"
        + (", de forma que la predicción cae dentro del intervalo observado y los controles describen bien la prueba."
           if ic[0] <= esp <= ic[1] else
           ", de forma que el laboratorio observa más reensayos conformes de los que los controles explican, lo que apunta "
           "a una especificidad efectiva menor que la de los controles." if obs > esp else
           ", de forma que se observan menos reensayos conformes de los que los controles predicen."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("FinPositivosPrueba", len(pos), "positivos (no conformes) en la prueba"),
        ("FinVPPxMedio", g, "VPP(x) medio de los positivos de prueba", 3),
        ("FinUnoDeCadaFalso", 1 / max(1 - g, 1e-9), "uno de cada cuántos positivos sería falso (prueba)", 0),
        ("FinReensayados", len(re), "positivos de prueba reensayados"),
        ("FinReensayoConformeObservadoPct", 100 * obs, "reensayos conformes observados (%) en la prueba", 1),
        ("FinReensayoConformeObservadoInfPct", 100 * ic[0], "límite inferior de Wilson (%)", 1),
        ("FinReensayoConformeObservadoSupPct", 100 * ic[1], "límite superior de Wilson (%)", 1),
        ("FinReensayoConformeEsperadoPct", 100 * esp, "reensayos conformes esperados (%) en la prueba", 1),
    ])
    latex.exportar_tabla(vista, "final_bayes", "VPP y reensayos conformes en la prueba, por ensayo", decimales=3)
    return r


def auditoria(c):
    r = Resultado()
    verdad = carga.leer_verdad()
    c.auditoria_prueba = None
    if verdad is None:
        return r.agregar(nota("No hay verdad oculta: la auditoría no aplica y la única referencia es el reensayo.",
                              "simulacion"))
    auditadas = c.positivas.merge(verdad["muestras"][["id_muestra", "contaminada_real"]], on="id_muestra", how="inner")
    real = auditadas.groupby("ensayo", observed=True)["contaminada_real"].mean()
    t = c.tabla_bayes[["ensayo", "positivos", "vpp_desarrollo", "vpp_x_medio"]].assign(
        vpp_real=lambda x: x["ensayo"].map(real))
    vpp_real = float(auditadas["contaminada_real"].mean())
    brier_x = metricas.brier(auditadas["contaminada_real"], auditadas["vpp_x"])
    brier_e = metricas.brier(auditadas["contaminada_real"], auditadas["ensayo"].map(c.vpp_desarrollo).to_numpy(dtype=float))
    vista = t.rename(columns={"ensayo": "Ensayo", "positivos": "Positivos", "vpp_desarrollo": "VPP del ensayo (desarrollo)",
                              "vpp_x_medio": "VPP(x) medio", "vpp_real": "Fracción realmente contaminada"})
    r.agregar(tabla_html(vista, "Auditoría en prueba: VPP estimado frente a la verdad de la simulación", decimales=3))
    g = c.vpp_x_global
    r.agregar(nota(
        f"En la prueba la fracción de positivos realmente contaminados es {vpp_real:.3f}, frente a un VPP(x) medio de {g:.3f}"
        + (", de forma que la estimación con controles es optimista: la especificidad de un control estéril supera a la de "
           "una muestra que se manipula." if g > vpp_real + 0.02 else
           ", de forma que la estimación con controles es conservadora." if g < vpp_real - 0.02 else
           ", prácticamente la misma cifra.")
        + f" Como probabilidad individual, el Brier del VPP(x) es {brier_x:.4f} frente a {brier_e:.4f} del VPP por ensayo"
        + (", así que la previa por muestra aporta." if brier_x < brier_e else
           ", así que la previa por muestra no mejora al VPP por ensayo en la prueba."),
        "simulacion", titulo="Lo que dice la verdad oculta",
    ))
    c.auditoria_prueba = {"vpp_real": vpp_real, "vpp_x_medio": g, "brier_vpp_x": brier_x, "brier_vpp_ensayo": brier_e,
                          "por_ensayo": t.to_dict("records")}
    latex.registrar_varias([
        ("FinAudVPPReal", vpp_real, "fracción de positivos de prueba realmente contaminados", 3),
        ("FinAudBrierVPPx", brier_x, "Brier del VPP(x) en la prueba", 4),
        ("FinAudBrierVPPEnsayo", brier_e, "Brier del VPP por ensayo en la prueba", 4),
    ])
    latex.exportar_tabla(vista, "final_auditoria", "Auditoría en prueba del VPP frente a la verdad", decimales=3)
    return r


def regla_reensayo(c):
    r = Resultado()
    c.regla = regla = c.clf["bayes"]["regla_reensayo"]
    variable, borde = regla["variable"], regla["borde"]
    c.borde, c.regla_prueba = borde, None
    pr = c.prueba_reensayo
    if borde is None or variable not in pr.columns:
        return r.agregar(nota("La etapa 05 no fijó un borde para la regla de reensayo, de forma que no hay nada que "
                              "comprobar.", "info"))
    con_razon = pr[pr[variable].notna()]
    bajo = con_razon[variable] <= borde
    tb = float(con_razon.loc[bajo, "falso_positivo"].mean()) if bajo.any() else np.nan
    ta = float(con_razon.loc[~bajo, "falso_positivo"].mean()) if (~bajo).any() else np.nan
    cap = float(con_razon.loc[bajo, "falso_positivo"].sum() / max(con_razon["falso_positivo"].sum(), 1))
    inf, sup = bayes.intervalo_wilson(int(con_razon.loc[bajo, "falso_positivo"].sum()), int(bajo.sum()))
    c.regla_prueba = rp = {"n": int(len(con_razon)), "n_bajo": int(bajo.sum()), "tasa_bajo": tb,
                           "tasa_bajo_inf": float(np.asarray(inf)), "tasa_bajo_sup": float(np.asarray(sup)),
                           "tasa_alto": ta, "captura": cap}
    tabla = pd.DataFrame({
        "Periodo": ["Desarrollo", "Prueba"],
        f"Conformes con razón ≤ {borde:.2f}": [regla["tasa_bajo"], tb],
        f"Conformes con razón > {borde:.2f}": [regla["tasa_alto"], ta],
        "Fracción de conformes capturada": [regla["captura"], cap],
    })
    r.agregar(tabla_html(tabla, f"Regla de reensayo (recuento por debajo de {10 ** borde:.2f} veces el límite)",
                         decimales=3))
    r.agregar(nota(
        f"En la prueba hay {len(con_razon)} reensayos con recuento, {int(bajo.sum())} de ellos por debajo del borde. Ahí "
        f"sale conforme el {100 * tb:.0f} % (IC 95 %: {100 * rp['tasa_bajo_inf']:.0f} a {100 * rp['tasa_bajo_sup']:.0f} %), "
        f"frente al {100 * ta:.0f} % por encima"
        + (", de forma que la regla se sostiene: un recuento apenas por encima del límite sigue siendo el caso donde el "
           "falso positivo es lo más probable." if tb >= 0.5 and tb > ta else
           ", de forma que la regla ordena el riesgo pero en la prueba el grupo bajo el borde ya no es mayoritariamente "
           "conforme." if tb > ta else
           ", de forma que la regla no se sostiene en la prueba.")
        + f" El grupo concentra el {100 * cap:.0f} % de los reensayos conformes (en el desarrollo "
          f"{100 * regla['captura']:.0f} %).",
        "hallazgo", titulo="Regla para el reensayo",
    ))
    latex.registrar_varias([
        ("FinReglaReensayosConRecuento", len(con_razon), "reensayos de prueba con recuento"),
        ("FinReglaBajoBorde", int(bajo.sum()), "reensayos de prueba bajo el borde"),
        ("FinReglaTasaBajoPct", 100 * tb, "reensayos conformes (%) bajo el borde en la prueba", 0),
        ("FinReglaTasaBajoInfPct", 100 * rp["tasa_bajo_inf"], "límite inferior de Wilson (%)", 0),
        ("FinReglaTasaBajoSupPct", 100 * rp["tasa_bajo_sup"], "límite superior de Wilson (%)", 0),
        ("FinReglaTasaAltoPct", 100 * ta, "reensayos conformes (%) sobre el borde en la prueba", 0),
        ("FinReglaCapturaPct", 100 * cap, "reensayos conformes que concentra el grupo en la prueba (%)", 0),
    ])
    latex.exportar_tabla(tabla, "final_regla_reensayo", "Regla de reensayo en desarrollo y prueba", decimales=3)
    return r


def agrupamiento(c):
    r = Resultado()
    grp, comp = c.grp["seleccionado"], c.grp.get("complementario")
    fila = next(f for f in c.cribado["tabla_cribado"] if f["analisis"] == grp["analisis"])
    c.silueta_cribado, c.silueta_etapa = sc, se = float(fila["silueta"]), float(grp["kmeans"]["silueta"])
    c.grp_sel, c.grp_comp = grp, comp
    nombres = [s["nombre"] for s in grp["segmentos"]]
    r.agregar(nota(
        f"En «{grp['analisis']}» la etapa 06 encontró {grp['kmeans']['k']} grupos con K-means "
        f"({', '.join(f'«{n}»' for n in nombres)}), con una silueta de {se:.3f} y un estadístico de Hopkins de "
        f"{grp['hopkins']:.3f}; DBSCAN, con otra lógica, dejó {grp['dbscan']['grupos']} grupos y un "
        f"{100 * grp['dbscan']['proporcion_ruido']:.0f} % de ruido, y los dos acuerdan poco (ARI "
        f"{grp['ari_kmeans_dbscan']:.2f}). "
        + (f"En «{comp['analisis']}» quedaron {comp['kmeans']['k']} segmentos de clientes con silueta "
           f"{comp['kmeans']['silueta']:.3f}. " if comp else "")
        + "Las siluetas bajas dicen que hay tendencia a agrupar pero sin fronteras nítidas, así que los grupos sirven para "
          "describir y conversar con el laboratorio, no para decidir muestra a muestra."
        + f"\n\nEn el cribado (etapa 03) «{grp['analisis']}» tuvo una silueta de {sc:.3f}, frente a {se:.3f} con el "
          f"preprocesamiento de la etapa 06 (variables elegidas, logaritmo en {len(grp['con_log1p'])} variables asimétricas "
          f"y escalado), una diferencia de {sc - se:+.3f}. Las dos cifras no son comparables, y la que describe los grupos "
          "presentados es la de la etapa 06.",
        "hallazgo",
    ))
    return r


def resumen(c):
    """Tarjetas, preguntas del laboratorio, recomendaciones, limitaciones y siguientes pasos."""
    r = Resultado()
    res, rp, grp, comp = c.res_clf, c.regla_prueba, c.grp_sel, c.grp_comp
    r.agregar(tarjetas([
        {"titulo": "Carga diaria", "valor": c.rmse_prueba, "decimales": 1, "prefijo": "RMSE ",
         "detalle": f"habilidad {c.habilidad_prueba:.3f} frente a «{c.referencia}»"},
        {"titulo": "Riesgo en recepción", "valor": c.auc_prueba[0], "decimales": 3, "prefijo": "AUC ", "color": "naranja",
         "detalle": f"{c.alertas_semana:.1f} alertas por semana, precisión {res['Precisión']:.3f}"},
        {"titulo": "Positivos reales", "valor": 100 * c.vpp_x_global, "decimales": 0, "sufijo": " %", "color": "verde",
         "detalle": f"VPP(x) medio; reensayo conforme {100 * c.observado_global:.0f} %"},
        {"titulo": "Regla de reensayo", "valor": 100 * rp["tasa_bajo"] if rp else np.nan, "decimales": 0, "sufijo": " %",
         "color": "morado", "detalle": "reensayos conformes bajo el borde" if rp else "sin regla"},
    ], titulo="Cifras de la prueba"))
    pocos = c.por_ensayo[c.por_ensayo["positivos"] < MINIMO_POSITIVOS].index.tolist()
    texto_pocos = " y ".join(f"«{e}» ({int(c.por_ensayo.loc[e, 'positivos'])})" for e in pocos)
    g = c.vpp_x_global
    preguntas = pd.DataFrame([
        {"Pregunta del laboratorio": "¿Cuántas muestras llegarán mañana?",
         "Respuesta": f"{c.nombre_modelo_reg} con un error típico de {c.mae_prueba:.1f} muestras sobre una media de "
                      f"{c.media_prueba:.1f}",
         "Evidencia": f"RMSE {c.rmse_prueba:.2f} en prueba ({c.rmse_val:.2f} en validación); habilidad "
                      f"{c.habilidad_prueba:.3f}",
         "Limitación": f"el {100 * c.fuera_modelo:.0f} % de los días queda fuera de la tolerancia"},
        {"Pregunta del laboratorio": "¿Qué muestras están en riesgo?",
         "Respuesta": f"alerta en recepción con umbral {c.umbral:.3f}, unas {c.alertas_semana:.0f} por semana",
         "Evidencia": f"AUC {c.auc_prueba[0]:.3f}; costo {res['Costo']:.0f} frente a "
                      f"{min(c.costo_nada, c.costo_todo):.0f} sin modelo",
         "Limitación": f"solo el {100 * res['Precisión']:.0f} % de las alertas acierta"},
        {"Pregunta del laboratorio": "¿Qué tan probable es un falso positivo?",
         "Respuesta": f"uno de cada {1 / max(1 - g, 1e-9):.0f} positivos, según controles y clasificador"
                      + (f"; reensayar si el recuento es menor que {10 ** c.borde:.2f} veces el límite" if rp else ""),
         "Evidencia": f"reensayo conforme {100 * c.observado_global:.1f} % observado frente a "
                      f"{100 * c.esperado_global:.1f} % esperado",
         "Limitación": "el reensayo no es una verdad: también falla"},
        {"Pregunta del laboratorio": "¿Qué grupos existen?",
         "Respuesta": f"{grp['kmeans']['k']} perfiles de muestra" + (f" y {comp['kmeans']['k']} segmentos de clientes"
                                                                      if comp else ""),
         "Evidencia": f"silueta {c.silueta_etapa:.3f}, Hopkins {grp['hopkins']:.3f}",
         "Limitación": "fronteras difusas; uso descriptivo"},
    ])
    r.agregar(tabla_html(preguntas, "Pregunta del laboratorio, respuesta, evidencia y limitación"))
    recomendaciones = [
        ("Planear la dotación del día siguiente con la predicción",
         f"sumando un margen de unas {c.mae_prueba:.0f} muestras, el error típico medido en la prueba"
         if c.ic_reg["Habilidad"][1] > 0 else
         f"solo como apoyo, porque en la prueba no supera con claridad a «{c.referencia}»"),
        ("Marcar en recepción las muestras sobre el umbral",
         f"para leerlas antes o asignarles segundo analista; son unas {c.alertas_semana:.0f} por semana y acierta el "
         f"{100 * res['Precisión']:.0f} %" if c.ahorro > 0 else
         "no se recomienda todavía: en la prueba la alerta no supera a la mejor política sin modelo"),
        ("Ordenar los positivos por su VPP(x)", "para decidir cuáles revisar primero, nunca para descartar un positivo sin "
                                                "reensayo"),
    ]
    if rp and rp["tasa_bajo"] > rp["tasa_alto"]:
        recomendaciones.append(("Reensayar antes de informar los recuentos apenas sobre el límite",
                                f"por debajo de {10 ** c.borde:.2f} veces el límite, donde en la prueba salió conforme el "
                                f"{100 * rp['tasa_bajo']:.0f} % de los reensayos"))
    recomendaciones.append(("Usar los perfiles y segmentos para conversar con el laboratorio",
                            "como descripción de la operación, no como regla de decisión"))
    r.agregar(pasos(recomendaciones, titulo="Recomendaciones"))
    limitaciones = [
        "Los datos son simulados: el procedimiento funciona de punta a punta, pero las cifras del laboratorio dependerán "
        "de su exportación, que se procesará con el mismo código.",
        f"Brechas medidas entre validación y prueba: RMSE {c.rmse_val:.2f} → {c.rmse_prueba:.2f} y ROC-AUC "
        f"{c.auc_val:.3f} → {c.auc_prueba[0]:.3f}.",
        (f"Pocos positivos de prueba en {texto_pocos}, de forma que su VPP y su tasa de reensayos conformes son poco "
         "precisos." if pocos else f"Todos los ensayos tienen al menos {MINIMO_POSITIVOS} positivos de prueba."),
        f"Siluetas bajas en el agrupamiento ({c.silueta_etapa:.3f} en perfiles de muestra), y la del cribado "
        f"({c.silueta_cribado:.3f}) no es comparable con la de la etapa 06.",
        "Los intervalos bootstrap remuestrean filas como si fueran independientes; en la serie diaria hay autocorrelación, "
        "de forma que los intervalos de la regresión probablemente son más estrechos de lo que deberían.",
    ]
    r.agregar(nota("\n\n".join(f"**{i}.** {t}" for i, t in enumerate(limitaciones, start=1)), "limite",
                   titulo="Limitaciones"))
    r.agregar(pasos([
        ("Copiar la exportación real", "en `datos/reales/` y completar `configuracion/mapeo_columnas.json` con los nombres "
                                       "de columnas y valores del laboratorio"),
        ("Reconstruir", "con `PROYECTO_ORIGEN=real make notebooks`, que vuelve a ejecutar las siete etapas"),
        ("Revisar primero la etapa 01", "la tabla de cobertura dice qué columnas opcionales faltan y qué análisis quedan "
                                        "fuera"),
        ("Revisar las fechas de corte", "`CORTE_VALIDACION` y `CORTE_PRUEBA` en `proyecto/base/config.py` deben caer "
                                        "dentro del periodo exportado"),
        ("Revisar los controles", "sin controles de calidad no hay sensibilidad ni especificidad y la parte de Bayes no "
                                  "aplica"),
    ], titulo="Siguientes pasos con la exportación real"))
    if config.ORIGEN_DATOS == "simulacion":
        r.agregar(nota("Todas las cifras de este proyecto provienen de una base **simulada** con la estructura de la del "
                       "laboratorio; ninguna describe todavía su operación real.", "simulacion"))
    latex.registrar_varias([
        ("FinAgrK", int(grp["kmeans"]["k"]), "perfiles de muestra de K-means"),
        ("FinAgrSilueta", c.silueta_etapa, "silueta de los perfiles de muestra (etapa 06)", 3),
        ("FinAgrSiluetaCribado", c.silueta_cribado, "silueta del mismo análisis en el cribado", 3),
        ("FinAgrHopkins", float(grp["hopkins"]), "Hopkins de los perfiles de muestra", 3),
        ("FinEnsayosPocosPositivos", len(pocos), f"ensayos con menos de {MINIMO_POSITIVOS} positivos de prueba"),
        ("FinMinimoPositivos", MINIMO_POSITIVOS, "mínimo de positivos de prueba por ensayo para una cifra precisa"),
    ])
    if comp:
        latex.registrar("FinAgrSegmentos", int(comp["kmeans"]["k"]), "segmentos de clientes de K-means")
    latex.exportar_tabla(preguntas, "final_preguntas", "Pregunta del laboratorio, respuesta, evidencia y limitación")
    resumen_final = pd.DataFrame([
        {"Análisis": "Regresión (carga diaria)", "Métrica": "RMSE", "Validación": c.rmse_val, "Prueba": c.rmse_prueba},
        {"Análisis": "Regresión (carga diaria)", "Métrica": "Habilidad", "Validación": c.fila_val_reg["Habilidad"],
         "Prueba": c.habilidad_prueba},
        {"Análisis": "Clasificación (riesgo)", "Métrica": "ROC-AUC", "Validación": c.auc_val, "Prueba": c.auc_prueba[0]},
        {"Análisis": "Clasificación (riesgo)", "Métrica": "Precisión", "Validación": c.umbral_val["Precisión"],
         "Prueba": res["Precisión"]},
        {"Análisis": "Clasificación (riesgo)", "Métrica": "Recall", "Validación": c.umbral_val["Recall"],
         "Prueba": res["Recall"]},
        {"Análisis": "Clasificación (riesgo)", "Métrica": "Alertas por semana",
         "Validación": c.umbral_val["Alertas por semana"], "Prueba": c.alertas_semana},
        {"Análisis": "Bayes (falso positivo)", "Métrica": "Reensayo conforme observado",
         "Validación": c.clf["bayes"]["reensayo_conforme"]["observado"], "Prueba": c.observado_global},
        {"Análisis": "Bayes (falso positivo)", "Métrica": "Reensayo conforme esperado",
         "Validación": c.clf["bayes"]["reensayo_conforme"]["esperado"], "Prueba": c.esperado_global},
    ])
    latex.exportar_tabla(resumen_final, "final_resumen", "Resumen final: validación (o desarrollo) frente a prueba",
                         decimales=3)
    _guardar_artefacto(c)
    return r


def _guardar_artefacto(c):
    c.segundos = time.perf_counter() - c.inicio
    f = c.fila_val_reg
    guardar_json({
        "regresion": {
            "analisis": c.nombre_reg, "modelo": c.reg["modelo"], "referencia": c.referencia, "dias": int(len(c.y_reg)),
            "tabla": c.tabla_reg.to_dict("records"),
            "intervalos": {m: list(map(float, v)) for m, v in c.ic_reg.items()},
            "validacion": {"RMSE": c.rmse_val, "MAE": f["MAE"], "R2": f["R2"], "Habilidad": f["Habilidad"]},
            "operacion": {"media_prueba": c.media_prueba, "tolerancia": c.tolerancia, "dias_fuera": c.fuera_modelo,
                          "dias_fuera_referencia": c.fuera_ref},
            "fin_de_anio": {"dias": c.n_ventana, "residuo_ventana": c.residuo_ventana, "residuo_resto": c.residuo_resto},
        },
        "clasificacion": {
            "analisis": c.nombre_clf, "modelo": c.clf["modelo"], "umbral": c.umbral, "muestras": int(len(c.y_clf)),
            "semanas": c.semanas, "metricas": c.res_clf, "alertas_por_semana": c.alertas_semana,
            "intervalos": {m: list(map(float, v)) for m, v in c.ic_clf.items()},
            "validacion": {m: (None if pd.isna(v) else float(v)) for m, v in c.valores_validacion.items()},
            "costo_politicas": {"no_alertar": c.costo_nada, "alertar_todo": c.costo_todo},
            "retrospectiva_no_usada": {"umbral": c.umbral_retro, "costo": c.costo_retro},
        },
        "bayes": {
            "positivos": int(len(c.positivas)), "vpp_x_medio": c.vpp_x_global, "reensayados": int(len(c.reensayadas)),
            "reensayo_conforme": {"observado": c.observado_global, "esperado": c.esperado_global,
                                  "ic_observado": list(c.ic_observado)},
            "por_ensayo": c.tabla_bayes.to_dict("records"), "auditoria": c.auditoria_prueba,
            "regla_reensayo": c.regla_prueba,
        },
        "agrupamiento": {"analisis": c.grp_sel["analisis"], "k": c.grp_sel["kmeans"]["k"], "silueta": c.silueta_etapa,
                         "silueta_cribado": c.silueta_cribado},
        "segundos_etapa": c.segundos,
    }, rutas.DIR_ARTEFACTOS / "07_evaluacion.json")


def terminar(c):
    cerrar({"rmse_prueba": c.rmse_prueba, "habilidad_prueba": c.habilidad_prueba, "auc_prueba": c.auc_prueba[0],
            "costo_prueba": c.res_clf["Costo"], "vpp_x_medio": c.vpp_x_global, "segundos": c.segundos})
