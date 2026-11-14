"""Capa 8 · Mecánica compartida del modelado a fondo (etapas 04 y 05).

Responsabilidad única: lo que las etapas de regresión y clasificación hacen
igual (variables discretas, tuberías de las corridas del diseño, vistas de
niveles y confusiones del L16, corrida de confirmación con su tolerancia,
factorial completo, rejillas de hiperparámetros) sin ningún texto: la
narración de cada etapa queda en su propio módulo.
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from proyecto.cribado import arreglos, corridas
from proyecto.etapas.comun import legible
from proyecto.evaluacion import validacion
from proyecto.modelos import registro, tuberias
from proyecto.variables.comunes import columnas_de_grupo

LIMITE_SEGUNDOS_COMPLETO = 60.0


def discretas(entrenamiento, numericas):
    """Numéricas enteras con 3 a 12 valores distintos: el factor de codificación las trata como
    orden (ordinal) o como categorías sin orden (one-hot)."""
    return [x for x in numericas
            if 3 <= entrenamiento[x].dropna().nunique() <= 12
            and np.all(np.mod(entrenamiento[x].dropna().to_numpy(dtype=float), 1) == 0)]


def constructor(tarea, papeles, columnas_discretas):
    """Función `fila → tubería` para las corridas del diseño."""
    def construir(fila):
        num, cat = columnas_de_grupo(papeles, fila["grupo"])
        disc = [x for x in columnas_discretas if x in num]
        extra = ({"objetivo_log": bool(fila["objetivo_log"])} if "objetivo_log" in fila.index
                 else {"desbalance": fila["desbalance"]})
        return tuberias.construir_tuberia(
            tarea, fila["modelo"], [x for x in num if x not in disc], list(cat) + disc,
            escalado=fila["escalado"], codificacion=fila["codificacion"], seleccion=fila["seleccion"],
            hiper=fila.get("hiper"), **extra)
    return construir


def vista_niveles(tabla, factores):
    """Copia de la tabla con los niveles de los factores como texto."""
    vista = tabla.copy()
    for factor in factores:
        if factor in vista.columns:
            vista[factor] = [legible(v) for v in vista[factor]]
    return vista


def diseno_l16(factores, columnas):
    arreglo = arreglos.arreglo_l16()
    return arreglo, arreglos.asignar_factores(arreglo, factores, columnas)


def vista_confusiones(confusiones):
    return confusiones.assign(
        columnas_interaccion=[", ".join(x) for x in confusiones["columnas_interaccion"]],
        confundida_con=[", ".join(x) if x else "libre" for x in confusiones["confundida_con"]],
    ).rename(columns={"factor_1": "Factor 1", "factor_2": "Factor 2", "columnas_interaccion": "Columnas de la interacción",
                      "confundida_con": "Confundida con"})


def interacciones_libres(confusiones):
    return [f"{a} × {b}" for a, b, x in confusiones[["factor_1", "factor_2", "confundida_con"]].itertuples(index=False)
            if not x]


def respuesta_legible(respuesta, factores):
    vista = vista_niveles(respuesta.reset_index().rename(columns={"factor": "Factor"}), factores)
    for columna in ("mejor_nivel", "peor_nivel"):
        vista[columna] = [legible(v) for v in vista[columna]]
    return vista.sort_values("rango")


def confirmar(niveles_optimo, construir, X, y, bloques, metrica, resultados, efectos, factores):
    """Corrida de confirmación del óptimo aditivo y tolerancia de dos veces el error residual del
    modelo aditivo en las corridas del diseño. Devuelve un diccionario."""
    indice = pd.Index(["confirmación"])
    diseno = pd.DataFrame({f: pd.Series([v], index=indice, dtype=object) for f, v in niveles_optimo.items()},
                          index=indice)
    fila = corridas.ejecutar_diseno(diseno, construir, X, y, bloques, metrica).iloc[0]
    media_sn = float(resultados["sn"].mean())
    media_por_nivel = {(f, repr(n)): m for f, n, m in efectos[["factor", "nivel", "media"]].itertuples(index=False)}
    nombres = list(factores)
    sn_aditiva = media_sn + sum(
        np.array([media_por_nivel[(f, repr(v))] for v in resultados[f]]) - media_sn for f in nombres)
    grados = len(resultados) - 1 - sum(len(v) - 1 for v in factores.values())
    error = float(np.sqrt(np.sum((resultados["sn"].to_numpy() - sn_aditiva) ** 2) / grados))
    return {"fila": fila, "sn_observada": float(fila["sn"]), "tolerancia": 2 * error}


def factorial_completo(factores, segundos_l16, n_l16, construir, X, y, bloques, metrica, respuesta, niveles_optimo):
    """Ejecuta el factorial completo solo si su tiempo estimado no pasa del límite."""
    diseno = arreglos.factorial_completo(factores)
    estimado = segundos_l16 * len(diseno) / n_l16
    resumen = {"ejecutado": bool(estimado <= LIMITE_SEGUNDOS_COMPLETO), "segundos_estimados": estimado,
               "corridas": len(diseno)}
    if not resumen["ejecutado"]:
        return resumen, None
    resultados = corridas.ejecutar_diseno(diseno, construir, X, y, bloques, metrica)
    respuesta_completo = corridas.tabla_respuesta(resultados, list(factores))
    rho, _ = spearmanr(respuesta["delta"], respuesta_completo.loc[respuesta.index, "delta"])
    optimo = respuesta_completo["mejor_nivel"].to_dict()
    resumen.update({"segundos": float(resultados["segundos"].sum()), "correlacion_rangos": float(rho),
                    "factores_mismo_optimo": [f for f in factores if repr(optimo[f]) == repr(niveles_optimo[f])]})
    return resumen, {"resultados": resultados, "respuesta": respuesta_completo}


def configuracion_y_candidatos(confirmacion, niveles_optimo, resultados, efectos, factores):
    """Configuración base (mayor S/N observada entre la confirmación y el L16) y segunda familia."""
    mejor = resultados["sn"].idxmax()
    if confirmacion["sn_observada"] >= resultados["sn"].max():
        configuracion, origen = dict(niveles_optimo), "corrida de confirmación"
    else:
        configuracion = {f: resultados.loc[mejor, f] for f in factores}
        origen = f"corrida {mejor} del L16"
    por_modelo = efectos[efectos["factor"] == "modelo"].sort_values("media", ascending=False)
    segunda = next(m for m in por_modelo["nivel"] if m != configuracion["modelo"])
    candidatos = {configuracion["modelo"]: configuracion, segunda: {**configuracion, "modelo": segunda}}
    return configuracion, origen, candidatos


def buscar_rejillas(tarea, candidatos, construir, X, y, bloques, metrica, n_trabajos=None):
    """`GridSearchCV` por candidato sobre los bloques temporales. Devuelve búsquedas, tablas y segundos."""
    busquedas, tablas, segundos = {}, {}, {}
    for modelo, cfg in candidatos.items():
        inicio = time.perf_counter()
        objetivo_log = bool(cfg.get("objetivo_log", False))
        estimador = construir(pd.Series(cfg, dtype=object))
        rejilla = registro.rejilla(tarea, cfg["modelo"], objetivo_log=objetivo_log)
        if cfg["modelo"] == "arbol":
            prefijo = registro.prefijo_modelo(tarea, "arbol", objetivo_log=objetivo_log)
            rejilla[prefijo + "ccp_alpha"] = [float(a) for a in tuberias.alfas_de_poda(estimador, X, y)]
        argumentos = {"n_trabajos": n_trabajos} if n_trabajos is not None else {}
        busquedas[modelo], tablas[modelo] = validacion.buscar_en_rejilla(estimador, rejilla, X, y, bloques, metrica,
                                                                         **argumentos)
        segundos[modelo] = time.perf_counter() - inicio
    return busquedas, tablas, segundos


def hiper_de(busqueda):
    """Hiperparámetros elegidos, sin prefijos."""
    return {clave.split("__")[-1]: valor for clave, valor in busqueda.best_params_.items()}


def taguchi_para_artefacto(factores, columnas, respuesta, niveles_optimo, sn_predicha, confirmacion, coincide,
                           segundos_l16, resumen_completo):
    return {
        "factores": {f: [legible(v) for v in niveles] for f, niveles in factores.items()},
        "columnas": columnas,
        "tabla_respuesta": respuesta.reset_index().assign(
            mejor_nivel=lambda t: [legible(v) for v in t["mejor_nivel"]],
            peor_nivel=lambda t: [legible(v) for v in t["peor_nivel"]]).to_dict("records"),
        "optimo": {f: legible(v) for f, v in niveles_optimo.items()},
        "sn_predicha": sn_predicha, "sn_observada": confirmacion["sn_observada"],
        "tolerancia": confirmacion["tolerancia"], "coincide": bool(coincide), "segundos_l16": segundos_l16,
        "factorial_completo": resumen_completo,
    }
