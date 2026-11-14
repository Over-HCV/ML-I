"""Capa 6 · Corridas de un diseño y análisis de Taguchi.

Responsabilidad única: ejecutar las corridas de un diseño de experimentos
(cada corrida es una configuración de tubería evaluada en los bloques
temporales) y resumirlas con las herramientas de Taguchi: razón señal/ruido,
efectos principales, tabla de respuesta y predicción del óptimo.

Los bloques temporales hacen de factor de ruido: la razón señal/ruido premia
una configuración que rinde bien **y** de forma estable entre bloques.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from proyecto.evaluacion.metricas import resolver_metrica
from proyecto.evaluacion.validacion import puntuar_por_bloques


def senal_ruido(valores, sentido):
    """Razón señal/ruido de Taguchi, en decibelios (siempre conviene mayor).

    - `sentido="mayor"` (mayor es mejor): −10·log10( media(1 / y²) ).
      Exige respuestas estrictamente positivas; con una respuesta ≤ 0 la
      razón no está definida y se lanza `ValueError`.
    - `sentido="menor"` (menor es mejor): −10·log10( media(y²) ).
    """
    y = np.asarray(valores, dtype=float)
    if y.size == 0 or np.isnan(y).any():
        return float("nan")
    if sentido == "mayor":
        if (y <= 0).any():
            raise ValueError("La razón señal/ruido «mayor es mejor» exige respuestas positivas.")
        return float(-10 * np.log10(np.mean(1 / y**2)))
    if sentido == "menor":
        with np.errstate(divide="ignore"):
            return float(-10 * np.log10(np.mean(y**2)))
    raise ValueError(f"Sentido desconocido: {sentido!r} (use 'mayor' o 'menor').")


def ejecutar_diseno(diseno, constructor, X, y, bloques, metrica, sentido=None):
    """Ejecuta todas las corridas de un diseño.

    `diseno` tiene una fila por corrida y una columna por factor;
    `constructor(fila)` recibe esa fila (una Serie) y devuelve el estimador
    sin ajustar. Cada corrida se evalúa con `puntuar_por_bloques` y la
    `metrica` indicada; `sentido` se toma del registro de métricas si no se
    da.

    Devuelve una tabla con el índice del diseño y, por corrida: los niveles
    de los factores, `bloque_1..k`, `media`, `desv` (desviación muestral
    entre bloques), `sn` (razón señal/ruido), `segundos`, `avisos` (avisos de
    no convergencia) y `error`. Si una corrida falla, su fila queda con NaN y
    el texto del error, y la ejecución continúa con la siguiente.
    """
    funcion, sentido = resolver_metrica(metrica, sentido)
    bloques = list(bloques)
    filas = []
    for registro in diseno.to_dict("records"):
        inicio = time.perf_counter()
        puntajes = np.full(len(bloques), np.nan)
        error, sn, avisos = "", float("nan"), 0
        try:
            with warnings.catch_warnings(record=True) as capturados:
                warnings.simplefilter("always")
                estimador = constructor(pd.Series(registro, dtype=object))
                puntajes = puntuar_por_bloques(estimador, X, y, bloques, (funcion, sentido))
            avisos = sum(issubclass(a.category, ConvergenceWarning) for a in capturados)
            sn = senal_ruido(puntajes, sentido)
        except Exception as excepcion:  # una corrida fallida no detiene el diseño
            error = f"{type(excepcion).__name__}: {excepcion}"
        fila = {f"bloque_{i + 1}": p for i, p in enumerate(puntajes)}
        fila.update({
            "media": float(np.mean(puntajes)),
            "desv": float(np.std(puntajes, ddof=1)) if len(puntajes) > 1 else float("nan"),
            "sn": sn,
            "segundos": time.perf_counter() - inicio,
            "avisos": avisos,
            "error": error,
        })
        filas.append(fila)
    # Los niveles se copian del diseño tal cual (así un nivel `None` no se
    # convierte en NaN) y al lado van las columnas calculadas.
    return pd.concat([diseno, pd.DataFrame(filas, index=diseno.index)], axis=1)


def _niveles(columna):
    """Niveles distintos de un factor, en orden de aparición, con la máscara
    de las corridas de cada uno. Admite etiquetas no ordenables o None."""
    claves = [repr(valor) for valor in columna]
    vistos = {}
    for clave, valor in zip(claves, columna):
        vistos.setdefault(clave, valor)
    claves = np.array(claves, dtype=object)
    return [(valor, claves == clave) for clave, valor in vistos.items()]


def _tabla_con_niveles(filas, columnas, de_nivel):
    """Arma una tabla dejando las columnas de niveles como objetos, para que
    pandas no convierta etiquetas como `None` o tuplas."""
    tabla = pd.DataFrame(index=range(len(filas)))
    for columna in columnas:
        valores = [fila[columna] for fila in filas]
        tabla[columna] = pd.Series(valores, dtype=object) if columna in de_nivel else valores
    return tabla


def efectos_principales(resultados, factores, respuesta="sn"):
    """Media de la respuesta en cada nivel de cada factor.

    Devuelve una tabla larga con columnas `factor`, `nivel`, `media` y `n`
    (corridas con respuesta válida). Las corridas sin respuesta (NaN) se
    excluyen; en ese caso el diseño deja de estar balanceado y los efectos
    son solo aproximados.
    """
    valores = resultados[respuesta].to_numpy(dtype=float)
    filas = []
    for factor in factores:
        for nivel, mascara in _niveles(resultados[factor]):
            validos = valores[mascara & ~np.isnan(valores)]
            filas.append({
                "factor": factor, "nivel": nivel,
                "media": float(validos.mean()) if len(validos) else float("nan"),
                "n": int(len(validos)),
            })
    return _tabla_con_niveles(filas, ["factor", "nivel", "media", "n"], ["nivel"])


def tabla_respuesta(resultados, factores, respuesta="sn"):
    """Tabla de respuesta de Taguchi: por factor, el nivel con la mayor y la
    menor media de la respuesta, la diferencia entre ambas (`delta`) y el
    `rango` (1 = factor con mayor delta, el más influyente)."""
    efectos = efectos_principales(resultados, factores, respuesta)
    filas = []
    for factor in factores:
        propios = efectos.loc[efectos["factor"] == factor].dropna(subset=["media"])
        mejor = propios.loc[propios["media"].idxmax()]
        peor = propios.loc[propios["media"].idxmin()]
        filas.append({
            "factor": factor,
            "mejor_nivel": mejor["nivel"], "media_mejor": mejor["media"],
            "peor_nivel": peor["nivel"], "media_peor": peor["media"],
            "delta": mejor["media"] - peor["media"],
        })
    columnas = ["factor", "mejor_nivel", "media_mejor", "peor_nivel", "media_peor", "delta"]
    tabla = _tabla_con_niveles(filas, columnas, ["mejor_nivel", "peor_nivel"]).set_index("factor")
    tabla["rango"] = tabla["delta"].rank(ascending=False, method="min").astype(int)
    return tabla


def predecir_optimo(resultados, factores, respuesta="sn"):
    """Combinación óptima y respuesta predicha con el modelo aditivo.

    Para cada factor se elige el nivel con mayor media de la respuesta (la
    razón señal/ruido siempre se maximiza). La predicción es la media
    general más la suma, sobre los factores, de (media del mejor nivel −
    media general). Supone que los factores no interactúan; por eso debe
    contrastarse con una corrida de confirmación. Para predecir solo con los
    factores influyentes basta pasar esa sublista en `factores`.

    Devuelve (niveles, predicha): un diccionario {factor: nivel elegido} y la
    respuesta predicha.
    """
    media_general = float(np.nanmean(resultados[respuesta].to_numpy(dtype=float)))
    tabla = tabla_respuesta(resultados, factores, respuesta)
    niveles = {factor: tabla.loc[factor, "mejor_nivel"] for factor in factores}
    predicha = media_general + float((tabla["media_mejor"] - media_general).sum())
    return niveles, predicha
