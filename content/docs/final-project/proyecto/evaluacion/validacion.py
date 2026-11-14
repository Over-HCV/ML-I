"""Capa 5 · Validación ordenada en el tiempo.

Responsabilidad única: partir los datos de desarrollo respetando el orden
temporal (nunca se entrena con el futuro para predecir el pasado) y puntuar
estimadores sobre esos bloques, a mano o con `GridSearchCV`.

Los bloques son de ventana creciente: cada uno entrena con todo lo anterior a
una fecha de corte y valida con el tramo siguiente. Los cortes caen siempre
entre fechas distintas, de modo que un mismo día nunca queda repartido entre
entrenamiento y validación.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from proyecto.base.config import BLOQUES_VALIDACION_CRUZADA, CORTE_PRUEBA, CORTE_VALIDACION
from proyecto.evaluacion.metricas import prediccion_de, resolver_metrica


def bloques_temporales(fechas, n_bloques=BLOQUES_VALIDACION_CRUZADA, minimo=0.4):
    """Bloques de validación cruzada de ventana creciente.

    `fechas` trae una fecha por fila, en cualquier orden. El primer bloque
    entrena con (al menos) la fracción `minimo` más antigua de las filas; el
    resto del periodo se reparte en `n_bloques` tramos de validación con un
    número parecido de filas. El bloque i entrena con todas las filas
    anteriores a su corte y valida con las del tramo i.

    Devuelve una lista de pares (índices de entrenamiento, índices de
    validación) **posicionales**, lista para usarse como `cv=` en
    scikit-learn.
    """
    dias = pd.DatetimeIndex(pd.to_datetime(np.asarray(fechas))).normalize()
    if len(dias) == 0 or dias.isna().any():
        raise ValueError("Las fechas no pueden estar vacías ni tener valores faltantes.")
    if not 0 < minimo < 1 or n_bloques < 1:
        raise ValueError("Se requiere 0 < minimo < 1 y n_bloques >= 1.")

    unicas, codigo, conteos = np.unique(dias.values, return_inverse=True, return_counts=True)
    codigo = np.asarray(codigo).ravel()
    # Fracción de filas estrictamente anteriores a cada fecha distinta.
    fraccion_previa = (np.cumsum(conteos) - conteos) / len(dias)
    objetivos = minimo + (1 - minimo) * np.arange(n_bloques) / n_bloques
    cortes = [int(c) for c in np.searchsorted(fraccion_previa, objetivos, side="left")]
    cortes[0] = max(cortes[0], 1)
    for i in range(1, n_bloques):
        cortes[i] = max(cortes[i], cortes[i - 1] + 1)
    cortes.append(len(unicas))
    if cortes[-2] >= cortes[-1]:
        raise ValueError(
            f"No hay suficientes fechas distintas ({len(unicas)}) para {n_bloques} bloques."
        )

    bloques = []
    for inicio, fin in zip(cortes[:-1], cortes[1:]):
        entrenar = np.flatnonzero(codigo < inicio)
        validar = np.flatnonzero((codigo >= inicio) & (codigo < fin))
        bloques.append((entrenar, validar))
    return bloques


def separar_desarrollo(tabla, columna_fecha="fecha"):
    """Separa una tabla de desarrollo en entrenamiento (fecha anterior a
    `CORTE_VALIDACION`) y validación (desde ese corte y antes de
    `CORTE_PRUEBA`). Devuelve (entrenamiento, validacion) con índice nuevo."""
    fechas = pd.to_datetime(tabla[columna_fecha])
    corte_validacion, corte_prueba = pd.Timestamp(CORTE_VALIDACION), pd.Timestamp(CORTE_PRUEBA)
    entrenamiento = tabla.loc[fechas < corte_validacion].reset_index(drop=True)
    validacion = tabla.loc[(fechas >= corte_validacion) & (fechas < corte_prueba)]
    return entrenamiento, validacion.reset_index(drop=True)


def _filas(X, indices):
    return X.iloc[indices] if hasattr(X, "iloc") else np.asarray(X)[indices]


def puntuar_por_bloques(estimador, X, y, bloques, metrica):
    """Ajusta una copia del estimador en el entrenamiento de cada bloque y
    devuelve la métrica obtenida en su tramo de validación (un valor por
    bloque). `metrica` es un nombre de `METRICAS` o un par (función, sentido)."""
    funcion, _ = resolver_metrica(metrica, sentido="mayor" if callable(metrica) else None)
    y = np.asarray(y)
    puntajes = []
    for entrenar, validar in bloques:
        modelo = clone(estimador).fit(_filas(X, entrenar), y[entrenar])
        puntajes.append(funcion(y[validar], prediccion_de(modelo, _filas(X, validar))))
    return np.array(puntajes, dtype=float)


class _Puntuador:
    """Adaptador de una métrica del proyecto a la firma de scikit-learn
    (estimador, X, y) → puntaje, donde mayor siempre es mejor."""

    def __init__(self, funcion, signo):
        self.funcion = funcion
        self.signo = signo

    def __call__(self, estimador, X, y):
        return self.signo * self.funcion(np.asarray(y), prediccion_de(estimador, X))


def buscar_en_rejilla(estimador, rejilla, X, y, bloques, metrica, n_trabajos=None):
    """Búsqueda en rejilla sobre los bloques temporales.

    Envoltura delgada de `GridSearchCV`: prueba cada combinación de `rejilla`
    en los bloques, se queda con la de mejor métrica media y reajusta el
    estimador con todas las filas recibidas. Devuelve (búsqueda ajustada,
    tabla), donde la tabla tiene una fila por combinación, ordenada de mejor
    a peor, con la métrica en su escala natural (sin el cambio de signo que
    usa scikit-learn): `media`, `desv`, `bloque_i`, `segundos_ajuste`,
    `rango`.
    """
    funcion, sentido = resolver_metrica(metrica)
    signo = 1.0 if sentido == "mayor" else -1.0
    bloques = list(bloques)
    busqueda = GridSearchCV(
        estimador, rejilla, scoring=_Puntuador(funcion, signo), cv=bloques,
        n_jobs=n_trabajos, refit=True, error_score=np.nan,
    )
    busqueda.fit(X, np.asarray(y))

    resultados = busqueda.cv_results_
    cortos = [nombre.split("__")[-1] for nombre in rejilla]
    usar_cortos = len(set(cortos)) == len(cortos)
    tabla = pd.DataFrame(index=range(len(resultados["params"])))
    for nombre, corto in zip(rejilla, cortos):
        tabla[corto if usar_cortos else nombre] = pd.Series(
            [combinacion[nombre] for combinacion in resultados["params"]], dtype=object
        )
    tabla["media"] = signo * resultados["mean_test_score"]
    tabla["desv"] = resultados["std_test_score"]
    for i in range(len(bloques)):
        tabla[f"bloque_{i + 1}"] = signo * resultados[f"split{i}_test_score"]
    tabla["segundos_ajuste"] = resultados["mean_fit_time"]
    tabla["rango"] = resultados["rank_test_score"]
    tabla = tabla.sort_values("rango", kind="mergesort").reset_index(drop=True)
    return busqueda, tabla
