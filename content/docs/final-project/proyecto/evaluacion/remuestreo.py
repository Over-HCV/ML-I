"""Capa 5 · Intervalos por remuestreo.

Responsabilidad única: cuantificar con bootstrap la incertidumbre de
cualquier cantidad calculada sobre predicciones ya congeladas (no reajusta
ningún modelo).
"""
import numpy as np

from proyecto.base.config import SEMILLA


def intervalo_bootstrap(funcion, *arreglos, n=1000, nivel=0.95, semilla=SEMILLA):
    """Intervalo bootstrap percentil de `funcion(*arreglos)`.

    Los arreglos (por ejemplo `y` y la predicción) tienen la misma longitud y
    se remuestrean **juntos**, por filas y con reemplazo, `n` veces. En cada
    réplica se recalcula la cantidad; el intervalo son los percentiles
    (1 − nivel)/2 y 1 − (1 − nivel)/2 de las réplicas. Las réplicas en las
    que la cantidad no está definida (NaN, por ejemplo un AUC con una sola
    clase) se descartan.

    Devuelve (estimado, inferior, superior), donde `estimado` es la cantidad
    en los datos originales. Es determinista para una misma `semilla`.
    """
    if not arreglos:
        raise ValueError("Se necesita al menos un arreglo.")
    arreglos = [np.asarray(a) for a in arreglos]
    tamano = len(arreglos[0])
    if tamano == 0 or any(len(a) != tamano for a in arreglos):
        raise ValueError("Los arreglos deben tener la misma longitud y no estar vacíos.")

    estimado = float(funcion(*arreglos))
    generador = np.random.default_rng(semilla)
    replicas = np.full(n, np.nan)
    for i in range(n):
        indices = generador.integers(0, tamano, tamano)
        try:
            replicas[i] = funcion(*(a[indices] for a in arreglos))
        except ValueError:  # réplica degenerada: se descarta
            pass
    if np.isnan(replicas).all():
        return estimado, float("nan"), float("nan")
    cola = (1 - nivel) / 2
    inferior, superior = np.nanquantile(replicas, [cola, 1 - cola])
    return estimado, float(inferior), float(superior)
