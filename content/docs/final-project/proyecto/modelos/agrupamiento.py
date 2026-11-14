"""Capa 4 · Agrupamiento.

Responsabilidad única: explorar y ajustar K-means y DBSCAN sobre una matriz
numérica ya preparada (escalada y codificada, por ejemplo con
`tuberias.preprocesador`), medir si los datos tienen tendencia a agruparse y
describir los grupos encontrados.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from proyecto.base.config import SEMILLA

# La silueta compara todos los pares de puntos; con tablas grandes se calcula
# sobre una muestra aleatoria de este tamaño.
MUESTRA_SILUETA = 5000


def _matriz(X):
    matriz = np.asarray(X, dtype=float)
    if matriz.ndim != 2 or np.isnan(matriz).any():
        raise ValueError("Se espera una matriz numérica de dos dimensiones y sin faltantes.")
    return matriz


def _silueta(matriz, etiquetas, semilla):
    """Silueta media; NaN si hay menos de dos grupos o tantos como puntos."""
    grupos = len(np.unique(etiquetas))
    if grupos < 2 or grupos >= len(matriz):
        return float("nan")
    muestra = MUESTRA_SILUETA if len(matriz) > MUESTRA_SILUETA else None
    return float(silhouette_score(matriz, etiquetas, sample_size=muestra, random_state=semilla))


def ajustar_kmeans(X, k, semilla=SEMILLA):
    """Ajusta K-means con `k` grupos (10 inicios). Devuelve (modelo,
    etiquetas)."""
    modelo = KMeans(n_clusters=int(k), n_init=10, random_state=semilla).fit(_matriz(X))
    return modelo, modelo.labels_


def explorar_kmeans(X, ks, semilla=SEMILLA):
    """Tabla con `k`, `inercia` (suma de distancias al cuadrado al centro,
    para el método del codo) y `silueta` de K-means para cada k de `ks`."""
    matriz = _matriz(X)
    filas = []
    for k in ks:
        modelo, etiquetas = ajustar_kmeans(matriz, k, semilla)
        filas.append({"k": int(k), "inercia": float(modelo.inertia_),
                      "silueta": _silueta(matriz, etiquetas, semilla)})
    return pd.DataFrame(filas, columns=["k", "inercia", "silueta"])


def curva_k_distancia(X, k):
    """Distancia de cada punto a su k-ésimo vecino más cercano (sin contarse
    a sí mismo), ordenada de menor a mayor. El «codo» de esta curva orienta
    la elección de `eps` en DBSCAN con `min_samples` cercano a k."""
    matriz = _matriz(X)
    vecinos = NearestNeighbors(n_neighbors=int(k) + 1).fit(matriz)
    distancias, _ = vecinos.kneighbors(matriz)
    return np.sort(distancias[:, -1])


def ajustar_dbscan(X, eps, min_muestras):
    """Ajusta DBSCAN. Devuelve (modelo, etiquetas); la etiqueta −1 es ruido."""
    modelo = DBSCAN(eps=float(eps), min_samples=int(min_muestras)).fit(_matriz(X))
    return modelo, modelo.labels_


def explorar_dbscan(X, eps_lista, min_muestras_lista, semilla=SEMILLA):
    """Tabla con una fila por combinación (`eps`, `min_muestras`): número de
    `grupos` (sin contar el ruido), `proporcion_ruido` y `silueta` calculada
    solo con los puntos que no son ruido (NaN si quedan menos de dos grupos)."""
    matriz = _matriz(X)
    filas = []
    for min_muestras in min_muestras_lista:
        for eps in eps_lista:
            _, etiquetas = ajustar_dbscan(matriz, eps, min_muestras)
            con_grupo = etiquetas != -1
            filas.append({
                "eps": float(eps), "min_muestras": int(min_muestras),
                "grupos": int(len(np.unique(etiquetas[con_grupo]))),
                "proporcion_ruido": float(1 - con_grupo.mean()),
                "silueta": _silueta(matriz[con_grupo], etiquetas[con_grupo], semilla),
            })
    return pd.DataFrame(filas, columns=["eps", "min_muestras", "grupos", "proporcion_ruido", "silueta"])


def estadistico_hopkins(X, semilla=SEMILLA, m=None):
    """Estadístico de Hopkins de tendencia al agrupamiento.

    Se toman `m` puntos reales al azar (por defecto el 10 % de las filas,
    entre 10 y 500) y `m` puntos artificiales uniformes dentro del rango
    observado de cada variable. Con u = distancia de cada punto artificial al
    dato real más cercano y w = distancia de cada punto real muestreado a su
    vecino real más cercano,

        H = Σu / (Σu + Σw)

    Convención implementada: H cercano a 0.5 indica datos sin estructura
    (como puntos al azar), H hacia 1 indica grupos marcados y H por debajo de
    0.5, puntos repartidos de forma regular. Se usan las distancias sin
    elevar a la dimensión.
    """
    matriz = _matriz(X)
    n, dimension = matriz.shape
    if n < 3:
        raise ValueError("Se necesitan al menos tres filas.")
    m = int(min(max(10, 0.1 * n), 500, n - 1)) if m is None else int(min(m, n - 1))
    generador = np.random.default_rng(semilla)
    vecinos = NearestNeighbors(n_neighbors=2).fit(matriz)

    artificiales = generador.uniform(matriz.min(axis=0), matriz.max(axis=0), size=(m, dimension))
    u = vecinos.kneighbors(artificiales, n_neighbors=1)[0][:, 0]
    elegidos = generador.choice(n, size=m, replace=False)
    w = vecinos.kneighbors(matriz[elegidos], n_neighbors=2)[0][:, 1]
    total = u.sum() + w.sum()
    return float(u.sum() / total) if total > 0 else float("nan")


def perfilar_grupos(tabla, etiquetas, columnas):
    """Describe cada grupo en las variables originales (sin escalar).

    Devuelve una fila por grupo con `n`, `proporcion` y, por cada columna de
    `columnas`, la media si es numérica o el nivel más frecuente si es
    categórica. El ruido de DBSCAN aparece como el grupo −1.
    """
    etiquetas = np.asarray(etiquetas)
    if len(etiquetas) != len(tabla):
        raise ValueError("Se necesita una etiqueta por fila de la tabla.")
    filas = []
    for grupo in np.unique(etiquetas):
        miembros = tabla.loc[etiquetas == grupo]
        fila = {"grupo": grupo, "n": int(len(miembros)), "proporcion": len(miembros) / len(tabla)}
        for columna in columnas:
            valores = miembros[columna]
            if pd.api.types.is_bool_dtype(valores) or pd.api.types.is_numeric_dtype(valores):
                fila[columna] = float(valores.astype(float).mean())
            else:
                modas = valores.mode(dropna=True)
                fila[columna] = modas.iloc[0] if len(modas) else None
        filas.append(fila)
    return pd.DataFrame(filas).set_index("grupo")
