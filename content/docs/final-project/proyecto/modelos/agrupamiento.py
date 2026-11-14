"""Capa 4 · Agrupamiento.

Responsabilidad única: explorar y ajustar K-means y DBSCAN sobre una matriz
numérica ya preparada (escalada y codificada, por ejemplo con
`tuberias.preprocesador`), medir si los datos tienen tendencia a agruparse y
describir los grupos encontrados; además, las reglas de elección de la etapa
de agrupamiento (log1p de conteos, número de grupos, parámetros de DBSCAN,
estabilidad, perfiles estandarizados y acuerdo con una partición de
referencia), aplicadas de punta a punta por `analizar_agrupamiento`.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from proyecto.base.config import SEMILLA
from proyecto.modelos.tuberias import preprocesador

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


def pasos_kmeans(X, k, semilla=SEMILLA, maximo_pasos=15):
    """K-means iteración por iteración (algoritmo de Lloyd con
    `KMeans(init=centros, n_init=1, max_iter=1)`), partiendo de `k` filas al
    azar. Devuelve (centros por paso: pasos × k × columnas, etiquetas por paso:
    pasos × filas); se detiene cuando ninguna fila cambia de grupo. Sirve
    para mostrar cómo converge, no para elegir grupos."""
    X = _matriz(X)
    centros = X[np.random.default_rng(semilla).choice(len(X), int(k), replace=False)]
    historia_centros = [centros]
    historia_etiquetas = [np.argmin(((X[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2), axis=1)]
    for _ in range(maximo_pasos):
        modelo = KMeans(n_clusters=int(k), init=centros, n_init=1, max_iter=1).fit(X)
        centros = modelo.cluster_centers_
        historia_centros.append(centros)
        historia_etiquetas.append(modelo.predict(X))
        if np.array_equal(historia_etiquetas[-1], historia_etiquetas[-2]):
            break
    return np.array(historia_centros), np.array(historia_etiquetas)


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


# ---------------------------------------------------------------------------
# Preparación y reglas de elección de la etapa de agrupamiento
# ---------------------------------------------------------------------------

def aplicar_log_a_conteos(tabla, numericas, asimetria=1.0):
    """Aplica log1p a las variables de conteo con cola larga.

    Una columna de `numericas` recibe log1p si no tiene valores negativos, su
    máximo es mayor que 1 y su asimetría supera `asimetria`. Devuelve
    (copia de la tabla con las columnas transformadas, lista de columnas
    transformadas)."""
    transformada = tabla.copy()
    con_log = []
    for columna in numericas:
        valores = transformada[columna].dropna()
        if len(valores) and valores.min() >= 0 and valores.max() > 1 and valores.skew() > asimetria:
            transformada[columna] = np.log1p(transformada[columna])
            con_log.append(columna)
    return transformada, con_log


def preparar_matrices(tabla, numericas, categoricas, asimetria=1.0):
    """Matrices para agrupar a partir de una tabla de modelado.

    Devuelve (matriz escalada tras la regla de log1p, matriz sin escalar ni
    transformar, columnas que recibieron log1p). Ambas matrices pasan por
    `tuberias.preprocesador` (imputación y codificación)."""
    columnas = list(numericas) + list(categoricas)
    transformada, con_log = aplicar_log_a_conteos(tabla[columnas], numericas, asimetria)
    matriz = preprocesador(numericas, categoricas, escalado="estandar").fit_transform(transformada)
    sin_escalar = preprocesador(numericas, categoricas, escalado="ninguno").fit_transform(tabla[columnas])
    return np.asarray(matriz, dtype=float), np.asarray(sin_escalar, dtype=float), con_log


def reparto_de_varianza(tabla, numericas):
    """Proporción de la varianza total (sin escalar) que aporta cada
    variable numérica, de mayor a menor."""
    varianza = tabla[numericas].var()
    return (varianza / varianza.sum()).sort_values(ascending=False)


def elegir_k(tabla_k, tolerancia):
    """Regla del número de grupos: el k de mayor silueta.

    Devuelve (k, k alternativo), donde el alternativo es el k ≥ 3 de mayor
    silueta a menos de `tolerancia` del ganador cuando este es k = 2, y None
    en cualquier otro caso (se discute, pero no cambia la regla)."""
    mejor = tabla_k.loc[tabla_k["silueta"].idxmax()]
    k = int(mejor["k"])
    alternativo = None
    if k == 2:
        cercanos = tabla_k[(tabla_k["k"] >= 3) & (mejor["silueta"] - tabla_k["silueta"] <= tolerancia)]
        if len(cercanos):
            alternativo = int(cercanos.loc[cercanos["silueta"].idxmax(), "k"])
    return k, alternativo


def estabilidad_kmeans(X, k, referencia, semillas):
    """Índice de Rand ajustado entre la partición de `referencia` y las de
    K-means con el mismo k y cada semilla de `semillas` (1 = la misma
    partición)."""
    matriz = _matriz(X)
    return np.array([adjusted_rand_score(referencia, ajustar_kmeans(matriz, k, s)[1]) for s in semillas])


def explorar_dbscan_por_cuantiles(X, cuantiles):
    """Rejilla de DBSCAN con min_samples ∈ {max(3, d), 2d} (d = columnas) y,
    para cada uno, eps en los `cuantiles` de su curva de k-distancias.

    Devuelve (tabla de `explorar_dbscan`, curvas por min_samples, lista de
    eps por min_samples)."""
    matriz = _matriz(X)
    dimension = matriz.shape[1]
    tablas, curvas, rejillas = [], {}, {}
    for min_muestras in sorted({max(3, dimension), 2 * dimension}):
        curva = curva_k_distancia(matriz, min_muestras)
        eps_lista = sorted(set(np.round(np.quantile(curva, cuantiles), 2)))
        curvas[min_muestras], rejillas[min_muestras] = curva, eps_lista
        tablas.append(explorar_dbscan(matriz, eps_lista, [min_muestras]))
    return pd.concat(tablas, ignore_index=True), curvas, rejillas


def elegir_dbscan(tabla, ruido_maximo, ruido_respaldo):
    """Regla de DBSCAN sobre la salida de `explorar_dbscan`.

    Primero, la mayor silueta entre las soluciones con dos o más grupos y
    ruido ≤ `ruido_maximo` (criterio «regla»). Si ninguna califica, respaldo:
    DBSCAN como detector de atípicos, con el menor eps que deja al menos un
    grupo y ruido ≤ `ruido_respaldo` (criterio «respaldo»). Devuelve (fila
    elegida, criterio) o (None, None)."""
    validas = tabla[(tabla["grupos"] >= 2) & (tabla["proporcion_ruido"] <= ruido_maximo)]
    if len(validas):
        return validas.loc[validas["silueta"].idxmax()], "regla"
    respaldo = tabla[(tabla["grupos"] >= 1) & (tabla["proporcion_ruido"] <= ruido_respaldo)]
    if len(respaldo):
        return respaldo.sort_values(["eps", "min_muestras"]).iloc[0], "respaldo"
    return None, None


def perfil_estandarizado(tabla, etiquetas, columnas):
    """Perfil de cada grupo (`perfilar_grupos`) y sus medias numéricas en
    desviaciones estándar respecto a la media general. Devuelve (perfil,
    perfil estandarizado)."""
    perfil = perfilar_grupos(tabla, etiquetas, columnas)
    numericas = [c for c in columnas if pd.api.types.is_numeric_dtype(tabla[c])]
    desviacion = tabla[numericas].std().replace(0, np.nan)
    return perfil, (perfil[numericas] - tabla[numericas].mean()) / desviacion


def desviaciones_de_filas(tabla, mascara, numericas):
    """Valores de las filas marcadas por `mascara` en desviaciones estándar
    respecto a la media y la desviación de toda la tabla."""
    return (tabla.loc[mascara, numericas] - tabla[numericas].mean()) / tabla[numericas].std().replace(0, np.nan)


def acuerdo_con_referencia(referencia, etiquetas):
    """Compara una partición con una de referencia (por ejemplo, un segmento
    conocido). Devuelve (tabla de contingencia referencia × grupo, índice de
    Rand ajustado, pureza = proporción de casos que caen en la categoría de
    referencia mayoritaria de su grupo)."""
    contingencia = pd.crosstab(pd.Series(referencia, name="referencia"), pd.Series(etiquetas, name="grupo"))
    pureza = float(contingencia.max(axis=0).sum() / contingencia.to_numpy().sum())
    return contingencia, float(adjusted_rand_score(referencia, etiquetas)), pureza


def analizar_agrupamiento(tabla, numericas, categoricas, *, valores_k, tolerancia, semillas, cuantiles_eps,
                          ruido_maximo, ruido_respaldo, asimetria, semilla=SEMILLA):
    """Aplica de punta a punta las reglas de la etapa de agrupamiento.

    Prepara las matrices (`preparar_matrices`), mide la tendencia a agruparse
    (Hopkins), explora y elige K-means (`elegir_k`), mide su estabilidad con
    `semillas` y su acuerdo con K-means sin escalar, explora y elige DBSCAN
    (`elegir_dbscan`), perfila los grupos de K-means, proyecta en dos
    componentes principales y compara K-means con DBSCAN. Devuelve un
    diccionario con todos los resultados (las claves son las de
    `resumen_para_artefacto`, más las matrices y etiquetas)."""
    matriz, sin_escalar, con_log = preparar_matrices(tabla, numericas, categoricas, asimetria)
    reparto = reparto_de_varianza(tabla, numericas)
    tabla_k = explorar_kmeans(matriz, valores_k, semilla)
    k, alternativo = elegir_k(tabla_k, tolerancia)
    _, etiquetas_km = ajustar_kmeans(matriz, k, semilla)
    _, etiquetas_crudo = ajustar_kmeans(sin_escalar, k, semilla)
    tabla_db, curvas, rejillas = explorar_dbscan_por_cuantiles(matriz, cuantiles_eps)
    fila_db, criterio = elegir_dbscan(tabla_db, ruido_maximo, ruido_respaldo)
    etiquetas_db = None if fila_db is None else ajustar_dbscan(matriz, fila_db["eps"], fila_db["min_muestras"])[1]
    _, perfil_z = perfil_estandarizado(tabla, etiquetas_km, numericas)
    pca = PCA(n_components=2, random_state=semilla).fit(matriz)
    ari_km_db = ari_km_db_sin_ruido = None
    if etiquetas_db is not None:
        con_grupo = etiquetas_db != -1
        ari_km_db = float(adjusted_rand_score(etiquetas_km, etiquetas_db))
        ari_km_db_sin_ruido = float(adjusted_rand_score(etiquetas_km[con_grupo], etiquetas_db[con_grupo]))
    return {
        "tabla": tabla, "numericas": list(numericas), "categoricas": list(categoricas), "con_log": con_log,
        "matriz": matriz, "sin_escalar": sin_escalar, "reparto": reparto, "dominante": str(reparto.index[0]),
        "hopkins": estadistico_hopkins(matriz, semilla),
        "tabla_k": tabla_k, "k": k, "k_alternativo": alternativo,
        "silueta_k": float(tabla_k.set_index("k").loc[k, "silueta"]), "etiquetas_km": etiquetas_km,
        "estabilidad": estabilidad_kmeans(matriz, k, etiquetas_km, semillas),
        "ari_sin_escalar": float(adjusted_rand_score(etiquetas_km, etiquetas_crudo)),
        "tabla_db": tabla_db, "curvas": curvas, "rejillas": rejillas, "fila_db": fila_db, "criterio_db": criterio,
        "etiquetas_db": etiquetas_db, "perfil_z": perfil_z,
        "pca": pca.transform(matriz), "pca_varianza": pca.explained_variance_ratio_,
        "ari_km_db": ari_km_db, "ari_km_db_sin_ruido": ari_km_db_sin_ruido,
    }


def resumen_para_artefacto(analisis):
    """Resumen serializable (JSON) de un `analizar_agrupamiento` completado
    con `nombre`, `perfil`, `nombres` y `descripciones` de los grupos."""
    fila = analisis["fila_db"]
    perfil = analisis["perfil"]
    return {
        "analisis": analisis["nombre"],
        "filas": int(len(analisis["tabla"])),
        "variables": analisis["numericas"] + analisis["categoricas"],
        "con_log1p": analisis["con_log"],
        "variable_dominante_sin_escalar": analisis["dominante"],
        "proporcion_varianza_dominante": float(analisis["reparto"].iloc[0]),
        "hopkins": analisis["hopkins"],
        "kmeans": {"exploracion": analisis["tabla_k"].to_dict("records"), "k": analisis["k"],
                   "silueta": analisis["silueta_k"], "k_alternativo": analisis["k_alternativo"],
                   "ari_estabilidad": analisis["estabilidad"].tolist(), "ari_sin_escalar": analisis["ari_sin_escalar"]},
        "dbscan": {"rejilla": analisis["tabla_db"].to_dict("records"), "criterio": analisis["criterio_db"],
                   "eps": None if fila is None else float(fila["eps"]),
                   "min_samples": None if fila is None else int(fila["min_muestras"]),
                   "grupos": None if fila is None else int(fila["grupos"]),
                   "proporcion_ruido": None if fila is None else float(fila["proporcion_ruido"]),
                   "silueta": None if fila is None or pd.isna(fila["silueta"]) else float(fila["silueta"])},
        "ari_kmeans_dbscan": analisis["ari_km_db"],
        "ari_kmeans_dbscan_sin_ruido": analisis["ari_km_db_sin_ruido"],
        "pca_varianza": analisis["pca_varianza"].tolist(),
        "segmentos": [{"grupo": int(g), "nombre": analisis["nombres"][g], "n": int(perfil.loc[g, "n"]),
                       "proporcion": float(perfil.loc[g, "proporcion"]),
                       "descripcion": analisis["descripciones"][g]} for g in perfil.index],
    }
