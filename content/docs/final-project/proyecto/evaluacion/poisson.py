"""Capa 5 · Pruebas y ajustes para conteos de llegadas.

Responsabilidad única: decir si unos conteos (muestras o lotes por cliente y
semana, tamaño de los lotes) se comportan como una variable de Poisson. Da el
índice de dispersión (varianza sobre media), la prueba de dispersión con su
valor p, el ajuste por máxima verosimilitud de Poisson frente a binomial
negativa con su AIC, la mezcla de Poisson con una tasa por cliente y los
ajustes del tamaño de lote (uno más un conteo) con sus frecuencias esperadas.
No lee archivos ni dibuja: recibe arreglos o tablas ya armadas.

Notación: la binomial negativa se escribe con media `mu` y tamaño `r`, de
modo que su varianza es `mu + mu**2 / r`. Con `r` grande vuelve a Poisson.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

# Límites de log(r) en la búsqueda de la binomial negativa (r de 0.01 a 1e6).
_LOG_R = (np.log(1e-2), np.log(1e6))


def _conteos(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if np.any(x < 0) or np.any(x != np.round(x)):
        raise ValueError("Los conteos deben ser enteros no negativos.")
    return x


def indice_dispersion(x):
    """Varianza muestral sobre media. Cerca de 1 en una Poisson, mayor que 1
    si hay sobredispersión. NaN si la media es cero o hay menos de dos datos."""
    x = _conteos(x)
    if len(x) < 2 or x.mean() == 0:
        return float("nan")
    return float(x.var(ddof=1) / x.mean())


def prueba_dispersion(x):
    """Prueba de dispersión: D = suma (x − media)² / media sigue una ji
    cuadrado con n − 1 grados de libertad si los conteos son Poisson con una
    misma tasa. Devuelve n, media, varianza, índice, D, grados de libertad y
    los valores p de cada cola (sobredispersión y subdispersión)."""
    x = _conteos(x)
    n, media = len(x), float(x.mean())
    estadistico = float(((x - media) ** 2).sum() / media) if media > 0 else float("nan")
    gl = n - 1
    return {"n": n, "media": media, "varianza": float(x.var(ddof=1)) if n > 1 else float("nan"),
            "indice": indice_dispersion(x), "estadistico": estadistico, "gl": gl,
            "p_sobredispersion": float(stats.chi2.sf(estadistico, gl)),
            "p_subdispersion": float(stats.chi2.cdf(estadistico, gl))}


def _logver_nbinom(x, mu, r):
    return float(stats.nbinom.logpmf(x, r, r / (r + mu)).sum())


def ajustar_conteos(x):
    """Poisson frente a binomial negativa por máxima verosimilitud.

    En las dos la media estimada es la media muestral; la binomial negativa
    agrega `r`, que se busca en escala logarítmica. Devuelve un diccionario
    con `mu`, `r`, la log-verosimilitud y el AIC de cada modelo, `mejor` (el
    de menor AIC) y la diferencia de AIC."""
    x = _conteos(x)
    mu = float(x.mean())
    logver_poisson = float(stats.poisson.logpmf(x, mu).sum())
    if mu == 0 or x.var() <= mu:
        r, logver_nb = float("inf"), logver_poisson   # sin sobredispersión: la binomial negativa es Poisson
    else:
        busqueda = optimize.minimize_scalar(lambda lr: -_logver_nbinom(x, mu, np.exp(lr)), bounds=_LOG_R,
                                            method="bounded")
        r, logver_nb = float(np.exp(busqueda.x)), -float(busqueda.fun)
    aic_poisson, aic_nb = 2 * 1 - 2 * logver_poisson, 2 * 2 - 2 * logver_nb
    return {"n": len(x), "mu": mu, "r": r,
            "logver_poisson": logver_poisson, "logver_binomial_negativa": logver_nb,
            "aic_poisson": aic_poisson, "aic_binomial_negativa": aic_nb,
            "mejor": "Poisson" if aic_poisson <= aic_nb else "binomial negativa",
            "diferencia_aic": abs(aic_poisson - aic_nb)}


def dispersion_por_fila(ancha):
    """Media, varianza, índice de dispersión y total de cada fila de una tabla
    ancha de conteos (por ejemplo, cliente × semana)."""
    valores = ancha.to_numpy(dtype=float)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)   # filas con una sola semana: varianza NaN
        media = np.nanmean(valores, axis=1)
        varianza = np.nanvar(valores, axis=1, ddof=1)
        indice = np.where(media > 0, varianza / media, np.nan)
    return pd.DataFrame({"media": media, "varianza": varianza, "indice": indice,
                         "total": np.nansum(valores, axis=1), "semanas": np.sum(~np.isnan(valores), axis=1)},
                        index=ancha.index)


def frecuencias_mezcla_poisson(ancha, maximo):
    """Frecuencias observadas de los conteos de todas las celdas de `ancha`
    (sin contar las NaN) frente a las esperadas si cada fila fuera Poisson con
    su propia media: la esperada en k es el promedio, sobre las celdas, de la
    probabilidad de Poisson de k con la media de su fila. Los conteos mayores
    que `maximo` se juntan en la última clase. Devuelve una tabla con `valor`,
    `observada` y `esperada` (proporciones)."""
    valores = ancha.to_numpy(dtype=float)
    medias = np.nanmean(valores, axis=1)
    celdas = ~np.isnan(valores)
    por_fila = celdas.sum(axis=1)
    observados = np.minimum(valores[celdas], maximo).astype(int)
    k = np.arange(maximo + 1)
    pmf = stats.poisson.pmf(k[None, :], medias[:, None])
    pmf[:, -1] = stats.poisson.sf(maximo - 1, medias)            # la última clase es "maximo o más"
    esperada = (pmf * por_fila[:, None]).sum(axis=0) / por_fila.sum()
    observada = np.bincount(observados, minlength=maximo + 1) / len(observados)
    return pd.DataFrame({"valor": k, "observada": observada, "esperada": esperada})


# ----------------------------------------------------------------------
# Tamaño de lote: 1 + un conteo
# ----------------------------------------------------------------------
MODELOS_TAMANO = ("1 + Poisson", "1 + geométrica", "1 + binomial negativa")


def ajustar_tamano_lote(tamanos):
    """Tres modelos para un tamaño de lote que vale al menos 1: el tamaño menos
    uno es Poisson (lo que hace la simulación), geométrico o binomial negativo.
    Ajuste por máxima verosimilitud. Devuelve una tabla con el modelo, sus
    parámetros, la log-verosimilitud, el AIC y la diferencia de AIC con el
    mejor, y el nombre del mejor."""
    tamanos = _conteos(tamanos)
    if np.any(tamanos < 1):
        raise ValueError("Un lote tiene al menos una muestra.")
    y = tamanos - 1
    media = float(y.mean())
    p_geo = 1 / (1 + media)
    conteos = ajustar_conteos(y)
    filas = [
        {"modelo": "1 + Poisson", "parametros": 1, "media_extra": media, "r": float("inf"),
         "logver": float(stats.poisson.logpmf(y, media).sum())},
        {"modelo": "1 + geométrica", "parametros": 1, "media_extra": media, "r": 1.0,
         "logver": float(stats.nbinom.logpmf(y, 1, p_geo).sum())},
        {"modelo": "1 + binomial negativa", "parametros": 2, "media_extra": media, "r": conteos["r"],
         "logver": conteos["logver_binomial_negativa"]},
    ]
    tabla = pd.DataFrame(filas)
    tabla["aic"] = 2 * tabla["parametros"] - 2 * tabla["logver"]
    tabla["delta_aic"] = tabla["aic"] - tabla["aic"].min()
    return tabla, str(tabla.loc[tabla["aic"].idxmin(), "modelo"])


def probabilidad_tamano(modelo, valores, media_extra, r):
    """Probabilidad de cada tamaño de `valores` (≥ 1) según un modelo de
    `ajustar_tamano_lote`."""
    y = np.asarray(valores, dtype=float) - 1
    if modelo == "1 + Poisson":
        return stats.poisson.pmf(y, media_extra)
    tamano = 1.0 if modelo == "1 + geométrica" else r
    if not np.isfinite(tamano):
        return stats.poisson.pmf(y, media_extra)
    return stats.nbinom.pmf(y, tamano, tamano / (tamano + media_extra))


def frecuencias_tamano(tamanos, tabla_ajustes, cortes):
    """Frecuencias observadas y esperadas por cada modelo en clases de tamaño.
    `cortes` son los límites inferiores de las clases (la última queda
    abierta); devuelve una fila por clase con `clase`, `observada` y una
    columna por modelo, como proporciones."""
    tamanos = np.asarray(tamanos, dtype=int)
    tope = int(max(tamanos.max(), cortes[-1])) + 1
    valores = np.arange(1, tope + 1)
    filas = []
    for i, desde in enumerate(cortes):
        hasta = cortes[i + 1] - 1 if i + 1 < len(cortes) else None
        en_clase = (valores >= desde) & ((valores <= hasta) if hasta is not None else True)
        fila = {"clase": f"{desde}" if hasta == desde else (f"{desde} a {hasta}" if hasta else f"{desde} o más"),
                "observada": float(((tamanos >= desde) & ((tamanos <= hasta) if hasta else True)).mean())}
        for _, ajuste in tabla_ajustes.iterrows():
            probabilidades = probabilidad_tamano(ajuste["modelo"], valores, ajuste["media_extra"], ajuste["r"])
            fila[ajuste["modelo"]] = (float(probabilidades[en_clase].sum()) if hasta else
                                      float(1 - probabilidades[valores < desde].sum()))
        filas.append(fila)
    return pd.DataFrame(filas)


def momentos_poisson_compuesto(tasa_lotes, tamanos):
    """Media y varianza de S = X_1 + ... + X_N con N Poisson de media
    `tasa_lotes` y tamaños X independientes como los de `tamanos`:
    E[S] = λ E[X] y Var[S] = λ E[X²]. Devuelve también el índice de dispersión
    que predice, Var[S] / E[S] = E[X²] / E[X]."""
    x = np.asarray(tamanos, dtype=float)
    ex, ex2 = float(x.mean()), float((x ** 2).mean())
    return {"media": tasa_lotes * ex, "varianza": tasa_lotes * ex2, "indice": ex2 / ex}
