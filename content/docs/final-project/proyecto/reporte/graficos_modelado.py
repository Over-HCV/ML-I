"""Capa 7 · Figuras del modelado supervisado.

Responsabilidad única: dibujar las figuras de las etapas de modelado
(regresión y clasificación) a partir de tablas ya calculadas: errores por
bloque temporal, recta de la regresión simple, coeficientes, efectos
principales del diseño de Taguchi, comparación con el factorial completo y
residuos. Cada función recibe datos en memoria, no lee archivos y devuelve
la figura de matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proyecto.base.config import COLORES, PALETA


def _ejes(ax, rejilla="y"):
    """Estilo común: sin bordes superior ni derecho y rejilla tenue."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if rejilla:
        ax.grid(axis=rejilla, color=COLORES["gris"], alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=COLORES["tinta"], labelsize=9)
    return ax


def _texto_nivel(nivel):
    """Etiqueta legible de un nivel de factor (None → «ninguna»)."""
    if nivel is None or (isinstance(nivel, float) and np.isnan(nivel)):
        return "ninguna"
    if isinstance(nivel, (bool, np.bool_)):
        return "sí" if nivel else "no"
    return str(nivel)


def figura_por_bloque(tabla, etiqueta_metrica, titulo=None, resaltar=None):
    """Métrica de cada candidato en cada bloque temporal, una línea por
    candidato.

    `tabla` tiene una fila por candidato (índice = nombre) y una columna por
    bloque. `resaltar` es una lista de candidatos que se dibujan con trazo
    grueso; el resto queda tenue para no competir con ellos.
    """
    resaltar = set(resaltar or [])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bloques = np.arange(1, tabla.shape[1] + 1)
    for i, (nombre, fila) in enumerate(tabla.iterrows()):
        destacado = not resaltar or nombre in resaltar
        ax.plot(bloques, fila.to_numpy(dtype=float), marker="o", color=PALETA[i % len(PALETA)],
                linewidth=2.2 if destacado else 1.2, alpha=1.0 if destacado else 0.55,
                markersize=6 if destacado else 4, label=str(nombre))
    ax.set_xticks(bloques)
    ax.set_xticklabels([f"Bloque {b}" for b in bloques])
    ax.set_ylabel(etiqueta_metrica)
    if titulo:
        ax.set_title(titulo, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    _ejes(ax)
    fig.tight_layout()
    return fig


def figura_recta_simple(x, y, pendiente, intercepto, nombre_x, nombre_y):
    """Nube de puntos con la recta de la regresión simple, dibujada solo en
    el rango observado de `x` (fuera de él la recta no tiene respaldo)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(x, y, s=12, alpha=0.35, color=COLORES["azul"], edgecolor="none", label="días de entrenamiento")
    malla = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    ax.plot(malla, intercepto + pendiente * malla, color=COLORES["rojo"], linewidth=2.2,
            label=f"ŷ = {intercepto:.2f} + {pendiente:.3f} x")
    ax.set_xlabel(nombre_x)
    ax.set_ylabel(nombre_y)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _ejes(ax, "both")
    fig.tight_layout()
    return fig


def figura_coeficientes(coeficientes, titulo="Coeficientes estandarizados", maximo=12):
    """Barras horizontales de los coeficientes de mayor valor absoluto
    (`coeficientes` es una Serie nombre → valor); azul si suman, rojo si
    restan."""
    serie = pd.Series(coeficientes, dtype=float)
    serie = serie.reindex(serie.abs().sort_values(ascending=False).index).head(maximo)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(serie) + 1.4))
    colores = [COLORES["azul"] if v >= 0 else COLORES["rojo"] for v in serie]
    ax.barh(serie.index.astype(str), serie.to_numpy(), color=colores, height=0.6)
    ax.axvline(0, color=COLORES["tinta"], linewidth=0.8)
    ax.set_xlabel("Cambio en el objetivo por una desviación estándar de la variable")
    ax.set_title(titulo, fontsize=11, loc="left")
    _ejes(ax, "x")
    fig.tight_layout()
    return fig


def figura_efectos_principales(efectos, media_general, etiqueta="S/N (dB)", optimo=None):
    """Gráfico de efectos principales de un diseño de Taguchi: un panel por
    factor con la media de la respuesta en cada nivel y la media general como
    línea de referencia. `efectos` es la tabla de
    `cribado.corridas.efectos_principales`; `optimo` ({factor: nivel}) marca
    el nivel elegido de cada factor."""
    factores = list(dict.fromkeys(efectos["factor"]))
    fig, ejes = plt.subplots(1, len(factores), figsize=(2.1 * len(factores) + 1, 3.6), sharey=True)
    ejes = np.atleast_1d(ejes)
    for ax, factor in zip(ejes, factores):
        propios = efectos[efectos["factor"] == factor]
        posiciones = np.arange(len(propios))
        ax.plot(posiciones, propios["media"].to_numpy(dtype=float), color=COLORES["azul"], marker="o",
                linewidth=1.8)
        if optimo is not None and factor in optimo:
            elegido = [i for i, nivel in enumerate(propios["nivel"])
                       if _texto_nivel(nivel) == _texto_nivel(optimo[factor])]
            ax.scatter(posiciones[elegido], propios["media"].to_numpy(dtype=float)[elegido], s=90,
                       facecolor="none", edgecolor=COLORES["rojo"], linewidth=1.8, zorder=3)
        ax.axhline(media_general, color=COLORES["gris"], linestyle="--", linewidth=0.9)
        ax.set_xticks(posiciones)
        ax.set_xticklabels([_texto_nivel(n) for n in propios["nivel"]], rotation=30, ha="right", fontsize=8)
        ax.set_xlim(-0.5, len(propios) - 0.5)
        ax.set_title(factor, fontsize=10)
        _ejes(ax)
    ejes[0].set_ylabel(etiqueta)
    fig.tight_layout()
    return fig


def figura_deltas(deltas, titulo=None):
    """Delta de cada factor según dos o más diseños, en barras agrupadas.
    `deltas` es una tabla con un factor por fila y una columna por diseño."""
    tabla = pd.DataFrame(deltas, dtype=float)
    tabla = tabla.loc[tabla.iloc[:, 0].sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(tabla) + 1.4))
    alto = 0.8 / tabla.shape[1]
    posiciones = np.arange(len(tabla))
    for i, columna in enumerate(tabla.columns):
        ax.barh(posiciones + (i - (tabla.shape[1] - 1) / 2) * alto, tabla[columna].to_numpy(), height=alto,
                color=PALETA[i % len(PALETA)], label=str(columna))
    ax.set_yticks(posiciones)
    ax.set_yticklabels(tabla.index.astype(str))
    ax.set_xlabel("Delta de S/N entre el mejor y el peor nivel (dB)")
    if titulo:
        ax.set_title(titulo, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _ejes(ax, "x")
    fig.tight_layout()
    return fig


def figura_residuos(fechas, y, pred, unidad="objetivo"):
    """Dos paneles: el residuo (observado − predicho) a lo largo del tiempo y
    el residuo frente al valor predicho. Una media móvil ayuda a ver si el
    error tiene tendencia o estacionalidad sin modelar."""
    fechas = pd.to_datetime(pd.Series(np.asarray(fechas)))
    residuo = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    izq.plot(fechas, residuo, color=COLORES["azul"], linewidth=0.8, alpha=0.6, marker="o", markersize=2.5)
    suavizado = pd.Series(residuo).rolling(10, min_periods=3, center=True).mean()
    izq.plot(fechas, suavizado, color=COLORES["rojo"], linewidth=2, label="media móvil de 10 días")
    izq.axhline(0, color=COLORES["tinta"], linewidth=0.8)
    izq.set_ylabel(f"Residuo ({unidad})")
    izq.legend(frameon=False, fontsize=8, loc="upper left")
    izq.tick_params(axis="x", rotation=30)
    _ejes(izq)
    der.scatter(pred, residuo, s=14, alpha=0.5, color=COLORES["azul"], edgecolor="none")
    der.axhline(0, color=COLORES["tinta"], linewidth=0.8)
    der.set_xlabel(f"Valor predicho ({unidad})")
    der.set_ylabel("Residuo")
    _ejes(der, "both")
    fig.tight_layout()
    return fig


def figura_serie_prediccion(fechas, y, predicciones, unidad="objetivo"):
    """Serie observada y una o más predicciones (`predicciones` es un
    diccionario nombre → arreglo) en el mismo periodo."""
    fechas = pd.to_datetime(pd.Series(np.asarray(fechas)))
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.plot(fechas, np.asarray(y, dtype=float), color=COLORES["tinta"], linewidth=1.2, label="observado")
    for i, (nombre, pred) in enumerate(predicciones.items()):
        ax.plot(fechas, np.asarray(pred, dtype=float), color=PALETA[i % len(PALETA)], linewidth=1.5,
                alpha=0.9, label=str(nombre), linestyle="-" if i == 0 else "--")
    ax.set_ylabel(unidad)
    ax.legend(frameon=False, fontsize=8, ncol=len(predicciones) + 1, loc="upper left")
    ax.tick_params(axis="x", rotation=30)
    _ejes(ax)
    fig.tight_layout()
    return fig
