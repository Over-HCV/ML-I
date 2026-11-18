"""Capa 7 · Figuras de la etapa de datos reales.

Responsabilidad única: dibujar las figuras de las llegadas reales a partir
de tablas ya calculadas: conteos semanales observados frente a una mezcla de
Poisson (lotes al lado de muestras), distribución del tamaño de lote por
tipo con los modelos ajustados y la nube media frente a varianza por cliente
con la recta de Poisson (varianza igual a la media). Cada función recibe
datos en memoria, no lee archivos y devuelve la figura de matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np

from proyecto.base.config import COLORES
from proyecto.reporte.graficos_modelado import _ejes

COLOR_MUESTRAS = COLORES["naranja"]
COLOR_LOTES = COLORES["azul"]
ESTILO_MODELOS = {"1 + Poisson": (COLORES["gris"], ":"), "1 + geométrica": (COLORES["verde"], "--"),
                  "1 + binomial negativa": (COLORES["rojo"], "-")}


def _panel_conteos(ax, tabla, color, titulo, rotulo_x):
    x = tabla["valor"].to_numpy()
    ax.bar(x, tabla["observada"], color=color, alpha=0.75, width=0.8, label="Observado")
    ax.plot(x, tabla["esperada"], color=COLORES["tinta"], marker="o", markersize=4, linewidth=1.4,
            label="Poisson con la tasa de cada cliente")
    etiquetas = [str(v) for v in x[:-1]] + [f"{x[-1]}+"]
    paso = max(1, len(x) // 10)
    ax.set_xticks(x[::paso], etiquetas[::paso])
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("Proporción de semanas")
    ax.set_title(titulo, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    _ejes(ax)


def figura_llegadas(frec_lotes, frec_muestras, indice_lotes, indice_muestras, tipo):
    """Dos paneles con las semanas de los clientes de un tipo de lote: lotes
    por semana y muestras por semana, observados (barras) frente a lo que
    daría una Poisson con la tasa de cada cliente (línea). Los índices de
    dispersión (medianas entre clientes) van en el título de cada panel."""
    fig, (ax_l, ax_m) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    _panel_conteos(ax_l, frec_lotes, COLOR_LOTES, f"Lotes por semana (dispersión {indice_lotes:.2f})",
                   "Lotes del cliente en la semana")
    _panel_conteos(ax_m, frec_muestras, COLOR_MUESTRAS, f"Muestras por semana (dispersión {indice_muestras:.2f})",
                   "Muestras del cliente en la semana")
    fig.suptitle(f"Llegadas de {tipo}: Los lotes siguen a Poisson y las muestras no", fontsize=12)
    fig.tight_layout()
    return fig


def figura_tamano_lote(por_tipo):
    """Un panel por tipo de lote con la proporción observada de cada tamaño
    (puntos) y la probabilidad de los modelos ajustados (líneas), en ejes
    logarítmicos para ver la cola. `por_tipo`: tipo → (tamaños observados,
    tabla de `poisson.ajustar_tamano_lote`, función probabilidad(modelo,
    valores, media_extra, r))."""
    fig, ejes = plt.subplots(1, len(por_tipo), figsize=(10.5, 4.0), sharey=True)
    for ax, (tipo, (tamanos, ajustes, probabilidad)) in zip(np.atleast_1d(ejes), por_tipo.items()):
        tamanos = np.asarray(tamanos, dtype=int)
        valores, cuantos = np.unique(tamanos, return_counts=True)
        ax.scatter(valores, cuantos / len(tamanos), s=14, color=COLOR_LOTES, alpha=0.8, label="Observado", zorder=3)
        rejilla = np.arange(1, tamanos.max() + 1)
        for _, ajuste in ajustes.iterrows():
            color, trazo = ESTILO_MODELOS[ajuste["modelo"]]
            p = probabilidad(ajuste["modelo"], rejilla, ajuste["media_extra"], ajuste["r"])
            visibles = p > 1e-5
            ax.plot(rejilla[visibles], p[visibles], color=color, linestyle=trazo, linewidth=1.6,
                    label=f"{ajuste['modelo']} (dAIC {ajuste['delta_aic']:.0f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("Muestras en el lote")
        ax.set_title(f"Lotes de {tipo} (media {tamanos.mean():.1f}, mediana {np.median(tamanos):.0f})", fontsize=11)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        _ejes(ax, rejilla="both")
    np.atleast_1d(ejes)[0].set_ylabel("Proporción de lotes")
    fig.suptitle("Tamaño de los lotes: Ejes logarítmicos para ver la cola", fontsize=12)
    fig.tight_layout()
    return fig


def figura_dispersion_clientes(muestras, lotes, titulo=None):
    """Media frente a varianza semanal por cliente en ejes logarítmicos: un
    punto por cliente para las muestras y otro para los lotes. Sobre la recta
    varianza = media caen los conteos de Poisson; por encima hay
    sobredispersión. `muestras` y `lotes` son salidas de
    `poisson.dispersion_por_fila`."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for tabla, color, rotulo in ((muestras, COLOR_MUESTRAS, "Muestras por semana"),
                                 (lotes, COLOR_LOTES, "Lotes por semana")):
        validos = tabla[(tabla["media"] > 0) & (tabla["varianza"] > 0)]
        ax.scatter(validos["media"], validos["varianza"], s=26, color=color, alpha=0.75, edgecolor="white",
                   linewidth=0.5, label=f"{rotulo} (mediana del índice {validos['indice'].median():.2f})")
    medias = np.concatenate([muestras["media"].to_numpy(), lotes["media"].to_numpy()])
    medias = medias[np.isfinite(medias) & (medias > 0)]
    recta = np.array([medias.min() / 1.5, medias.max() * 1.5])
    ax.plot(recta, recta, color=COLORES["tinta"], linestyle="--", linewidth=1.2, label="Varianza = media (Poisson)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Media semanal del cliente")
    ax.set_ylabel("Varianza semanal del cliente")
    ax.set_title(titulo or "Cada cliente dos veces: Lotes sobre la recta y muestras muy por encima", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _ejes(ax, rejilla="both")
    fig.tight_layout()
    return fig
