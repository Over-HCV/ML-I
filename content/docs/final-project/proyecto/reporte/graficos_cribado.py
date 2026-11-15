"""Capa 7 · Figuras del cribado de análisis.

Responsabilidad única: dibujar las figuras de la etapa de cribado a partir
de la tabla que devuelve `cribado.analisis.cribar`. Cada función recibe esa
tabla (y los umbrales de la regla, leídos por quien llama), no lee archivos
y devuelve la figura de matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from proyecto.base.config import COLORES

SUPERVISADOS = ("regresion", "clasificacion")
TITULOS_TIPO = {"regresion": "Regresión", "clasificacion": "Clasificación", "agrupamiento": "Agrupamiento"}
ETIQUETA_HABILIDAD = {"regresion": "Habilidad = 1 − RMSE / RMSE de referencia",
                      "clasificacion": "Habilidad = 2 AUC − 1"}


def _ejes(ax, rejilla="x"):
    """Estilo común: sin bordes superior ni derecho y rejilla tenue."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if rejilla:
        ax.grid(axis=rejilla, color=COLORES["gris"], alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=COLORES["tinta"], labelsize=9)
    return ax


def _aviso(mensaje):
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.axis("off")
    ax.text(0.5, 0.5, mensaje, ha="center", va="center", fontsize=10, color=COLORES["tinta"], wrap=True)
    return fig


def _cribados(tabla, tipos):
    return tabla[(tabla["estado"] == "cribado") & tabla["tipo"].isin(tipos)]


def habilidades_por_bloque(fila):
    """Habilidad de cada bloque temporal del mejor modelo de la prueba rápida
    (la lista que corresponde a `mejor_modelo`)."""
    columna = "habilidades_arbol" if fila["mejor_modelo"] == "arbol" else "habilidades_lineal"
    valores = fila[columna]
    if valores is None or (np.isscalar(valores) and pd.isna(valores)):
        return np.array([])
    return np.asarray(valores, dtype=float)


def _color_estado(fila, nota_regla):
    if bool(fila["elegible"]):
        return COLORES["verde"]
    return COLORES["naranja"] if fila["observacion"] == nota_regla else COLORES["gris"]


def figura_habilidad(tabla_cribado, nota_regla="basta una regla simple"):
    """Habilidad de cada candidato supervisado: un punto por bloque temporal,
    la media como barra y el rango entre bloques como línea. El color marca
    si el candidato es elegible, si lo descarta la regla simple o si lo
    descarta otro criterio."""
    datos = _cribados(tabla_cribado, SUPERVISADOS)
    if not len(datos):
        return _aviso("No hay candidatos supervisados cribados.")
    tipos = [t for t in SUPERVISADOS if (datos["tipo"] == t).any()]
    alturas = [max(int((datos["tipo"] == t).sum()), 1) for t in tipos]
    fig, ejes = plt.subplots(len(tipos), 1, figsize=(9.5, 1.2 + 0.55 * sum(alturas)),
                             gridspec_kw={"height_ratios": alturas}, squeeze=False)
    for ax, tipo in zip(ejes[:, 0], tipos):
        grupo = datos[datos["tipo"] == tipo].sort_values("habilidad_media")
        for posicion, (_, fila) in enumerate(grupo.iterrows()):
            color = _color_estado(fila, nota_regla)
            bloques = habilidades_por_bloque(fila)
            ax.barh(posicion, fila["habilidad_media"], color=color, alpha=0.35, height=0.6)
            if len(bloques):
                ax.hlines(posicion, bloques.min(), bloques.max(), color=color, linewidth=1.6)
                ax.scatter(bloques, np.full(len(bloques), posicion), color=color, s=18, zorder=3,
                           edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(grupo)))
        ax.set_yticklabels(grupo["analisis"], fontsize=9)
        ax.axvline(0, color=COLORES["tinta"], linewidth=0.8)
        ax.set_xlabel(ETIQUETA_HABILIDAD[tipo], fontsize=9)
        ax.set_title(TITULOS_TIPO[tipo], loc="left", fontsize=11)
        _ejes(ax)
    rotulos = {COLORES["verde"]: "Elegible", COLORES["naranja"]: "Descartado: basta una regla simple",
               COLORES["gris"]: "Descartado por otro criterio"}
    presentes = {_color_estado(fila, nota_regla) for _, fila in datos.iterrows()}
    leyenda = [Line2D([0], [0], color=color, marker="o", label=rotulo)
               for color, rotulo in rotulos.items() if color in presentes]
    fig.legend(handles=leyenda, loc="lower center", ncol=len(leyenda), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def figura_regla_simple(tabla_cribado, umbral):
    """Proporción de la habilidad que logra un árbol con una sola variable,
    frente al umbral de la regla. Cada barra lleva el nombre de esa variable."""
    datos = _cribados(tabla_cribado, SUPERVISADOS).dropna(subset=["proporcion_regla"])
    if not len(datos):
        return _aviso("No hay proporciones de regla simple que mostrar.")
    datos = datos.sort_values("proporcion_regla")
    fig, ax = plt.subplots(figsize=(9.5, 0.9 + 0.45 * len(datos)))
    colores = [COLORES["naranja"] if p >= umbral else COLORES["azul"] for p in datos["proporcion_regla"]]
    posiciones = np.arange(len(datos))
    ax.barh(posiciones, datos["proporcion_regla"], color=colores, height=0.6)
    for posicion, (_, fila) in zip(posiciones, datos.iterrows()):
        ax.text(max(fila["proporcion_regla"], 0) + 0.01, posicion, f"{fila['proporcion_regla']:.2f}, {fila['variable_regla']}",
                va="center", fontsize=8, color=COLORES["tinta"], zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    ax.axvline(umbral, color=COLORES["rojo"], linestyle="--", linewidth=1)
    ax.text(umbral, len(datos) - 0.4, f" umbral {umbral:.2f}", color=COLORES["rojo"], fontsize=8, va="bottom")
    ax.set_yticks(posiciones)
    ax.set_yticklabels(datos["analisis"], fontsize=9)
    ax.set_xlim(min(0, datos["proporcion_regla"].min()), max(1.0, datos["proporcion_regla"].max()) * 1.35)
    ax.set_xlabel("Habilidad con una sola variable / habilidad del mejor modelo rápido", fontsize=9)
    ax.set_title("¿Basta una regla simple?", loc="left", fontsize=11)
    _ejes(ax)
    fig.tight_layout()
    return fig


def figura_agrupamiento(tabla_cribado, umbral_hopkins, elegido=None):
    """Candidatos de agrupamiento: estadístico de Hopkins contra la mejor
    silueta de K-means. El tamaño del punto es el valor de negocio y la línea
    vertical es el umbral de Hopkins de la regla."""
    datos = _cribados(tabla_cribado, ("agrupamiento",)).dropna(subset=["hopkins", "silueta"])
    if not len(datos):
        return _aviso("No hay candidatos de agrupamiento cribados.")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for _, fila in datos.iterrows():
        es_elegido = fila["analisis"] == elegido
        color = COLORES["verde"] if es_elegido else (COLORES["azul"] if bool(fila["elegible"]) else COLORES["gris"])
        ax.scatter(fila["hopkins"], fila["silueta"], s=90 * fila["valor_negocio"], color=color, alpha=0.85,
                   edgecolor=COLORES["tinta"] if es_elegido else "white", linewidth=1, zorder=3)
        ax.annotate(f"{fila['analisis']} (k = {int(fila['mejor_k'])}, valor {int(fila['valor_negocio'])})",
                    (fila["hopkins"], fila["silueta"]), textcoords="offset points", xytext=(9, 4), fontsize=8,
                    color=COLORES["tinta"])
    ax.axvline(umbral_hopkins, color=COLORES["rojo"], linestyle="--", linewidth=1)
    ax.set_xlim(min(umbral_hopkins - 0.05, datos["hopkins"].min() - 0.05), 1.0)
    ax.set_ylim(0, max(datos["silueta"].max() * 1.25, 0.1))
    ax.text(umbral_hopkins, 0.01, f" umbral de Hopkins {umbral_hopkins:.2f}", color=COLORES["rojo"],
            fontsize=8, va="bottom")
    ax.set_xlabel("Estadístico de Hopkins (0.5 = sin estructura, cerca de 1 = tendencia a agruparse)", fontsize=9)
    ax.set_ylabel("Mejor silueta de K-means", fontsize=9)
    ax.set_title("Candidatos de agrupamiento", loc="left", fontsize=11)
    leyenda = [Line2D([0], [0], color=COLORES["verde"], marker="o", linestyle="", label="Elegido por la regla"),
               Line2D([0], [0], color=COLORES["azul"], marker="o", linestyle="", label="Elegible"),
               Line2D([0], [0], color=COLORES["gris"], marker="o", linestyle="", label="No elegible")]
    ax.legend(handles=leyenda, fontsize=8, frameon=False, loc="lower right")
    _ejes(ax, rejilla="both")
    fig.tight_layout()
    return fig
