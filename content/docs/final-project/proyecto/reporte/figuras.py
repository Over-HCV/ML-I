"""Capa 7 · Estilo y guardado de figuras.

Responsabilidad única: dar a todas las figuras del proyecto el mismo aspecto
(paleta de `config`, fondo blanco, punto decimal) y un único camino para
guardarlas y mostrarlas. No calcula nada: recibe figuras ya construidas.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from proyecto.base import rutas
from proyecto.base.config import COLORES, PALETA
from proyecto.base.salida import mostrar

PUNTOS_POR_PULGADA = 200


def configurar_estilo():
    """Fija el estilo común de matplotlib. Uso: `configurar_estilo()` al
    inicio de cada etapa."""
    mpl.rcParams.update({
        "figure.figsize": (8.0, 4.5),
        "figure.dpi": 110,
        "figure.facecolor": "white",      # legible también en temas oscuros
        "savefig.facecolor": "white",
        "savefig.dpi": PUNTOS_POR_PULGADA,
        "savefig.bbox": "tight",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORES["gris"],
        "axes.labelcolor": COLORES["tinta"],
        "axes.titlecolor": COLORES["tinta"],
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=PALETA),
        "axes.formatter.use_locale": False,   # punto decimal en todos los equipos
        "grid.color": "#E4E8EE",
        "grid.linewidth": 0.8,
        "text.color": COLORES["tinta"],
        "xtick.color": COLORES["tinta"],
        "ytick.color": COLORES["tinta"],
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "font.family": "DejaVu Sans",         # incluida en matplotlib; cubre tildes y ñ
    })


def guardar_figura(fig, nombre):
    """Guarda la figura como PNG en `figuras/` y devuelve la ruta.
    Uso: `guardar_figura(fig, "02_estacionalidad")`."""
    rutas.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = rutas.DIR_FIGURAS / f"{nombre}.png"
    fig.savefig(ruta, dpi=PUNTOS_POR_PULGADA, bbox_inches="tight", facecolor="white")
    return ruta


def mostrar_figura(fig, nombre=None):
    """Muestra una figura de matplotlib una sola vez y la cierra (si queda
    abierta, el notebook la dibuja de nuevo al terminar la celda). Con
    `nombre`, además la guarda. No devuelve nada, para que pueda ser la
    última línea de una celda. Uso: `mostrar_figura(fig, "02_volumen")`."""
    if nombre:
        guardar_figura(fig, nombre)
    mostrar(fig)
    plt.close(fig)


def mostrar_plotly(fig, con_estilo=True):
    """Muestra una figura de plotly incrustada como HTML, de forma que se vea
    igual en Colab, Jupyter y VS Code (la biblioteca se carga desde su CDN).
    Uso: `mostrar_plotly(px.line(tabla, x="semana", y="muestras"))`."""
    from IPython.display import HTML

    if con_estilo:
        fig.update_layout(
            template="plotly_white", colorway=PALETA,
            font={"color": COLORES["tinta"]},
            paper_bgcolor="white", plot_bgcolor="white",
            separators=". ",                  # punto decimal, sin separador de miles
            margin={"l": 50, "r": 20, "t": 50, "b": 45},
        )
    mostrar(HTML(fig.to_html(full_html=False, include_plotlyjs="cdn")))
