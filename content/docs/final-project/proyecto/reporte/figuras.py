"""Capa 7 · Estilo y guardado de figuras.

Responsabilidad única: dar a todas las figuras del proyecto el mismo aspecto
(el sistema editorial de la presentación: papel claro, filetes finos, Fraunces
en los títulos, IBM Plex Sans en el texto y JetBrains Mono en las cifras de
los ejes) y un único camino para guardarlas y mostrarlas. No calcula nada:
recibe figuras ya construidas.

Las tres familias se buscan primero entre las fuentes del sistema; si faltan,
se descargan una sola vez del repositorio oficial `google/fonts` a una
carpeta de caché (`~/.cache/proyecto_fuentes/`, o la que indique la variable
de entorno `PROYECTO_FUENTES`). Sin conexión, todo cae en silencio a DejaVu,
que viene con matplotlib.
"""
import os
import re
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager

from proyecto.base import rutas
from proyecto.base.config import COLORES, PALETA
from proyecto.base.salida import mostrar

PUNTOS_POR_PULGADA = 200
PAPEL = "#FCFBF8"        # fondo de papel de las figuras en pantalla (las guardadas van en blanco)
TINTA = "#1B1F24"
TENUE = "#5B6470"
FILETE = "#D8D4CA"       # filetes y rejilla

# Familia → (nombre que registra matplotlib, archivo variable en google/fonts, instancias fijas).
# matplotlib no elige ejes de una fuente variable (usaría la instancia por omisión, que en
# Fraunces es la más gruesa), así que se recortan instancias estáticas con fontTools, que ya
# viene con matplotlib. Fraunces: tamaño óptico de titular, sin «soft» ni «wonk».
FUENTES = {
    "titulos": ("Fraunces", "ofl/fraunces/Fraunces%5BSOFT,WONK,opsz,wght%5D.ttf",
                {"normal": {"wght": 560, "opsz": 48, "SOFT": 0, "WONK": 0},
                 "bold": {"wght": 650, "opsz": 48, "SOFT": 0, "WONK": 0}}),
    "texto": ("IBM Plex Sans", "ofl/ibmplexsans/IBMPlexSans%5Bwdth,wght%5D.ttf",
              {"normal": {"wght": 400, "wdth": 100}, "bold": {"wght": 600, "wdth": 100}}),
    "cifras": ("JetBrains Mono", "ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
               {"normal": {"wght": 400}, "bold": {"wght": 600}}),
}
RESPALDO = {"titulos": "DejaVu Serif", "texto": "DejaVu Sans", "cifras": "DejaVu Sans Mono"}
URL_FUENTES = "https://raw.githubusercontent.com/google/fonts/main/"
SEGUNDOS_DESCARGA = 6
_familias = {}


def _carpeta_cache():
    return Path(os.environ.get("PROYECTO_FUENTES", Path.home() / ".cache" / "proyecto_fuentes"))


def _registrada(nombre):
    return any(f.name == nombre for f in font_manager.fontManager.ttflist)


def _instancias(nombre, archivo, instancias, descargar):
    """Rutas de las instancias estáticas en la caché (las crea si hace falta)."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    carpeta = _carpeta_cache()
    variable = carpeta / Path(archivo.replace("%5B", "[").replace("%5D", "]")).name
    fijas = {peso: carpeta / f"{nombre.replace(' ', '')}-{peso}.ttf" for peso in instancias}
    if all(r.exists() for r in fijas.values()):
        return list(fijas.values())
    if not variable.exists():
        if not descargar:
            return []
        carpeta.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL_FUENTES + archivo, timeout=SEGUNDOS_DESCARGA) as respuesta:
            contenido = respuesta.read()
        temporal = variable.with_suffix(".parcial")
        temporal.write_bytes(contenido)
        temporal.replace(variable)
    for peso, coordenadas in instancias.items():
        fuente = instantiateVariableFont(TTFont(variable), coordenadas)
        fuente["OS/2"].usWeightClass = 700 if peso == "bold" else 400   # lo que matplotlib entiende por peso
        temporal = fijas[peso].with_suffix(".parcial")
        fuente.save(temporal)
        temporal.replace(fijas[peso])
    return list(fijas.values())


def registrar_fuentes(descargar=True):
    """Hace disponibles en matplotlib las tres familias del diseño y devuelve
    `{papel: familia usada}` (la de respaldo si no hubo forma de obtenerla).
    Se ejecuta una sola vez por sesión; nunca lanza errores por la red."""
    if _familias:
        return dict(_familias)
    for papel, (nombre, archivo, instancias) in FUENTES.items():
        if not _registrada(nombre):
            try:
                for ruta in _instancias(nombre, archivo, instancias, descargar):
                    font_manager.fontManager.addfont(str(ruta))
            except Exception:  # noqa: BLE001 — sin red, sin permisos o sin fontTools: se usa el respaldo
                pass
        _familias[papel] = nombre if _registrada(nombre) else RESPALDO[papel]
    return dict(_familias)


def familia(papel):
    """Familia efectiva para `papel` ∈ {"titulos", "texto", "cifras"}."""
    return registrar_fuentes()[papel]


def configurar_estilo():
    """Fija el estilo común de matplotlib. Uso: `configurar_estilo()` al
    inicio de cada etapa."""
    familias = registrar_fuentes()
    mpl.rcParams.update({
        "figure.figsize": (8.0, 4.5),
        "figure.dpi": 110,
        "figure.facecolor": PAPEL,        # en pantalla, sobre el papel de las diapositivas
        "savefig.facecolor": "white",
        "savefig.dpi": PUNTOS_POR_PULGADA,
        "savefig.bbox": "tight",
        "axes.facecolor": "none",
        "axes.edgecolor": "#9A958A",
        "axes.linewidth": 0.7,
        "axes.labelcolor": TINTA,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 13,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=PALETA),
        "axes.formatter.use_locale": False,   # punto decimal en todos los equipos
        "grid.color": "#E7E3DA",
        "grid.linewidth": 0.6,
        "text.color": TINTA,
        "xtick.color": TENUE,
        "ytick.color": TENUE,
        "xtick.labelcolor": TINTA,
        "ytick.labelcolor": TINTA,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "font.family": [familias["texto"], "DejaVu Sans"],   # DejaVu cubre los glifos que falten
        "mathtext.fontset": "stixsans",       # incluida en matplotlib; casa con un texto sin serifa
        "animation.embed_limit": 40.0,
    })


_CIFRAS = re.compile(r"[\s\d.,:;%+\-\u2212×$^{}()]*")


def _eje_numerico(eje):
    """True si las etiquetas de las marcas del eje son cifras o fechas (no palabras)."""
    try:
        formato, posiciones = eje.get_major_formatter(), eje.get_majorticklocs()
        formato.set_locs(posiciones)
        textos = [str(formato(v, i)).replace("\\mathdefault", "") for i, v in enumerate(posiciones)]
    except Exception:  # noqa: BLE001 — ante cualquier duda, se deja la familia del texto
        return False
    textos = [t for t in textos if t.strip()]
    return bool(textos) and all(_CIFRAS.fullmatch(t) for t in textos)


def tipografia(fig):
    """Aplica las familias del diseño a una figura ya construida: títulos en
    Fraunces y cifras de los ejes numéricos en JetBrains Mono. No cambia
    datos ni escalas. Uso: `tipografia(fig)` antes de guardar o mostrar."""
    titulos, cifras = familia("titulos"), familia("cifras")
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_fontfamily(titulos)
    for ax in fig.get_axes():
        if ax.images:                      # mapas de calor: la rejilla cruzaría las celdas
            ax.grid(False)
        for titulo in (ax.title, ax._left_title, ax._right_title):
            if titulo.get_text():
                titulo.set_fontfamily(titulos)
        for eje, nombre in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            if _eje_numerico(eje):
                ax.tick_params(axis=nombre, labelfontfamily=cifras)
    return fig


def guardar_figura(fig, nombre):
    """Guarda la figura como PNG en `figuras/` y devuelve la ruta.
    Uso: `guardar_figura(fig, "01_estacionalidad")`."""
    rutas.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = rutas.DIR_FIGURAS / f"{nombre}.png"
    tipografia(fig)
    fig.savefig(ruta, dpi=PUNTOS_POR_PULGADA, bbox_inches="tight", facecolor="white")
    return ruta


def mostrar_figura(fig, nombre=None):
    """Muestra una figura de matplotlib una sola vez y la cierra (si queda
    abierta, el notebook la dibuja de nuevo al terminar la celda). Con
    `nombre`, además la guarda. No devuelve nada, para que pueda ser la
    última línea de una celda. Uso: `mostrar_figura(fig, "01_volumen")`."""
    tipografia(fig)
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
            font={"color": COLORES["tinta"], "family": "IBM Plex Sans, system-ui, sans-serif"},
            paper_bgcolor=PAPEL, plot_bgcolor=PAPEL,
            separators=". ",                  # punto decimal, sin separador de miles
            margin={"l": 50, "r": 20, "t": 50, "b": 45},
        )
    mostrar(HTML(fig.to_html(full_html=False, include_plotlyjs="cdn")))
