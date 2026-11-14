"""Capa 7 · Figuras y animaciones de la presentación.

Responsabilidad única: dibujar, con el estilo editorial del proyecto, las
figuras que la presentación muestra junto a su código (bloques temporales,
arquitectura del perceptrón, funciones de activación, curva de pérdida, VPP
según la previa) y tres animaciones de `matplotlib.animation` que se
incrustan como HTML con `to_jshtml()` (se reproducen sin núcleo): el umbral
que recorre las probabilidades de validación, la red que aprende época por
época y K-means que converge. No calcula resultados: recibe arreglos ya
calculados por las etapas o por `proyecto/etapas/presentacion.py`.

Los colores de los términos de las ecuaciones (`TINTAS`) son los mismos que
usan las celdas de texto con `\\color{...}` en MathJax, para que una
ecuación y la figura que la sigue hablen el mismo idioma.
"""
import base64
import io
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from PIL import Image

from proyecto.base.config import COLORES
from proyecto.reporte.figuras import PAPEL, TENUE, TINTA, configurar_estilo, familia, tipografia

# Colores de los términos de las ecuaciones (los mismos hexadecimales van en las celdas de MathJax).
TINTAS = {
    "previa": "#3B64A8",        # p: probabilidad previa
    "sensibilidad": "#2F8A4E",  # S: verdaderos positivos
    "falsos": "#B8423F",        # (1 − E)(1 − p): masa de falsos positivos; también FN y pérdidas
    "costo": "#C46A2B",         # costo y umbral
    "pesos": "#6E5BA8",         # pesos de la red
    "conformes": "#6B6B6B",     # muestras conformes (FP cuando reciben alerta)
}
PPP_ANIMACION = 80              # puntos por pulgada de los cuadros (cada cuadro es un PNG en base64)
COLORES_CUADRO = 64             # colores de la paleta de cada cuadro


def _figura(ancho, alto, columnas=1, **opciones):
    configurar_estilo()
    fig, ejes = plt.subplots(1, columnas, figsize=(ancho, alto), **opciones)
    return fig, ejes


def texto_coloreado(ax, partes, ubicacion="upper left", tamano=13, **opciones):
    """Escribe una ecuación por piezas, cada una con su color (mathtext no
    admite `\\color`). `partes` es una lista de `(texto, color)`."""
    cajas = [TextArea(t, textprops={"color": c or TINTA, "fontsize": tamano}) for t, c in partes]
    fila = HPacker(children=cajas, align="baseline", pad=0, sep=2)
    caja = AnchoredOffsetbox(loc=ubicacion, child=fila, frameon=False, pad=0.2, **opciones)
    ax.add_artist(caja)
    return caja


def _paleta_reducida(coincidencia):
    """Recomprime un cuadro PNG con una paleta de `COLORES_CUADRO` colores (las figuras son planas: el cambio
    no se nota y el cuadro pesa la mitad o menos)."""
    datos = base64.b64decode(re.sub(r"[^A-Za-z0-9+/=]", "", coincidencia.group(1)))
    imagen = Image.open(io.BytesIO(datos)).convert("RGB").quantize(colors=COLORES_CUADRO, method=Image.Quantize.MEDIANCUT)
    salida = io.BytesIO()
    imagen.save(salida, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(salida.getvalue()).decode("ascii")


def como_html(anim, fps):
    """Animación → HTML autocontenido (cuadros en base64 y controles en JavaScript)."""
    # `Animation.save` toma la resolución de `savefig.dpi` (200 en el proyecto): aquí se baja a la de pantalla,
    # y sin recorte «tight» para que todos los cuadros midan lo mismo.
    with mpl.rc_context({"savefig.dpi": PPP_ANIMACION, "savefig.bbox": None, "animation.frame_format": "png"}):
        html = anim.to_jshtml(fps=fps, default_mode="once")
    plt.close(anim._fig)
    html = re.sub(r"data:image/png;base64,([A-Za-z0-9+/=\\\n]+)", _paleta_reducida, html)
    return HTML(f'<div style="background:{PAPEL};padding:8px 0;max-width:980px">{html}</div>')


# ----------------------------------------------------------------------
# Figuras estáticas
# ----------------------------------------------------------------------
def figura_bloques(fechas, bloques):
    """Ventanas crecientes de la validación cruzada temporal: en cada bloque se
    ajusta con el pasado (azul) y se valida con el tramo siguiente (naranja)."""
    fechas = np.asarray(fechas, dtype="datetime64[D]")
    fig, ax = _figura(8.6, 0.55 * len(bloques) + 1.3)
    for i, (entrenar, validar) in enumerate(bloques):
        for indices, color, nombre in ((entrenar, COLORES["azul"], "ajustar"), (validar, COLORES["naranja"], "validar")):
            desde, hasta = fechas[indices].min(), fechas[indices].max()
            ax.barh(i, (hasta - desde).astype(int), left=desde, height=0.5, color=color,
                    label=nombre if i == 0 else None)
        ax.text(fechas[validar].max(), i, f"  {len(entrenar)} → {len(validar)}", va="center", fontsize=9,
                fontfamily=familia("cifras"), color=TENUE)
    ax.set_yticks(range(len(bloques)), [f"bloque {i + 1}" for i in range(len(bloques))])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0), handlelength=1.2)
    ax.set_title("Validación por bloques: se ajusta con el pasado y se valida con lo que sigue", pad=26)
    ax.set_xlim(right=fechas.max() + np.timedelta64(140, "D"))
    return tipografia(fig)


def figura_arquitectura(n_entradas, ocultas, activacion="relu", maximo_dibujadas=10):
    """Diagrama del perceptrón multicapa con las capas que de verdad se usaron
    (`hidden_layer_sizes`). Las capas grandes se dibujan recortadas con «⋮»."""
    capas = [int(n_entradas), *[int(c) for c in ocultas], 1]
    nombres = ["entrada"] + [f"oculta {i + 1}" if len(ocultas) > 1 else "oculta" for i in range(len(ocultas))] + ["salida"]
    fig, ax = _figura(8.6, 3.6)
    ax.axis("off")
    posiciones = []
    for j, n in enumerate(capas):
        dibujadas = min(n, maximo_dibujadas)
        ys = np.linspace(1, -1, dibujadas) * (dibujadas - 1) / (maximo_dibujadas - 1) if dibujadas > 1 else np.zeros(1)
        posiciones.append((j, ys, n > dibujadas))
    for (x0, ys0, _), (x1, ys1, _) in zip(posiciones[:-1], posiciones[1:]):
        for a in ys0:
            for b in ys1:
                ax.plot([x0, x1], [a, b], color=TINTAS["pesos"], alpha=0.12, lw=0.6, zorder=1)
    for (x, ys, recortada), n, nombre in zip(posiciones, capas, nombres):
        color = COLORES["gris"] if x == 0 else TINTAS["pesos"] if x < len(capas) - 1 else TINTAS["falsos"]
        ax.scatter(np.full(len(ys), x), ys, s=90, color=PAPEL, edgecolor=color, linewidth=1.6, zorder=3)
        if recortada:
            ax.text(x, 0, "⋮", ha="center", va="center", fontsize=18, color=TENUE, zorder=4,
                    bbox={"boxstyle": "square,pad=0.1", "fc": PAPEL, "ec": "none"})
        ax.text(x, -1.32, nombre, ha="center", fontsize=10.5, color=TINTA)
        ax.text(x, -1.55, f"{n}", ha="center", fontsize=10.5, color=TENUE, fontfamily=familia("cifras"))
    funcion = {"relu": "ReLU", "logistic": "logística", "tanh": "tanh"}.get(activacion, activacion)
    for x in range(1, len(capas) - 1):
        ax.text(x, 1.18, funcion, ha="center", fontsize=10, color=TINTAS["pesos"], style="italic")
    ax.text(len(capas) - 1, 0.25, "  σ → P(no conforme)", ha="left", va="bottom", fontsize=10, color=TINTAS["falsos"])
    ax.set_xlim(-0.4, len(capas) - 0.2)
    ax.set_ylim(-1.7, 1.35)
    ax.set_title("Perceptrón multicapa usado en la pasada didáctica (dibujado desde hidden_layer_sizes)")
    return tipografia(fig)


def figura_activaciones():
    """ReLU, logística y tanh, con los colores de sus ecuaciones."""
    fig, ejes = _figura(8.6, 2.4, columnas=3, sharey=False)
    z = np.linspace(-4, 4, 200)
    for ax, (nombre, valores, color) in zip(ejes, (
            (r"ReLU: $\max(0, z)$", np.maximum(0, z), TINTAS["pesos"]),
            (r"logística: $\sigma(z) = 1/(1+e^{-z})$", 1 / (1 + np.exp(-z)), TINTAS["falsos"]),
            (r"tanh: $\tanh(z)$", np.tanh(z), TINTAS["previa"]))):
        ax.axhline(0, color=TENUE, lw=0.6)
        ax.axvline(0, color=TENUE, lw=0.6)
        ax.plot(z, valores, color=color, lw=2.2)
        ax.set_title(nombre, fontsize=11)
        ax.set_xlabel("z = w·x + b")
    fig.tight_layout()
    return tipografia(fig)


def figura_curva_perdida(perdida, puntaje_validacion, paciencia, fraccion_validacion, tasa_mayoritaria=None):
    """`loss_curve_` (pérdida logarítmica de entrenamiento) y `validation_scores_`
    (exactitud en la fracción apartada) de un `MLPClassifier` con parada temprana.
    Con `tasa_mayoritaria`, dibuja la exactitud de responder siempre la clase
    mayoritaria: con clases desbalanceadas la exactitud queda cerca de ella y no
    dice cuánto ordena el modelo (para eso, el ROC-AUC de la animación)."""
    fig, (a1, a2) = _figura(8.6, 3.0, columnas=2)
    epocas = np.arange(1, len(perdida) + 1)
    a1.plot(epocas, perdida, color=TINTAS["falsos"])
    a1.set_title("loss_curve_: pérdida de entrenamiento", fontsize=11)
    a1.set_xlabel("época")
    a2.plot(epocas, puntaje_validacion, color=TINTAS["previa"])
    mejor = int(np.argmax(puntaje_validacion)) + 1
    a2.axvline(mejor, color=TENUE, ls=":", lw=1)
    a2.text(0.98, 0.04, f"mejor época: {mejor}\nse detiene {paciencia} épocas después", transform=a2.transAxes,
            ha="right", va="bottom", fontsize=9, color=TENUE)
    if tasa_mayoritaria is not None:
        a2.axhline(tasa_mayoritaria, color=TINTAS["conformes"], ls="--", lw=1)
        a2.annotate(f"siempre «conforme»: {tasa_mayoritaria:.3f}", (1, tasa_mayoritaria), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8.5,
                    color=TINTAS["conformes"])
    a2.set_title(f"validation_scores_: exactitud en el {100 * fraccion_validacion:.0f} % apartado", fontsize=11)
    a2.set_xlabel("época")
    fig.tight_layout()
    if tasa_mayoritaria is not None:
        fig.text(0.5, -0.04, "La parada temprana de scikit-learn vigila la exactitud, que aquí apenas supera a la clase "
                 "mayoritaria; la métrica del proyecto es el ROC-AUC (animación siguiente).", ha="center", va="top",
                 fontsize=9, color=TENUE)
    return tipografia(fig)


def figura_vpp(previas, vpp, ensayos, marcas=None):
    """VPP según la probabilidad previa para cada ensayo (curvas de Bayes), con
    la ecuación coloreada. `marcas` = {ensayo: (previa, vpp)} observados."""
    fig, ax = _figura(8.6, 3.8)
    grises = np.linspace(0.25, 0.7, len(ensayos))
    for j, ensayo in enumerate(ensayos):
        color = plt.cm.Greens(1 - grises[j] + 0.25)
        ax.plot(previas, vpp[:, j], color=color, lw=1.8, label=ensayo)
        if marcas and ensayo in marcas:
            ax.scatter(*marcas[ensayo], color=color, edgecolor=TINTA, linewidth=0.6, zorder=4, s=36)
    ax.set_xscale("log")
    ax.set_xlabel("probabilidad previa de contaminación  p", color=TINTAS["previa"])
    ax.set_ylabel("VPP = P(contaminada | positivo)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8.5)
    texto_coloreado(ax, [("VPP = ", None), (r"$S\,p$", TINTAS["sensibilidad"]), ("  /  (", None),
                         (r"$S\,p$", TINTAS["sensibilidad"]), (" + ", None),
                         (r"$(1-E)(1-p)$", TINTAS["falsos"]), (")", None)], tamano=12)
    ax.set_title("El mismo positivo vale distinto según la previa", pad=12)
    return tipografia(fig)


# ----------------------------------------------------------------------
# Animaciones
# ----------------------------------------------------------------------
def animacion_umbral(proba, y, umbrales, costo_fp, costo_fn, elegido, fps=4):
    """El umbral recorre las probabilidades de validación: a la izquierda, la
    distribución de cada clase y la línea del umbral; a la derecha, el costo
    `costo_fp·FP + costo_fn·FN` y los cuatro conteos de la matriz de confusión."""
    proba, y = np.asarray(proba, dtype=float), np.asarray(y, dtype=int)
    umbrales = np.asarray(umbrales, dtype=float)
    conteos = []
    for t in umbrales:
        alerta = proba >= t
        conteos.append(((~alerta & (y == 0)).sum(), (alerta & (y == 0)).sum(), (~alerta & (y == 1)).sum(),
                        (alerta & (y == 1)).sum()))
    conteos = np.array(conteos)
    costos = costo_fp * conteos[:, 1] + costo_fn * conteos[:, 2]
    fig, (a1, a2) = _figura(8.6, 3.3, columnas=2, gridspec_kw={"width_ratios": [1.15, 1]})
    tope = float(np.quantile(proba, 0.995))
    bordes = np.linspace(0, tope, 36)
    a1.hist(proba[y == 0], bins=bordes, color=TINTAS["conformes"], alpha=0.55, density=True, label="conformes")
    a1.hist(proba[y == 1], bins=bordes, color=TINTAS["falsos"], alpha=0.65, density=True, label="no conformes")
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlabel("probabilidad predicha (validación)")
    a1.legend(loc="upper right", fontsize=8.5)
    a1.set_title("Umbral de alerta", fontsize=12)
    linea = a1.axvline(umbrales[0], color=TINTAS["costo"], lw=2)
    a2.plot(umbrales, costos, color=TINTAS["costo"], lw=1.6)
    a2.axvline(elegido, color=TENUE, ls=":", lw=1)
    punto, = a2.plot([], [], "o", color=TINTAS["costo"], ms=7)
    a2.set_xlabel("umbral")
    a2.set_title(f"Costo = {costo_fp}·FP + {costo_fn}·FN", fontsize=12)
    texto = a2.text(0.98, 0.95, "", transform=a2.transAxes, ha="right", va="top", fontsize=9.5,
                    fontfamily=familia("cifras"), color=TINTA, linespacing=1.4)
    fig.tight_layout()
    tipografia(fig)

    def cuadro(i):
        linea.set_xdata([umbrales[i], umbrales[i]])
        punto.set_data([umbrales[i]], [costos[i]])
        vn, fp, fn, vp = conteos[i]
        texto.set_text(f"umbral {umbrales[i]:.3f}\nFP {fp:5d}   FN {fn:4d}\nVP {vp:5d}   VN {vn:4d}\ncosto {costos[i]:6.0f}")
        return linea, punto, texto

    anim = animation.FuncAnimation(fig, cuadro, frames=len(umbrales), blit=False)
    return como_html(anim, fps)


def animacion_red(perdida, auc, histogramas, bordes, fps=5):
    """La red aprende época por época: pérdida de entrenamiento y ROC-AUC de
    validación que se dibujan a medida que avanzan, y la distribución de la
    probabilidad predicha para cada clase que se separa."""
    epocas = np.arange(1, len(perdida) + 1)
    centros = (bordes[:-1] + bordes[1:]) / 2
    fig, (a1, a2) = _figura(8.6, 3.3, columnas=2)
    a1.set_xlim(0.5, len(epocas) + 0.5)
    a1.set_ylim(min(perdida) * 0.97, max(perdida) * 1.03)
    a1.set_xlabel("época")
    a1.set_ylabel("pérdida (entrenamiento)", color=TINTAS["falsos"])
    a1b = a1.twinx()
    a1b.set_ylim(min(auc) - 0.02, max(auc) + 0.02)
    a1b.set_ylabel("ROC-AUC (validación)", color=TINTAS["previa"])
    a1b.grid(False)
    a1b.spines["right"].set_visible(True)
    l_perdida, = a1.plot([], [], color=TINTAS["falsos"], lw=2)
    l_auc, = a1b.plot([], [], color=TINTAS["previa"], lw=2)
    a1.set_title("Pérdida y ROC-AUC por época", fontsize=12)
    tope = max(h.max() for h in histogramas.reshape(-1, histogramas.shape[-1]))
    a2.set_xlim(bordes[0], bordes[-1])
    a2.set_ylim(0, tope * 1.08)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel("probabilidad predicha (validación)")
    l0, = a2.plot([], [], drawstyle="steps-mid", color=TINTAS["conformes"], lw=1.8, label="conformes")
    l1, = a2.plot([], [], drawstyle="steps-mid", color=TINTAS["falsos"], lw=1.8, label="no conformes")
    a2.legend(loc="upper right", fontsize=8.5)
    titulo = a2.set_title("Época 1: ¿se separan las clases?", fontsize=12)
    fig.tight_layout()
    tipografia(fig)
    a1b.tick_params(axis="y", labelfontfamily=familia("cifras"))

    def cuadro(i):
        l_perdida.set_data(epocas[: i + 1], perdida[: i + 1])
        l_auc.set_data(epocas[: i + 1], auc[: i + 1])
        l0.set_data(centros, histogramas[i, 0])
        l1.set_data(centros, histogramas[i, 1])
        titulo.set_text(f"Época {i + 1}: ¿se separan las clases?")
        return l_perdida, l_auc, l0, l1, titulo

    anim = animation.FuncAnimation(fig, cuadro, frames=len(epocas), blit=False)
    return como_html(anim, fps)


def animacion_kmeans(puntos, etiquetas, centros, fps=2):
    """K-means paso a paso sobre dos componentes principales: cada cuadro es una
    iteración de Lloyd (asignar al centro más cercano y mover cada centro a la
    media de su grupo). `etiquetas` (pasos × n) y `centros` (pasos × k × 2)."""
    paleta = [COLORES[c] for c in ("azul", "naranja", "verde", "rojo", "morado", "gris")]
    k = centros.shape[1]
    colores = np.array([paleta[i % len(paleta)] for i in range(k)])
    fig, ax = _figura(6.4, 4.2)
    nube = ax.scatter(puntos[:, 0], puntos[:, 1], s=12, c=colores[etiquetas[0]], alpha=0.55, linewidths=0)
    rastros = [ax.plot([], [], color=TINTA, lw=0.8, alpha=0.6)[0] for _ in range(k)]
    marcas = ax.scatter(centros[0, :, 0], centros[0, :, 1], s=170, marker="X", c=colores, edgecolor=TINTA,
                        linewidth=1.2, zorder=5)
    ax.set_xlabel("componente principal 1")
    ax.set_ylabel("componente principal 2")
    titulo = ax.set_title("K-means, iteración 0: asignar y promediar")
    fig.tight_layout()
    tipografia(fig)

    def cuadro(i):
        nube.set_color(colores[etiquetas[i]])
        marcas.set_offsets(centros[i])
        for j, rastro in enumerate(rastros):
            rastro.set_data(centros[: i + 1, j, 0], centros[: i + 1, j, 1])
        estado = "convergió" if i == len(centros) - 1 else "asignar y promediar"
        titulo.set_text(f"K-means, iteración {i}: {estado}")
        return nube, marcas, titulo, *rastros

    anim = animation.FuncAnimation(fig, cuadro, frames=len(centros), blit=False)
    return como_html(anim, fps)
