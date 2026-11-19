"""Capa 7 · Paneles interactivos de la presentación.

Responsabilidad única: armar, con `ipywidgets`, los paneles con controles que
la presentación usa en vivo para mover un parámetro del experimento y ver al
instante cómo cambia la métrica que se optimizó y lo que depende de ella:
umbral de alerta, red neuronal, Bayes, K-means, KNN y árbol podado. Cada
`panel_*(d)` recibe el espacio de nombres que devuelve
`proyecto.etapas.presentacion.iniciar()` (arreglos pequeños del desarrollo,
nunca de la prueba) y devuelve un contenedor de widgets: controles arriba y
la figura, con el estilo editorial del proyecto, debajo. Cada panel marca el
valor que el proyecto eligió (congelado) junto al valor actual. La excepción
es `panel_taguchi(taguchi, congelada)`, que no reajusta nada: recibe la
sección `taguchi` del artefacto de una etapa con L16 (03 o 04) y la
configuración congelada, y lo usan los notebooks de esas etapas.

Sin núcleo (notebook guardado y abierto sin ejecutar), la celda conserva la
figura del estado inicial como imagen estática: se publica aparte, con un
`display_id`, y se borra la primera vez que se mueve un control, cuando la
figura pasa a dibujarse dentro del área de salida del panel. Los reajustes
(red, KNN, árbol, K-means) se guardan en memoria por combinación, así que
volver a un valor ya visto es inmediato.
"""
import io
import time
import warnings
from contextlib import contextmanager

import ipywidgets as w
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, Image, clear_output, display
from matplotlib.patches import Rectangle

from proyecto.base.config import COLORES, COSTO_FP, SEMILLA
from proyecto.evaluacion import bayes, metricas
from proyecto.modelos import registro
from proyecto.modelos.agrupamiento import pasos_kmeans
from proyecto.reporte.figuras import FILETE, PAPEL, TENUE, TINTA, configurar_estilo, familia, tipografia
from proyecto.reporte.graficos_presentacion import TINTAS, texto_coloreado
from proyecto.reporte.html import COLORES_CONFUSION, MONO, SANS, SERIF

ANCHO = 8.6                     # pulgadas: el mismo ancho de las figuras de la presentación
PPP = 110                       # puntos por pulgada de la imagen del panel
RAZONES_COSTO = (1, 2, 5, 10)   # razón costo FN / costo FP que ofrece el panel del umbral
CAPAS_RED = ((8,), (32,), (64,), (32, 16))
ALFAS_RED = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
VECINOS = (1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 71, 101, 151)
MAXIMO_ALFAS_ARBOL = 24         # pasos del deslizador de poda
GRUPOS = range(2, 9)
MUESTRAS_ICONOS = 1000          # cuadritos del diagrama de Bayes (40 × 25)
NOMBRES_PESOS = {"uniform": "uniforme", "distance": "por distancia"}
TEXTO_SIN_NUCLEO = "Panel interactivo: Ejecute la celda con el núcleo en marcha para mover los controles."

_ESTILO = f"""<style>
.pf-int {{background:{PAPEL};border-top:2px solid {TINTA};padding:10px 6px 4px;max-width:980px;color:{TINTA}}}
.pf-int .pf-int-rotulo {{font-family:{SANS};font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:{TENUE}}}
.pf-int .pf-int-titulo {{font-family:{SERIF};font-size:21px;line-height:1.25;margin:2px 0 4px}}
.pf-int .pf-int-nota {{font-family:{SANS};font-size:13px;line-height:1.45;color:{TENUE};max-width:860px}}
.pf-int .pf-int-nota b {{color:{TINTA};font-weight:600}}
.pf-int .widget-label, .pf-int .widget-readout, .pf-int .widget-toggle-button, .pf-int select,
.pf-int .widget-dropdown {{font-family:{SANS} !important;font-size:13px}}
.pf-int .widget-readout {{font-family:{MONO} !important}}
</style>"""


# ----------------------------------------------------------------------
# Panel genérico
# ----------------------------------------------------------------------
def _png(fig):
    """Figura → PNG (y la cierra), con la tipografía del proyecto."""
    tipografia(fig)
    salida = io.BytesIO()
    fig.savefig(salida, format="png", dpi=PPP, facecolor=PAPEL, bbox_inches="tight")
    plt.close(fig)
    return salida.getvalue()


def _figura(alto, columnas=1, **opciones):
    configurar_estilo()
    return plt.subplots(1, columnas, figsize=(ANCHO, alto), **opciones)


class Panel(w.VBox):
    """Controles + área de salida. `dibujar(**valores)` devuelve una figura de
    matplotlib; cada cambio de un control la vuelve a dibujar. Guarda el
    tiempo de cada dibujo (`tiempos`) y los errores (`errores`), que la
    verificación revisa sin pantalla."""

    def __init__(self, rotulo, titulo, explicacion, controles, dibujar):
        self.controles = controles
        self._dibujar = dibujar
        self.salida = w.Output()
        self.tiempos, self.errores = [], []
        self._pausa = False
        self._estatica = None
        cabecera = w.HTML(f'{_ESTILO}<div class="pf-int-rotulo">{rotulo} (interactivo)</div>'
                          f'<div class="pf-int-titulo">{titulo}</div><div class="pf-int-nota">{explicacion}</div>')
        fila = w.HBox(list(controles.values()), layout=w.Layout(flex_flow="row wrap", align_items="center"))
        super().__init__([cabecera, fila, self.salida])
        self.add_class("pf-int")
        for control in controles.values():
            control.observe(self._al_cambiar, names="value")
        self.imagen_inicial = self.imagen = self._calcular()

    def valores(self):
        return {nombre: control.value for nombre, control in self.controles.items()}

    def _calcular(self):
        inicio = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # avisos de convergencia de los reajustes rápidos
            png = _png(self._dibujar(**self.valores()))
        self.tiempos.append(time.perf_counter() - inicio)
        return png

    def _al_cambiar(self, _cambio=None):
        if not self._pausa:
            self.redibujar()

    @contextmanager
    def en_lote(self):
        """Cambia varios controles a la vez y redibuja una sola vez al final."""
        self._pausa = True
        try:
            yield
        finally:
            self._pausa = False
        self.redibujar()

    def redibujar(self):
        try:
            png = self._calcular()
        except Exception as error:  # noqa: BLE001 (el panel informa el error en vez de romper la celda)
            self.errores.append(f"{type(error).__name__}: {error}")
            with self.salida:
                clear_output(wait=True)
                display(HTML(f"<pre>No se pudo dibujar: {type(error).__name__}: {error}</pre>"))
            return
        if self._estatica is not None:   # la imagen estática del estado inicial deja paso a la viva
            self._estatica.update(HTML(""))
            self._estatica = None
        self.imagen = png
        with self.salida:
            clear_output(wait=True)
            display(Image(data=png, format="png"))

    def _ipython_display_(self, **opciones):
        padre = super()
        if hasattr(padre, "_repr_mimebundle_"):          # ipywidgets 8
            paquete = padre._repr_mimebundle_()
            paquete["text/plain"] = TEXTO_SIN_NUCLEO   # lo que ven los visores sin widgets (GitHub, nbviewer)
            display(paquete, raw=True)
        else:                                            # ipywidgets 7 (Colab)
            padre._ipython_display_(**opciones)
        self._estatica = display(Image(data=self.imagen_inicial, format="png"), display_id=True)


def _estilo_control(ancho="330px"):
    return {"style": {"description_width": "initial"}, "layout": w.Layout(width=ancho, margin="2px 18px 2px 0")}


def _marca_elegida(ax, x, texto, color=TENUE):
    """Línea punteada en el valor que eligió el proyecto, con su rótulo del lado que tenga espacio."""
    ax.axvline(x, color=color, ls=":", lw=1.2, zorder=1)
    ax.figure.canvas.draw_idle()
    fraccion = ax.transAxes.inverted().transform(ax.transData.transform((x, 0)))[0]
    derecha = fraccion > 0.6
    ax.annotate(texto, (x, 1), xycoords=("data", "axes fraction"), xytext=(-3 if derecha else 3, -2),
                textcoords="offset points", ha="right" if derecha else "left", va="top", fontsize=8.5, color=color)


def _pct(valor, decimales=1):
    return "" if np.isnan(valor) else f"{100 * valor:.{decimales}f} %"


def _cuadro_cifras(ax, lineas, x=0.98, y=0.96, ha="right"):
    ax.text(x, y, "\n".join(lineas), transform=ax.transAxes, ha=ha, va="top", fontsize=9.5,
            fontfamily=familia("cifras"), color=TINTA, linespacing=1.45,
            bbox={"boxstyle": "square,pad=0.4", "fc": PAPEL, "ec": FILETE, "lw": 0.6})


def _submuestras(d):
    """Submuestras de entrenamiento y validación (ya transformadas) en float64."""
    return (d.sub_X_ent.astype(float), d.sub_y_ent.astype(int), d.sub_X_val.astype(float), d.sub_y_val.astype(int))


def _texto_submuestra(d):
    return (f"Se reentrena en vivo con una submuestra fija de <b>{len(d.sub_y_ent)}</b> filas de entrenamiento y se "
            f"mide en <b>{len(d.sub_y_val)}</b> de validación (semilla del proyecto): Las cifras son de esa "
            "submuestra y no las del modelo congelado.")


# ----------------------------------------------------------------------
# 1. Umbral de decisión
# ----------------------------------------------------------------------
def _conteos(proba, y, umbrales):
    """VN, FP, FN, VP para cada umbral (alerta si proba ≥ t), sin recorrer filas."""
    pos = np.sort(proba[y == 1])
    neg = np.sort(proba[y == 0])
    fn = np.searchsorted(pos, umbrales, side="left")
    vn = np.searchsorted(neg, umbrales, side="left")
    return vn, len(neg) - vn, fn, len(pos) - fn


def _matriz(ax, vn, fp, fn, vp):
    rotulos = ("bien descartadas", "alertas innecesarias", "no conformes sin alerta", "no conformes detectadas")
    for i, (valor, rotulo, color) in enumerate(zip((vn, fp, fn, vp), rotulos, COLORES_CONFUSION)):
        fila, columna = divmod(i, 2)
        ax.add_patch(Rectangle((columna, 1 - fila), 0.96, 0.92, color=color, alpha=0.16, lw=0))
        ax.add_patch(Rectangle((columna, 1 - fila), 0.04, 0.92, color=color, lw=0))
        ax.text(columna + 0.5, 1 - fila + 0.55, f"{valor}", ha="center", va="center", fontsize=17, color=color,
                fontfamily=familia("cifras"), fontweight="bold")
        ax.text(columna + 0.5, 1 - fila + 0.22, rotulo, ha="center", va="center", fontsize=8.5, color=TINTA)
    ax.set_xlim(-0.02, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.48, 1.48], ["sin alerta", "alerta"])
    ax.set_yticks([1.46, 0.46], ["conforme", "no conforme"])
    ax.tick_params(length=0)
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)


def panel_umbral(d):
    """Deslizador del umbral t sobre las probabilidades de validación del
    clasificador elegido (sin reajustar): matriz de confusión, precisión,
    recall, alertas por semana y costo `COSTO_FP FP + COSTO_FN FN`, con la
    curva de costo, el umbral actual y el elegido en la etapa 04. La razón
    FN/FP se puede cambiar para ver cómo se mueve el umbral óptimo."""
    datos = d.interactivo
    proba, y = np.asarray(d.proba_validacion, dtype=float), np.asarray(d.y_validacion, dtype=int)
    elegido, costos, semanas = datos["umbral"], datos["costos"], datos["semanas_validacion"]
    razon_elegida = costos["falso_negativo"] / costos["falso_positivo"]
    rejilla = np.asarray(datos["umbrales"], dtype=float)   # recorrido de la etapa 04: ahí se busca el óptimo
    _, fp_r, fn_r, _ = _conteos(proba, y, rejilla)
    fina = np.round(np.arange(0.02, 0.6005, 0.0025), 4)    # solo para dibujar la curva
    _, fp_f, fn_f, _ = _conteos(proba, y, fina)
    razones = sorted(set(RAZONES_COSTO) | {int(razon_elegida)})

    def dibujar(t, razon):
        costo_fn = COSTO_FP * razon
        curva = COSTO_FP * fp_r + costo_fn * fn_r
        optimo = float(rejilla[np.flatnonzero(curva == curva.min())[-1]])   # empate → umbral más alto
        vn, fp, fn, vp = (int(v[0]) for v in _conteos(proba, y, np.array([t])))
        costo = COSTO_FP * fp + costo_fn * fn
        precision = vp / (vp + fp) if vp + fp else float("nan")
        recall = vp / (vp + fn) if vp + fn else float("nan")
        fig, (a1, a2) = _figura(3.9, 2, gridspec_kw={"width_ratios": [1, 1.35]})
        _matriz(a1, vn, fp, fn, vp)
        a1.set_title(f"Validación con t = {t:.3f}", fontsize=12)
        a2.plot(fina, COSTO_FP * fp_f + costo_fn * fn_f, color=TINTAS["costo"], lw=1.8)
        a2.plot(rejilla, curva, "o", color=TINTAS["costo"], ms=2.5, alpha=0.6)
        a2.plot([t], [costo], "o", color=TINTAS["costo"], ms=8, zorder=5)
        _marca_elegida(a2, elegido, f"elegido {elegido:.3f}")
        if razon != razon_elegida:
            a2.axvline(optimo, color=TINTAS["costo"], ls="--", lw=1, alpha=0.7)
        a2.set_xlabel("umbral t")
        a2.set_ylabel("costo en validación")
        a2.set_title(f"Costo = {COSTO_FP} FP + {costo_fn:g} FN, óptimo t = {optimo:.3f}", fontsize=12)
        a2.set_xlim(0, 0.62)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        cifras = (f"precisión {_pct(precision)},   recall {_pct(recall)},   {(vp + fp) / semanas:.1f} alertas por "
                  f"semana,   costo {costo:.0f}")
        fig.text(0.5, 0.02, cifras, ha="center", va="bottom", fontsize=11, fontfamily=familia("cifras"), color=TINTA,
                 bbox={"boxstyle": "square,pad=0.5", "fc": PAPEL, "ec": FILETE, "lw": 0.6})
        return fig

    controles = {
        "t": w.FloatSlider(value=elegido, min=0.02, max=0.6, step=0.001, readout_format=".3f",
                           description="Umbral t", continuous_update=False, **_estilo_control("380px")),
        "razon": w.Dropdown(options=[(f"{r} : 1", r) for r in razones], value=int(razon_elegida),
                            description="Costo FN / costo FP", **_estilo_control("230px")),
    }
    explicacion = (f"Se alerta cuando la probabilidad pasa <b>t</b>. En la etapa 04 elegimos t = <b>{elegido:.3f}</b> "
                   f"porque minimiza el costo cuando perder una no conforme cuesta <b>{razon_elegida:g}</b> veces más que "
                   "una falsa alarma (línea punteada). Al mover t se ve qué se gana y qué se pierde. Al cambiar la razón "
                   "de costos se ve dónde quedaría el óptimo (línea discontinua). Son las probabilidades de validación "
                   "del modelo elegido, sin reajustar.")
    return Panel("Clasificación", "Umbral de alerta: precisión, recall y costo", explicacion, controles, dibujar)


# ----------------------------------------------------------------------
# 2. Red neuronal
# ----------------------------------------------------------------------
def panel_red(d):
    """Capas ocultas y regularización α de un `MLPClassifier` reentrenado en
    vivo sobre la submuestra fija: pérdida de entrenamiento y ROC-AUC de
    validación época a época, frente a la regresión logística ajustada con
    las mismas filas. Las combinaciones ya vistas quedan en memoria."""
    from sklearn.linear_model import LogisticRegression

    X_ent, y_ent, X_val, y_val = _submuestras(d)
    partida = registro.crear_modelo("clasificacion", "mlp").get_params()
    epocas = int(partida["max_iter"])
    referencia = metricas.roc_auc(y_val, LogisticRegression(max_iter=1000).fit(X_ent, y_ent).predict_proba(X_val)[:, 1])
    elegida = (tuple(d.mlp["ocultas"]), float(d.mlp["alpha"]))
    memoria = {}

    def entrenar(ocultas, alfa):
        if (ocultas, alfa) not in memoria:
            red = registro.crear_modelo("clasificacion", "mlp", hidden_layer_sizes=ocultas, alpha=alfa,
                                        early_stopping=False)
            perdida, auc = [], []
            for _ in range(epocas):
                red.partial_fit(X_ent, y_ent, classes=np.array([0, 1]))
                perdida.append(red.loss_)
                auc.append(metricas.roc_auc(y_val, red.predict_proba(X_val)[:, 1]))
            memoria[(ocultas, alfa)] = (np.array(perdida), np.array(auc))
        return memoria[(ocultas, alfa)]

    def dibujar(ocultas, alfa):
        perdida, auc = entrenar(ocultas, alfa)
        auc_elegida = entrenar(*elegida)[1][-1]
        fig, (a1, a2) = _figura(3.4, 2, gridspec_kw={"width_ratios": [1.5, 1]})
        x = np.arange(1, epocas + 1)
        a1.plot(x, perdida, color=TINTAS["falsos"], lw=1.6)
        a1.set_xlabel("época")
        a1.set_ylabel("pérdida (entrenamiento)", color=TINTAS["falsos"])
        a1b = a1.twinx()
        a1b.plot(x, auc, color=TINTAS["previa"], lw=1.6)
        a1b.axhline(referencia, color=TENUE, ls=":", lw=1)
        a1b.annotate("logística", (1, referencia), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                     textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=TENUE)
        a1b.set_ylabel("ROC-AUC (validación)", color=TINTAS["previa"])
        a1b.grid(False)
        a1b.spines["right"].set_visible(True)
        mejor = int(np.argmax(auc)) + 1
        a1b.plot([mejor], [auc[mejor - 1]], "o", color=TINTAS["previa"], ms=5)
        a1.set_title(f"Mejor época {mejor}: La red sigue bajando su pérdida", fontsize=11.5)
        nombres = ["Logística (referencia)", f"Red elegida {elegida[0]}, α={elegida[1]:g}", f"Red actual {ocultas}, α={alfa:g}"]
        valores = [referencia, auc_elegida, auc[-1]]
        colores = [COLORES["gris"], TINTAS["pesos"], TINTAS["previa"]]
        a2.barh(range(3), valores, color=colores, height=0.55)
        for i, v in enumerate(valores):
            a2.text(v, i, f" {v:.3f}", va="center", fontsize=9.5, fontfamily=familia("cifras"))
        a2.set_yticks(range(3), nombres, fontsize=9)
        a2.invert_yaxis()
        a2.set_xlim(min(0.5, min(valores) - 0.02), max(valores) + 0.06)
        a2.set_title(f"ROC-AUC tras {epocas} épocas", fontsize=11.5)
        a2.grid(axis="y", visible=False)
        fig.tight_layout()
        tipografia(fig)
        a1b.tick_params(axis="y", labelfontfamily=familia("cifras"))
        return fig

    opciones_capas = sorted(set(CAPAS_RED) | {elegida[0]}, key=lambda c: (len(c), c))
    opciones_alfa = sorted(set(ALFAS_RED) | {elegida[1]})
    controles = {
        "ocultas": w.ToggleButtons(options=[(" → ".join(map(str, c)), c) for c in opciones_capas], value=elegida[0],
                                   description="Capas ocultas", style={"description_width": "initial",
                                                                       "button_width": "64px"}),
        "alfa": w.SelectionSlider(options=[(f"{a:g}", a) for a in opciones_alfa], value=elegida[1],
                                  description="Regularización α", continuous_update=False, **_estilo_control()),
    }
    explicacion = (f"En la pasada didáctica la red tiene una capa oculta de <b>{elegida[0][0]}</b> unidades y α = "
                   f"<b>{elegida[1]:g}</b>. Aquí entrena <b>{epocas}</b> épocas sin parada temprana para que se vea el "
                   "sobreajuste: Más unidades o menos α bajan la pérdida de entrenamiento sin mejorar la validación. "
                   f"{_texto_submuestra(d)}")
    return Panel("Red neuronal", "Tamaño de la capa oculta y regularización", explicacion, controles, dibujar)


# ----------------------------------------------------------------------
# 3. Bayes
# ----------------------------------------------------------------------
def _iconos(ax, vp, fp, fn, total=MUESTRAS_ICONOS, columnas=40):
    """Diagrama de 1000 cuadritos: verdaderos positivos, falsos positivos, falsos negativos y el resto."""
    colores = ([TINTAS["sensibilidad"]] * vp + [TINTAS["falsos"]] * fp + ["#A9CDB4"] * fn
               + [FILETE] * (total - vp - fp - fn))
    filas = int(np.ceil(total / columnas))
    for i, color in enumerate(colores):
        fila, col = divmod(i, columnas)
        ax.add_patch(Rectangle((col, filas - 1 - fila), 0.82, 0.82, color=color, lw=0))
    ax.set_xlim(-0.2, columnas)
    ax.set_ylim(-0.2, filas)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_bayes(d):
    """Previa p (escala logarítmica), sensibilidad S y especificidad E: VPP y
    probabilidad de falso positivo en grande, y 1000 muestras como cuadritos
    (verdaderos positivos, falsos positivos, falsos negativos). El menú
    carga los valores de cada ensayo tal como los estimó la etapa 04."""
    ensayos = {e["ensayo"]: e for e in sorted(d.interactivo["ensayos"], key=lambda e: e["vpp"])}
    primero = next(iter(ensayos.values()))

    def dibujar(ensayo, p, s, e):
        vpp = float(bayes.valor_predictivo_positivo(p, s, e))
        falso = float(bayes.prob_falso_positivo(p, s, e))
        contaminadas = int(round(MUESTRAS_ICONOS * p))
        vp = int(round(contaminadas * s))
        fp = int(round((MUESTRAS_ICONOS - contaminadas) * (1 - e)))
        fn = contaminadas - vp
        ref = ensayos[ensayo]
        fig, (a1, a2) = _figura(3.5, 2, gridspec_kw={"width_ratios": [1, 1.45]})
        a1.axis("off")
        a1.text(0, 0.98, "VPP = P(contaminada | positivo)", fontsize=10, color=TENUE, va="top")
        a1.text(0, 0.86, _pct(vpp), fontsize=36, color=TINTAS["sensibilidad"], va="top", fontfamily=familia("titulos"))
        a1.text(0, 0.56, "P(falso positivo) = 1 − VPP", fontsize=10, color=TENUE, va="top")
        a1.text(0, 0.44, _pct(falso), fontsize=36, color=TINTAS["falsos"], va="top", fontfamily=familia("titulos"))
        a1.text(0, 0.12, f"{ensayo} en la etapa 04:\np = {_pct(ref['prevalencia_corregida'])}, VPP = {_pct(ref['vpp'])}",
                fontsize=8.5, color=TENUE, va="top", wrap=True)
        texto_coloreado(a2, [("De 1000 muestras: ", None), (f"{vp} verdaderos positivos", TINTAS["sensibilidad"]),
                             (", ", None), (f"{fp} falsos positivos", TINTAS["falsos"]), (" y ", None),
                             (f"{fn} sin detectar", "#5E9A72")], ubicacion="lower left", tamano=9.5,
                        bbox_to_anchor=(0, 1.0), bbox_transform=a2.transAxes)
        _iconos(a2, vp, fp, fn)
        fig.tight_layout()
        return fig

    def control_deslizador(clase, **opciones):
        return clase(continuous_update=False, **_estilo_control("330px"), **opciones)

    controles = {
        "ensayo": w.Dropdown(options=list(ensayos), value=primero["ensayo"], description="Ensayo",
                             **_estilo_control("330px")),
        "p": control_deslizador(w.FloatLogSlider, value=max(primero["prevalencia_corregida"], 1e-3), base=10,
                                min=-3, max=float(np.log10(0.3)), step=0.02, readout_format=".2%",
                                description="Previa p"),
        "s": control_deslizador(w.FloatSlider, value=primero["sensibilidad"], min=0.5, max=1.0, step=0.001,
                                readout_format=".3f", description="Sensibilidad S"),
        "e": control_deslizador(w.FloatSlider, value=primero["especificidad"], min=0.8, max=1.0, step=0.001,
                                readout_format=".3f", description="Especificidad E"),
    }
    explicacion = ("Al elegir un ensayo se cargan la sensibilidad y la especificidad que estimamos con sus controles en "
                   "la etapa 04 y la previa corregida de su desarrollo. Después se puede mover la previa: Con una "
                   "contaminación rara hasta un ensayo muy específico puede dar más falsos positivos que verdaderos.")
    panel = Panel("Bayes", "¿Qué tan probable es que un positivo sea falso?", explicacion, controles, dibujar)

    def cargar(cambio):
        ref = ensayos[cambio["new"]]
        with panel.en_lote():
            controles["p"].value = max(ref["prevalencia_corregida"], 1e-3)
            controles["s"].value = ref["sensibilidad"]
            controles["e"].value = ref["especificidad"]

    controles["ensayo"].unobserve(panel._al_cambiar, names="value")
    controles["ensayo"].observe(cargar, names="value")
    return panel


# ----------------------------------------------------------------------
# 4. K-means
# ----------------------------------------------------------------------
def panel_kmeans(d):
    """k = 2…8 sobre la misma muestra de perfiles de la animación: nube en
    dos componentes principales coloreada por grupo, silueta del k actual y
    curva de silueta según k (muestra y matriz completa de la etapa 05)."""
    from sklearn.metrics import silhouette_score

    X = np.asarray(d.kmeans_muestra, dtype=float)
    plano = (X - d.kmeans_media) @ d.kmeans_componentes.T
    elegido = int(d.kmeans["k"])
    completa = {f["k"]: f["silueta"] for f in d.interactivo["kmeans_exploracion"] if f["k"] in GRUPOS}
    memoria = {}

    def agrupar(k):
        if k not in memoria:
            centros, etiquetas = pasos_kmeans(X, k)
            memoria[k] = (centros[-1], etiquetas[-1], float(silhouette_score(X, etiquetas[-1], random_state=SEMILLA)),
                          len(centros) - 1)
        return memoria[k]

    siluetas = {k: agrupar(k)[2] for k in GRUPOS}
    paleta = [COLORES[c] for c in ("azul", "naranja", "verde", "rojo", "morado", "gris")] + ["#A07A3B", "#3E8E8E"]

    def dibujar(k):
        centros, etiquetas, silueta, pasos = agrupar(k)
        fig, (a1, a2) = _figura(3.6, 2, gridspec_kw={"width_ratios": [1.25, 1]})
        colores = np.array(paleta)[etiquetas % len(paleta)]
        a1.scatter(plano[:, 0], plano[:, 1], s=11, c=colores, alpha=0.6, linewidths=0)
        c2 = (centros - d.kmeans_media) @ d.kmeans_componentes.T
        a1.scatter(c2[:, 0], c2[:, 1], s=150, marker="X", c=np.array(paleta)[np.arange(k) % len(paleta)],
                   edgecolor=TINTA, linewidth=1.1, zorder=5)
        a1.set_xlabel("componente principal 1")
        a1.set_ylabel("componente principal 2")
        a1.set_title(f"k = {k}: Silueta {silueta:.3f} ({pasos} iteraciones)", fontsize=11.5)
        ks = list(GRUPOS)
        a2.plot(ks, [siluetas[i] for i in ks], "o-", color=TINTAS["previa"], ms=4, label=f"muestra ({len(X)} filas)")
        if completa:
            a2.plot(list(completa), list(completa.values()), "s--", color=TENUE, ms=3, lw=1,
                    label="matriz completa (etapa 05)")
        a2.plot([k], [silueta], "o", color=TINTAS["costo"], ms=9, zorder=5)
        _marca_elegida(a2, elegido, f"elegido k = {elegido}")
        a2.set_xlabel("número de grupos k")
        a2.set_ylabel("silueta (más es mejor)")
        a2.legend(loc="lower right", fontsize=8)
        a2.set_title("Silueta según k", fontsize=11.5)
        fig.tight_layout()
        return fig

    controles = {"k": w.IntSlider(value=elegido, min=min(GRUPOS), max=max(GRUPOS), step=1, description="Grupos k",
                                  continuous_update=False, **_estilo_control())}
    explicacion = (f"En la etapa 05 elegimos <b>k = {elegido}</b> perfiles de muestra. Al cambiar k se rehace el mismo "
                   f"K-means paso a paso de la animación sobre la misma muestra de {len(X)} perfiles y la curva muestra "
                   "la silueta de cada k (cohesión frente a separación). La curva de la muestra parte de una sola "
                   "inicialización así que es más ruidosa que la de la matriz completa de la etapa 05, que es con la que "
                   "elegimos k. Si la silueta es baja en todo el rango los grupos son poco nítidos.")
    return Panel("Agrupamiento", "Número de grupos de K-means", explicacion, controles, dibujar)


# ----------------------------------------------------------------------
# 5. KNN
# ----------------------------------------------------------------------
def _proba_vecinos(distancias, indices, y, k, pesos):
    """P(clase 1) de KNN con los k primeros vecinos ya buscados (igual que `predict_proba`)."""
    votos = y[indices[:, :k]].astype(float)
    if pesos == "uniform":
        return votos.mean(axis=1)
    d = distancias[:, :k]
    with np.errstate(divide="ignore"):
        inverso = 1 / d
    ceros = (d == 0)
    filas_cero = ceros.any(axis=1)
    inverso[filas_cero] = ceros[filas_cero].astype(float)   # distancia 0: solo cuentan esos vecinos
    return (inverso * votos).sum(axis=1) / inverso.sum(axis=1)


def panel_knn(d):
    """k vecinos (impares, pasos casi logarítmicos) y ponderación: ROC-AUC de
    entrenamiento y de validación según k sobre la submuestra fija. Con k
    pequeño la diferencia muestra el sobreajuste; con k grande, el subajuste."""
    from sklearn.neighbors import NearestNeighbors

    X_ent, y_ent, X_val, y_val = _submuestras(d)
    tope = min(max(VECINOS), len(y_ent))
    vecinos = NearestNeighbors(n_neighbors=tope).fit(X_ent)
    busqueda = {"validacion": vecinos.kneighbors(X_val), "entrenamiento": vecinos.kneighbors(X_ent)}
    elegido = d.interactivo["knn"]
    opciones = sorted({k for k in VECINOS if k <= tope} | {elegido["k"]})
    memoria = {}

    def curvas(pesos):
        if pesos not in memoria:
            memoria[pesos] = {
                parte: np.array([metricas.roc_auc(y, _proba_vecinos(*busqueda[parte], y_ent, k, pesos)) for k in opciones])
                for parte, y in (("validacion", y_val), ("entrenamiento", y_ent))}
        return memoria[pesos]

    def dibujar(k, pesos):
        c = curvas(pesos)
        i = opciones.index(k)
        fig, ax = _figura(3.4)
        ax.plot(opciones, c["entrenamiento"], "o-", color=TENUE, ms=3, lw=1.2, label="entrenamiento (las mismas filas)")
        ax.plot(opciones, c["validacion"], "o-", color=TINTAS["previa"], ms=4, label="validación")
        ax.plot([k], [c["validacion"][i]], "o", color=TINTAS["costo"], ms=10, zorder=5)
        _marca_elegida(ax, elegido["k"], f"pasada didáctica: k = {elegido['k']}, {NOMBRES_PESOS.get(elegido['pesos'], elegido['pesos'])}")
        ax.set_xscale("log")
        ax.set_xticks(opciones, [str(v) for v in opciones])
        ax.minorticks_off()
        ax.set_xlabel("vecinos k (escala logarítmica)")
        ax.set_ylabel("ROC-AUC")
        mejor = opciones[int(np.argmax(c["validacion"]))]
        ax.set_title(f"k = {k}, {NOMBRES_PESOS[pesos]}: ROC-AUC de validación {c['validacion'][i]:.3f} "
                     f"(el mejor k aquí es {mejor})", fontsize=11.5)
        ax.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()
        return fig

    controles = {
        "k": w.SelectionSlider(options=opciones, value=elegido["k"], description="Vecinos k", continuous_update=False,
                               **_estilo_control("380px")),
        "pesos": w.ToggleButtons(options=[(v, k) for k, v in NOMBRES_PESOS.items()], value=elegido["pesos"],
                                 description="Ponderación", style={"description_width": "initial",
                                                                  "button_width": "110px"}),
    }
    explicacion = ("Con <b>k = 1</b> cada fila de entrenamiento es su propio vecino: El ROC-AUC de entrenamiento es "
                   "perfecto y el de validación cae (sobreajuste). Con k muy grande el modelo promedia demasiado y "
                   f"pierde detalle (subajuste). {_texto_submuestra(d)}")
    return Panel("Clasificación", "KNN: número de vecinos", explicacion, controles, dibujar)


# ----------------------------------------------------------------------
# 6. Árbol de decisión
# ----------------------------------------------------------------------
def _dibujar_arbol(ax, arbol, nombres, profundidad=3):
    """Dibuja los primeros niveles de un `DecisionTreeClassifier` en el estilo de la casa:
    cada nodo es una regla "variable ≤ umbral" con el % de filas y una barra de la proporción
    de no conformes; las ramas más profundas se resumen como "… k hojas"."""
    t = arbol.tree_
    total = t.weighted_n_node_samples[0]
    tasa_global = t.value[0][0][1] / t.value[0][0].sum()
    posiciones, aristas, hoja = {}, [], [0]

    def hojas_debajo(nodo):
        if t.children_left[nodo] == -1:
            return 1
        return hojas_debajo(t.children_left[nodo]) + hojas_debajo(t.children_right[nodo])

    def ubicar(nodo, nivel):
        es_hoja = t.children_left[nodo] == -1
        if es_hoja or nivel == profundidad:
            posiciones[nodo] = (hoja[0], nivel, es_hoja)
            hoja[0] += 1
            return
        for hijo, rotulo in ((t.children_left[nodo], "sí"), (t.children_right[nodo], "no")):
            ubicar(hijo, nivel + 1)
            aristas.append((nodo, hijo, rotulo))
        x = (posiciones[t.children_left[nodo]][0] + posiciones[t.children_right[nodo]][0]) / 2
        posiciones[nodo] = (x, nivel, False)

    ubicar(0, 0)
    ancho_x = max(hoja[0] - 1, 1)
    ax.set_xlim(-0.6, ancho_x + 0.6)
    ax.set_ylim(profundidad + 0.75, -0.55)
    ax.axis("off")
    for padre, hijo, rotulo in aristas:
        x0, y0, _ = posiciones[padre]
        x1, y1, _ = posiciones[hijo]
        ax.plot([x0, x0, x1, x1], [y0 + 0.50, y0 + 0.62, y0 + 0.62, y1 - 0.30], color=FILETE, lw=0.9, zorder=1)
        ax.text(x0 + 0.5 * (x1 - x0), y0 + 0.60, rotulo, ha="center", va="bottom", fontsize=7.5, color=TENUE,
                style="italic", bbox=dict(facecolor=PAPEL, edgecolor="none", pad=0.6))
    for nodo, (x, nivel, es_hoja) in posiciones.items():
        valores = t.value[nodo][0]
        tasa = valores[1] / valores.sum()
        filas = t.weighted_n_node_samples[nodo] / total
        recortado = (not es_hoja) and t.children_left[nodo] != -1 and nivel == profundidad
        if t.children_left[nodo] != -1 and not recortado:
            variable = nombres[t.feature[nodo]].replace("_", " ")
            ax.text(x, nivel - 0.06, f"{variable} ≤ {t.threshold[nodo]:.2f}", ha="center", va="bottom",
                    fontsize=8.3, color=TINTA, fontweight="bold")
        elif recortado:
            ax.text(x, nivel - 0.06, f"… {hojas_debajo(nodo)} hojas", ha="center", va="bottom",
                    fontsize=8, color=TENUE, style="italic")
        else:
            ax.text(x, nivel - 0.06, "hoja", ha="center", va="bottom", fontsize=8, color=TENUE)
        color = TINTAS["falsos"] if tasa > tasa_global else TINTAS["conformes"]
        ancho = 0.30
        ax.add_patch(plt.Rectangle((x - ancho, nivel + 0.02), 2 * ancho, 0.08, color=FILETE, lw=0, zorder=2))
        ax.add_patch(plt.Rectangle((x - ancho, nivel + 0.02), 2 * ancho * min(tasa / 0.5, 1), 0.08, color=color,
                                   lw=0, zorder=3))
        ax.text(x, nivel + 0.15, f"{100 * filas:.1f} % de filas\n{100 * tasa:.1f} % no conformes", ha="center",
                va="top", fontsize=7.2, color=TENUE, linespacing=1.25)


def panel_arbol(d):
    """`ccp_alpha` recorre el camino de poda por costo-complejidad del árbol
    crecido sobre la submuestra: número de hojas y ROC-AUC de entrenamiento y
    validación según α, y los tres primeros niveles del árbol podado."""
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    X_ent, y_ent, X_val, y_val = _submuestras(d)
    hiper = d.interactivo["arbol"]
    nombres = d.interactivo["variables"]

    def crear(alfa):
        return DecisionTreeClassifier(min_samples_leaf=hiper["min_samples_leaf"], ccp_alpha=alfa, random_state=SEMILLA)

    camino = np.unique(crear(0.0).cost_complexity_pruning_path(X_ent, y_ent).ccp_alphas.clip(min=0))[:-1]
    posiciones = np.unique(np.round(np.linspace(0, len(camino) - 1, min(MAXIMO_ALFAS_ARBOL, len(camino)))).astype(int))
    alfas = [float(a) for a in camino[posiciones]]
    elegido = hiper["ccp_alpha"]
    inicial = min(alfas, key=lambda a: abs(np.log10(a + 1e-6) - np.log10(elegido + 1e-6)))
    memoria = {}

    def ajustar(alfa):
        if alfa not in memoria:
            arbol = crear(alfa).fit(X_ent, y_ent)
            memoria[alfa] = (arbol, arbol.get_n_leaves(), metricas.roc_auc(y_ent, arbol.predict_proba(X_ent)[:, 1]),
                             metricas.roc_auc(y_val, arbol.predict_proba(X_val)[:, 1]))
        return memoria[alfa]

    tabla = np.array([ajustar(a)[1:] for a in alfas])
    eje_x = np.array(alfas) + 1e-6   # α = 0 en escala logarítmica

    def dibujar(alfa):
        arbol, hojas, auc_ent, auc_val = ajustar(alfa)
        configurar_estilo()
        fig = plt.figure(figsize=(ANCHO, 9.6))
        rejilla = fig.add_gridspec(2, 1, height_ratios=[1, 2.9], hspace=0.30)
        a1 = fig.add_subplot(rejilla[0])
        a1.plot(eje_x, tabla[:, 1], "o-", color=TENUE, ms=3, lw=1.2, label="ROC-AUC entrenamiento")
        a1.plot(eje_x, tabla[:, 2], "o-", color=TINTAS["previa"], ms=4, label="ROC-AUC validación")
        a1.plot([alfa + 1e-6], [auc_val], "o", color=TINTAS["costo"], ms=10, zorder=5)
        _marca_elegida(a1, elegido + 1e-6, f"rejilla de la etapa 04: {elegido:.2g}")
        a1.set_xscale("log")
        a1.set_xlabel("ccp_alpha (escala logarítmica; α = 0 a la izquierda)")
        a1.set_ylabel("ROC-AUC")
        a1b = a1.twinx()
        a1b.step(eje_x, tabla[:, 0], where="mid", color=TINTAS["pesos"], lw=1, alpha=0.7)
        a1b.set_ylabel("hojas", color=TINTAS["pesos"])
        a1b.grid(False)
        a1b.spines["right"].set_visible(True)
        a1.legend(loc="lower left", fontsize=8.5)
        a1.set_title(f"ccp_alpha = {alfa:.2g}: {hojas} hojas, ROC-AUC validación {auc_val:.3f} "
                     f"(entrenamiento {auc_ent:.3f})", fontsize=11.5)
        a2 = fig.add_subplot(rejilla[1])
        _dibujar_arbol(a2, arbol, nombres, profundidad=3)
        a2.set_title("Tres primeros niveles del árbol podado. Barra: % de no conformes (rojo si supera la tasa "
                     "global). Umbrales en escala estandarizada", fontsize=10)
        tipografia(fig)
        a1b.tick_params(axis="y", labelfontfamily=familia("cifras"))
        return fig

    controles = {"alfa": w.SelectionSlider(options=[(f"{a:.2g}", a) for a in alfas], value=inicial,
                                           description="Poda ccp_alpha", continuous_update=False,
                                           **_estilo_control("420px"))}
    explicacion = (f"Sin poda (α = 0) el árbol crece hasta hojas de <b>{hiper['min_samples_leaf']}</b> filas y "
                   "memoriza el entrenamiento. Al subir α se cortan primero las ramas que menos reducen la impureza por "
                   "cada hoja que agregan, hasta que solo queda la raíz. La rejilla de la etapa 04 eligió α con "
                   f"validación cruzada (línea punteada). {_texto_submuestra(d)}")
    return Panel("Clasificación", "Árbol de decisión: poda por costo-complejidad", explicacion, controles, dibujar)


# ----------------------------------------------------------------------
# Taguchi: modelo aditivo frente a lo observado
# ----------------------------------------------------------------------
def _contribuciones(taguchi, niveles):
    """S/N que predice el modelo aditivo para `niveles` y lo que aporta cada factor (nivel − media)."""
    media = taguchi["media_sn"]
    aportes = {f: taguchi["efectos"][f][nivel] - media for f, nivel in niveles.items()}
    return media + sum(aportes.values()), aportes


def _corrida_igual(taguchi, niveles):
    return next((c for c in taguchi["corridas"] if all(c[f] == v for f, v in niveles.items())), None)


def panel_taguchi(taguchi, congelada=None, rotulo="Regresión", etapa="03"):
    """Un selector por factor del L16. Con cada combinación muestra lo que aporta cada nivel según el modelo
    aditivo (barras) y la S/N predicha frente a la observada en la corrida del L16 que tenga esa misma
    combinación (si existe) y frente a la corrida de confirmación del óptimo. No reajusta nada: todo sale de la
    sección `taguchi` del artefacto de la etapa (`efectos`, `media_sn`, `corridas`). `congelada` es la
    configuración que se congeló (niveles ya legibles), que se marca y es el estado inicial."""
    factores = taguchi["factores"]
    optimo = taguchi["optimo"]
    congelada = {f: congelada.get(f, optimo[f]) for f in factores} if congelada else dict(optimo)
    todas = [c["sn"] for c in taguchi["corridas"]]

    def dibujar(**niveles):
        predicha, aportes = _contribuciones(taguchi, niveles)
        corrida = _corrida_igual(taguchi, niveles)
        fig, (a1, a2) = _figura(4.3, 2, gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})
        nombres = list(aportes)[::-1]
        valores = [aportes[f] for f in nombres]
        a1.barh(nombres, valores, color=[TINTAS["sensibilidad"] if v >= 0 else TINTAS["falsos"] for v in valores],
                height=0.55)
        a1.axvline(0, color=TINTA, lw=0.8)
        for i, (f, v) in enumerate(zip(nombres, valores)):
            marca = "" if niveles[f] == congelada[f] else f" (congelado: {congelada[f]})"
            a1.annotate(f"{niveles[f]}{marca}", (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=8.5, color=TENUE)
        limite = max(0.5, 1.6 * max(abs(v) for v in valores))
        a1.set_xlim(-limite, limite)
        a1.set_xlabel("aporte del nivel a la S/N (dB)")
        a1.set_title("Qué suma cada factor sobre la media"
                     + ("\n(la configuración congelada)" if niveles == congelada else ""), fontsize=11)
        a2.scatter(todas, np.zeros(len(todas)), marker="|", s=260, color=FILETE, zorder=1)
        filas = [("predicha", predicha, TINTAS["previa"])]
        if corrida is not None:
            filas.append((f"corrida {corrida['corrida']} del L16", corrida["sn"], TINTAS["conformes"]))
        filas.append(("confirmación del óptimo", taguchi["sn_observada"], TINTAS["costo"]))
        for j, (texto, valor, color) in enumerate(filas, start=1):
            a2.plot([valor], [j], "o", ms=9, color=color, zorder=3)
            a2.annotate(f"{texto}: {valor:.2f} dB", (valor, j), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=color)
        a2.axvspan(taguchi["sn_observada"] - taguchi["tolerancia"], taguchi["sn_observada"] + taguchi["tolerancia"],
                   color=TINTAS["costo"], alpha=0.10, lw=0)
        a2.set_yticks([0], ["16 corridas"])
        a2.set_ylim(-0.6, len(filas) + 0.9)
        margen = 0.4 + taguchi["tolerancia"]
        a2.set_xlim(min(todas + [predicha]) - margen, max(todas + [predicha, taguchi["sn_observada"]]) + margen)
        a2.set_xlabel("S/N (dB, más es mejor)")
        a2.grid(axis="y", visible=False)
        estado = ("Esta combinación está en el L16" if corrida is not None else
                  "Esta combinación no está en el L16: Solo hay predicción")
        a2.set_title(estado, fontsize=11)
        return fig

    controles = {f: w.Dropdown(options=list(niveles), value=congelada[f], description=f, **_estilo_control("230px"))
                 for f, niveles in factores.items()}
    explicacion = (f"El modelo aditivo de Taguchi predice la S/N de cualquier combinación como la media de las 16 "
                   f"corridas (<b>{taguchi['media_sn']:.2f} dB</b>) más lo que aporta el nivel elegido de cada factor. "
                   "Cambie los niveles y compare la predicción con lo observado: Si la combinación está entre las 16 "
                   "corridas aparece su S/N medida y la franja muestra la tolerancia de la corrida de confirmación "
                   f"(±{taguchi['tolerancia']:.2f} dB). El estado inicial es la configuración que congelamos en la "
                   f"etapa {etapa}.")
    return Panel(rotulo, "Taguchi: lo que predice el modelo aditivo frente a lo observado", explicacion, controles,
                 dibujar)
