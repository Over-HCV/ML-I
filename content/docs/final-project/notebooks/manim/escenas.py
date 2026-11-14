"""Escenas de Manim para la presentación: Bayes término a término y retropropagación en el MLP.

Responsabilidad única: animar, con el estilo editorial del proyecto (papel `PAPEL`, tinta `TINTA` y los
colores `TINTAS` de las ecuaciones de la presentación), dos ideas que en el notebook son una ecuación
coloreada. No calcula resultados: lee al renderizar las cifras que ya guardaron las etapas
(`artefactos/05_clasificacion.json` para la sensibilidad, la especificidad, la previa y el VPP del ensayo
de esterilidad; `artefactos/presentacion_datos.npz` para la arquitectura y la α de la red). Los colores y el
papel se leen del código fuente de `proyecto/reporte/` (sin importarlo), para que no haya dos copias.

Manim no es dependencia del proyecto: se instala en un entorno aparte y los videos ya renderizados se
versionan en `figuras/manim/`. Para rehacerlos, desde la carpeta del proyecto:

    uv venv ~/.cache/proyecto_manim_venv --python 3.12
    VIRTUAL_ENV=~/.cache/proyecto_manim_venv uv pip install manim   # trae PyAV (FFmpeg), pycairo y manimpango
    make manim                                   # o make manim MANIM_VENV=/otra/ruta; requiere latex y dvisvgm

`make manim` renderiza las dos escenas a 720p30 (`-qm`), las recodifica con x264 a CRF alto (cada MP4 queda
en unos 600 kB) y guarda el último cuadro como portada (`figuras/manim/<nombre>.png`). A mano, una escena
(`-ql` para pruebas rápidas), y luego la compresión con el Python del mismo entorno:

    manim -qm --media_dir /tmp/medios notebooks/manim/escenas.py EscenaBayes -o bayes
    python notebooks/manim/escenas.py /tmp/medios/videos/escenas/720p30 figuras/manim

Las fuentes (Fraunces, IBM Plex Sans, JetBrains Mono) se registran desde `~/.cache/proyecto_fuentes/`, la
carpeta donde las deja `proyecto/reporte/figuras.py`; si no están, Pango usa su fuente por omisión.
"""
import ast
import json
from pathlib import Path

import manimpango
import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Circle, Create, FadeIn, FadeOut, GrowArrow,
                   LaggedStart, Line, MathTex, RoundedRectangle, Scene, ShowPassingFlash, SurroundingRectangle,
                   Text, TransformMatchingTex, VGroup, Write, config)

RAIZ = Path(__file__).resolve().parents[2]
ARTEFACTOS = RAIZ / "artefactos"
CARPETA_FUENTES = Path.home() / ".cache" / "proyecto_fuentes"


def _constantes(ruta, nombres):
    """Valores literales asignados en el nivel superior de un módulo, leídos con `ast` (sin importarlo)."""
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    valores = {}
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 and isinstance(nodo.targets[0], ast.Name):
            if nodo.targets[0].id in nombres:
                valores[nodo.targets[0].id] = ast.literal_eval(nodo.value)
    return valores


_ESTILO = _constantes(RAIZ / "proyecto" / "reporte" / "figuras.py", {"PAPEL", "TINTA", "TENUE", "FILETE"})
TINTAS = _constantes(RAIZ / "proyecto" / "reporte" / "graficos_presentacion.py", {"TINTAS"})["TINTAS"]
PAPEL, TINTA, TENUE, FILETE = (_ESTILO[n] for n in ("PAPEL", "TINTA", "TENUE", "FILETE"))

for _archivo in sorted(CARPETA_FUENTES.glob("*-*.ttf")):   # estáticas: Fraunces-normal.ttf, IBMPlexSans-bold.ttf...
    manimpango.register_font(str(_archivo))
TITULOS, TEXTO, CIFRAS = "Fraunces", "IBM Plex Sans", "JetBrains Mono"

config.background_color = PAPEL


def _texto(contenido, tamano=24, color=TINTA, fuente=TEXTO, **opciones):
    return Text(contenido, font=fuente, font_size=tamano, color=color, **opciones)


def _encabezado(titulo, subtitulo):
    arriba = _texto(titulo, 40, fuente=TITULOS).to_edge(UP, buff=0.35)
    abajo = _texto(subtitulo, 20, TENUE).next_to(arriba, DOWN, buff=0.15)
    return VGroup(arriba, abajo)


def _cifra(x, decimales=4):
    return f"{x:.{decimales}f}"


def _ensayo(nombre):
    clasificacion = json.loads((ARTEFACTOS / "05_clasificacion.json").read_text(encoding="utf-8"))
    return next(e for e in clasificacion["bayes"]["vpp_por_ensayo"] if e["ensayo"] == nombre)


def _meta_red():
    with np.load(ARTEFACTOS / "presentacion_datos.npz") as archivo:
        return json.loads(str(archivo["meta"]))["mlp"]


# ----------------------------------------------------------------------
# Escena 1 · Bayes, término a término
# ----------------------------------------------------------------------
def _nodo(nombre, cantidad, color, ancho=2.5):
    rotulo = VGroup(_texto(nombre, 20), _texto(cantidad, 22, color, CIFRAS)).arrange(RIGHT, buff=0.25)
    caja = RoundedRectangle(corner_radius=0.12, width=max(ancho, rotulo.width + 0.4), height=0.62,
                            stroke_color=color, stroke_width=2.5, fill_color=PAPEL, fill_opacity=1)
    return VGroup(caja, rotulo.move_to(caja))


def _rama(origen, destino, rotulo, color):
    linea = Line(origen.get_right(), destino.get_left(), buff=0.08, stroke_color=color, stroke_width=3)
    arriba = destino.get_y() > origen.get_y()
    etiqueta = rotulo.next_to(linea.point_from_proportion(0.6), (UP + LEFT) if arriba else (DOWN + LEFT), buff=0.02)
    return linea, etiqueta


class EscenaBayes(Scene):
    """P(contaminada | positivo) en el ensayo de esterilidad: árbol de 1000 muestras → fórmula → cifras."""

    def construct(self):
        e = _ensayo("esterilidad")
        S, E, p = e["sensibilidad"], e["especificidad"], e["prevalencia_corregida"]
        total = 1000
        contaminadas = round(total * p)
        limpias = total - contaminadas
        vp = round(total * p * S)
        fp = round(total * (1 - p) * (1 - E))
        fn, vn = contaminadas - vp, limpias - fp
        vpp = S * p / (S * p + (1 - E) * (1 - p))
        assert abs(vpp - e["vpp"]) < 1e-9, "el VPP de la fórmula no coincide con el de la etapa 05"
        verde, rojo, azul, gris = TINTAS["sensibilidad"], TINTAS["falsos"], TINTAS["previa"], TINTAS["conformes"]

        encabezado = _encabezado("Bayes, término a término",
                                 f"ensayo de esterilidad · {total} muestras imaginarias con las cifras de la etapa 05")
        self.play(FadeIn(encabezado, shift=0.2 * DOWN), run_time=1.2)

        # Árbol: previa y luego resultado del ensayo
        raiz = _nodo("muestras", str(total), TINTA, 2.3).move_to([-5.55, 0.45, 0])
        n_cont = _nodo("contaminadas", str(contaminadas), azul).move_to([-2.35, 1.55, 0])
        n_limp = _nodo("limpias", str(limpias), gris).move_to([-2.35, -0.65, 0])
        hojas = {
            "vp": _nodo("positivo", str(vp), verde, 2.2).move_to([2.2, 2.1, 0]),
            "fn": _nodo("negativo", str(fn), TENUE, 2.2).move_to([2.2, 1.0, 0]),
            "fp": _nodo("positivo", str(fp), rojo, 2.2).move_to([2.2, -0.1, 0]),
            "vn": _nodo("negativo", str(vn), TENUE, 2.2).move_to([2.2, -1.2, 0]),
        }
        nombres_hojas = {
            "vp": _texto("verdaderos positivos", 20, verde).next_to(hojas["vp"], RIGHT, buff=0.25),
            "fn": _texto("falsos negativos", 20, TENUE).next_to(hojas["fn"], RIGHT, buff=0.25),
            "fp": _texto("falsos positivos", 20, rojo).next_to(hojas["fp"], RIGHT, buff=0.25),
            "vn": _texto("verdaderos negativos", 20, TENUE).next_to(hojas["vn"], RIGHT, buff=0.25),
        }
        r_cont = _rama(raiz, n_cont, MathTex("p", "=", _cifra(p), font_size=28, color=azul), azul)
        r_limp = _rama(raiz, n_limp, MathTex("1-p", font_size=28, color=azul), azul)
        r_vp = _rama(n_cont, hojas["vp"], MathTex("S", "=", _cifra(S), font_size=28, color=verde), verde)
        r_fn = _rama(n_cont, hojas["fn"], MathTex("1-S", font_size=28, color=TENUE), TENUE)
        r_fp = _rama(n_limp, hojas["fp"], MathTex("1-E", "=", _cifra(1 - E), font_size=28, color=rojo), rojo)
        r_vn = _rama(n_limp, hojas["vn"], MathTex("E", "=", _cifra(E), font_size=28, color=TENUE), TENUE)

        nota = _texto("La previa p separa las contaminadas de las limpias", 24).move_to([0, -2.6, 0])
        self.play(FadeIn(raiz), run_time=0.6)
        self.play(*(Create(r[0]) for r in (r_cont, r_limp)), FadeIn(n_cont), FadeIn(n_limp),
                  *(FadeIn(r[1]) for r in (r_cont, r_limp)), FadeIn(nota), run_time=1.6)
        self.wait(1.6)
        nota_ensayo = _texto("El ensayo acierta con S en las contaminadas y se equivoca con 1 − E en las limpias",
                             24).move_to(nota)
        ramas = (r_vp, r_fn, r_fp, r_vn)
        self.play(LaggedStart(*(AnimationGroup(Create(r[0]), FadeIn(r[1]), FadeIn(h), FadeIn(nombres_hojas[k]))
                                for r, (k, h) in zip(ramas, hojas.items())), lag_ratio=0.25),
                  FadeOut(nota), FadeIn(nota_ensayo), run_time=2.6)
        self.wait(2.0)

        # Solo los positivos: los conteos se vuelven la fórmula
        apagados = VGroup(hojas["fn"], hojas["vn"], nombres_hojas["fn"], nombres_hojas["vn"], *r_fn, *r_vn)
        marcos = VGroup(*(SurroundingRectangle(VGroup(hojas[k], nombres_hojas[k]), color=c, buff=0.1,
                                               corner_radius=0.1, stroke_width=2) for k, c in (("vp", verde), ("fp", rojo))))
        conteo = MathTex(r"P(\text{contaminada}\mid\text{positivo})", "=", "{", str(vp), r"\over", str(vp), "+",
                         str(fp), "}", font_size=40, color=TINTA).move_to([0, -3.0, 0])
        for i, c in ((3, verde), (5, verde), (7, rojo)):
            conteo[i].set_color(c)
        self.play(apagados.animate.set_opacity(0.25), Create(marcos), FadeOut(nota_ensayo), run_time=1.2)
        self.play(Write(conteo), run_time=1.6)
        self.wait(2.0)

        division = _texto(f"÷ {total} en el numerador y en el denominador", 20, TENUE)
        formula = MathTex(r"\mathrm{VPP}", "=", "{", "S", r"\,", "p", r"\over", "S", r"\,", "p", "+", r"(1-E)",
                          r"(1-", "p", ")", "}", font_size=42, color=TINTA)
        for i, c in ((3, verde), (5, azul), (7, verde), (9, azul), (11, rojo), (12, rojo), (13, azul), (14, rojo)):
            formula[i].set_color(c)
        numeros = MathTex("=", "{", _cifra(S), r"\cdot", _cifra(p), r"\over", _cifra(S), r"\cdot", _cifra(p), "+",
                          _cifra(1 - E), r"\cdot", _cifra(1 - p), "}", "=", _cifra(e["vpp"], 3), font_size=40, color=TINTA)
        for i, c in ((2, verde), (4, azul), (6, verde), (8, azul), (10, rojo), (12, rojo), (15, verde)):
            numeros[i].set_color(c)
        fila = VGroup(formula, numeros).arrange(RIGHT, buff=0.25)
        if fila.width > 13.4:
            fila.scale_to_fit_width(13.4)
        fila.move_to([0, -3.0, 0])
        division.next_to(formula, UP, buff=0.12).align_to(formula, LEFT)
        self.play(FadeIn(division), TransformMatchingTex(conteo, formula), run_time=2.0)
        self.wait(1.8)

        # Las cifras del ensayo de esterilidad
        self.play(FadeOut(division), Write(numeros[:-2]), run_time=2.0)
        self.play(Write(numeros[-2:]), run_time=0.8)
        self.wait(1.5)

        # Cierre: el falso positivo y por qué
        ecuaciones = VGroup(formula, numeros)
        falso = MathTex(r"P(\text{limpia}\mid\text{positivo})", "=", r"1-\mathrm{VPP}", "=",
                        _cifra(e["prob_falso_positivo"], 3), font_size=40, color=TINTA)
        falso[4].set_color(rojo)
        self.play(FadeOut(VGroup(raiz, n_cont, n_limp, *hojas.values(), *nombres_hojas.values(), marcos,
                                 *(m for r in (r_cont, r_limp, *ramas) for m in r))),
                  ecuaciones.animate.move_to([0, 0.95, 0]), run_time=1.4)
        falso.next_to(ecuaciones, DOWN, buff=0.55)
        self.play(Write(falso), run_time=1.4)
        por_cada = limpias / contaminadas
        cierre = _texto(
            f"Con una previa baja, hasta un ensayo muy específico (E = {_cifra(E, 3)}) da muchos falsos positivos:\n"
            f"hay unas {por_cada:.0f} limpias por cada contaminada, y su {100 * (1 - E):.1f} % de error "
            f"({fp} de cada {total}) compite\ncon los {vp} verdaderos positivos.", 24, TINTA,
            line_spacing=1.0, t2c={"falsos positivos": rojo, "verdaderos positivos": verde, "previa baja": azul})
        cierre.next_to(falso, DOWN, buff=0.6)
        if cierre.width > 13:
            cierre.scale_to_fit_width(13)
        self.play(FadeIn(cierre, shift=0.15 * UP), run_time=1.4)
        self.wait(5.0)


# ----------------------------------------------------------------------
# Escena 2 · Retropropagación en el perceptrón multicapa
# ----------------------------------------------------------------------
def _capa(x, dibujadas, color, radio=0.14, alto=4.2):
    ys = np.linspace(alto / 2, -alto / 2, dibujadas) if dibujadas > 1 else np.zeros(1)
    nodos = VGroup(*(Circle(radius=radio, stroke_color=color, stroke_width=2.5, fill_color=PAPEL, fill_opacity=1)
                     .move_to([x, y + 0.2, 0]) for y in ys))
    return nodos


def _conexiones(izquierda, derecha, color, ancho=1.0, opacidad=0.28):
    return VGroup(*(Line(a.get_center(), b.get_center(), stroke_color=color, stroke_width=ancho,
                         stroke_opacity=opacidad) for a in izquierda for b in derecha))


def _destello(lineas, color, al_reves=False):
    copias = [Line(l.get_end(), l.get_start()) if al_reves else Line(l.get_start(), l.get_end()) for l in lineas]
    return AnimationGroup(*(ShowPassingFlash(c.set_stroke(color, width=3.5), time_width=0.6) for c in copias))


class EscenaRetropropagacion(Scene):
    """Paso hacia adelante, pérdida, gradientes hacia atrás con la regla de la cadena y actualización de pesos."""

    def construct(self):
        red = _meta_red()
        n_entradas, ocultas, alfa = red["n_entradas"], red["ocultas"], red["alpha"]
        morado, rojo, azul = TINTAS["pesos"], TINTAS["falsos"], TINTAS["previa"]
        capas = " → ".join([f"{n_entradas} entradas", *(f"{h} ReLU" for h in ocultas), "1 sigmoide"])
        encabezado = _encabezado("Retropropagación en el perceptrón multicapa", f"{capas} · α = {alfa:g}")
        self.play(FadeIn(encabezado, shift=0.2 * DOWN), run_time=1.2)

        # Red esquemática (capas grandes recortadas, como en la figura de la arquitectura)
        entrada = _capa(-5.9, 6, TENUE, alto=3.6)
        oculta = _capa(-3.6, 8, morado, alto=4.2)
        salida = _capa(-1.3, 1, rojo, radio=0.2)
        primera = _conexiones(entrada, oculta, morado)
        segunda = _conexiones(oculta, salida, morado, ancho=1.4, opacidad=0.4)
        rotulos = VGroup(
            _texto(f"x · {n_entradas}", 20, TENUE).next_to(entrada, DOWN, buff=0.35),
            _texto(f"h · {ocultas[0]} ReLU", 20, morado).next_to(oculta, DOWN, buff=0.2),
            MathTex(r"\hat p", font_size=34, color=rojo).next_to(salida, RIGHT, buff=0.15),
        )
        rotulos[0].set_y(rotulos[1].get_y())
        self.play(Create(primera), Create(segunda), FadeIn(entrada), FadeIn(oculta), FadeIn(salida),
                  FadeIn(rotulos), run_time=2.0)

        # Columna de ecuaciones
        columna_x = 0.2

        def paso(titulo, color):
            return _texto(titulo, 22, color, TITULOS)

        def ubicar(objeto, anterior, separacion=0.22):
            objeto.next_to(anterior, DOWN, buff=separacion)
            objeto.set_x(columna_x + objeto.width / 2)
            if objeto.get_right()[0] > 6.9:
                objeto.scale_to_fit_width(6.9 - columna_x).set_x(columna_x + objeto.width / 2)
            return objeto

        t1 = paso("1 · hacia adelante", TINTA).move_to([columna_x, 2.55, 0], aligned_edge=LEFT)
        e_h = ubicar(MathTex("h", "=", r"\mathrm{ReLU}(", "W_1", "x", "+", "b_1", ")", font_size=32, color=TINTA), t1, 0.15)
        e_p = ubicar(MathTex(r"\hat p", "=", r"\sigma(", "w_2", r"^\top h", "+", "b_2", ")", font_size=32, color=TINTA), e_h, 0.12)
        for m in (e_h[3], e_h[6], e_p[3], e_p[6]):
            m.set_color(morado)
        e_p[2].set_color(rojo)
        self.play(FadeIn(t1), entrada.animate.set_fill(TENUE, 0.8), run_time=0.6)
        self.play(_destello(primera, morado), Write(e_h), run_time=1.6)
        self.play(oculta.animate.set_fill(morado, 0.75), run_time=0.4)
        self.play(_destello(segunda, morado), Write(e_p), run_time=1.3)
        self.play(salida.animate.set_fill(rojo, 0.85), run_time=0.4)

        e_l = ubicar(MathTex("L", "=", r"-\big[\,y\log\hat p+(1-y)\log(1-\hat p)\,\big]", "+", r"\alpha", r"\,\lVert", "W",
                             r"\rVert^2", font_size=30, color=TINTA), e_p, 0.3)
        e_l[0].set_color(rojo)
        e_l[4].set_color(azul)
        e_l[6].set_color(morado)
        objetivo = _texto("y = 1 si la muestra salió no conforme, 0 si no", 17, TENUE)
        objetivo.next_to(rotulos, DOWN, buff=0.3).align_to(rotulos[0], LEFT)
        anillo = Circle(radius=0.34, stroke_color=rojo, stroke_width=3).move_to(salida)
        self.play(Write(e_l), FadeIn(objetivo), Create(anillo), run_time=1.6)
        self.wait(1.2)

        # Hacia atrás
        t2 = ubicar(paso("2 · hacia atrás: regla de la cadena", rojo), e_l, 0.35)
        e_g2 = ubicar(MathTex(r"{\partial L \over \partial w_2}", "=", r"(\hat p - y)\,h", "+", r"2\,\alpha", r"\,w_2",
                              font_size=30, color=TINTA), t2, 0.15)
        e_g1 = ubicar(MathTex(r"{\partial L \over \partial W_1}", "=",
                              r"{\partial L \over \partial \hat p}\,{\partial \hat p \over \partial h}\,{\partial h \over \partial W_1}",
                              "+", r"2\,\alpha", r"\,W_1", font_size=30, color=TINTA), e_g2, 0.12)
        for g in (e_g2, e_g1):
            g[0].set_color(rojo)
            g[4].set_color(azul)
            g[5].set_color(morado)
        self.play(FadeIn(t2), FadeOut(anillo), run_time=0.5)
        self.play(_destello(segunda, rojo, al_reves=True), Write(e_g2), oculta.animate.set_fill(rojo, 0.35), run_time=1.6)
        self.play(_destello(primera, rojo, al_reves=True), Write(e_g1), entrada.animate.set_fill(rojo, 0.25), run_time=2.0)
        self.wait(1.2)

        # Actualización
        t3 = ubicar(paso("3 · actualización de los pesos", morado), e_g1, 0.35)
        e_a = ubicar(MathTex("W_1", r"\leftarrow", "W_1", "-", r"\eta", r"\,{\partial L \over \partial W_1}",
                             font_size=32, color=TINTA), t3, 0.15)
        e_a[0].set_color(morado)
        e_a[2].set_color(morado)
        e_a[5].set_color(rojo)
        azar = np.random.default_rng(0)   # solo para dibujar pesos que cambian; no es un resultado
        nuevos = [l.animate.set_stroke(width=float(azar.uniform(0.4, 3.2)), opacity=float(azar.uniform(0.2, 0.7)))
                  for l in (*primera, *segunda)]
        self.play(FadeIn(t3), run_time=0.4)
        self.play(Write(e_a), *nuevos, entrada.animate.set_fill(PAPEL, 1), oculta.animate.set_fill(PAPEL, 1),
                  salida.animate.set_fill(PAPEL, 1), run_time=1.8)
        nota = _texto(f"scikit-learn usa Adam: η se adapta peso a peso;\nel ciclo se repite por lotes durante "
                      f"{red['epocas']}\népocas (parada temprana: la mejor fue la {red['mejor_epoca']}).", 17, TENUE)
        nota.next_to(objetivo, DOWN, buff=0.15).align_to(objetivo, LEFT)
        self.play(FadeIn(nota), run_time=0.8)
        for _ in range(2):   # otra pasada completa, más rápida
            self.play(_destello(primera, morado), run_time=0.6)
            self.play(_destello(segunda, morado), run_time=0.4)
            self.play(_destello(segunda, rojo, al_reves=True), run_time=0.4)
            self.play(_destello(primera, rojo, al_reves=True), run_time=0.6)
        self.wait(3.0)


# ----------------------------------------------------------------------
# Compresión y portada (se ejecuta con el Python del entorno de Manim)
# ----------------------------------------------------------------------
CALIDAD_CRF = 32          # x264: más alto, más liviano (las escenas son planas y casi estáticas)
ANCHO_PORTADA = 960       # píxeles de la imagen de portada (último cuadro del video)
COLORES_PORTADA = 256      # con menos, la cuantización corre los tonos de TINTAS hacia el gris


def comprimir(entrada, salida, crf=CALIDAD_CRF):
    """Recodifica un MP4 de Manim con x264 (CRF alto, `faststart`) y guarda su último cuadro como portada PNG
    junto a él (mismo nombre, extensión .png). Devuelve (kB del video, kB de la portada, segundos)."""
    import av

    salida = Path(salida)
    with av.open(str(entrada)) as origen, av.open(str(salida), "w", options={"movflags": "+faststart"}) as destino:
        video = origen.streams.video[0]
        flujo = destino.add_stream("libx264", rate=video.average_rate)
        flujo.width, flujo.height, flujo.pix_fmt = video.width, video.height, "yuv420p"
        flujo.options = {"crf": str(crf), "preset": "slow", "tune": "animation"}
        ultimo = None
        for cuadro in origen.decode(video):
            ultimo = cuadro.to_image()
            for paquete in flujo.encode(av.VideoFrame.from_image(ultimo)):
                destino.mux(paquete)
        for paquete in flujo.encode():
            destino.mux(paquete)
        segundos = float(video.duration * video.time_base)
    alto = round(ultimo.height * ANCHO_PORTADA / ultimo.width)
    portada = ultimo.resize((ANCHO_PORTADA, alto)).quantize(colors=COLORES_PORTADA, method=2)   # 2 = octree
    portada.save(salida.with_suffix(".png"), optimize=True)
    return salida.stat().st_size / 1024, salida.with_suffix(".png").stat().st_size / 1024, segundos


if __name__ == "__main__":
    import sys

    # python escenas.py <carpeta con los MP4 de Manim> <carpeta de destino> [crf]
    origen_videos, destino_videos = Path(sys.argv[1]), Path(sys.argv[2])
    destino_videos.mkdir(parents=True, exist_ok=True)
    for nombre in ("bayes", "red"):
        kb_video, kb_portada, duracion = comprimir(origen_videos / f"{nombre}.mp4", destino_videos / f"{nombre}.mp4",
                                                   *(int(a) for a in sys.argv[3:4]))
        print(f"{nombre}.mp4: {duracion:.1f} s, {kb_video:.0f} kB; portada {kb_portada:.0f} kB")
