"""Verificación independiente del proyecto.

Se ejecuta desde la raíz del proyecto final con `make verificar`. Cada
comprobación imprime una línea «OK» o «FALLA» con su motivo, y el guion
termina con código distinto de cero si alguna falla. No escribe en `datos/`
ni en `artefactos/`: solo lee y recalcula.
"""
import ast
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

RAIZ = Path.cwd()
PAQUETE = RAIZ / "proyecto"
FUENTES = RAIZ / "notebooks" / "src"
CAPAS = {"base": 0, "datos": 1, "preparacion": 2, "variables": 3, "modelos": 4, "evaluacion": 5, "cribado": 6,
         "reporte": 7, "etapas": 8}
MODULO_PRUEBA = PAQUETE / "etapas" / "evaluacion_final.py"   # único módulo que puede leer la prueba
MAXIMO_LINEAS_CELDA = 8                                        # celdas de código cortas en los guiones
ETAPA_PRUEBA = "07_evaluacion_final"

# Números con punto decimal permitidos en las celdas de texto de las etapas: definiciones, ejemplos
# didácticos con valores hipotéticos y constantes de regla fijadas antes de ver resultados. Ninguno es
# un resultado.
PERMITIDOS_TEXTO = {
    "03_cribado_analisis.py": {"0.5"},           # Hopkins ≈ 0.5 sin estructura (definición)
    "06_agrupamiento.py": {"0.5"},               # Hopkins > 0.5 (definición)
    "05_clasificacion_bayes.py": {
        "1.5",                                    # ejemplo de razón de odds
        "0.5",                                    # ROC-AUC de un modelo que no ordena (definición)
        "0.005",                                  # tolerancia de empate de la regla de elección
        "0.2",                                    # ejemplo de probabilidad calibrada
        "0.02", "0.50",                           # rejilla de umbrales declarada
        "0.98", "0.05", "0.95",                   # ejemplo didáctico del efecto de la especificidad
    },
}

resultados = []


def comprobar(nombre, funcion):
    """Ejecuta una comprobación; `funcion` devuelve un detalle o lanza AssertionError."""
    inicio = time.perf_counter()
    try:
        detalle = funcion()
        resultados.append(True)
        print(f"OK     {nombre}" + (f" · {detalle}" if detalle else "") + f" ({time.perf_counter() - inicio:.1f} s)")
    except AssertionError as error:
        resultados.append(False)
        print(f"FALLA  {nombre} · {error}")
    except Exception as error:  # noqa: BLE001 — cualquier error es una falla de la comprobación
        resultados.append(False)
        print(f"FALLA  {nombre} · error inesperado: {type(error).__name__}: {error}")


def modulos():
    return sorted(p for p in PAQUETE.rglob("*.py") if "__pycache__" not in p.parts)


def capa_de(ruta):
    partes = ruta.relative_to(PAQUETE).parts
    return partes[0] if len(partes) > 1 else None


def importaciones(ruta):
    """Lista de módulos importados (nombres absolutos) por un archivo."""
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    nombres = []
    paquete = ".".join(("proyecto",) + ruta.relative_to(PAQUETE).parts[:-1]) if PAQUETE in ruta.parents else ""
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            nombres.extend(a.name for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom):
            if nodo.level:
                base = ".".join(paquete.split(".")[: len(paquete.split(".")) - nodo.level + 1])
                modulo = f"{base}.{nodo.module}" if nodo.module else base
            else:
                modulo = nodo.module or ""
            nombres.append(modulo)
            nombres.extend(f"{modulo}.{a.name}" for a in nodo.names)
    return nombres


# ----------------------------------------------------------------------
# Arquitectura
# ----------------------------------------------------------------------
def regla_de_capas():
    violaciones = []
    for ruta in modulos():
        propia = capa_de(ruta)
        if propia is None:
            continue
        for nombre in importaciones(ruta):
            partes = nombre.split(".")
            if partes[0] != "proyecto" or len(partes) < 2 or partes[1] not in CAPAS:
                continue
            if CAPAS[partes[1]] > CAPAS[propia]:
                violaciones.append(f"{ruta.relative_to(RAIZ)} importa {nombre}")
    assert not violaciones, "; ".join(sorted(set(violaciones)))
    return f"{len(modulos())} archivos revisados"


def simulacion_aislada():
    archivos = modulos() + sorted(FUENTES.glob("*.py")) + sorted((RAIZ / "verificacion").glob("*.py"))
    infractores = []
    for ruta in archivos:
        if PAQUETE / "datos" in ruta.parents:
            continue
        if any(n.endswith("simulacion") or ".simulacion." in n for n in importaciones(ruta)):
            infractores.append(str(ruta.relative_to(RAIZ)))
    assert not infractores, "importan la simulación: " + ", ".join(infractores)
    return f"{len(archivos)} archivos revisados"


def encabezados():
    malos = []
    for ruta in modulos():
        texto = ruta.read_text(encoding="utf-8")
        if ruta.name == "__init__.py" and not texto.strip():
            continue  # paquetes vacíos: no son módulos con responsabilidad propia
        doc = ast.get_docstring(ast.parse(texto)) or ""
        coincide = re.match(r"Capa (\d+) · \S", doc)
        capa = capa_de(ruta)
        if not coincide:
            malos.append(f"{ruta.relative_to(RAIZ)} (sin «Capa N ·»)")
        elif capa is not None and int(coincide.group(1)) != CAPAS[capa]:
            malos.append(f"{ruta.relative_to(RAIZ)} (dice capa {coincide.group(1)}, está en {capa})")
        if "Responsabilidad única:" not in doc:
            malos.append(f"{ruta.relative_to(RAIZ)} (sin «Responsabilidad única:»)")
    assert not malos, "; ".join(malos)


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
def generador_determinista():
    from proyecto.datos import carga

    primera, _ = carga.obtener_tablas_crudas("simulacion")
    segunda, _ = carga.obtener_tablas_crudas("simulacion")
    h1, h2 = carga.huella(primera), carga.huella(segunda)
    assert h1 == h2, f"huellas distintas: {h1} y {h2}"
    return f"huella {h1}"


def festivos():
    from proyecto.datos import calendario

    # Ley 51 de 1983 (traslado al lunes siguiente) y Pascua de 2025 = 20 de abril.
    assert calendario.domingo_de_pascua(2025) == date(2025, 4, 20), "Pascua de 2025 mal calculada"
    esperados = {
        date(2025, 1, 6): "Reyes Magos (lunes)",
        date(2025, 3, 24): "San José (19 de marzo, miércoles → lunes)",
        date(2025, 4, 17): "Jueves Santo (Pascua − 3)",
        date(2025, 4, 18): "Viernes Santo (Pascua − 2)",
        date(2025, 6, 2): "Ascensión (Pascua + 39 → lunes)",
        date(2025, 6, 23): "Corpus Christi (Pascua + 60 → lunes)",
        date(2025, 6, 30): "Sagrado Corazón y San Pedro (coinciden)",
        date(2025, 8, 18): "Asunción (15 de agosto, viernes → lunes)",
        date(2025, 11, 17): "Independencia de Cartagena (11 de noviembre, martes → lunes)",
    }
    no_festivos = [date(2025, 3, 19), date(2025, 8, 15), date(2025, 11, 11)]
    del_anio = calendario.festivos_colombia(2025)
    faltan = [f"{f} {n}" for f, n in esperados.items() if f not in del_anio]
    sobran = [str(f) for f in no_festivos if f in del_anio]
    assert not faltan and not sobran, f"faltan {faltan}; sobran {sobran}"
    return f"{len(esperados)} festivos y {len(no_festivos)} días trasladados"


def ortogonalidad_l16():
    from proyecto.cribado import arreglos

    arreglo = arreglos.arreglo_l16()
    assert len(arreglo) == 16, f"{len(arreglo)} corridas"
    assert arreglos.verificar_ortogonalidad(arreglo), "el L16 no está balanceado por pares"
    roto = arreglo.copy()
    columna = arreglos.COLUMNAS_DOS_NIVELES[0]
    otro_nivel = next(v for v in sorted(set(arreglo[columna])) if v != arreglo[columna].iloc[0])
    roto.loc[roto.index[0], columna] = otro_nivel
    assert not arreglos.verificar_ortogonalidad(roto), "una copia rota pasa la verificación"
    return "balanceado; la copia con una celda cambiada falla"


def bayes_a_mano():
    from proyecto.evaluacion import bayes

    previa, s, e = 0.1, 0.9, 0.95
    vpp_mano = (s * previa) / (s * previa + (1 - e) * (1 - previa))           # 0.09 / 0.135
    tras_negativo = ((1 - s) * vpp_mano) / ((1 - s) * vpp_mano + e * (1 - vpp_mano))
    assert abs(vpp_mano - 2 / 3) < 1e-12
    assert abs(float(bayes.valor_predictivo_positivo(previa, s, e)) - vpp_mano) < 1e-12, "VPP distinto"
    assert abs(float(bayes.prob_falso_positivo(previa, s, e)) - (1 - vpp_mano)) < 1e-12, "1 − VPP distinto"
    assert abs(float(bayes.posterior_secuencial(previa, s, e, [1])) - vpp_mano) < 1e-12, "posterior([+]) ≠ VPP"
    assert abs(float(bayes.posterior_secuencial(previa, s, e, [1, 0])) - tras_negativo) < 1e-12, "posterior([+, −])"
    p_observada = s * previa + (1 - e) * (1 - previa)                              # 0.135
    assert abs(float(bayes.previa_corregida(p_observada, s, e)) - previa) < 1e-12, "Rogan–Gladen no invierte"
    return f"VPP {vpp_mano:.4f}, posterior (+, −) {tras_negativo:.4f}"


def sin_fugas():
    import pandas as pd

    from proyecto.variables.comunes import auditar_momentos

    carpeta = RAIZ / "datos" / "modelado"
    papeles = sorted(carpeta.glob("*.json"))
    assert papeles, "no hay tablas de modelado (ejecute `make notebooks`)"
    hallazgos = {}
    for ruta in papeles:
        tabla = pd.read_parquet(ruta.with_suffix(".parquet"))
        encontrados = auditar_momentos(tabla, json.loads(ruta.read_text(encoding="utf-8")))
        if encontrados:
            hallazgos[ruta.stem] = encontrados
    assert not hallazgos, json.dumps(hallazgos, ensure_ascii=False)[:500]
    return f"{len(papeles)} tablas sin hallazgos"


# ----------------------------------------------------------------------
# Protocolo y narrativa
# ----------------------------------------------------------------------
def custodia_prueba():
    registro = RAIZ / "artefactos" / "_etapas" / "lecturas_prueba.log"
    assert registro.exists(), "no existe el registro de lecturas (ejecute `make notebooks`)"
    lineas = [l.split("\t") for l in registro.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lineas, "el registro está vacío"
    etapas = {l[1] for l in lineas}
    assert etapas == {ETAPA_PRUEBA}, f"leyeron la prueba: {sorted(etapas)}"
    sospechosos = []
    candidatos = [g for g in sorted(FUENTES.glob("0*.py")) if g.stem < ETAPA_PRUEBA]
    candidatos += [m for m in sorted((PAQUETE / "etapas").glob("*.py")) if m != MODULO_PRUEBA]
    for guion in candidatos:
        # Solo el código: el texto de la etapa 03 puede nombrar la carpeta al explicar que la escribe
        # (con `guardar_modelado`) sin abrirla.
        texto = "\n".join(l for l in guion.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))
        for patron in ("leer_prueba", "datos/prueba", "DIR_PRUEBA", "/ \"prueba\"", "'prueba'"):
            if patron in texto:
                sospechosos.append(f"{guion.name}: {patron}")
    assert not sospechosos, "; ".join(sospechosos)
    return f"{len(lineas)} lecturas, todas en {ETAPA_PRUEBA}; {len(candidatos)} guiones y módulos sin acceso"


def supuestos_narrativos():
    artefactos = RAIZ / "artefactos"
    leer = lambda nombre: json.loads((artefactos / nombre).read_text(encoding="utf-8"))  # noqa: E731
    seleccion = leer("03_cribado.json")["seleccion"]
    reportado = {
        "regresion": leer("04_regresion.json")["analisis"],
        "clasificacion": leer("05_clasificacion.json")["analisis"],
        "agrupamiento": leer("06_agrupamiento.json")["seleccionado"]["analisis"],
    }
    distintos = [f"{t}: cribado «{seleccion[t]}», etapa «{reportado[t]}»" for t in reportado
                 if seleccion[t] != reportado[t]]
    if (artefactos / "07_evaluacion.json").exists():
        final = leer("07_evaluacion.json")
        for tarea in ("regresion", "clasificacion"):
            if final[tarea]["analisis"] != seleccion[tarea]:
                distintos.append(f"07 evalúa «{final[tarea]['analisis']}» en {tarea}")
    assert not distintos, "; ".join(distintos)
    return ", ".join(f"{t} = {n}" for t, n in seleccion.items())


def guiones_narrados():
    """Guiones de etapa y el de la presentación (los que se convierten en notebooks)."""
    return sorted(FUENTES.glob("0*.py")) + sorted(FUENTES.glob("presentacion.py"))


def sin_cifras_en_texto():
    halladas = []
    for guion in guiones_narrados():
        permitidos = PERMITIDOS_TEXTO.get(guion.name, set())
        en_texto = False
        for numero, linea in enumerate(guion.read_text(encoding="utf-8").splitlines(), start=1):
            if linea.startswith("# %%"):
                en_texto = "[markdown]" in linea
                continue
            if not en_texto or not linea.startswith("#"):
                continue
            contenido = linea[1:].strip()
            if contenido.startswith("#"):
                continue  # títulos numerados («## 3.1 ...»)
            for cifra in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", contenido):
                if cifra not in permitidos:
                    halladas.append(f"{guion.name}:{numero} «{cifra}»")
    assert not halladas, "cifras escritas a mano: " + "; ".join(halladas)


def guiones_delgados():
    """Los guiones de etapa solo narran y llaman: sin funciones ni clases y con celdas de código cortas."""
    problemas, celdas = [], 0
    for guion in guiones_narrados():
        texto = guion.read_text(encoding="utf-8")
        for nodo in ast.walk(ast.parse(texto)):
            if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                problemas.append(f"{guion.name}:{nodo.lineno} define {type(nodo).__name__}")
        for bloque in re.split(r"^# %%.*$", texto, flags=re.MULTILINE)[1:]:
            lineas = [l for l in bloque.strip().splitlines() if l.strip() and not l.startswith("#")]
            if lineas:
                celdas += 1
                if len(lineas) > MAXIMO_LINEAS_CELDA:
                    problemas.append(f"{guion.name}: celda de {len(lineas)} líneas («{lineas[0][:40]}»)")
    assert not problemas, "; ".join(problemas)
    return f"{celdas} celdas de código, ninguna con más de {MAXIMO_LINEAS_CELDA} líneas"


def macros_del_informe():
    """Cada etapa dejó su registro de cifras y `resultados.tex` las define todas, una vez cada una."""
    artefactos = RAIZ / "artefactos"
    registros = {r.stem.removeprefix("macros_"): json.loads(r.read_text(encoding="utf-8"))
                 for r in sorted(artefactos.glob("macros_*.json"))}
    faltan = [e for e in ("01_datos", "02_exploracion", "03_cribado_analisis", "04_regresion",
                          "05_clasificacion_bayes", "06_agrupamiento", ETAPA_PRUEBA) if e not in registros]
    assert not faltan, f"etapas sin registro de cifras: {faltan}"
    nombres = [n for r in registros.values() for n in r["macros"]]
    assert len(nombres) == len(set(nombres)), "macros repetidas entre etapas"
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in nombres), "nombres de macro con caracteres no permitidos"
    generado = RAIZ / "informe" / "generado"
    definidas = re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}", (generado / "resultados.tex").read_text(encoding="utf-8"),
                           flags=re.MULTILINE)
    assert sorted(definidas) == sorted(nombres), "resultados.tex no coincide con los registros de las etapas"
    tablas = [n for r in registros.values() for n in r["tablas"]]
    sin_archivo = [n for n in tablas if not (generado / f"tabla_{n}.tex").exists()]
    assert not sin_archivo, f"tablas registradas sin archivo: {sin_archivo}"
    assert (generado / "indice_macros.tex").exists(), "falta indice_macros.tex"
    return f"{len(nombres)} macros y {len(tablas)} tablas de {len(registros)} etapas"


def presentacion_completa():
    """La presentación arma sus diapositivas solo con los artefactos guardados, una pieza por diapositiva."""
    from proyecto.etapas import presentacion

    faltan = presentacion.comprobar_artefactos()
    assert not faltan, f"faltan artefactos: {faltan}"
    piezas = [f() for f in presentacion.DIAPOSITIVAS]
    assert len(piezas) == presentacion.TOTAL, f"{len(piezas)} diapositivas, se esperaban {presentacion.TOTAL}"
    malas = [f.__name__ for f, r in zip(presentacion.DIAPOSITIVAS, piezas) if len(r.salidas) != 1]
    assert not malas, f"diapositivas con más de una pieza: {malas}"
    contadores = [f"{i} / {presentacion.TOTAL}" for i in range(1, presentacion.TOTAL + 1)]
    fuera = [c for c, r in zip(contadores, piezas) if c not in r.salidas[0].data]
    assert not fuera, f"contador fuera de orden: {fuera}"
    return f"{len(piezas)} diapositivas, {sum(len(r.salidas[0].data) for r in piezas) // 1024} kB de HTML"


MAXIMO_CELDAS_PRESENTACION = 40
MAXIMO_KB_DATOS_PRESENTACION = 500


def datos_de_presentacion():
    """Los arreglos de la presentación son pequeños, vienen del desarrollo (nunca de la prueba), reproducen
    el ROC-AUC y el costo de validación de la etapa 05 y el guion no pasa de su número de celdas."""
    import numpy as np

    from proyecto.base import config
    from proyecto.etapas import presentacion

    ruta = RAIZ / "artefactos" / presentacion.ARCHIVO_DATOS
    assert ruta.exists(), f"falta {ruta.name}"
    kb = ruta.stat().st_size / 1024
    assert kb <= MAXIMO_KB_DATOS_PRESENTACION, f"{ruta.name} pesa {kb:.0f} kB"
    with np.load(ruta) as archivo:
        meta = json.loads(str(archivo["meta"]))
        fechas = archivo["fechas_entrenamiento"]
        n_val = len(archivo["proba_validacion"])
    assert fechas.max() < np.datetime64(str(config.CORTE_VALIDACION)[:10]), "fechas de entrenamiento fuera del corte"
    clf = json.loads((RAIZ / "artefactos" / "05_clasificacion.json").read_text(encoding="utf-8"))
    tabla = clf["tabla_umbrales"][0]
    assert n_val == tabla["TN"] + tabla["FP"] + tabla["FN"] + tabla["TP"], "la validación no tiene las filas de la etapa 05"
    assert abs(meta["auc_reproducido"] - meta["comprobacion"]["auc_artefacto"]) < 1e-6, "ROC-AUC de validación distinto"
    assert meta["costo_reproducido"] == meta["comprobacion"]["costo_artefacto"], "costo de validación distinto"
    guion = (FUENTES / "presentacion.py").read_text(encoding="utf-8")
    celdas = len(re.findall(r"^# %%", guion, flags=re.MULTILINE))
    assert celdas <= MAXIMO_CELDAS_PRESENTACION, f"la presentación tiene {celdas} celdas"
    return f"{kb:.0f} kB, {n_val} filas de validación, ROC-AUC {meta['auc_reproducido']:.4f}, {celdas} celdas en el guion"


def colores_de_ecuaciones():
    """Los colores de las ecuaciones de la presentación (MathJax) son los mismos de sus figuras."""
    from proyecto.reporte.graficos_presentacion import TINTAS

    guion = "".join(g.read_text(encoding="utf-8") for g in guiones_narrados())   # presentación y etapas
    usados = set(re.findall(r"\\color\{(#[0-9A-Fa-f]{6})\}", guion))
    ajenos = sorted(usados - set(TINTAS.values()))
    assert usados, "las ecuaciones no usan colores"
    assert not ajenos, f"colores de ecuación que no están en TINTAS: {ajenos}"
    from proyecto.reporte.html import COLORES_CONFUSION
    _, fp, fn, vp = COLORES_CONFUSION
    assert (fp, fn, vp) == (TINTAS["conformes"], TINTAS["falsos"], TINTAS["sensibilidad"]), \
        "la matriz de confusión no usa los colores de FP, FN y VP de las ecuaciones"
    return f"{len(usados)} colores, todos de TINTAS"


# Notebooks de etapa cuyo texto ya sigue la guía de redacción (la lista crece a medida que se revisan las demás).
ETAPAS_ESTILO = ["01_datos", "02_exploracion", "03_cribado_analisis", "04_regresion", "05_clasificacion_bayes",
                 "06_agrupamiento", "07_evaluacion_final", "08_datos_reales", "Presentacion"]
PROHIBIDOS_ESTILO = {"·": "punto medio", "«": "comilla angular", "»": "comilla angular", "—": "raya",
                     "\\cdot": "\\cdot"}


def _texto_visible(html):
    """Lo que lee una persona en una salida HTML: sin hojas de estilo, guiones ni imágenes en base64."""
    html = re.sub(r"<(style|script)\b.*?</\1>", " ", html, flags=re.S | re.I)
    html = re.sub(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+", " ", html)
    return re.sub(r"<[^>]+>", " ", html)


def estilo_de_redaccion():
    """Las celdas de texto y lo que muestran las salidas de los notebooks de `ETAPAS_ESTILO` no usan punto medio,
    comillas angulares, rayas ni \\cdot (el código de las celdas queda fuera: es código)."""
    import nbformat

    hallados = []
    for etapa in ETAPAS_ESTILO:
        ruta = RAIZ / "notebooks" / f"{etapa}.ipynb"
        assert ruta.exists(), f"falta {ruta.name}: construya los notebooks"
        cuaderno = nbformat.read(ruta, as_version=4)
        for i, celda in enumerate(cuaderno.cells):
            textos = []
            if celda.cell_type == "markdown":
                textos.append(celda.source)
            for salida in celda.get("outputs", []):
                datos = salida.get("data", {})
                if "text/html" in datos:
                    textos.append(_texto_visible("".join(datos["text/html"])))
                elif "text/plain" in datos:
                    textos.append("".join(datos["text/plain"]))
                if "text" in salida:
                    textos.append("".join(salida["text"]))
            for texto in textos:
                for simbolo, nombre in PROHIBIDOS_ESTILO.items():
                    for hallazgo in re.finditer(re.escape(simbolo), texto):
                        contexto = texto[max(0, hallazgo.start() - 30):hallazgo.end() + 20].replace("\n", " ")
                        hallados.append(f"{etapa} celda {i}: {nombre} en «{contexto.strip()}»")
    assert not hallados, f"{len(hallados)} hallazgos: " + "; ".join(hallados[:12])
    return f"{len(ETAPAS_ESTILO)} notebooks sin punto medio, comillas angulares, rayas ni \\cdot"


MAXIMO_SEGUNDOS_PANEL = 1.5   # por redibujo (objetivo: menos de 1 s en un portátil)


def _artefacto_regresion(nombre="04_regresion"):
    return json.loads((RAIZ / "artefactos" / f"{nombre}.json").read_text(encoding="utf-8"))


def _valores_de_prueba(control):
    """Al menos dos valores distintos del actual para mover un control: extremos y centro."""
    import ipywidgets as w

    if isinstance(control, w.FloatLogSlider):
        candidatos = [control.base ** control.max, control.base ** control.min, control.base ** ((control.min + control.max) / 2)]
    elif hasattr(control, "_options_values"):
        opciones = list(control._options_values)
        candidatos = [opciones[-1], opciones[0], opciones[len(opciones) // 2]]
    else:
        candidatos = [control.max, control.min, type(control.value)((control.min + control.max) / 2)]
    unicos = []
    for valor in candidatos:
        if valor != control.value and valor not in unicos:
            unicos.append(valor)
    if len(unicos) < 2:   # dos opciones: ir a la otra y volver
        unicos.append(control.value)
    return unicos


def paneles_interactivos():
    """Cada panel de la presentación se arma sin pantalla, cada control se mueve al menos dos veces, cada
    cambio vuelve a dibujar la figura (otra imagen) sin errores y en menos de `MAXIMO_SEGUNDOS_PANEL`."""
    import contextlib
    import io

    import matplotlib

    matplotlib.use("Agg")
    from proyecto.etapas import presentacion
    from proyecto.reporte import interactivo

    from proyecto.etapas.comun import legible

    d = presentacion.iniciar()
    guion = (FUENTES / "presentacion.py").read_text(encoding="utf-8")
    nombres = sorted(set(re.findall(r"ui\.(panel_\w+)\(d\)", guion)))
    # Paneles que no reciben `d` sino un artefacto de etapa: se arman con los datos de su etapa.
    de_etapa = {"panel_taguchi": lambda: interactivo.panel_taguchi(
        _artefacto_regresion()["taguchi"],
        {f: legible(v) for f, v in _artefacto_regresion()["configuracion"].items()}),
        "panel_taguchi (etapa 05)": lambda: interactivo.panel_taguchi(
            _artefacto_regresion("05_clasificacion")["taguchi"],
            {f: legible(v) for f, v in _artefacto_regresion("05_clasificacion")["configuracion"].items()},
            rotulo="Clasificación", etapa="05")}
    definidos = sorted(n for n in dir(interactivo) if n.startswith("panel_") and n not in de_etapa)
    assert nombres == definidos, f"paneles en el guion {nombres} y en el módulo {definidos}"
    usados = "".join(g.read_text(encoding="utf-8") for g in guiones_narrados())
    sin_usar = [n for n in de_etapa if f"ui.{n.split()[0]}(" not in usados]
    assert not sin_usar, f"paneles de etapa que ningún guion usa: {sin_usar}"
    informe = []
    for nombre in nombres + sorted(de_etapa):
        inicio = time.perf_counter()
        panel = de_etapa[nombre]() if nombre in de_etapa else getattr(interactivo, nombre)(d)
        armado = time.perf_counter() - inicio
        with contextlib.redirect_stdout(io.StringIO()):   # fuera de un notebook, display escribe texto
            panel._ipython_display_()
            for clave, control in panel.controles.items():
                valores = _valores_de_prueba(control)
                assert len(valores) >= 2, f"{nombre}.{clave}: menos de dos valores para probar"
                for valor in valores:
                    antes, imagen = len(panel.tiempos), panel.imagen
                    control.value = valor
                    assert not panel.errores, f"{nombre}.{clave}={valor!r}: {panel.errores[-1]}"
                    assert len(panel.tiempos) > antes, f"{nombre}.{clave}={valor!r}: no se volvió a dibujar"
                    assert panel.imagen != imagen, f"{nombre}.{clave}={valor!r}: la figura no cambió"
        cambios = panel.tiempos[1:]
        assert max(cambios) <= MAXIMO_SEGUNDOS_PANEL, f"{nombre}: un redibujo tardó {max(cambios):.2f} s"
        informe.append(f"{nombre.removeprefix('panel_')} {len(cambios)}× máx {max(cambios):.2f} s (armado {armado:.1f} s)")
    return "; ".join(informe)


def main():
    inicio = time.perf_counter()
    print(f"Verificación del proyecto en {RAIZ}")
    comprobar("Regla de capas (importaciones por AST)", regla_de_capas)
    comprobar("Solo proyecto/datos importa la simulación", simulacion_aislada)
    comprobar("Encabezado «Capa N ·» y «Responsabilidad única:» en cada módulo", encabezados)
    comprobar("Generador determinista (dos corridas, misma huella)", generador_determinista)
    comprobar("Festivos de Colombia en 2025", festivos)
    comprobar("Ortogonalidad del arreglo L16", ortogonalidad_l16)
    comprobar("Bayes contra el cálculo a mano", bayes_a_mano)
    comprobar("Sin fugas por momento en las tablas de modelado", sin_fugas)
    comprobar("Custodia de la partición de prueba", custodia_prueba)
    comprobar("Supuestos narrativos (análisis elegidos)", supuestos_narrativos)
    comprobar("Sin cifras escritas a mano en el texto de las etapas", sin_cifras_en_texto)
    comprobar("Guiones de etapa y de la presentación delgados (sin def, celdas cortas)", guiones_delgados)
    comprobar("Macros y tablas del informe completas y sin repetir", macros_del_informe)
    comprobar("Presentación armada con los artefactos guardados", presentacion_completa)
    comprobar("Datos pequeños de la presentación (desarrollo, reproducibles)", datos_de_presentacion)
    comprobar("Colores de las ecuaciones (presentación y etapas) iguales a los de las figuras", colores_de_ecuaciones)
    comprobar("Paneles interactivos de la presentación y de las etapas (controles movidos sin pantalla)", paneles_interactivos)
    comprobar("Estilo de redacción en los notebooks revisados", estilo_de_redaccion)
    fallas = resultados.count(False)
    print(f"\n{len(resultados) - fallas} de {len(resultados)} comprobaciones superadas en "
          f"{time.perf_counter() - inicio:.1f} s.")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())
