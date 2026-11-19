"""Construcción de los notebooks del proyecto.

Responsabilidad única: convertir los guiones de etapa (`notebooks/src/NN_*.py`,
formato percent de jupytext) en notebooks ejecutados y autocontenidos.

    python notebooks/src/build.py [--no-execute] [--solo NN] [--presentacion]
                                  [--prueba-colab DIR] [--sin-construir]

Qué hace, en orden:

1. Sin `--solo`, borra las marcas de etapa (`artefactos.reiniciar_etapas`).
2. Ejecuta cada guion con nbclient (núcleo `python3`, directorio de trabajo
   = raíz del proyecto, paquete `proyecto/` real del repositorio).
3. Después de ejecutar, inserta tras la primera celda de texto el arranque,
   **sin ejecutar**: un título plegable y una celda corta que busca
   `notebooks/codigo/<notebook>.tar.xz` (se escribe aquí y se versiona: un
   .tar.xz determinista, entradas ordenadas y fecha fija) o lo descarga del
   repositorio si el notebook está solo. El archivo lleva el paquete, `configuracion/mapeo_columnas.json` y, en los notebooks
   encadenados, los guiones de las etapas previas (para que
   `iniciar_etapa` pueda reconstruir lo que falte en una sesión nueva de
   Colab). La celda comprueba el SHA-256 y no reemplaza archivos que ya
   existan, de forma que ejecutarla en el repositorio nunca reescribe los
   fuentes.
4. Escribe `notebooks/<etapa>.ipynb` y el notebook único
   `notebooks/Proyecto_Final.ipynb` (arranque una sola vez y todas las
   etapas en orden).
5. Con `--prueba-colab DIR` copia los notebooks a una carpeta vacía y los
   ejecuta allí completos (arranque incluido, con su archivo de código en
   `codigo/` como si se subieran juntos a Colab) en un entorno donde el paquete
   del repositorio no es importable. Con `--sin-construir` hace solo esta
   prueba sobre los notebooks ya construidos.

6. Al final (o sola, con `--presentacion`) arma `notebooks/Presentacion.ipynb`
   desde `notebooks/src/presentacion.py`: once diapositivas que leen los
   artefactos ya calculados, con fragmentos de código y animaciones entre
   ellas. Antes de ejecutarla, `presentacion.preparar_datos()` escribe
   `artefactos/presentacion_datos.npz` (arreglos pequeños sacados de las
   tablas de desarrollo, nunca de la prueba) y comprueba que reproduce el
   ROC-AUC y el costo de validación de la etapa 04. Su arranque recrea el
   paquete y esos artefactos (unos cientos de kB), de forma que en Colab se
   ejecuta en segundos sin repetir las etapas. Antes de escribirla se comprueba que las seis
   etapas estén terminadas y que los artefactos no cambien durante la
   ejecución (misma huella SHA-256 antes y después).

Un guion de etapa que todavía no existe se omite con un aviso.
"""
import argparse
import copy
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

RAIZ = Path(__file__).resolve().parents[2]
DIR_FUENTES = RAIZ / "notebooks" / "src"
DIR_NOTEBOOKS = RAIZ / "notebooks"
DIR_PAQUETE = RAIZ / "proyecto"
NOMBRE_FUSIONADO = "Proyecto_Final"
NOMBRE_PRESENTACION = "Presentacion"
GUION_PRESENTACION = DIR_FUENTES / "presentacion.py"
DIR_ARTEFACTOS = RAIZ / "artefactos"
TIEMPO_MAXIMO = 1200

# Las rutas del proyecto son relativas al directorio de trabajo.
os.chdir(RAIZ)
sys.path.insert(0, str(RAIZ))
from proyecto.base import artefactos  # noqa: E402
from proyecto.base.config import ETAPAS, TITULOS_ETAPAS  # noqa: E402

ORDEN_CAPAS = ["base", "datos", "preparacion", "variables", "modelos",
               "evaluacion", "cribado", "reporte", "etapas"]
NUCLEO = {"name": "python3", "display_name": "Python 3", "language": "python"}
TITULO_PROYECTO = "Aprendizaje Automático de Máquina I, proyecto final"


# ----------------------------------------------------------------------
# Celdas
# ----------------------------------------------------------------------
def _marcar(celda, papel, identificador):
    """Etiqueta una celda insertada por la construcción (para poder
    retirarla al releer un notebook ya construido) y le da un id estable."""
    celda.metadata["proyecto"] = papel
    celda["id"] = identificador
    return celda


def _texto(fuente, papel, identificador):
    return _marcar(nbformat.v4.new_markdown_cell(fuente), papel, identificador)


def _codigo(fuente, papel, identificador, oculta=False):
    celda = nbformat.v4.new_code_cell(fuente)
    if oculta:  # JupyterLab muestra la celda plegada; el código sigue ahí
        celda.metadata["jupyter"] = {"source_hidden": True}
    return _marcar(celda, papel, identificador)


def modulos_del_paquete():
    """Módulos de `proyecto/`, ordenados por capa y luego por nombre."""
    def clave(ruta):
        partes = ruta.relative_to(DIR_PAQUETE).parts
        capa = partes[0] if len(partes) > 1 else ""
        orden = ORDEN_CAPAS.index(capa) if capa in ORDEN_CAPAS else len(ORDEN_CAPAS)
        return (0 if capa == "" else 1, orden, partes)

    rutas = [r for r in DIR_PAQUETE.rglob("*.py") if "__pycache__" not in r.parts]
    return sorted(rutas, key=clave)


ARCHIVO_MAPEO = RAIZ / "configuracion" / "mapeo_columnas.json"
# Mismos archivos → mismos bytes: entradas ordenadas, fecha 0, dueño y permisos fijos.

# El código de cada notebook viaja en notebooks/codigo/<notebook>.tar.xz (se versiona) y la celda
# de arranque solo lo busca al lado o lo descarga de aquí si el notebook está solo (Colab).
DIR_CODIGO = DIR_NOTEBOOKS / "codigo"
URL_CODIGO = "https://raw.githubusercontent.com/Over-HCV/ML-I/main/content/docs/final-project/notebooks/codigo/"

CODIGO_ARRANQUE = """\
# Código del proyecto ({archivos} archivos, {tamano} kB): {contenido}.
# En una copia del repositorio o en la carpeta de la entrega ya está y no se toca nada.
# En Colab se extrae de codigo/{archivo} si se subió junto al notebook o se descarga
# del repositorio; en los dos casos se comprueba su huella SHA-256 antes de extraerlo.
# Un archivo local con otro contenido se conserva (SOBRESCRIBIR = True lo reemplaza).
import hashlib, importlib, io, os, sys, tarfile, urllib.request
from pathlib import Path

ARCHIVO, HUELLA, SOBRESCRIBIR = "{archivo}", "{huella}", False
URL = "{url}" + ARCHIVO

if Path.cwd().name == "notebooks" and Path("../proyecto/__init__.py").exists():
    os.chdir("..")  # abierto desde notebooks/: se trabaja en la raíz del proyecto


def recrear_proyecto():
    origen = next((Path(d) / ARCHIVO for d in ("notebooks/codigo", "codigo", ".") if (Path(d) / ARCHIVO).exists()), None)
    if origen is None and Path("proyecto/__init__.py").exists():
        escritos, iguales, conservados = 0, None, []
    else:
        datos = origen.read_bytes() if origen else urllib.request.urlopen(URL, timeout=60).read()
        if hashlib.sha256(datos).hexdigest() != HUELLA:
            raise ValueError(f"{{ARCHIVO}} no corresponde a este notebook: su SHA-256 no coincide.")
        escritos, iguales, conservados = 0, 0, []
        with tarfile.open(fileobj=io.BytesIO(datos), mode="r:xz") as archivo:
            for miembro in archivo.getmembers():
                destino = Path(miembro.name)
                if destino.is_absolute() or ".." in destino.parts:
                    raise ValueError(f"Ruta no permitida en el archivo: {{miembro.name}}")
                contenido = archivo.extractfile(miembro).read()
                if destino.exists() and destino.read_bytes() == contenido:
                    iguales += 1
                elif destino.exists() and not SOBRESCRIBIR:
                    conservados.append(miembro.name)
                else:
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    destino.write_bytes(contenido)
                    escritos += 1
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    importlib.invalidate_caches()
    print("Código del proyecto:", "se usa la copia local." if iguales is None
          else f"{{escritos}} archivos escritos, {{iguales}} ya estaban iguales.")
    if conservados:
        print(f"Se conservaron {{len(conservados)}} archivos locales distintos:", ", ".join(conservados))


recrear_proyecto()"""


def archivos_de_arranque(etapas_previas=()):
    """Archivos que recrea el arranque: el paquete, el mapeo y los guiones previos."""
    guiones = [DIR_FUENTES / f"{e}.py" for e in etapas_previas if (DIR_FUENTES / f"{e}.py").exists()]
    return modulos_del_paquete() + [ARCHIVO_MAPEO] + guiones, guiones


def archivo_determinista(rutas_archivos):
    """.tar.xz con entradas ordenadas, fecha 0, dueño vacío y permisos 644: mismos archivos, mismos bytes
    (xz comprime el código bastante más que el ZIP, y Colab lo lee con la biblioteca estándar)."""
    memoria = io.BytesIO()
    with tarfile.open(fileobj=memoria, mode="w:xz", preset=9, format=tarfile.USTAR_FORMAT) as archivo:
        for ruta in sorted(rutas_archivos, key=lambda r: r.relative_to(RAIZ).as_posix()):
            contenido = ruta.read_bytes()
            info = tarfile.TarInfo(ruta.relative_to(RAIZ).as_posix())
            info.size, info.mtime, info.mode = len(contenido), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archivo.addfile(info, io.BytesIO(contenido))
    return memoria.getvalue()


ARTEFACTOS_EXTRA = ["presentacion_datos.npz", "modelos/regresion.joblib"]


def artefactos_de_presentacion():
    """Artefactos pequeños que lee la presentación: el JSON de cada etapa, su registro de cifras, los arreglos
    que preparó `presentacion.preparar_datos` y el modelo de regresión congelado (unos pocos kB)."""
    extra = [DIR_ARTEFACTOS / n for n in ARTEFACTOS_EXTRA if (DIR_ARTEFACTOS / n).exists()]
    return sorted(DIR_ARTEFACTOS.glob("0*.json")) + sorted(DIR_ARTEFACTOS.glob("macros_*.json")) + extra + videos_manim()


def videos_manim():
    """Videos de Manim ya renderizados y sus portadas (si existen). Sin ellos, al volver a ejecutar en Colab
    las celdas de video mostrarían solo el aviso."""
    carpeta = RAIZ / "figuras" / "manim"
    return sorted(carpeta.glob("*.mp4")) + sorted(carpeta.glob("*.png"))


# Etapas que muestran videos de Manim: su arranque (y el del notebook único) los lleva.
ETAPAS_CON_VIDEO = ["04_clasificacion_bayes"]


def celdas_de_arranque(nombre, etapas_previas=(), presentacion=False, videos=False):
    """Título plegable y una sola celda que recrea el paquete (y los guiones previos, o los
    artefactos que lee la presentación, y los videos de Manim si la etapa los muestra)."""
    rutas_archivos, guiones = archivos_de_arranque(etapas_previas)
    if presentacion:
        rutas_archivos = rutas_archivos + artefactos_de_presentacion()
    elif videos:
        rutas_archivos = rutas_archivos + videos_manim()
    datos = archivo_determinista(rutas_archivos)
    modulos = [m for m in modulos_del_paquete() if m.stat().st_size > 0]
    explicacion = (
        "## Código del proyecto\n\n"
        "La celda siguiente prepara el código del proyecto. En una copia del repositorio no hace nada y en Google Colab "
        "trae el paquete `proyecto/` desde `notebooks/codigo/` (si se subió junto al notebook) o desde el repositorio, "
        "así que basta con ejecutar todo (*Entorno de ejecución → Ejecutar todas*). La sección se puede plegar desde "
        "su título."
    )
    if presentacion:
        explicacion += (" Trae también los artefactos pequeños que leen las diapositivas (JSON, arreglos y el modelo de regresión), así que no hace "
                        "falta volver a ejecutar las etapas.")
    if videos and not presentacion and videos_manim():
        explicacion += " Trae también los videos de Manim que muestra la etapa."
    if guiones:
        explicacion += (" Trae también los guiones de las etapas anteriores: Si sus resultados no están en la sesión, "
                        "la etapa los vuelve a calcular en silencio antes de empezar.")
    archivo = f"{nombre}.tar.xz"
    DIR_CODIGO.mkdir(exist_ok=True)
    ruta = DIR_CODIGO / archivo
    if not ruta.exists() or ruta.read_bytes() != datos:
        ruta.write_bytes(datos)
    contenido = f"el paquete proyecto/ ({len(modulos)} módulos) y el mapeo de columnas"
    if guiones:
        contenido += f", los guiones de las {len(guiones)} etapas previas"
    if presentacion:
        contenido += f", los {len(artefactos_de_presentacion())} artefactos que lee la presentación"
    if videos and not presentacion and videos_manim():
        contenido += " y los videos de Manim"
    codigo = CODIGO_ARRANQUE.format(
        archivos=len(rutas_archivos), tamano=round(len(datos) / 1024), contenido=contenido,
        archivo=archivo, huella=hashlib.sha256(datos).hexdigest(), url=URL_CODIGO,
    )
    return [_texto(explicacion, "arranque", "arranque-000"),
            _codigo(codigo, "arranque", "arranque-001", oculta=True)]


def _rotulo(etapa):
    return f"Etapa {etapa[:2]}: {TITULOS_ETAPAS.get(etapa, etapa)}"


def celda_de_navegacion(etapa, disponibles, identificador):
    """Enlaces al notebook anterior y al siguiente de la cadena."""
    posicion = disponibles.index(etapa)
    partes = []
    if posicion > 0:
        previa = disponibles[posicion - 1]
        partes.append(f"← Anterior: [{_rotulo(previa)}]({previa}.ipynb)")
    partes.append(f"[Notebook completo]({NOMBRE_FUSIONADO}.ipynb)")
    if posicion < len(disponibles) - 1:
        siguiente = disponibles[posicion + 1]
        partes.append(f"Siguiente: [{_rotulo(siguiente)}]({siguiente}.ipynb) →")
    return _texto("<br>".join(partes), "navegacion", identificador)


def _posicion_tras_primer_texto(celdas):
    for i, celda in enumerate(celdas):
        if celda.cell_type == "markdown":
            return i + 1
    return 0


def _empieza_con_titulo(celdas):
    return bool(celdas) and celdas[0].cell_type == "markdown" \
        and celdas[0].source.lstrip().startswith("#")


def _nuevo_notebook(celdas):
    nb = nbformat.v4.new_notebook()
    nb.cells = celdas
    nb.metadata["kernelspec"] = dict(NUCLEO)
    nb.metadata["language_info"] = {"name": "python"}
    # Colab: la sección de arranque se abre plegada.
    nb.metadata["colab"] = {"provenance": [], "collapsed_sections": ["arranque-000"]}
    return nb


def armar_encadenado(etapa, celdas_etapa, disponibles):
    """Notebook de una etapa: título, navegación, arranque y contenido."""
    corte = _posicion_tras_primer_texto(celdas_etapa)
    resto = celdas_etapa[corte:]
    celdas = list(celdas_etapa[:corte])
    celdas.append(celda_de_navegacion(etapa, disponibles, f"nav-{etapa[:2]}-inicio"))
    celdas += celdas_de_arranque(etapa, ETAPAS[: ETAPAS.index(etapa)], videos=etapa in ETAPAS_CON_VIDEO)
    if not _empieza_con_titulo(resto):  # cierra la sección plegable del arranque
        celdas.append(_texto("## Desarrollo de la etapa", "navegacion",
                             f"nav-{etapa[:2]}-desarrollo"))
    celdas += resto
    celdas.append(celda_de_navegacion(etapa, disponibles, f"nav-{etapa[:2]}-fin"))
    return _nuevo_notebook(celdas)


def armar_fusionado(celdas_por_etapa):
    """Notebook único: portada, arranque una vez y todas las etapas."""
    indice = "\n".join(f"{i}. {TITULOS_ETAPAS.get(e, e)}"
                       for i, e in enumerate(celdas_por_etapa, start=1))
    portada = (f"# {TITULO_PROYECTO}\n\n"
               "Este notebook reúne en orden todas las etapas del proyecto. "
               "Cada etapa guarda sus resultados en disco y la siguiente los "
               "lee de allí.\n\n" + indice)
    celdas = [_texto(portada, "navegacion", "portada")] + celdas_de_arranque(
        NOMBRE_FUSIONADO, videos=any(e in ETAPAS_CON_VIDEO for e in celdas_por_etapa))
    contador = 0
    for etapa, celdas_etapa in celdas_por_etapa.items():
        if not _empieza_con_titulo(celdas_etapa):
            celdas.append(_texto(f"# {_rotulo(etapa)}", "navegacion",
                                 f"nav-{etapa[:2]}-titulo"))
        for celda in copy.deepcopy(celdas_etapa):
            if celda.cell_type == "code" and celda.get("execution_count"):
                contador += 1
                celda["execution_count"] = contador
                for salida in celda.outputs:
                    if "execution_count" in salida:
                        salida["execution_count"] = contador
            celdas.append(celda)
    return _nuevo_notebook(celdas)


def armar_presentacion(celdas):
    """Notebook de la presentación: título, arranque (paquete y artefactos) y diapositivas."""
    corte = _posicion_tras_primer_texto(celdas)
    nuevas = list(celdas[:corte]) + celdas_de_arranque(NOMBRE_PRESENTACION, presentacion=True)
    nuevas.append(_texto("## Diapositivas", "navegacion", "pres-diapositivas"))
    return _nuevo_notebook(nuevas + list(celdas[corte:]))


# ----------------------------------------------------------------------
# Ejecución
# ----------------------------------------------------------------------
def _entorno(con_paquete):
    """Entorno del núcleo: con el paquete del repositorio (construcción) o
    sin ningún acceso a él (prueba de Colab)."""
    entorno = dict(os.environ)
    entorno.pop("PROYECTO_RAIZ", None)
    otras = [p for p in entorno.get("PYTHONPATH", "").split(os.pathsep)
             if p and Path(p).resolve() != RAIZ]
    if con_paquete:
        otras.insert(0, str(RAIZ))
    if otras:
        entorno["PYTHONPATH"] = os.pathsep.join(otras)
    else:
        entorno.pop("PYTHONPATH", None)
    return entorno


def _ejecutar(nb, carpeta, con_paquete):
    cliente = NotebookClient(nb, timeout=TIEMPO_MAXIMO, kernel_name="python3",
                             resources={"metadata": {"path": str(carpeta)}})
    cliente.execute(env=_entorno(con_paquete))
    return nb


def leer_guion(etapa):
    """Guion de etapa → celdas con ids estables (sin salidas)."""
    nb = jupytext.read(DIR_FUENTES / f"{etapa}.py", fmt="py:percent")
    for i, celda in enumerate(nb.cells):
        celda["id"] = f"e{etapa[:2]}-{i:03d}"
        celda.metadata.pop("lines_to_next_cell", None)
    nb.metadata["kernelspec"] = dict(NUCLEO)
    return nb


def leer_construido(etapa):
    """Celdas propias de una etapa ya construida (sin arranque ni navegación)."""
    ruta = DIR_NOTEBOOKS / f"{etapa}.ipynb"
    if not ruta.exists():
        return None
    nb = nbformat.read(ruta, as_version=4)
    return [c for c in nb.cells if "proyecto" not in c.metadata]


def _escribir(nb, ruta):
    nbformat.validate(nb)
    nbformat.write(nb, ruta)
    try:  # firma local: JupyterLab muestra el HTML completo sin pedir "confiar"
        from nbformat.sign import NotebookNotary
        NotebookNotary().sign(nbformat.read(ruta, as_version=4))
    except Exception:  # la firma es una comodidad, nunca un requisito
        pass
    print(f"  escrito {ruta.relative_to(RAIZ)}  ({ruta.stat().st_size / 1024:.0f} kB)")


def construir(ejecutar=True, solo=None):
    """Ejecuta las etapas y escribe los notebooks. Devuelve False si falla."""
    con_guion = [e for e in ETAPAS if (DIR_FUENTES / f"{e}.py").exists()]
    for etapa in ETAPAS:
        if etapa not in con_guion:
            print(f"Etapa {etapa}: no existe notebooks/src/{etapa}.py; se omite.")
    if not con_guion:
        print("No hay ningún guion de etapa en notebooks/src/. Nada que construir.")
        return False

    elegidas = con_guion
    if solo is not None:
        elegidas = [e for e in con_guion if e.startswith(solo)]
        if not elegidas:
            print(f"--solo {solo}: ninguna etapa con guion empieza por «{solo}».")
            return False
    elif ejecutar:
        artefactos.reiniciar_etapas()

    celdas_por_etapa = {}
    for etapa in con_guion:
        if etapa in elegidas:
            nb = leer_guion(etapa)
            if ejecutar:
                print(f"Etapa {etapa}: ejecutando...", flush=True)
                inicio = time.time()
                try:
                    _ejecutar(nb, RAIZ, con_paquete=True)
                except CellExecutionError as error:
                    print(f"\nLa etapa {etapa} falló:\n{error}")
                    return False
                print(f"  terminada en {time.time() - inicio:.1f} s")
            else:
                print(f"Etapa {etapa}: sin ejecutar (--no-execute).")
            celdas_por_etapa[etapa] = list(nb.cells)
        else:
            previas = leer_construido(etapa)
            if previas is None:
                print(f"Etapa {etapa}: no se ha construido todavía; "
                      "no entra en el notebook completo.")
            else:
                celdas_por_etapa[etapa] = previas

    disponibles = list(celdas_por_etapa)
    print("Escribiendo notebooks:")
    for etapa, celdas in celdas_por_etapa.items():
        _escribir(armar_encadenado(etapa, celdas, disponibles),
                  DIR_NOTEBOOKS / f"{etapa}.ipynb")
    _escribir(armar_fusionado(celdas_por_etapa),
              DIR_NOTEBOOKS / f"{NOMBRE_FUSIONADO}.ipynb")
    return True


def _huella_artefactos():
    return hashlib.sha256(archivo_determinista(artefactos_de_presentacion())).hexdigest()


def problemas_de_salida(nb):
    """Errores, avisos o ecos de texto en las salidas de un notebook ejecutado."""
    problemas = []
    for i, celda in enumerate(nb.cells):
        if celda.metadata.get("proyecto") == "arranque":   # el arranque informa qué archivos escribió
            continue
        for salida in celda.get("outputs", []):
            tipo = salida.get("output_type")
            if tipo == "error":
                problemas.append(f"celda {i}: error {salida.get('ename')}")
            elif tipo == "stream":
                problemas.append(f"celda {i}: texto en {salida.get('name')}: {salida.get('text', '')[:80]!r}")
            elif tipo == "execute_result":
                problemas.append(f"celda {i}: eco de un valor ({list(salida.get('data', {}))})")
            elif tipo == "display_data" and set(salida.get("data", {})) == {"text/plain"}:
                problemas.append(f"celda {i}: salida solo de texto")
    return problemas


def construir_presentacion(ejecutar=True):
    """Ejecuta el guion de la presentación sobre los artefactos actuales y escribe
    `notebooks/Presentacion.ipynb`. Devuelve False si algo falla."""
    if not GUION_PRESENTACION.exists():
        print("Presentación: no existe notebooks/src/presentacion.py; se omite.")
        return True
    sin_terminar = [e for e in ETAPAS if not artefactos.etapa_completa(e)]
    if sin_terminar:
        print(f"Presentación: faltan etapas terminadas ({', '.join(sin_terminar)}); ejecute antes `make notebooks`.")
        return False
    if ejecutar:
        from proyecto.etapas.presentacion import preparar_datos
        resumen = preparar_datos()
        print(f"Presentación: datos pequeños en {resumen['segundos']:.1f} s ({resumen['kB']:.0f} kB); "
              f"ROC-AUC de validación reproducido {resumen['auc'][0]:.6f} (artefacto {resumen['auc'][1]:.6f}), "
              f"costo {resumen['costo'][0]} (artefacto {resumen['costo'][1]:.0f})")
        if abs(resumen["auc"][0] - resumen["auc"][1]) > 1e-6 or resumen["costo"][0] != resumen["costo"][1]:
            print("Presentación: las probabilidades de validación no reproducen la etapa 04; se detiene.")
            return False
    nb = jupytext.read(GUION_PRESENTACION, fmt="py:percent")
    for i, celda in enumerate(nb.cells):
        celda["id"] = f"pres-{i:03d}"
        celda.metadata.pop("lines_to_next_cell", None)
    huella = _huella_artefactos()
    if ejecutar:
        print("Presentación: ejecutando...", flush=True)
        inicio = time.time()
        try:
            _ejecutar(nb, RAIZ, con_paquete=True)
        except CellExecutionError as error:
            print(f"\nLa presentación falló:\n{error}")
            return False
        problemas = problemas_de_salida(nb)
        if problemas:
            print("  FALLO en las salidas:\n  " + "\n  ".join(problemas))
            return False
        print(f"  terminada en {time.time() - inicio:.1f} s")
    if _huella_artefactos() != huella:
        print("Presentación: los artefactos cambiaron durante la construcción; vuelva a ejecutarla.")
        return False
    print(f"  artefactos embebidos: {len(artefactos_de_presentacion())} archivos, huella {huella[:12]}")
    armado = armar_presentacion(nb.cells)
    if "widgets" in nb.metadata:   # estado de los paneles: sin núcleo, JupyterLab y VS Code dibujan los controles
        armado.metadata["widgets"] = nb.metadata["widgets"]
    _escribir(armado, DIR_NOTEBOOKS / f"{NOMBRE_PRESENTACION}.ipynb")
    return True


# ----------------------------------------------------------------------
# Prueba de autocontención (simula una sesión nueva de Colab)
# ----------------------------------------------------------------------
def _huella(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _probar_notebook(nombre, carpeta, etapas_esperadas, guiones_esperados, originales=None,
                     revisar_salidas=False):
    """Ejecuta un notebook construido, solo, en una carpeta vacía. `originales` son los
    archivos que el arranque debe recrear idénticos (por omisión, paquete, mapeo y guiones)."""
    carpeta.mkdir(parents=True)
    shutil.copy(DIR_NOTEBOOKS / f"{nombre}.ipynb", carpeta)
    (carpeta / "codigo").mkdir()
    shutil.copy(DIR_CODIGO / f"{nombre}.tar.xz", carpeta / "codigo")
    entorno = _entorno(con_paquete=False)

    # El paquete no debe poder importarse antes de ejecutar el arranque.
    sonda = subprocess.run([sys.executable, "-c", "import proyecto"], cwd=carpeta,
                           env=entorno, capture_output=True)
    if sonda.returncode == 0:
        print("  FALLO: `proyecto` ya es importable en el entorno de prueba.")
        return False

    nb = nbformat.read(carpeta / f"{nombre}.ipynb", as_version=4)
    inicio = time.time()
    try:
        _ejecutar(nb, carpeta, con_paquete=False)
    except CellExecutionError as error:
        print(f"  FALLO al ejecutar {nombre}.ipynb:\n{error}")
        return False
    nbformat.write(nb, carpeta / f"{nombre}.ipynb")
    print(f"  {nombre}.ipynb ejecutado en {time.time() - inicio:.1f} s "
          f"({len(nb.cells)} celdas)")

    problemas = []
    if originales is None:
        originales, _ = archivos_de_arranque(guiones_esperados)
    if revisar_salidas:
        problemas += problemas_de_salida(nb)
    for original in originales:
        copia = carpeta / original.relative_to(RAIZ)
        if not copia.exists():
            problemas.append(f"no se recreó {original.relative_to(RAIZ)}")
        elif _huella(copia) != _huella(original):
            problemas.append(f"{original.relative_to(RAIZ)} difiere del original")
    for etapa in etapas_esperadas:
        if not (carpeta / "artefactos" / "_etapas" / f"{etapa}.json").exists():
            problemas.append(f"falta la marca de la etapa {etapa}")
    for problema in problemas:
        print(f"  FALLO: {problema}")
    if not problemas:
        print(f"  {len(originales)} archivos recreados idénticos byte a byte"
              + (f"; marcas de etapa presentes: {', '.join(etapas_esperadas)}" if etapas_esperadas else ""))
    return not problemas


def probar_colab(destino):
    """Dos escenarios: (a) el notebook completo solo; (b) el último notebook
    encadenado solo, que debe reconstruir en silencio sus etapas previas."""
    destino = Path(destino).resolve()
    if destino.exists() and any(destino.iterdir()):
        print(f"--prueba-colab: {destino} no está vacío; indique una carpeta vacía o nueva.")
        return False
    if RAIZ == destino or RAIZ in destino.parents:
        print("--prueba-colab: la carpeta debe estar fuera del proyecto.")
        return False
    construidas = [e for e in ETAPAS if (DIR_NOTEBOOKS / f"{e}.ipynb").exists()
                   and (DIR_FUENTES / f"{e}.py").exists()]
    if not construidas or not (DIR_NOTEBOOKS / f"{NOMBRE_FUSIONADO}.ipynb").exists():
        print("--prueba-colab: no hay notebooks construidos; ejecute antes la construcción.")
        return False
    ultima = construidas[-1]
    previas = ETAPAS[: ETAPAS.index(ultima)]

    print(f"Prueba de autocontención en {destino}")
    print(f"(a) {NOMBRE_FUSIONADO}.ipynb solo, en carpeta vacía:")
    bien_a = _probar_notebook(NOMBRE_FUSIONADO, destino / "fusionado", construidas, [])
    print(f"(b) {ultima}.ipynb solo, en carpeta vacía (reconstruye etapas previas):")
    bien_b = _probar_notebook(ultima, destino / "encadenado", previas + [ultima], previas)
    bien_c = probar_presentacion(destino / "presentacion", rotulo="(c)")
    print("Prueba de autocontención:", "CORRECTA" if bien_a and bien_b and bien_c else "CON FALLOS")
    return bien_a and bien_b and bien_c


def probar_presentacion(carpeta, rotulo=""):
    """La presentación sola, en una carpeta vacía: el arranque debe bastar (sin etapas)."""
    if not (DIR_NOTEBOOKS / f"{NOMBRE_PRESENTACION}.ipynb").exists():
        print(f"{rotulo} {NOMBRE_PRESENTACION}.ipynb no está construido; se omite.".strip())
        return True
    carpeta = Path(carpeta).resolve()
    if carpeta.exists() and any(carpeta.iterdir()):
        print(f"--prueba-colab: {carpeta} no está vacío; indique una carpeta vacía o nueva.")
        return False
    print(f"{rotulo} {NOMBRE_PRESENTACION}.ipynb solo, en carpeta vacía (sin ejecutar etapas):".strip())
    originales, _ = archivos_de_arranque()
    return _probar_notebook(NOMBRE_PRESENTACION, carpeta, [], [],
                            originales=originales + artefactos_de_presentacion(), revisar_salidas=True)


def principal():
    analizador = argparse.ArgumentParser(description="Construye los notebooks del proyecto.")
    analizador.add_argument("--no-execute", action="store_true",
                            help="arma los notebooks sin ejecutar las etapas")
    analizador.add_argument("--solo", metavar="NN",
                            help="ejecuta solo la etapa cuyo nombre empieza por NN")
    analizador.add_argument("--prueba-colab", metavar="DIR",
                            help="ejecuta los notebooks construidos en una carpeta vacía")
    analizador.add_argument("--sin-construir", action="store_true",
                            help="con --prueba-colab: prueba los notebooks ya construidos")
    analizador.add_argument("--presentacion", action="store_true",
                            help="arma solo la presentación (con --prueba-colab, prueba solo la presentación)")
    opciones = analizador.parse_args()

    bien = True
    if opciones.presentacion:
        if not (opciones.sin_construir and opciones.prueba_colab):
            bien = construir_presentacion(ejecutar=not opciones.no_execute)
        if bien and opciones.prueba_colab:
            bien = probar_presentacion(opciones.prueba_colab)
        sys.exit(0 if bien else 1)
    if not (opciones.sin_construir and opciones.prueba_colab):
        bien = construir(ejecutar=not opciones.no_execute, solo=opciones.solo)
        if bien and not opciones.no_execute:
            bien = construir_presentacion()
    if bien and opciones.prueba_colab:
        bien = probar_colab(opciones.prueba_colab)
    sys.exit(0 if bien else 1)


if __name__ == "__main__":
    principal()
