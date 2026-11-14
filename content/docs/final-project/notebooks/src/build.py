"""Construcción de los notebooks del proyecto.

Responsabilidad única: convertir los guiones de etapa (`notebooks/src/NN_*.py`,
formato percent de jupytext) en notebooks ejecutados y autocontenidos.

    python notebooks/src/build.py [--no-execute] [--solo NN]
                                  [--prueba-colab DIR] [--sin-construir]

Qué hace, en orden:

1. Sin `--solo`, borra las marcas de etapa (`artefactos.reiniciar_etapas`).
2. Ejecuta cada guion con nbclient (núcleo `python3`, directorio de trabajo
   = raíz del proyecto, paquete `proyecto/` real del repositorio).
3. Después de ejecutar, inserta tras la primera celda de texto las celdas de
   arranque, **sin ejecutar**: una celda que crea las carpetas del paquete y
   una celda `%%writefile` por módulo. En los notebooks encadenados se
   añaden además los guiones de las etapas previas, para que
   `iniciar_etapa` pueda reconstruir lo que falte en una sesión nueva de
   Colab. Como estas celdas no se ejecutan aquí, construir nunca reescribe
   los fuentes del repositorio.
4. Escribe `notebooks/<etapa>.ipynb` y el notebook único
   `notebooks/Proyecto_Final.ipynb` (arranque una sola vez y todas las
   etapas en orden).
5. Con `--prueba-colab DIR` copia los notebooks a una carpeta vacía y los
   ejecuta allí completos (arranque incluido) en un entorno donde el paquete
   del repositorio no es importable. Con `--sin-construir` hace solo esta
   prueba sobre los notebooks ya construidos.

Un guion de etapa que todavía no existe se omite con un aviso.
"""
import argparse
import copy
import hashlib
import os
import shutil
import subprocess
import sys
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
TIEMPO_MAXIMO = 1200

# Las rutas del proyecto son relativas al directorio de trabajo.
os.chdir(RAIZ)
sys.path.insert(0, str(RAIZ))
from proyecto.base import artefactos  # noqa: E402
from proyecto.base.config import ETAPAS, TITULOS_ETAPAS  # noqa: E402

ORDEN_CAPAS = ["base", "datos", "preparacion", "variables", "modelos",
               "evaluacion", "cribado", "reporte"]
NUCLEO = {"name": "python3", "display_name": "Python 3", "language": "python"}
TITULO_PROYECTO = "Proyecto final · Aprendizaje Automático de Máquina I"


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


def _celda_escribir(ruta, numero):
    """Celda `%%writefile` que reproduce un archivo de texto.

    IPython garantiza un salto de línea al final de la celda, así que se
    quita uno del contenido: la copia queda idéntica byte a byte siempre que
    el archivo termine en salto de línea."""
    relativa = ruta.relative_to(RAIZ).as_posix()
    contenido = ruta.read_text(encoding="utf-8")
    if not contenido.endswith("\n"):
        print(f"  aviso: {relativa} no termina en salto de línea; "
              "la copia de Colab tendrá uno de más.")
    else:
        contenido = contenido[:-1]
    return _codigo(f"%%writefile {relativa}\n{contenido}", "arranque",
                   f"arranque-{numero:03d}", oculta=True)


def celdas_de_arranque(etapas_previas=()):
    """Celdas que recrean el paquete (y los guiones previos) desde cero."""
    modulos = modulos_del_paquete()
    carpetas = sorted({m.parent.relative_to(RAIZ).as_posix() for m in modulos}
                      | {DIR_FUENTES.relative_to(RAIZ).as_posix()})
    inicios = sorted(m.relative_to(RAIZ).as_posix() for m in modulos
                     if m.name == "__init__.py")
    con_contenido = [m for m in modulos if m.stat().st_size > 0]
    guiones = [DIR_FUENTES / f"{e}.py" for e in etapas_previas
               if (DIR_FUENTES / f"{e}.py").exists()]

    explicacion = (
        "## Código del proyecto\n\n"
        "Las celdas de esta sección recrean el paquete `proyecto/` "
        f"({len(con_contenido)} módulos) dentro de la sesión, de modo que el "
        "notebook funciona en Google Colab sin clonar ningún repositorio. "
        "Basta con ejecutarlas una vez (*Entorno de ejecución → Ejecutar "
        "todas*). El código se muestra completo para que pueda revisarse; "
        "la sección puede plegarse desde su título."
    )
    if guiones:
        explicacion += (
            "\n\nTambién se guardan los guiones de las etapas anteriores: si "
            "sus resultados no están en la sesión, la etapa los vuelve a "
            "calcular en silencio antes de empezar."
        )
    crear = (
        "# Crea las carpetas del paquete y sus archivos __init__.py\n"
        "import importlib\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        f"for carpeta in {carpetas!r}:\n"
        "    Path(carpeta).mkdir(parents=True, exist_ok=True)\n"
        f"for inicio in {inicios!r}:\n"
        "    Path(inicio).touch()\n"
        "if os.getcwd() not in sys.path:\n"
        "    sys.path.insert(0, os.getcwd())\n"
        "importlib.invalidate_caches()"
    )
    celdas = [_texto(explicacion, "arranque", "arranque-000"),
              _codigo(crear, "arranque", "arranque-001")]
    for numero, ruta in enumerate(con_contenido + guiones, start=2):
        celdas.append(_celda_escribir(ruta, numero))
    return celdas


def _rotulo(etapa):
    return f"{etapa[:2]} · {TITULOS_ETAPAS.get(etapa, etapa)}"


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
    return _texto(" &nbsp;·&nbsp; ".join(partes), "navegacion", identificador)


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
    celdas += celdas_de_arranque(ETAPAS[: ETAPAS.index(etapa)])
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
               "Notebook completo: reúne, en orden, todas las etapas del "
               "proyecto. Cada etapa guarda sus resultados en disco y la "
               "siguiente los lee de allí.\n\n" + indice)
    celdas = [_texto(portada, "navegacion", "portada")] + celdas_de_arranque()
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
    try:  # firma local: JupyterLab muestra el HTML completo sin pedir «confiar»
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


# ----------------------------------------------------------------------
# Prueba de autocontención (simula una sesión nueva de Colab)
# ----------------------------------------------------------------------
def _huella(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _probar_notebook(nombre, carpeta, etapas_esperadas, guiones_esperados):
    """Ejecuta un notebook construido, solo, en una carpeta vacía."""
    carpeta.mkdir(parents=True)
    shutil.copy(DIR_NOTEBOOKS / f"{nombre}.ipynb", carpeta)
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
    originales = modulos_del_paquete() + [DIR_FUENTES / f"{e}.py" for e in guiones_esperados]
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
        print(f"  {len(originales)} archivos recreados idénticos byte a byte; "
              f"marcas de etapa presentes: {', '.join(etapas_esperadas)}")
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
    print("Prueba de autocontención:", "CORRECTA" if bien_a and bien_b else "CON FALLOS")
    return bien_a and bien_b


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
    opciones = analizador.parse_args()

    bien = True
    if not (opciones.sin_construir and opciones.prueba_colab):
        bien = construir(ejecutar=not opciones.no_execute, solo=opciones.solo)
    if bien and opciones.prueba_colab:
        bien = probar_colab(opciones.prueba_colab)
    sys.exit(0 if bien else 1)


if __name__ == "__main__":
    principal()
