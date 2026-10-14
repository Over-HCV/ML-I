"""
Arma el notebook final concatenando, en orden, los ficheros .py en
formato "percent" (jupytext) de este directorio, y lo ejecuta para
producir un .ipynb con las salidas (tablas y figuras) ya generadas.

Puede lanzarse desde cualquier carpeta; la ejecución del notebook se
hace siempre con `notebook/` como directorio de trabajo, de modo que las
rutas relativas usadas en las celdas (p. ej. `../latex/figuras`) apunten
al lugar correcto:

    uv run python content/docs/workshop-02/notebook/src/build.py [--no-execute]

Requiere: jupytext, nbclient, nbformat, ipykernel (kernel "python3").
"""
import glob
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

SRC_DIR = Path(__file__).parent
OUT_DIR = SRC_DIR.parent  # notebook/
COMBINED_PY = SRC_DIR / "_combined_notebook.py"
OUTPUT_IPYNB = OUT_DIR / "Taller_2_Notebook.ipynb"


def combinar_fuentes():
    """Concatena, en orden alfabético, todos los .py fuente (excluyendo
    este mismo script y el archivo combinado de una corrida previa)."""
    fuentes = sorted(
        f for f in glob.glob(str(SRC_DIR / "*.py"))
        if Path(f).name not in {"build.py", COMBINED_PY.name}
    )
    if not fuentes:
        raise SystemExit("No se encontraron ficheros fuente .py en este directorio.")

    print("Ficheros fuente, en orden:")
    for f in fuentes:
        print(" -", Path(f).name)

    partes = [Path(f).read_text(encoding="utf-8") for f in fuentes]
    COMBINED_PY.write_text("\n\n".join(partes), encoding="utf-8")
    print(f"\nCombinado escrito en: {COMBINED_PY}")


def construir_notebook(ejecutar: bool = True):
    nb = jupytext.read(COMBINED_PY, fmt="py:percent")
    nb.metadata["kernelspec"] = {
        "name": "python3", "display_name": "Python 3", "language": "python",
    }
    if ejecutar:
        print("\nEjecutando el notebook (cwd = notebook/)...")
        NotebookClient(
            nb, timeout=600, kernel_name="python3",
            resources={"metadata": {"path": str(OUT_DIR)}},
        ).execute()
    nbformat.write(nb, OUTPUT_IPYNB)
    print(f"\nNotebook final: {OUTPUT_IPYNB}")


if __name__ == "__main__":
    ejecutar = "--no-execute" not in sys.argv
    combinar_fuentes()
    construir_notebook(ejecutar=ejecutar)
