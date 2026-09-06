"""
Arma el notebook final concatenando, en orden, los ficheros .py en
formato "percent" (jupytext) de este directorio, y lo convierte + ejecuta
para producir un .ipynb con las salidas (tablas y figuras) ya generadas.

IMPORTANTE — ejecutar desde la carpeta `notebook/` (la de arriba), no
desde `src/`, para que el directorio de trabajo coincida con el de
`cs-training.csv` y el notebook resultante:

    cd notebook
    python3 src/build.py [--no-execute]

Requiere: jupytext, nbclient, nbformat, ipykernel (kernel "python3"
registrado con `python3 -m ipykernel install --user --name python3`).
"""
import glob
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent
OUT_DIR = SRC_DIR.parent  # notebook/ — donde vive cs-training.csv
COMBINED_PY = SRC_DIR / "_combined_notebook.py"
OUTPUT_IPYNB = OUT_DIR / "Taller_1_KNN_Notebook.ipynb"


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
    cmd = ["jupytext", "--to", "notebook"]
    if ejecutar:
        cmd.append("--execute")
    cmd += [str(COMBINED_PY), "-o", str(OUTPUT_IPYNB)]
    print("\nEjecutando:", " ".join(cmd))
    # cwd=OUT_DIR asegura que, al ejecutar el notebook, la ruta relativa
    # "cs-training.csv" (usada dentro de las celdas) apunte a notebook/
    # sin importar desde dónde se haya lanzado este script.
    subprocess.run(cmd, check=True, cwd=OUT_DIR)
    print(f"\nNotebook final: {OUTPUT_IPYNB}")


if __name__ == "__main__":
    ejecutar = "--no-execute" not in sys.argv
    combinar_fuentes()
    construir_notebook(ejecutar=ejecutar)
