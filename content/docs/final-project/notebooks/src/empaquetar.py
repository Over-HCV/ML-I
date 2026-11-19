"""Arma el archivo de entrega del aula virtual: entrega/TallerFinal_Grupo<G>.zip.

Uso (desde la carpeta del proyecto, después de `make notebooks` y `make pdf`):

    make entrega            (o GRUPO=07 make entrega)

Contenido del zip:
    Informe_Grupo<G>.pdf        informe/main.pdf
    Modelo_Grupo<G>.ipynb       notebooks/Proyecto_Final.ipynb (ejecutado y autocontenido)
    notebooks/                  los notebooks de etapa y la presentación, ejecutados
    datos/                      cinco filas anonimizadas y el diccionario de la base real
    README.md, requirements.txt
    proyecto/, notebooks/src/, verificacion/, configuracion/, informe/ (fuentes), Makefile

La exportación real del laboratorio es confidencial y NUNCA entra al zip: el
script falla si encuentra un parquet o una tabla de `datos/reales/`, o si
algún archivo de texto del zip contiene un código de cliente de la exportación.
"""
import os
import re
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from proyecto.base import rutas  # noqa: E402
from proyecto.datos import muestra_publica, rodam  # noqa: E402

GRUPO = os.environ.get("GRUPO", "XX")
DESTINO = RAIZ / "entrega"
CARPETAS_CODIGO = ["proyecto", "notebooks/src", "verificacion", "configuracion"]
EXCLUIR = re.compile(r"(__pycache__|\.pyc$|\.ipynb_checkpoints|\.DS_Store|notebooks/src/idea/|"
                     r"informe/.*\.(aux|log|out|toc|fls|fdb_latexmk|bbl|blg|synctex\.gz)$)")
NOTA_DATOS = """# Datos

La exportación real del laboratorio (38126 muestras, junio de 2024 a agosto de 2026) es confidencial
y no se entrega. Aquí van solo dos archivos para entender su forma:

- `muestra_anonimizada.csv`: cinco filas de muestras distintas con los clientes, puntos de muestreo y
  analistas reemplazados por rótulos genéricos, los textos libres omitidos y los identificadores renumerados.
- `diccionario_datos.csv`: las 36 columnas con su tipo, descripción, papel en la clasificación de no
  conformidad y momento en que se conoce.

Las etapas 01 a 07 usan la base simulada que genera `proyecto/datos/simulacion.py` (semilla 42), así que
se reproducen completas. La etapa 08 necesita el archivo real en `datos/reales/`; sin él cada sección
muestra una nota en lugar de cifras.
"""


def _archivos_de(carpeta):
    for ruta in sorted((RAIZ / carpeta).rglob("*")):
        relativa = ruta.relative_to(RAIZ).as_posix()
        if ruta.is_file() and not EXCLUIR.search(relativa):
            yield ruta, relativa


def contenido():
    """Lista de (origen o bytes, ruta dentro del zip)."""
    piezas = [
        (RAIZ / "informe" / "main.pdf", f"Informe_Grupo{GRUPO}.pdf"),
        (RAIZ / "notebooks" / "Proyecto_Final.ipynb", f"Modelo_Grupo{GRUPO}.ipynb"),
        (RAIZ / "README.md", "README.md"),
        (RAIZ / "requirements.txt", "requirements.txt"),
        (RAIZ / "Makefile", "Makefile"),
    ]
    piezas += [(r, f"notebooks/{r.name}") for r in sorted((RAIZ / "notebooks").glob("*.ipynb"))
               if r.name != "Proyecto_Final.ipynb"]
    for carpeta in CARPETAS_CODIGO:
        piezas += list(_archivos_de(carpeta))
    for ruta in sorted((RAIZ / "informe").rglob("*")):
        relativa = ruta.relative_to(RAIZ).as_posix()
        if ruta.is_file() and not EXCLUIR.search(relativa) and ruta.name != "main.pdf":
            piezas.append((ruta, relativa))
    faltan = [str(o) for o, _ in piezas if isinstance(o, Path) and not o.exists()]
    if faltan:
        raise SystemExit(f"Faltan archivos (ejecute make notebooks y make pdf): {faltan}")
    return piezas


def _codigos_cliente():
    if not rodam.hay_exportacion():
        return set()
    return set(rodam.leer_exportacion(["cliente"])["cliente"].astype(str).unique())


def revisar(piezas, datos_publicos):
    """Falla si algo confidencial entraría al zip."""
    prohibidos = [d for _, d in piezas if d.endswith(".parquet") or d.startswith("datos/reales")]
    if prohibidos:
        raise SystemExit(f"Archivos de datos reales en el zip: {prohibidos}")
    codigos = _codigos_cliente()
    if not codigos:
        print("Aviso: sin la exportación real no se pueden buscar códigos de cliente en el zip.")
        return
    hallados = {}
    textos = [(o, d) for o, d in piezas if Path(d).suffix in {".ipynb", ".tex", ".md", ".py", ".json", ".csv"}]
    textos += [(o, f"datos/{o.name}") for o in datos_publicos]
    for origen, destino in textos:
        palabras = set(re.findall(r"\b[A-Z]{5}\b", origen.read_text(encoding="utf-8", errors="ignore")))
        if palabras & codigos:
            hallados[destino] = sorted(palabras & codigos)[:5]
    if hallados:
        raise SystemExit(f"Códigos de cliente en el zip: {hallados}")
    print(f"Revisión: {len(textos)} archivos de texto sin códigos de cliente; sin datos reales.")


def main():
    piezas = contenido()
    carpeta_datos = DESTINO / "_datos_publicos"
    if rodam.hay_exportacion():
        muestra_publica.escribir(carpeta_datos)
    datos_publicos = sorted(carpeta_datos.glob("*.csv")) if carpeta_datos.exists() else []
    if not datos_publicos:
        raise SystemExit(f"No hay muestra pública: copie la exportación en {rutas.DIR_REALES} y repita.")
    revisar(piezas, datos_publicos)
    DESTINO.mkdir(exist_ok=True)
    archivo = DESTINO / f"TallerFinal_Grupo{GRUPO}.zip"
    with zipfile.ZipFile(archivo, "w", zipfile.ZIP_DEFLATED) as z:
        for origen, destino in piezas:
            z.write(origen, destino)
        for ruta in datos_publicos:
            z.write(ruta, f"datos/{ruta.name}")
        z.writestr("datos/LEEME.md", NOTA_DATOS)
    print(f"Escrito {archivo.relative_to(RAIZ)} ({archivo.stat().st_size / 1e6:.1f} MB, "
          f"{len(piezas) + len(datos_publicos) + 1} archivos)")


if __name__ == "__main__":
    main()
