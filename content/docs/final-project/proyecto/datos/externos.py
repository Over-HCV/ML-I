"""Capa 1 · Fuentes externas opcionales.

Responsabilidad única: leer, si existe, una tabla pública de positividad que
el usuario haya descargado a mano en `datos/externos/` (por ejemplo, los
resultados de muestreo del USDA FSIS). Este módulo no descarga nada ni
contiene cifras: si el archivo no está, devuelve `None` y el proyecto sigue
sin esa comparación.
"""
import pandas as pd

from proyecto.base import rutas

# Nombre con el que debe guardarse el archivo descargado (sin la extensión).
NOMBRE_POSITIVIDAD = "positividad_externa"
_EXTENSIONES = (".csv", ".parquet")


def ruta_positividad_externa():
    """Ruta del archivo de positividad externa, o None si no se ha descargado."""
    for extension in _EXTENSIONES:
        ruta = rutas.DIR_EXTERNOS / f"{NOMBRE_POSITIVIDAD}{extension}"
        if ruta.exists():
            return ruta
    return None


def leer_positividad_externa():
    """Tabla de positividad externa tal como la descargó el usuario (sin
    transformar), o None si el archivo no existe. La tabla lleva en
    `attrs["fuente"]` el nombre del archivo leído, para citarlo."""
    ruta = ruta_positividad_externa()
    if ruta is None:
        return None
    if ruta.suffix == ".parquet":
        tabla = pd.read_parquet(ruta)
    else:
        tabla = pd.read_csv(ruta, sep=None, engine="python")
    tabla.attrs["fuente"] = ruta.name
    return tabla
