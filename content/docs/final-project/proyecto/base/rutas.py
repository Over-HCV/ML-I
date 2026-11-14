"""Capa 0 · Rutas del proyecto.

Responsabilidad única: decir dónde vive cada cosa. La raíz es el directorio
de trabajo (o la variable de entorno PROYECTO_RAIZ), de forma que las mismas
rutas relativas sirvan en el repositorio y en Google Colab.
"""
import os
from pathlib import Path

RAIZ = Path(os.environ.get("PROYECTO_RAIZ", Path.cwd()))

DIR_DATOS = RAIZ / "datos"
DIR_CRUDOS = DIR_DATOS / "crudos"          # tablas tal como llegan (ya con nombres canónicos)
DIR_REALES = DIR_DATOS / "reales"          # exportación real del laboratorio (si existe)
DIR_EXTERNOS = DIR_DATOS / "externos"      # fuentes públicas descargadas a mano
DIR_AUDITORIA = DIR_DATOS / "auditoria"    # verdad oculta de la simulación
DIR_LIMPIOS = DIR_DATOS / "limpios"        # tablas limpias e integradas
DIR_MODELADO = DIR_DATOS / "modelado"      # tablas de modelado (entrenamiento + validación)
DIR_PRUEBA = DIR_DATOS / "prueba"          # partición de prueba, separada físicamente

DIR_ARTEFACTOS = RAIZ / "artefactos"
DIR_ETAPAS = DIR_ARTEFACTOS / "_etapas"    # marcas de etapa terminada
DIR_FIGURAS = RAIZ / "figuras"
DIR_FUENTES = RAIZ / "notebooks" / "src"   # guiones de etapa (formato percent)
ARCHIVO_MAPEO = RAIZ / "configuracion" / "mapeo_columnas.json"

DIRECTORIOS = [
    DIR_CRUDOS, DIR_REALES, DIR_EXTERNOS, DIR_AUDITORIA, DIR_LIMPIOS, DIR_MODELADO,
    DIR_PRUEBA, DIR_ARTEFACTOS, DIR_ETAPAS, DIR_FIGURAS,
]


def crear_directorios():
    """Crea las carpetas de trabajo que falten."""
    for carpeta in DIRECTORIOS:
        carpeta.mkdir(parents=True, exist_ok=True)
