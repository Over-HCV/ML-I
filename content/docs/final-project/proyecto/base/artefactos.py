"""Capa 0 · Artefactos y control de etapas.

Responsabilidad única: leer y escribir lo que una etapa le entrega a la
siguiente (tablas, JSON, modelos) y llevar el registro de qué etapa está
corriendo. Con esto las etapas no comparten variables en memoria: cada una
carga del disco lo que necesita y guarda lo que produce.

También custodia la partición de prueba: solo las etapas declaradas en
`config.ETAPA_PRUEBA` pueden leerla, y cada lectura queda registrada (la de
la simulación con `leer_prueba` y la de la exportación real con
`registrar_lectura_prueba`, que llama `proyecto.datos.rodam`).
"""
import json
import runpy
from datetime import datetime

import joblib
import pandas as pd

from proyecto.base import rutas, salida
from proyecto.base.config import ETAPA_PRUEBA, ETAPAS

_PILA_ETAPAS = []


# ----------------------------------------------------------------------
# Etapas
# ----------------------------------------------------------------------
def etapa_actual():
    return _PILA_ETAPAS[-1] if _PILA_ETAPAS else None


def _marca(etapa):
    return rutas.DIR_ETAPAS / f"{etapa}.json"


def etapa_completa(etapa):
    return _marca(etapa).exists()


def ejecutar_en_silencio(etapa):
    """Ejecuta el guion de una etapa sin mostrar nada (ni tablas ni figuras)."""
    import matplotlib.pyplot as plt

    guion = rutas.DIR_FUENTES / f"{etapa}.py"
    if not guion.exists():
        raise FileNotFoundError(
            f"Falta el guion {guion}. La etapa «{etapa}» debe ejecutarse antes que esta."
        )
    previo = salida.en_silencio()
    salida.fijar_silencio(True)
    profundidad = len(_PILA_ETAPAS)
    try:
        runpy.run_path(str(guion), run_name="__etapa__")
    finally:
        del _PILA_ETAPAS[profundidad:]
        salida.fijar_silencio(previo)
        plt.close("all")


def iniciar_etapa(etapa):
    """Declara la etapa en curso. Si alguna etapa previa no dejó su marca
    (por ejemplo, al abrir un notebook suelto en una sesión nueva de Colab),
    la ejecuta primero en silencio."""
    if etapa not in ETAPAS:
        raise ValueError(f"Etapa desconocida: {etapa!r}")
    rutas.crear_directorios()
    for previa in ETAPAS[: ETAPAS.index(etapa)]:
        if not etapa_completa(previa):
            print(f"Ejecutando la etapa previa «{previa}» (no se encontraron sus artefactos)...")
            ejecutar_en_silencio(previa)
    _PILA_ETAPAS.append(etapa)


def cerrar_etapa(resumen=None):
    """Deja la marca de etapa terminada, con un resumen opcional."""
    etapa = etapa_actual()
    if etapa is None:
        raise RuntimeError("No hay ninguna etapa abierta.")
    guardar_json(
        {"etapa": etapa, "terminada": datetime.now().isoformat(timespec="seconds"),
         "resumen": resumen or {}},
        _marca(etapa),
    )
    _PILA_ETAPAS.pop()


def reiniciar_etapas():
    """Borra las marcas de etapa y el registro de lecturas de la prueba
    (reconstrucción completa)."""
    if rutas.DIR_ETAPAS.exists():
        for marca in rutas.DIR_ETAPAS.glob("*.json"):
            marca.unlink()
        registro = rutas.DIR_ETAPAS / "lecturas_prueba.log"
        if registro.exists():
            registro.unlink()


# ----------------------------------------------------------------------
# Lectura y escritura
# ----------------------------------------------------------------------
def guardar_tabla(tabla, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(ruta, index=False)
    return ruta


def leer_tabla(ruta):
    return pd.read_parquet(ruta)


def _a_json(valor):
    """Convierte tipos de numpy/pandas a tipos nativos de JSON."""
    if hasattr(valor, "item") and getattr(valor, "shape", None) == ():
        return valor.item()
    if hasattr(valor, "tolist"):
        return valor.tolist()
    if isinstance(valor, (pd.Timestamp, datetime)):
        return valor.isoformat()
    raise TypeError(f"No se puede serializar {type(valor)}")


def guardar_json(objeto, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(objeto, ensure_ascii=False, indent=2, default=_a_json),
                    encoding="utf-8")
    return ruta


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


def guardar_modelo(modelo, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)
    return ruta


def leer_modelo(ruta):
    return joblib.load(ruta)


# ----------------------------------------------------------------------
# Tablas de modelado y custodia de la prueba
# ----------------------------------------------------------------------
def guardar_modelado(nombre, desarrollo, prueba):
    """Guarda por separado el bloque de desarrollo (entrenamiento +
    validación) y el de prueba de una tabla de modelado."""
    guardar_tabla(desarrollo, rutas.DIR_MODELADO / f"{nombre}.parquet")
    guardar_tabla(prueba, rutas.DIR_PRUEBA / f"{nombre}.parquet")


def leer_modelado(nombre):
    """Bloque de desarrollo (entrenamiento + validación) de un análisis."""
    return leer_tabla(rutas.DIR_MODELADO / f"{nombre}.parquet")


def etapas_de_prueba():
    """Etapas que pueden leer la prueba (acepta una cadena o una colección)."""
    return (ETAPA_PRUEBA,) if isinstance(ETAPA_PRUEBA, str) else tuple(ETAPA_PRUEBA)


def registrar_lectura_prueba(nombre):
    """Comprueba que la etapa en curso puede leer la prueba y deja la lectura
    en `lecturas_prueba.log` (fecha, etapa y nombre). Lanza PermissionError
    en cualquier otra etapa. La usan `leer_prueba` y la lectura del bloque de
    prueba de la exportación real."""
    etapa = etapa_actual()
    if etapa not in etapas_de_prueba():
        raise PermissionError(
            f"La partición de prueba solo puede leerse en {', '.join(etapas_de_prueba())} "
            f"(etapa actual: {etapa!r})."
        )
    rutas.DIR_ETAPAS.mkdir(parents=True, exist_ok=True)
    registro = rutas.DIR_ETAPAS / "lecturas_prueba.log"
    with registro.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{etapa}\t{nombre}\n")


def leer_prueba(nombre):
    """Bloque de prueba de un análisis. Solo se permite en las etapas de
    `ETAPA_PRUEBA`."""
    registrar_lectura_prueba(nombre)
    return leer_tabla(rutas.DIR_PRUEBA / f"{nombre}.parquet")
