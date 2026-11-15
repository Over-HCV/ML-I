"""Capa 2 · Partición temporal.

Responsabilidad única: asignar cada fila a entrenamiento, validación o
prueba según su fecha y los cortes de `config`, y separar físicamente el
bloque de prueba. La partición es por tiempo (no al azar) porque los
modelos se usarán para predecir el futuro a partir del pasado.

    entrenamiento : fecha <  CORTE_VALIDACION
    validación    : CORTE_VALIDACION <= fecha < CORTE_PRUEBA
    prueba        : fecha >= CORTE_PRUEBA

Las filas sin fecha no pertenecen a ningún bloque y quedan fuera de ambos.
"""
import numpy as np
import pandas as pd

from proyecto.base.config import CORTE_PRUEBA, CORTE_VALIDACION

ENTRENAMIENTO, VALIDACION, PRUEBA = "entrenamiento", "validacion", "prueba"
PARTICIONES = [ENTRENAMIENTO, VALIDACION, PRUEBA]
# Rótulos para mostrar (tablas y figuras); las claves internas siguen sin tilde.
ROTULOS = {ENTRENAMIENTO: "entrenamiento", VALIDACION: "validación", PRUEBA: "prueba"}


def etiquetar_particion(fechas):
    """Serie con «entrenamiento», «validacion» o «prueba» para cada fecha
    (faltante si la fecha lo es)."""
    fechas = pd.to_datetime(pd.Series(fechas), errors="coerce")
    etiquetas = np.select(
        [fechas.isna().to_numpy(), (fechas < pd.Timestamp(CORTE_VALIDACION)).to_numpy(),
         (fechas < pd.Timestamp(CORTE_PRUEBA)).to_numpy()],
        ["", ENTRENAMIENTO, VALIDACION], default=PRUEBA)
    etiquetas = pd.Series(etiquetas, index=fechas.index, dtype=object, name="particion")
    return etiquetas.mask(etiquetas == "", None)


def separar(tabla, columna_fecha="fecha"):
    """Devuelve `(desarrollo, prueba)`: el bloque de desarrollo reúne
    entrenamiento y validación; el de prueba no debe abrirse hasta la etapa
    final."""
    particion = etiquetar_particion(tabla[columna_fecha]).to_numpy()
    desarrollo = tabla.loc[np.isin(particion, [ENTRENAMIENTO, VALIDACION])].reset_index(drop=True)
    prueba = tabla.loc[particion == PRUEBA].reset_index(drop=True)
    return desarrollo, prueba


def solo_desarrollo(tabla, columna_fecha="fecha"):
    """Filas de entrenamiento y validación (para explorar sin mirar la prueba)."""
    return separar(tabla, columna_fecha)[0]


def resumen_particion(tabla, columna_fecha="fecha"):
    """Filas y rango de fechas de cada bloque. Solo cuenta filas: no calcula
    nada sobre el contenido del bloque de prueba."""
    fechas = pd.to_datetime(tabla[columna_fecha], errors="coerce")
    particion = etiquetar_particion(fechas)
    filas = []
    for bloque in PARTICIONES:
        del_bloque = fechas[(particion == bloque).to_numpy()]
        filas.append({"Bloque": bloque, "Filas": len(del_bloque),
                      "Desde": del_bloque.min() if len(del_bloque) else pd.NaT,
                      "Hasta": del_bloque.max() if len(del_bloque) else pd.NaT})
    resumen = pd.DataFrame(filas)
    resumen["Porcentaje"] = 100 * resumen["Filas"] / max(int(resumen["Filas"].sum()), 1)
    return resumen
