"""Capa 1 · Lectura de la exportación real del laboratorio.

Responsabilidad única: leer la exportación del sistema del laboratorio
(un archivo parquet en `datos/reales/`, con una fila por muestra,
especificación y lectura) y entregarla en las unidades que usa el proyecto:
una fila por muestra, los lotes de llegada y las campañas de muestreo en
planta. Aquí no se limpia ni se modela nada; solo se cambia la granularidad.

El archivo es confidencial y nunca se versiona (`datos/` está en
`.gitignore`). Si no está, `hay_exportacion()` devuelve False y quien llame
decide cómo seguir sin él (por ejemplo, en Colab).

Definiciones:

- **Lote**: las muestras que un mismo cliente ingresa el mismo día y del
  mismo tipo. El tipo es «planta» si la muestra viene de un muestreo en planta
  (superficies, ambientes, personal) y «producto» en cualquier otro caso
  (producto terminado, materia prima, agua, granel, envase).
- **Campaña**: lotes de planta de un mismo cliente separados por a lo sumo
  `SEPARACION_CAMPANA_DIAS` días; describe una visita de muestreo que puede
  ocupar varios días seguidos.
"""
import numpy as np
import pandas as pd

from proyecto.base import rutas

PATRON_EXPORTACION = "muestras_*.parquet"
ORIGEN_PLANTA = "Muestreo en planta"
TIPOS_LOTE = ("producto", "planta")
SEPARACION_CAMPANA_DIAS = 3


# ----------------------------------------------------------------------
# Archivo
# ----------------------------------------------------------------------
def ruta_exportacion():
    """Ruta del parquet más reciente en `datos/reales/` (None si no hay)."""
    candidatos = sorted(rutas.DIR_REALES.glob(PATRON_EXPORTACION))
    return candidatos[-1] if candidatos else None


def hay_exportacion():
    return ruta_exportacion() is not None


def leer_exportacion(columnas=None):
    """Exportación completa (una fila por muestra × especificación × lectura)."""
    ruta = ruta_exportacion()
    if ruta is None:
        raise FileNotFoundError(
            f"No hay exportación real en {rutas.DIR_REALES} (se espera {PATRON_EXPORTACION}). "
            "El archivo es confidencial: pídalo al laboratorio y cópielo en esa carpeta."
        )
    return pd.read_parquet(ruta, columns=columnas)


# ----------------------------------------------------------------------
# Una fila por muestra
# ----------------------------------------------------------------------
_DE_LA_MUESTRA = [
    "fecha_ingreso", "fecha_inicio", "estado_muestra", "concepto_muestra", "origen_muestra", "cliente", "sector",
    "id_producto", "producto", "forma_farmaceutica", "id_cuadro_analitico", "cuadro_complejo",
]


def tabla_por_muestra(filas=None):
    """Una fila por muestra con lo que la describe y un resumen de sus lecturas.

    Columnas: las de la muestra (fechas, estado, dictamen, origen, cliente,
    sector, producto, cuadro), `n_especificaciones`, `n_lecturas`,
    `reanalisis` (alguna especificación con más de una lectura),
    `primera_lectura` y `ultima_lectura`, `analista` y `lote_medio` de la
    primera lectura, `no_conforme` (dictamen final «no cumple»; nulo si no hay
    dictamen), `dia` (fecha de ingreso sin hora) y `tipo_lote`."""
    filas = leer_exportacion() if filas is None else filas
    grupos = filas.groupby("id_muestra", sort=True)
    muestras = grupos[_DE_LA_MUESTRA].first()
    muestras["n_especificaciones"] = grupos["id_especificacion"].nunique()
    muestras["n_lecturas"] = grupos["id_lectura"].count()
    muestras["reanalisis"] = grupos["n_lectura"].max().fillna(0) > 1
    muestras["primera_lectura"] = grupos["fecha_lectura"].min()
    muestras["ultima_lectura"] = grupos["fecha_lectura"].max()
    primeras = (filas.dropna(subset=["fecha_lectura"]).sort_values(["id_muestra", "fecha_lectura"])
                .groupby("id_muestra")[["analista_lectura", "lote_de_medio_id"]].first())
    muestras = muestras.join(primeras.rename(columns={"analista_lectura": "analista",
                                                      "lote_de_medio_id": "lote_medio"}))
    concepto = muestras["concepto_muestra"].astype("boolean")
    muestras["no_conforme"] = (~concepto).astype("boolean")
    muestras["dia"] = muestras["fecha_ingreso"].dt.normalize()
    muestras["tipo_lote"] = np.where(muestras["origen_muestra"].astype(str) == ORIGEN_PLANTA, "planta", "producto")
    for columna in ("cliente", "sector", "origen_muestra", "forma_farmaceutica", "estado_muestra"):
        muestras[columna] = muestras[columna].astype(str)
    return muestras.reset_index()


# ----------------------------------------------------------------------
# Lotes y campañas
# ----------------------------------------------------------------------
def asignar_lotes(muestras, separacion_dias=SEPARACION_CAMPANA_DIAS):
    """Agrega a cada muestra `id_lote` (cliente × día × tipo) e `id_campana`
    (solo en planta; nulo en producto)."""
    muestras = muestras.sort_values(["cliente", "dia", "tipo_lote", "fecha_ingreso"]).copy()
    llave = ["cliente", "dia", "tipo_lote"]
    muestras["id_lote"] = muestras.groupby(llave, sort=False).ngroup() + 1
    planta = muestras[muestras["tipo_lote"] == "planta"].drop_duplicates("id_lote")[["cliente", "dia", "id_lote"]]
    salto = planta.groupby("cliente")["dia"].diff().dt.days
    planta["id_campana"] = (salto.isna() | (salto > separacion_dias)).cumsum()
    muestras = muestras.merge(planta[["id_lote", "id_campana"]], on="id_lote", how="left")
    muestras["id_campana"] = muestras["id_campana"].astype("Int64")
    return muestras.sort_values("id_muestra").reset_index(drop=True)


def lotes(muestras):
    """Una fila por lote: cliente, día, tipo, sector, campaña, número de
    muestras y de no conformes. Requiere `asignar_lotes`."""
    return (muestras.groupby("id_lote")
            .agg(cliente=("cliente", "first"), dia=("dia", "first"), tipo_lote=("tipo_lote", "first"),
                 sector=("sector", "first"), id_campana=("id_campana", "first"),
                 n_muestras=("id_muestra", "size"), no_conformes=("no_conforme", "sum"))
            .reset_index())


def campanas(muestras):
    """Una fila por campaña de muestreo en planta: cliente, primer y último
    día, días con ingreso, lotes y muestras. Requiere `asignar_lotes`."""
    planta = muestras.dropna(subset=["id_campana"])
    return (planta.groupby("id_campana")
            .agg(cliente=("cliente", "first"), desde=("dia", "min"), hasta=("dia", "max"),
                 dias=("dia", "nunique"), lotes=("id_lote", "nunique"), n_muestras=("id_muestra", "size"))
            .reset_index())


def conteos_por_semana(tabla, llave="cliente", fecha="dia", desde=None, hasta=None):
    """Conteo de filas de `tabla` por `llave` y semana (lunes), con ceros en las
    semanas sin filas. Devuelve una tabla ancha: llave × semana."""
    semana = tabla[fecha].dt.to_period("W-SUN").dt.start_time
    desde = semana.min() if desde is None else pd.Timestamp(desde)
    hasta = semana.max() if hasta is None else pd.Timestamp(hasta)
    semanas = pd.date_range(desde, hasta, freq="W-MON")
    ancha = tabla.assign(semana=semana).groupby([llave, "semana"]).size().unstack(fill_value=0)
    return ancha.reindex(columns=semanas, fill_value=0)
