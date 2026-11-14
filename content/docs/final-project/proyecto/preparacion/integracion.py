"""Capa 2 · Integración de tablas.

Responsabilidad única: unir las tablas limpias en una sola tabla por muestra
(`muestras` + atributos del cliente + marcas de calendario de la fecha de
recepción). Todas las columnas que añade están declaradas con su momento en
`esquema.DERIVADAS["muestras"]`, de forma que las capas siguientes pueden
seguir preguntando al contrato qué pueden usar.
"""
import numpy as np
import pandas as pd

from proyecto.datos import esquema
from proyecto.datos.calendario import tabla_calendario

DIAS_BORDE_MES = 5   # días hábiles que cuentan como cierre o inicio de mes
MARGEN_CALENDARIO = pd.Timedelta(days=7)
MARCAS_CALENDARIO = ["anio", "mes", "dia_mes", "dia_semana", "semana", "dias_habiles_semana",
                     "es_semana_santa", "tras_festivo", "cierre_mes", "inicio_mes"]


def banderas_calendario(calendario):
    """Una fila por día del calendario con las marcas que se integran a las
    muestras: año, mes, día, día de la semana (0 = lunes), lunes de su semana,
    días hábiles de la semana, Semana Santa, primer día hábil tras un festivo
    y cierre/inicio de mes (cinco días hábiles)."""
    cal = calendario.sort_values("fecha").reset_index(drop=True)
    fecha = cal["fecha"].astype("datetime64[ns]")
    habil = cal["es_habil"].astype(bool)
    festivo = cal["es_festivo"].astype(bool)
    marcas = pd.DataFrame({"fecha": fecha})
    marcas["anio"] = fecha.dt.year
    marcas["mes"] = fecha.dt.month
    marcas["dia_mes"] = fecha.dt.day
    marcas["dia_semana"] = fecha.dt.dayofweek
    marcas["semana"] = (fecha - pd.to_timedelta(fecha.dt.dayofweek, unit="D")).dt.normalize()
    marcas["dias_habiles_semana"] = habil.astype(int).groupby(marcas["semana"]).transform("sum")
    marcas["es_semana_santa"] = cal["es_semana_santa"].astype(int) if "es_semana_santa" in cal else 0
    # Festivos acumulados hasta cada día y hasta el día hábil anterior: si hay
    # alguno entre ambos, el día es el primero hábil después de un festivo.
    festivos_acumulados = festivo.astype(int).cumsum()
    hasta_habil_previo = festivos_acumulados.where(habil).ffill().shift(1)
    marcas["tras_festivo"] = (habil & (festivos_acumulados.shift(1) > hasta_habil_previo)).astype(int)
    mes = fecha.dt.to_period("M")
    habiles_hasta = habil.astype(int).groupby(mes).cumsum()
    habiles_mes = habil.astype(int).groupby(mes).transform("sum")
    marcas["cierre_mes"] = (habil & (habiles_mes - habiles_hasta < DIAS_BORDE_MES)).astype(int)
    marcas["inicio_mes"] = (habil & (habiles_hasta <= DIAS_BORDE_MES)).astype(int)
    return marcas


def integrar_muestras(limpias):
    """Tabla integrada: una fila por muestra con sus columnas limpias, los
    atributos del cliente que existan y las marcas de calendario de
    `fecha_recepcion`. No elimina ni duplica muestras."""
    muestras = limpias["muestras"]
    integrada = muestras.copy()

    if "clientes" in limpias:
        clientes = limpias["clientes"].drop_duplicates("id_cliente")
        atributos = [c for c in esquema.columnas("clientes")
                     if c in clientes.columns and c != "id_cliente" and c not in integrada.columns]
        integrada = integrada.merge(clientes[["id_cliente"] + atributos], on="id_cliente", how="left")
        if "fecha_vinculacion" in integrada.columns:
            integrada["antiguedad_dias"] = (
                integrada["fecha_recepcion"] - integrada["fecha_vinculacion"]).dt.days.astype(float)

    recepcion = integrada["fecha_recepcion"]
    calendario = limpias.get("calendario")
    con_fecha = recepcion.dropna()
    if len(con_fecha):
        # Las marcas miran los días vecinos (festivo previo, semana completa),
        # así que el calendario debe cubrir una semana de margen a cada lado.
        desde, hasta = con_fecha.min() - MARGEN_CALENDARIO, con_fecha.max() + MARGEN_CALENDARIO
        cubre = (calendario is not None and len(calendario)
                 and calendario["fecha"].min() <= desde and calendario["fecha"].max() >= hasta)
        if not cubre:   # el calendario guardado no alcanza: se recalcula para el rango necesario
            calendario = tabla_calendario(desde, hasta)
        marcas = banderas_calendario(calendario)
        integrada["fecha_recepcion"] = integrada["fecha_recepcion"].astype("datetime64[ns]")
        integrada = integrada.merge(marcas, left_on="fecha_recepcion", right_on="fecha", how="left")
        integrada = integrada.drop(columns="fecha")
    else:
        for marca in MARCAS_CALENDARIO:
            integrada[marca] = np.nan
    if len(integrada) != len(muestras):
        raise RuntimeError("La integración cambió el número de muestras: revise llaves duplicadas.")
    return integrada.reset_index(drop=True)
