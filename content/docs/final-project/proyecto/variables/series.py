"""Capa 3 · Tablas de modelado de series (volumen y carga).

Responsabilidad única: construir las tablas cuya unidad es un periodo de
tiempo: el volumen semanal de muestras por línea de negocio y la carga diaria
del laboratorio. Cada fila se predice con un paso de anticipación (antes de
que empiece la semana o el día), de modo que todas sus variables son retrasos
del propio objetivo o datos que se conocen de antemano (calendario y agenda de
muestreos).

Cada tabla incluye además columnas de referencia (`base_*`) con pronósticos
ingenuos; no son variables, sirven de línea base que el modelo debe superar.
"""
import numpy as np
import pandas as pd

from proyecto.variables import comunes
from proyecto.variables.comunes import MOMENTO_HISTORIA, a_fecha, con_datos

RETRASOS_SEMANALES = (1, 2, 3, 4, 8)
RETRASOS_DIARIOS = (1, 2, 5)
SEMANAS_CLIENTES_ACTIVOS = 8
SEMANAS_POR_ANIO = 52
SERIE_UNICA = "todas"


def _recepciones(integradas):
    """Fechas de recepción válidas con el cliente y la línea (sector)."""
    tabla = pd.DataFrame({"fecha": a_fecha(integradas["fecha_recepcion"]).dt.normalize(),
                          "id_cliente": integradas["id_cliente"].astype(object)})
    tabla["sector"] = integradas["sector"].astype(object) if con_datos(integradas, "sector") else SERIE_UNICA
    tabla["sector"] = tabla["sector"].fillna("sin sector")
    return tabla.dropna(subset=["fecha"]).reset_index(drop=True)


def _agenda(limpias, llave_sector=None):
    """Visitas de muestreo programadas: fecha, puntos previstos (1 por visita
    si la columna no existe) y sector del cliente. None si no hay agenda."""
    muestreos = limpias.get("muestreos") if limpias is not None else None
    if muestreos is None or not con_datos(muestreos, "fecha_programada"):
        return None
    agenda = pd.DataFrame({"fecha": a_fecha(muestreos["fecha_programada"]).dt.normalize(),
                           "id_cliente": muestreos["id_cliente"].astype(object)})
    agenda["puntos"] = (muestreos["puntos_programados"].astype(float).fillna(1.0).to_numpy()
                        if con_datos(muestreos, "puntos_programados") else 1.0)
    if llave_sector is not None:
        agenda["sector"] = agenda["id_cliente"].map(llave_sector)
    return agenda.dropna(subset=["fecha"])


# ---------------------------------------------------------------------------
# Volumen semanal por línea
# ---------------------------------------------------------------------------

def _semanas_completas(fechas):
    """Lunes de las semanas cubiertas por completo por los datos."""
    primera, ultima = fechas.min(), fechas.max()
    inicio = comunes.lunes_de([primera]).iloc[0]
    if primera.dayofweek > 0:           # la primera semana empezó antes que los datos
        inicio += pd.Timedelta(days=7)
    fin = comunes.lunes_de([ultima]).iloc[0]
    if ultima.dayofweek < 4:            # la última semana no llegó al viernes
        fin -= pd.Timedelta(days=7)
    return pd.date_range(inicio, fin, freq="7D")


def _calendario_semanal(limpias, semanas):
    """Marcas de cada semana que se conocen antes de que empiece."""
    marcas = comunes.calendario_con_marcas(limpias, semanas.min(), semanas.max())
    marcas["semana"] = a_fecha(marcas["semana"])
    laboral = marcas["fecha"].dt.dayofweek < 5
    resumen = pd.DataFrame({
        "dias_habiles_semana": marcas["es_habil"].groupby(marcas["semana"]).sum(),
        "festivos_semana": (marcas["es_festivo"] * laboral).groupby(marcas["semana"]).sum(),
        "es_semana_santa": marcas["es_semana_santa"].groupby(marcas["semana"]).max(),
        "dias_cierre_mes": marcas["cierre_mes"].groupby(marcas["semana"]).sum(),
        "dias_inicio_mes": marcas["inicio_mes"].groupby(marcas["semana"]).sum(),
    }).astype(float)
    return resumen.reindex(semanas)


def volumen_semanal_linea(limpias, integradas):
    """Una fila por semana y línea de negocio; objetivo `muestras_semana`
    (muestras recibidas esa semana); pronóstico a una semana vista. `fecha` es
    el lunes de la semana."""
    recepciones = _recepciones(integradas)
    recepciones["semana"] = comunes.lunes_de(recepciones["fecha"]).to_numpy()
    semanas = _semanas_completas(recepciones["fecha"])
    sectores = sorted(recepciones["sector"].unique())

    conteo = (recepciones.groupby(["sector", "semana"]).size()
              .unstack("sector").reindex(index=semanas, columns=sectores).fillna(0.0))
    retrasos = {k: conteo.shift(k) for k in RETRASOS_SEMANALES}
    media_4 = conteo.shift(1).rolling(4).mean()

    # Clientes distintos con muestras en las ocho semanas anteriores.
    presencia = (recepciones.drop_duplicates(["id_cliente", "semana"])
                 .assign(presente=1.0)
                 .pivot(index="semana", columns="id_cliente", values="presente")
                 .reindex(semanas).fillna(0.0))
    reciente = presencia.shift(1).rolling(SEMANAS_CLIENTES_ACTIVOS, min_periods=1).max()
    sector_de = recepciones.drop_duplicates("id_cliente").set_index("id_cliente")["sector"]
    activos = reciente.T.groupby(sector_de.reindex(reciente.columns).to_numpy()).sum().T.reindex(columns=sectores)

    # Tendencia interanual: últimas cuatro semanas frente a las mismas cuatro
    # de un año antes (faltante mientras no haya un año de historia).
    reciente_4 = conteo.shift(1).rolling(4).sum()
    tendencia = np.log((1.0 + reciente_4) / (1.0 + reciente_4.shift(SEMANAS_POR_ANIO)))
    hace_un_anio = conteo.shift(SEMANAS_POR_ANIO)

    # Sector de cada cliente para repartir la agenda: el de sus muestras y, si
    # aún no ha enviado ninguna, el de la tabla de clientes.
    sector_agenda = sector_de
    clientes = limpias.get("clientes") if limpias is not None else None
    if con_datos(integradas, "sector") and clientes is not None and con_datos(clientes, "sector"):
        maestro = clientes.drop_duplicates("id_cliente").set_index("id_cliente")["sector"].astype(object)
        sector_agenda = sector_de.combine_first(maestro)
    agenda = _agenda(limpias, sector_agenda)
    puntos = None
    if agenda is not None:
        agenda["semana"] = comunes.lunes_de(agenda["fecha"]).to_numpy()
        agenda["sector"] = agenda["sector"].fillna("sin sector") if con_datos(integradas, "sector") else SERIE_UNICA
        puntos = (agenda.groupby(["sector", "semana"])["puntos"].sum()
                  .unstack("sector").reindex(index=semanas, columns=sectores).fillna(0.0))

    def apilar(ancha, nombre):
        return comunes.apilar(ancha, nombre, "sector")

    columnas = [apilar(conteo, "muestras_semana")]
    columnas += [apilar(retrasos[k], f"retraso_{k}") for k in RETRASOS_SEMANALES]
    columnas += [apilar(media_4, "media_movil_4"), apilar(activos, "clientes_activos_8"),
                 apilar(tendencia, "tendencia_anual"), apilar(hace_un_anio, "retraso_52")]
    if puntos is not None:
        columnas.append(apilar(puntos, "puntos_programados_semana"))
    tabla = pd.concat(columnas, axis=1).reset_index()
    tabla["fecha"] = a_fecha(tabla["fecha"])

    calendario = _calendario_semanal(limpias, semanas)
    for columna in calendario.columns:
        tabla[columna] = calendario[columna].reindex(tabla["fecha"]).to_numpy()
    semana_anio = tabla["fecha"].dt.isocalendar().week.astype(float)
    tabla["semana_anio_sen"], tabla["semana_anio_cos"] = comunes.ciclicas(semana_anio - 1, 52.18)
    # Mes del jueves: el mes al que pertenece la mayor parte de la semana.
    tabla["mes"] = (tabla["fecha"] + pd.Timedelta(days=3)).dt.month.astype(float)
    tabla["base_semana_anterior"] = tabla["retraso_1"]
    tabla["base_media_4"] = tabla["media_movil_4"]

    # Sin el retraso más largo no hay pronóstico ingenuo comparable: se
    # descartan las primeras semanas de cada serie.
    tabla = tabla[tabla[f"retraso_{max(RETRASOS_SEMANALES)}"].notna()].copy()

    base_numericas = [f"retraso_{k}" for k in RETRASOS_SEMANALES] + ["media_movil_4"]
    base_categoricas = ["sector"] if len(sectores) > 1 else []
    nuevas = ["semana_anio_sen", "semana_anio_cos", "mes", "dias_habiles_semana", "festivos_semana",
              "es_semana_santa", "dias_cierre_mes", "dias_inicio_mes", "clientes_activos_8",
              "tendencia_anual", "retraso_52"]
    momentos = {c: MOMENTO_HISTORIA for c in base_numericas + nuevas}
    if len(sectores) > 1:
        momentos["sector"] = "maestro"
    if puntos is not None:
        nuevas.append("puntos_programados_semana")
        momentos["puntos_programados_semana"] = "programacion"
    papeles = comunes.armar_papeles("muestras_semana", "programacion", "semana × línea de negocio",
                                    base_numericas, base_categoricas, nuevas,
                                    identificadores=[] if len(sectores) > 1 else ["sector"],
                                    momentos_derivados=momentos,
                                    referencias=["base_semana_anterior", "base_media_4"])
    return comunes.ordenar_tabla(tabla, papeles), papeles


# ---------------------------------------------------------------------------
# Carga diaria del laboratorio
# ---------------------------------------------------------------------------

def carga_diaria(limpias, integradas):
    """Una fila por día hábil, todas las líneas juntas; objetivo
    `muestras_dia` (muestras recibidas ese día); pronóstico a un día hábil
    vista."""
    recepciones = _recepciones(integradas)
    marcas = comunes.calendario_con_marcas(limpias, recepciones["fecha"].min(), recepciones["fecha"].max())
    habiles = marcas[marcas["es_habil"] == 1].reset_index(drop=True)
    # Días calendario hasta el siguiente día hábil (víspera de festivo o puente).
    habiles["dias_hasta_siguiente_habil"] = (habiles["fecha"].shift(-1) - habiles["fecha"]).dt.days.astype(float)
    dentro = habiles["fecha"].between(recepciones["fecha"].min(), recepciones["fecha"].max())
    tabla = habiles[dentro].reset_index(drop=True)
    conteo = recepciones.groupby("fecha").size()
    tabla["muestras_dia"] = conteo.reindex(tabla["fecha"]).fillna(0.0).to_numpy()

    objetivo = tabla["muestras_dia"]
    base_numericas = []
    for k in RETRASOS_DIARIOS:
        tabla[f"retraso_{k}"] = objetivo.shift(k)
        base_numericas.append(f"retraso_{k}")
    tabla["media_movil_5"] = objetivo.shift(1).rolling(5).mean()
    tabla["media_movil_20"] = objetivo.shift(1).rolling(20).mean()
    base_numericas += ["media_movil_5", "media_movil_20", "dia_semana"]
    tabla["base_dia_anterior"] = tabla["retraso_1"]
    tabla["base_media_5"] = tabla["media_movil_5"]

    tabla["mes_sen"], tabla["mes_cos"] = comunes.ciclicas(tabla["fecha"].dt.month - 1, 12)
    nuevas = ["dia_mes", "mes_sen", "mes_cos", "dias_habiles_semana", "es_semana_santa", "tras_festivo",
              "cierre_mes", "inicio_mes", "dias_hasta_siguiente_habil"]
    momentos = {c: MOMENTO_HISTORIA for c in base_numericas + nuevas}
    agenda = _agenda(limpias)
    if agenda is not None:
        tabla["puntos_programados_dia"] = (agenda.groupby("fecha")["puntos"].sum()
                                           .reindex(tabla["fecha"]).fillna(0.0).to_numpy())
        nuevas.append("puntos_programados_dia")
        momentos["puntos_programados_dia"] = "programacion"

    for columna in base_numericas + nuevas:
        tabla[columna] = tabla[columna].astype(float)
    tabla = tabla[tabla["media_movil_20"].notna()].copy()
    papeles = comunes.armar_papeles("muestras_dia", "programacion", "día hábil", base_numericas, [], nuevas,
                                    momentos_derivados=momentos,
                                    referencias=["base_dia_anterior", "base_media_5"])
    return comunes.ordenar_tabla(tabla, papeles), papeles
