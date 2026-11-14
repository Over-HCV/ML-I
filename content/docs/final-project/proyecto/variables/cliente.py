"""Capa 3 · Tablas de modelado por cliente y tablas de agrupamiento.

Responsabilidad única: construir las tablas cuya unidad es el cliente en un
periodo (visitas de muestreo por mes, inactividad en el trimestre siguiente)
y las tablas descriptivas para agrupamiento (segmentos de clientes, perfiles
de muestra y días atípicos).

Las tablas supervisadas usan solo periodos ya cerrados para predecir el
siguiente. Las de agrupamiento no tienen objetivo y se construyen únicamente
con el periodo de desarrollo (fechas anteriores a `config.CORTE_PRUEBA`), de
modo que el bloque de prueba no interviene en ninguna descripción.
"""
import numpy as np
import pandas as pd

from proyecto.base.config import SEMILLA
from proyecto.datos import esquema
from proyecto.datos.mapeo import VALORES_CANONICOS
from proyecto.variables import comunes
from proyecto.variables.comunes import MOMENTO_HISTORIA, a_fecha, con_datos

MINIMO_MUESTRAS_SEGMENTO = 10    # actividad mínima para describir a un cliente
MAXIMO_PERFILES = 5000           # tope de filas de la tabla de perfiles de muestra
COBERTURA_MINIMA_PERFIL = 0.90   # un descriptor debe existir en casi todas las muestras
DIAS_POR_MES = 30.44
ORIGEN_PLANTA = VALORES_CANONICOS["muestras"]["origen"][1]   # «muestreo en planta»


# ---------------------------------------------------------------------------
# Piezas compartidas
# ---------------------------------------------------------------------------

def _atributos_cliente(limpias):
    """Atributos del cliente declarados como variables en el contrato.
    Devuelve (tabla indexada por cliente, numéricas, categóricas)."""
    clientes = limpias.get("clientes") if limpias is not None else None
    if clientes is None or "id_cliente" not in clientes.columns:
        return None, [], []
    numericas = [c for c in esquema.columnas("clientes", papel="variable", tipo="numero") if con_datos(clientes, c)]
    categoricas = [c for c in esquema.columnas("clientes", papel="variable", tipo="categoria")
                   if con_datos(clientes, c)]
    extras = ["fecha_vinculacion"] if con_datos(clientes, "fecha_vinculacion") else []
    atributos = clientes.drop_duplicates("id_cliente").set_index("id_cliente")[numericas + categoricas + extras]
    atributos.index = atributos.index.astype(object)
    return atributos, numericas, categoricas


def _muestras_marcadas(integradas):
    """Muestras con fecha de recepción y las marcas que se agregan por
    cliente o por día. El nivel «urgente» es cualquier nivel de urgencia
    distinto del habitual (el más frecuente en el periodo de entrenamiento)."""
    tabla = integradas.copy()
    tabla["fecha"] = a_fecha(tabla["fecha_recepcion"]).dt.normalize()
    tabla = tabla[tabla["fecha"].notna()].reset_index(drop=True)
    tabla["id_cliente"] = tabla["id_cliente"].astype(object)
    marcas = []
    if con_datos(tabla, "urgencia"):
        habitual = comunes.nivel_habitual(tabla["urgencia"], tabla["fecha"])
        if habitual is not None:
            tabla["_urgente"] = (tabla["urgencia"] != habitual).astype(float).where(tabla["urgencia"].notna())
            marcas.append("_urgente")
    if con_datos(tabla, "origen"):
        tabla["_planta"] = (tabla["origen"] == ORIGEN_PLANTA).astype(float).where(tabla["origen"].notna())
        marcas.append("_planta")
    llave = pd.Series(np.nan, index=tabla.index, dtype=object)
    for columna in ("id_lote_envio", "id_muestreo"):
        if con_datos(tabla, columna):
            llave = llave.where(llave.notna(), tabla[columna].astype(object))
    # Sin identificador de envío, un envío es un día de llegada del cliente.
    tabla["_envio"] = llave.where(llave.notna(), tabla["id_cliente"].astype(str) + "|"
                                  + tabla["fecha"].dt.strftime("%Y-%m-%d"))
    return tabla, marcas


def _ancha(tabla, valor, periodos, clientes):
    """Tabla periodo × cliente con ceros donde no hubo actividad."""
    return (tabla.groupby(["_periodo", "id_cliente"])[valor].sum().unstack("id_cliente")
            .reindex(index=periodos, columns=clientes).fillna(0.0))


def _larga(anchas):
    """Une varias tablas anchas (periodo × cliente) en una tabla larga."""
    series = [comunes.apilar(ancha, nombre, "id_cliente") for nombre, ancha in anchas.items()]
    larga = pd.concat(series, axis=1).reset_index()
    larga["fecha"] = a_fecha(larga["fecha"])
    return larga


def _unir_atributos(tabla, atributos):
    if atributos is None:
        return tabla
    return tabla.merge(atributos, left_on="id_cliente", right_index=True, how="left")


# ---------------------------------------------------------------------------
# Visitas de muestreo por cliente y mes
# ---------------------------------------------------------------------------

def visitas_mensuales_cliente(limpias, integradas):
    """Una fila por cliente con plan de muestreo y mes; objetivo `visitas_mes`
    (visitas programadas ese mes); se predice antes de armar la agenda del
    mes, con los meses anteriores y los atributos del cliente."""
    muestreos = limpias["muestreos"]
    visitas = pd.DataFrame({"id_cliente": muestreos["id_cliente"].astype(object),
                            "_periodo": comunes.inicio_de_mes(muestreos["fecha_programada"]).to_numpy(),
                            "uno": 1.0}).dropna(subset=["_periodo"])
    muestras, _ = _muestras_marcadas(integradas)
    muestras["_periodo"] = comunes.inicio_de_mes(muestras["fecha"]).to_numpy()
    muestras["uno"] = 1.0

    # Horizonte de los datos: el último mes cubierto por las recepciones.
    ultima = muestras["fecha"].max()
    ultimo_mes = comunes.inicio_de_mes([ultima]).iloc[0]
    if not comunes.periodo_completo(ultimo_mes + pd.offsets.MonthEnd(0), ultima):
        ultimo_mes -= pd.DateOffset(months=1)
    meses = pd.date_range(visitas["_periodo"].min(), ultimo_mes, freq="MS")
    clientes = sorted(visitas["id_cliente"].unique())

    conteo = _ancha(visitas, "uno", meses, clientes)
    # Un cliente entra a la tabla desde el mes de su primera visita programada.
    con_plan = conteo.cumsum() > 0
    conteo = conteo.where(con_plan)
    enviadas = _ancha(muestras, "uno", meses, clientes)
    hubo_visita = (conteo > 0).astype(float).where(con_plan)
    indice_mes = pd.DataFrame(np.arange(len(meses), dtype=float)[:, None].repeat(len(clientes), axis=1),
                              index=meses, columns=clientes)
    ultimo_con_visita = indice_mes.where(hubo_visita == 1).ffill().shift(1)
    primero = indice_mes.where(con_plan).min()

    anchas = {"visitas_mes": conteo}
    for k in (1, 2, 3, 12):
        anchas[f"retraso_{k}"] = conteo.shift(k)
    anchas["media_movil_3"] = conteo.shift(1).rolling(3, min_periods=1).mean()
    anchas["media_movil_6"] = conteo.shift(1).rolling(6, min_periods=1).mean()
    anchas["meses_desde_ultima_visita"] = indice_mes - ultimo_con_visita
    anchas["meses_con_plan"] = indice_mes - primero
    anchas["muestras_mes_anterior"] = enviadas.shift(1)
    tabla = _larga(anchas)
    tabla = tabla[tabla["visitas_mes"].notna() & tabla["retraso_1"].notna()].copy()

    marcas = comunes.calendario_con_marcas(limpias, meses.min(), meses.max() + pd.offsets.MonthEnd(0))
    habiles_mes = marcas["es_habil"].groupby(comunes.inicio_de_mes(marcas["fecha"]).to_numpy()).sum()
    tabla["dias_habiles_mes"] = habiles_mes.reindex(tabla["fecha"]).to_numpy().astype(float)
    tabla["mes_sen"], tabla["mes_cos"] = comunes.ciclicas(tabla["fecha"].dt.month - 1, 12)
    tabla["base_mes_anterior"] = tabla["retraso_1"]
    tabla["base_media_3"] = tabla["media_movil_3"]

    atributos, numericas, categoricas = _atributos_cliente(limpias)
    tabla = _unir_atributos(tabla, atributos)
    base_numericas = ["retraso_1", "retraso_2", "retraso_3", "retraso_12", "media_movil_3", "media_movil_6"]
    nuevas = ["meses_desde_ultima_visita", "meses_con_plan", "muestras_mes_anterior", "dias_habiles_mes",
              "mes_sen", "mes_cos"]
    momentos = {c: MOMENTO_HISTORIA for c in base_numericas + nuevas}
    momentos.update({c: "maestro" for c in numericas + categoricas})
    papeles = comunes.armar_papeles("visitas_mes", "maestro", "cliente × mes", base_numericas + numericas,
                                    categoricas, nuevas, identificadores=["id_cliente"],
                                    momentos_derivados=momentos,
                                    referencias=["base_mes_anterior", "base_media_3"])
    return comunes.ordenar_tabla(tabla, papeles), papeles


# ---------------------------------------------------------------------------
# Cliente inactivo en el trimestre siguiente
# ---------------------------------------------------------------------------

def cliente_inactivo_trimestre(limpias, integradas):
    """Una fila por cliente y trimestre con actividad; objetivo `inactivo` (no
    envía ninguna muestra en el trimestre siguiente). `fecha` es el primer día
    del trimestre siguiente, cuando empieza a observarse el desenlace; las
    variables resumen solo el trimestre que termina y los anteriores."""
    muestras, marcas = _muestras_marcadas(integradas)
    muestras["_periodo"] = comunes.inicio_de_trimestre(muestras["fecha"]).to_numpy()
    siguiente = a_fecha(muestras["_periodo"]) + pd.DateOffset(months=3)
    muestras["uno"] = 1.0
    muestras["_mes"] = muestras["fecha"].dt.month
    ultima = muestras["fecha"].max()
    trimestres = pd.date_range(muestras["_periodo"].min(), comunes.inicio_de_trimestre([ultima]).iloc[0],
                               freq="QS")
    clientes = sorted(muestras["id_cliente"].unique())

    # Resultados del trimestre que ya se conocían al cerrarlo.
    resultados = []
    if con_datos(muestras, "fecha_reporte"):
        conocido = a_fecha(muestras["fecha_reporte"]) < siguiente
        for origen, destino in (("no_conforme", "_nc"), ("entrega_tardia", "_tardia")):
            if con_datos(muestras, origen):
                muestras[destino] = muestras[origen].where(conocido)
                muestras[destino + "_n"] = muestras[destino].notna().astype(float)
                muestras[destino] = muestras[destino].fillna(0.0)
                resultados.append((origen, destino))

    volumen = _ancha(muestras, "uno", trimestres, clientes)
    activo = volumen > 0
    anchas = {
        "muestras_trimestre": volumen,
        "muestras_trimestre_previo": volumen.shift(1),
        "muestras_4_trimestres": volumen.rolling(4, min_periods=1).sum(),
        "trimestres_activos_previos": activo.astype(float).cumsum().shift(1),
        "_siguiente": volumen.shift(-1),
    }
    nuevas = ["variacion_trimestre", "muestras_4_trimestres", "trimestres_activos_previos"]
    por_trimestre = muestras.groupby(["_periodo", "id_cliente"])
    resumen = pd.DataFrame({
        "envios_trimestre": por_trimestre["_envio"].nunique().astype(float),
        "_ultima": por_trimestre["fecha"].max(),
        "meses_activos_trimestre": por_trimestre["_mes"].nunique().astype(float),
    })
    for marca, nombre in (("_urgente", "proporcion_urgente"), ("_planta", "proporcion_planta")):
        if marca in marcas:
            resumen[nombre] = por_trimestre[marca].mean()
            nuevas.append(nombre)
    for origen, destino in resultados:
        previa = comunes.media_de_entrenamiento(muestras[origen], muestras["fecha"])
        nombre = "tasa_no_conforme" if origen == "no_conforme" else "proporcion_tardia"
        resumen[nombre] = comunes.tasa_suavizada(por_trimestre[destino].sum(), por_trimestre[destino + "_n"].sum(),
                                                 previa)
        nuevas.append(nombre)
    resumen = resumen.rename_axis(index=["fecha", "id_cliente"]).reset_index()
    resumen["fecha"] = a_fecha(resumen["fecha"])

    tabla = _larga(anchas)
    tabla = tabla[tabla["muestras_trimestre"] > 0].merge(resumen, on=["fecha", "id_cliente"], how="left")
    # Se descartan los trimestres cuyo trimestre siguiente no está completo.
    fin_siguiente = tabla["fecha"] + pd.DateOffset(months=6) - pd.Timedelta(days=1)
    tabla = tabla[tabla["_siguiente"].notna() & (fin_siguiente <= ultima + pd.Timedelta(days=3))].copy()
    tabla["inactivo"] = (tabla["_siguiente"] == 0).astype(int)
    tabla["muestras_trimestre_previo"] = tabla["muestras_trimestre_previo"].fillna(0.0)
    tabla["trimestres_activos_previos"] = tabla["trimestres_activos_previos"].fillna(0.0)
    tabla["variacion_trimestre"] = np.log((1.0 + tabla["muestras_trimestre"])
                                          / (1.0 + tabla["muestras_trimestre_previo"]))
    tabla["fecha"] = tabla["fecha"] + pd.DateOffset(months=3)       # inicio del trimestre que se predice
    tabla["dias_desde_ultima_muestra"] = (tabla["fecha"] - a_fecha(tabla["_ultima"])).dt.days.astype(float)
    tabla["trimestre_del_anio"] = ((tabla["fecha"].dt.month - 1) // 3 + 1).astype(float)
    nuevas += ["envios_trimestre", "meses_activos_trimestre", "dias_desde_ultima_muestra", "trimestre_del_anio"]

    atributos, numericas, categoricas = _atributos_cliente(limpias)
    tabla = _unir_atributos(tabla, atributos)
    base_numericas = ["muestras_trimestre", "muestras_trimestre_previo"] + numericas
    if "fecha_vinculacion" in tabla.columns:
        tabla["antiguedad_dias"] = (tabla["fecha"] - a_fecha(tabla["fecha_vinculacion"])).dt.days.astype(float)
        base_numericas.append("antiguedad_dias")
    momentos = {c: MOMENTO_HISTORIA for c in base_numericas + nuevas}
    momentos.update({c: "maestro" for c in numericas + categoricas})
    papeles = comunes.armar_papeles("inactivo", "maestro", "cliente × trimestre", base_numericas, categoricas,
                                    nuevas, identificadores=["id_cliente"], momentos_derivados=momentos)
    return comunes.ordenar_tabla(tabla, papeles), papeles


# ---------------------------------------------------------------------------
# Tablas de agrupamiento (solo periodo de desarrollo, sin objetivo)
# ---------------------------------------------------------------------------

def _papeles_descriptivos(unidad, numericas, identificadores=()):
    """Papeles de una tabla sin objetivo: todas las variables son resúmenes
    descriptivos de datos ya cerrados (no hay predicción ni momento que
    respetar), y los dos grupos coinciden."""
    return comunes.armar_papeles(None, "reporte", unidad, numericas, [], identificadores=identificadores,
                                 momentos_derivados={c: "reporte" for c in numericas})


def segmentos_clientes(limpias, integradas):
    """Una fila por cliente con al menos `MINIMO_MUESTRAS_SEGMENTO` muestras
    en el periodo de desarrollo, descrito por su volumen, su mezcla y su
    regularidad. `fecha` es la de su primera muestra."""
    muestras, marcas = _muestras_marcadas(integradas)
    muestras = muestras[comunes.es_desarrollo(muestras["fecha"])].copy()
    fin = muestras["fecha"].max()
    por_cliente = muestras.groupby("id_cliente")
    tabla = pd.DataFrame({"muestras": por_cliente.size().astype(float), "fecha": por_cliente["fecha"].min()})
    meses = ((fin - tabla["fecha"]).dt.days / DIAS_POR_MES).clip(lower=1.0)
    tabla["muestras_mes"] = tabla["muestras"] / meses
    tabla["muestras_mes_log"] = np.log10(tabla["muestras_mes"])
    tabla["tamano_lote_medio"] = tabla["muestras"] / por_cliente["_envio"].nunique()
    numericas = ["muestras_mes_log", "tamano_lote_medio"]

    producto = next((c for c in ("codigo_producto", "tipo_producto") if con_datos(muestras, c)), None)
    if producto:
        tabla["productos_distintos"] = por_cliente[producto].nunique().astype(float)
        numericas.append("productos_distintos")
    tabla["ensayos_distintos"] = por_cliente["ensayo"].nunique().astype(float)
    numericas.append("ensayos_distintos")
    medias = {"_urgente": "proporcion_urgente", "_planta": "proporcion_planta", "no_conforme": "tasa_no_conforme",
              "entrega_tardia": "proporcion_tardia", "temperatura_transporte_c": "temperatura_media"}
    for origen, nombre in medias.items():
        if con_datos(muestras, origen):
            tabla[nombre] = por_cliente[origen].mean()
            numericas.append(nombre)

    # Regularidad: coeficiente de variación de los días entre envíos.
    llegadas = muestras.drop_duplicates(["id_cliente", "fecha"]).sort_values(["id_cliente", "fecha"])
    brecha = llegadas.groupby("id_cliente")["fecha"].diff().dt.days
    por_brecha = brecha.groupby(llegadas["id_cliente"])
    tabla["regularidad_cv"] = por_brecha.std() / por_brecha.mean()
    # Concentración estacional: índice de Herfindahl del reparto por mes del
    # año (1/12 = repartido por igual, 1 = todo en un mes).
    reparto = pd.crosstab(muestras["id_cliente"], muestras["fecha"].dt.month, normalize="index")
    tabla["concentracion_estacional"] = (reparto ** 2).sum(axis=1)
    numericas += ["regularidad_cv", "concentracion_estacional"]

    atributos, _, _ = _atributos_cliente(limpias)
    inicio = tabla["fecha"]
    if atributos is not None and "fecha_vinculacion" in atributos.columns:
        inicio = a_fecha(atributos["fecha_vinculacion"].reindex(tabla.index)).fillna(inicio)
    tabla["antiguedad_dias"] = (fin - inicio).dt.days.astype(float)
    numericas.append("antiguedad_dias")

    tabla = tabla[tabla["muestras"] >= MINIMO_MUESTRAS_SEGMENTO].rename_axis("id_cliente").reset_index()
    papeles = _papeles_descriptivos("cliente", numericas, ["id_cliente"])
    return comunes.ordenar_tabla(tabla, papeles, extras=["muestras", "muestras_mes"]), papeles


def perfiles_muestra(limpias, integradas):
    """Submuestra aleatoria (con semilla) de hasta `MAXIMO_PERFILES` muestras
    del periodo de desarrollo, descritas por sus variables numéricas.

    Solo entran descriptores presentes en al menos `COBERTURA_MINIMA_PERFIL`
    de las muestras, y solo filas completas: con columnas muy incompletas los
    grupos reproducirían el patrón de faltantes y no el perfil de la muestra.
    """
    muestras, _ = _muestras_marcadas(integradas)
    muestras = muestras[comunes.es_desarrollo(muestras["fecha"])].copy()
    if "rechazada" in muestras.columns:
        muestras = muestras[muestras["rechazada"] == 0].copy()
    candidatas = [c for c in esquema.columnas("muestras", papel="variable", tipo="numero") if c in muestras.columns]
    if con_datos(muestras, "dias_entrega"):
        candidatas.append("dias_entrega")
    numericas = [c for c in candidatas if muestras[c].notna().mean() >= COBERTURA_MINIMA_PERFIL]
    tabla = muestras.dropna(subset=numericas)
    if len(tabla) > MAXIMO_PERFILES:
        tabla = tabla.sample(n=MAXIMO_PERFILES, random_state=SEMILLA)
    identificadores = [c for c in ("id_muestra", "id_cliente") if c in tabla.columns]
    papeles = _papeles_descriptivos("muestra aceptada", numericas, identificadores)
    extras = [c for c in ("matriz", "ensayo", "sector") if c in tabla.columns]
    return comunes.ordenar_tabla(tabla, papeles, extras=extras), papeles


def dias_atipicos(limpias, integradas):
    """Una fila por día hábil del periodo de desarrollo con muestras
    recibidas, descrito por su volumen y sus tasas; sirve para buscar días
    fuera de lo común."""
    muestras, marcas = _muestras_marcadas(integradas)
    muestras = muestras[comunes.es_desarrollo(muestras["fecha"])].copy()
    por_dia = muestras.groupby("fecha")
    tabla = pd.DataFrame({"muestras_dia": por_dia.size().astype(float)})
    numericas = ["muestras_dia"]
    medias = {"_urgente": "proporcion_urgente", "no_conforme": "tasa_no_conforme",
              "control_ambiental_ufc": "control_ambiental_medio", "carga_dia": "carga_dia_media",
              "entrega_tardia": "proporcion_tardia"}
    for origen, nombre in medias.items():
        if con_datos(muestras, origen):
            tabla[nombre] = por_dia[origen].mean()
            numericas.append(nombre)
    tabla = tabla.rename_axis("fecha").reset_index()
    calendario = comunes.calendario_con_marcas(limpias, tabla["fecha"].min(), tabla["fecha"].max())
    habiles = set(calendario.loc[calendario["es_habil"] == 1, "fecha"])
    tabla = tabla[tabla["fecha"].isin(habiles)].copy()
    papeles = _papeles_descriptivos("día hábil", numericas)
    return comunes.ordenar_tabla(tabla, papeles), papeles
