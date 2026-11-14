"""Capa 3 · Tablas de modelado por muestra.

Responsabilidad única: construir, a partir de la tabla integrada, las tablas
cuya unidad de análisis es la muestra (riesgo de no conformidad, falso
positivo, entrega tardía, rechazo, días de entrega y recuento), con sus
variables crudas tomadas del contrato según el momento de la predicción y sus
variables construidas con información estrictamente pasada.

Cada constructor recibe `(limpias, integradas)` y devuelve `(tabla,
papeles)`. Las columnas opcionales del contrato se consultan antes de usarse:
si la exportación real no las trae, se omiten las variables que dependen de
ellas y el resto de la tabla se construye igual.
"""
import numpy as np
import pandas as pd

from proyecto.variables import comunes
from proyecto.variables.comunes import a_fecha, con_datos

# Umbrales operativos de transporte, declarados de antemano (no se ajustan con
# los datos): cadena de frío rota y transporte de más de un día.
UMBRAL_TEMPERATURA_C = 8.0
UMBRAL_HORAS_TRANSPORTE = 24.0
# Meses de mayor lluvia en el régimen bimodal de la sabana de Bogotá (supuesto
# declarado; con otra sede debe revisarse).
MESES_LLUVIOSOS = (4, 5, 10, 11)
VENTANA_HISTORIA_DIAS = 90
DIAS_HABILES_CARGA = 5
HORIZONTE_FESTIVOS_DIAS = 7
CUANTIL_CONTROL_ALTO = 0.90


# ---------------------------------------------------------------------------
# Piezas compartidas
# ---------------------------------------------------------------------------

def _muestras_con_fecha(integradas, columna_fecha="fecha_recepcion"):
    """Copia de la tabla integrada con la columna `fecha` (la que ordena y
    parte la tabla), sin las filas que no la tienen."""
    tabla = integradas.copy()
    fecha = a_fecha(tabla["fecha_recepcion"])
    if columna_fecha != "fecha_recepcion" and con_datos(tabla, columna_fecha):
        fecha = a_fecha(tabla[columna_fecha]).fillna(fecha)
    tabla["fecha"] = fecha.dt.normalize()
    return tabla[tabla["fecha"].notna()].reset_index(drop=True)


def _identificadores(tabla):
    return [c for c in ("id_muestra", "id_cliente") if c in tabla.columns]


def _fecha_resultado(tabla):
    """Columna con la fecha en que el resultado de una muestra pasó a
    conocerse (el informe; si no existe, la lectura). None si no hay ninguna."""
    for columna in ("fecha_reporte", "fecha_lectura"):
        if con_datos(tabla, columna):
            return columna
    return None


def _agregar_transporte(tabla, momentos):
    """Interacción temperatura × horas y marcas de umbral."""
    nuevas = []
    tiene_temperatura = con_datos(tabla, "temperatura_transporte_c")
    tiene_horas = con_datos(tabla, "horas_transporte")
    if tiene_temperatura:
        temperatura = tabla["temperatura_transporte_c"]
        tabla["temperatura_alta"] = (temperatura > UMBRAL_TEMPERATURA_C).astype(float).where(temperatura.notna())
        nuevas.append("temperatura_alta")
    if tiene_horas:
        horas = tabla["horas_transporte"]
        tabla["transporte_largo"] = (horas > UMBRAL_HORAS_TRANSPORTE).astype(float).where(horas.notna())
        nuevas.append("transporte_largo")
    if tiene_temperatura and tiene_horas:
        tabla["temperatura_por_horas"] = tabla["temperatura_transporte_c"].clip(lower=0) * tabla["horas_transporte"]
        nuevas.append("temperatura_por_horas")
    momentos.update({c: "llegada" for c in nuevas})
    return nuevas


def _agregar_calendario(tabla, momentos):
    """Mes y día de la semana en forma cíclica y marca de mes lluvioso, a
    partir de la fecha de la fila."""
    fecha = tabla["fecha"]
    tabla["mes_sen"], tabla["mes_cos"] = comunes.ciclicas(fecha.dt.month - 1, 12)
    tabla["dia_semana_sen"], tabla["dia_semana_cos"] = comunes.ciclicas(fecha.dt.dayofweek, 7)
    tabla["mes_lluvioso"] = fecha.dt.month.isin(MESES_LLUVIOSOS).astype(float)
    nuevas = ["mes_sen", "mes_cos", "dia_semana_sen", "dia_semana_cos", "mes_lluvioso"]
    momentos.update({c: "llegada" for c in nuevas})
    return nuevas


def _agregar_tamano_lote(tabla, momentos):
    """Número de muestras que llegaron en el mismo envío (o en la misma visita
    de muestreo). El envío llega completo, así que se conoce a la llegada."""
    llave = pd.Series(np.nan, index=tabla.index, dtype=object)
    for columna in ("id_lote_envio", "id_muestreo"):
        if con_datos(tabla, columna):
            llave = llave.where(llave.notna(), tabla[columna].astype(object))
    if not llave.notna().any():
        return []
    tabla["tamano_lote"] = llave.groupby(llave).transform("size").astype(float).where(llave.notna())
    momentos["tamano_lote"] = "llegada"
    return ["tamano_lote"]


def _agregar_historia(tabla, llave, valor, prefijo, fecha_conocido, momentos, ventana=None, momento="llegada"):
    """Historia de `valor` dentro de `llave`: número de resultados conocidos
    antes de la fecha de la fila y tasa suavizada con la previa del periodo de
    entrenamiento (y lo mismo en la ventana reciente, si se pide)."""
    if not con_datos(tabla, llave) or not con_datos(tabla, valor) or fecha_conocido is None:
        return []
    previa = comunes.media_de_entrenamiento(tabla[valor], tabla["fecha"])
    if np.isnan(previa):
        return []
    historia = comunes.historial_conocido(tabla, llave, "fecha", fecha_conocido, valor, ventana_dias=ventana)
    tabla[f"{prefijo}_n"] = historia["n"]
    tabla[f"{prefijo}_tasa"] = comunes.tasa_suavizada(historia["suma"], historia["n"], previa)
    nuevas = [f"{prefijo}_n", f"{prefijo}_tasa"]
    if ventana:
        tabla[f"{prefijo}_n_{ventana}"] = historia["n_ventana"]
        tabla[f"{prefijo}_tasa_{ventana}"] = comunes.tasa_suavizada(
            historia["suma_ventana"], historia["n_ventana"], previa)
        nuevas += [f"{prefijo}_n_{ventana}", f"{prefijo}_tasa_{ventana}"]
    momentos.update({c: momento for c in nuevas})
    return nuevas


def _llave_producto(tabla):
    """Referencia de producto dentro de su cliente (None si no hay código)."""
    if not con_datos(tabla, "codigo_producto"):
        return None
    codigo = tabla["codigo_producto"]
    llave = tabla["id_cliente"].astype(str) + "|" + codigo.astype(str)
    return llave.where(codigo.notna()).astype(object)


# ---------------------------------------------------------------------------
# Riesgo a la recepción: no conformidad y recuento
# ---------------------------------------------------------------------------

def _tabla_riesgo(integradas):
    """Tabla común a `no_conforme_recepcion` y `recuento_log_recepcion`:
    todas las muestras con fecha, con las variables crudas hasta la recepción
    y las construidas. Devuelve (tabla, numéricas base, categóricas base,
    nuevas, momentos)."""
    tabla = _muestras_con_fecha(integradas)
    numericas, categoricas = comunes.variables_crudas(tabla, "recepcion")
    momentos, nuevas = {}, []
    conocido = _fecha_resultado(tabla)
    nuevas += _agregar_historia(tabla, "id_cliente", "no_conforme", "historia_cliente", conocido,
                                momentos, ventana=VENTANA_HISTORIA_DIAS)
    llave = _llave_producto(tabla)
    if llave is not None:
        tabla["_producto"] = llave
        nuevas += _agregar_historia(tabla, "_producto", "no_conforme", "historia_producto", conocido, momentos)
    nuevas += _agregar_transporte(tabla, momentos)
    nuevas += _agregar_calendario(tabla, momentos)
    nuevas += _agregar_tamano_lote(tabla, momentos)
    return tabla, numericas, categoricas, nuevas, momentos


def no_conforme_recepcion(limpias, integradas):
    """Muestras aceptadas; objetivo `no_conforme`; se predice al recibir la
    muestra, antes de sembrarla."""
    tabla, numericas, categoricas, nuevas, momentos = _tabla_riesgo(integradas)
    filas = tabla["no_conforme"].notna()
    if "rechazada" in tabla.columns:
        filas &= tabla["rechazada"] == 0
    tabla = tabla[filas].copy()
    tabla["no_conforme"] = tabla["no_conforme"].astype(int)
    papeles = comunes.armar_papeles("no_conforme", "recepcion", "muestra aceptada", numericas, categoricas,
                                    nuevas, identificadores=_identificadores(tabla),
                                    momentos_derivados=momentos)
    return comunes.ordenar_tabla(tabla, papeles), papeles


def recuento_log_recepcion(limpias, integradas):
    """Muestras aceptadas con ensayo cuantitativo; objetivo `recuento_log` =
    log10(1 + recuento en UFC); mismas variables que la no conformidad."""
    tabla, numericas, categoricas, nuevas, momentos = _tabla_riesgo(integradas)
    filas = tabla["recuento_ufc"].notna()
    if "es_cuantitativo" in tabla.columns:
        filas &= tabla["es_cuantitativo"] == 1
    if "rechazada" in tabla.columns:
        filas &= tabla["rechazada"] == 0
    tabla = tabla[filas].copy()
    tabla["recuento_log"] = np.log10(1.0 + tabla["recuento_ufc"].clip(lower=0))
    # En esta tabla todos los ensayos son cuantitativos: la marca no informa.
    numericas = [c for c in numericas if c != "es_cuantitativo"]
    papeles = comunes.armar_papeles("recuento_log", "recepcion", "muestra aceptada con recuento",
                                    numericas, categoricas, nuevas,
                                    identificadores=_identificadores(tabla), momentos_derivados=momentos)
    return comunes.ordenar_tabla(tabla, papeles), papeles


# ---------------------------------------------------------------------------
# Plazo de entrega: entrega tardía y días de entrega
# ---------------------------------------------------------------------------

def _agregar_carga_laboratorio(tabla, limpias, momentos):
    """Carga del laboratorio conocida al llegar la muestra y marcas del
    calendario que afectan el plazo."""
    fecha = tabla["fecha"]
    marcas = comunes.calendario_con_marcas(limpias, fecha.min(), fecha.max()).set_index("fecha")
    habiles = marcas.index[marcas["es_habil"] == 1]
    # Fecha del quinto día hábil anterior a la llegada (inicio de la ventana).
    previos = (marcas["habiles_acumulados"] - marcas["es_habil"]).reindex(fecha).to_numpy().astype(int)
    inicio = habiles.to_numpy()[np.clip(previos - DIAS_HABILES_CARGA, 0, len(habiles) - 1)]
    inicio = np.where(previos >= DIAS_HABILES_CARGA, inicio, marcas.index[0].to_datetime64())
    tabla["carga_previa_5"] = comunes.conteo_previo(fecha, fecha) - comunes.conteo_previo(fecha, inicio)
    nuevas = ["carga_previa_5"]
    if con_datos(tabla, "fecha_reporte"):
        # Muestras recibidas antes de hoy cuyo informe aún no había salido.
        con_informe = tabla["fecha_reporte"].notna()
        tabla["pendientes_al_llegar"] = (comunes.conteo_previo(fecha[con_informe], fecha)
                                         - comunes.conteo_previo(tabla.loc[con_informe, "fecha_reporte"], fecha))
        nuevas.append("pendientes_al_llegar")

    festivos = marcas["festivos_acumulados"]
    tabla["festivos_proximos"] = (festivos.reindex(fecha + pd.Timedelta(days=HORIZONTE_FESTIVOS_DIAS)).to_numpy()
                                  - festivos.reindex(fecha).to_numpy())
    nuevas.append("festivos_proximos")
    momentos.update({c: "llegada" for c in nuevas})
    if con_datos(tabla, "dias_prometidos"):
        plazo = pd.to_timedelta(tabla["dias_prometidos"].fillna(0).clip(lower=0, upper=30), unit="D")
        habiles_acumulados = marcas["habiles_acumulados"]
        en_plazo = habiles_acumulados.reindex(fecha + plazo).to_numpy() - habiles_acumulados.reindex(fecha).to_numpy()
        tabla["habiles_en_plazo"] = pd.Series(en_plazo, index=tabla.index).where(tabla["dias_prometidos"].notna())
        momentos["habiles_en_plazo"] = "recepcion"
        nuevas.append("habiles_en_plazo")
    return nuevas


def _tabla_entrega(limpias, integradas):
    tabla = _muestras_con_fecha(integradas)
    numericas, categoricas = comunes.variables_crudas(tabla, "recepcion")
    momentos = {}
    nuevas = _agregar_carga_laboratorio(tabla, limpias, momentos)
    nuevas += _agregar_calendario(tabla, momentos)
    nuevas += _agregar_tamano_lote(tabla, momentos)
    return tabla, numericas, categoricas, nuevas, momentos


def entrega_tardia_recepcion(limpias, integradas):
    """Muestras aceptadas; objetivo `entrega_tardia` (el informe salió después
    del plazo prometido); se predice al recibir la muestra."""
    tabla, numericas, categoricas, nuevas, momentos = _tabla_entrega(limpias, integradas)
    tabla = tabla[tabla["entrega_tardia"].notna()].copy()
    tabla["entrega_tardia"] = tabla["entrega_tardia"].astype(int)
    papeles = comunes.armar_papeles("entrega_tardia", "recepcion", "muestra aceptada", numericas, categoricas,
                                    nuevas, identificadores=_identificadores(tabla),
                                    momentos_derivados=momentos)
    return comunes.ordenar_tabla(tabla, papeles), papeles


def dias_entrega_recepcion(limpias, integradas):
    """Muestras aceptadas; objetivo `dias_entrega` (días calendario hasta el
    informe); mismas variables que la entrega tardía."""
    tabla, numericas, categoricas, nuevas, momentos = _tabla_entrega(limpias, integradas)
    tabla = tabla[tabla["dias_entrega"].notna()].copy()
    papeles = comunes.armar_papeles("dias_entrega", "recepcion", "muestra aceptada", numericas, categoricas,
                                    nuevas, identificadores=_identificadores(tabla),
                                    momentos_derivados=momentos)
    return comunes.ordenar_tabla(tabla, papeles), papeles


# ---------------------------------------------------------------------------
# Rechazo a la llegada
# ---------------------------------------------------------------------------

def rechazo_llegada(limpias, integradas):
    """Todas las muestras; objetivo `rechazada`; se predice con lo medido al
    llegar, antes de la decisión de recepción."""
    tabla = _muestras_con_fecha(integradas)
    tabla = tabla[tabla["rechazada"].notna()].copy()
    tabla["rechazada"] = tabla["rechazada"].astype(int)
    numericas, categoricas = comunes.variables_crudas(tabla, "llegada")
    momentos = {}
    # Un rechazo se conoce el mismo día de la llegada: cuenta para las
    # muestras del cliente que lleguen en días posteriores.
    nuevas = _agregar_historia(tabla, "id_cliente", "rechazada", "historia_rechazo", "fecha", momentos)
    nuevas += _agregar_transporte(tabla, momentos)
    nuevas += _agregar_calendario(tabla, momentos)
    nuevas += _agregar_tamano_lote(tabla, momentos)
    papeles = comunes.armar_papeles("rechazada", "llegada", "muestra recibida", numericas, categoricas,
                                    nuevas, identificadores=_identificadores(tabla),
                                    momentos_derivados=momentos)
    return comunes.ordenar_tabla(tabla, papeles), papeles


# ---------------------------------------------------------------------------
# Falso positivo a la lectura
# ---------------------------------------------------------------------------

def falso_positivo_lectura(limpias, integradas):
    """Muestras no conformes que se reensayaron; objetivo `falso_positivo`
    (el reensayo no confirmó el resultado); se predice al leer las placas."""
    tabla = _muestras_con_fecha(integradas, "fecha_lectura")
    numericas, categoricas = comunes.variables_crudas(
        tabla, "lectura", incluir=("recuento_ufc", "recuento_censurado"))
    momentos, nuevas = {}, []

    if con_datos(tabla, "recuento_ufc") and con_datos(tabla, "limite_ufc"):
        tabla["razon_log_recuento_limite"] = (np.log10(1.0 + tabla["recuento_ufc"].clip(lower=0))
                                              - np.log10(1.0 + tabla["limite_ufc"].clip(lower=0)))
        momentos["razon_log_recuento_limite"] = "lectura"
        nuevas.append("razon_log_recuento_limite")

    if con_datos(tabla, "analista"):
        siembra = a_fecha(tabla["fecha_siembra"]).fillna(tabla["fecha"]) if "fecha_siembra" in tabla.columns \
            else tabla["fecha"]
        debut = siembra.groupby(tabla["analista"]).transform("min")
        tabla["dias_desde_debut_analista"] = (siembra - debut).dt.days.astype(float)
        momentos["dias_desde_debut_analista"] = "siembra"
        nuevas.append("dias_desde_debut_analista")

    # El falso positivo solo se conoce cuando sale el informe del reensayo.
    conocido = "fecha_reporte" if con_datos(tabla, "fecha_reporte") else None
    for llave, prefijo in (("analista", "historia_fp_analista"), ("incubadora", "historia_fp_incubadora"),
                           ("lote_medio", "historia_fp_lote_medio")):
        nuevas += _agregar_historia(tabla, llave, "falso_positivo", prefijo, conocido, momentos,
                                    ventana=VENTANA_HISTORIA_DIAS if llave == "analista" else None,
                                    momento="lectura")

    if con_datos(tabla, "control_ambiental_ufc"):
        control = tabla["control_ambiental_ufc"]
        referencia = control[comunes.es_entrenamiento(tabla["fecha"])].quantile(CUANTIL_CONTROL_ALTO)
        if not np.isnan(referencia):
            tabla["control_ambiental_alto"] = (control > referencia).astype(float).where(control.notna())
            momentos["control_ambiental_alto"] = "lectura"
            nuevas.append("control_ambiental_alto")

    if con_datos(tabla, "fecha_lectura"):
        lectura = a_fecha(tabla["fecha_lectura"])
        tabla["lecturas_dia"] = lectura.groupby(lectura).transform("size").astype(float).where(lectura.notna())
        momentos["lecturas_dia"] = "lectura"
        nuevas.append("lecturas_dia")

    tabla = tabla[tabla["falso_positivo"].notna()].copy()
    tabla["falso_positivo"] = tabla["falso_positivo"].astype(int)
    # Tras filtrar pueden quedar columnas sin datos o constantes inútiles.
    numericas = [c for c in numericas if con_datos(tabla, c)]
    categoricas = [c for c in categoricas if con_datos(tabla, c)]
    papeles = comunes.armar_papeles("falso_positivo", "lectura", "resultado no conforme reensayado",
                                    numericas, categoricas, nuevas,
                                    identificadores=_identificadores(tabla), momentos_derivados=momentos)
    return comunes.ordenar_tabla(tabla, papeles), papeles
