"""Capa 3 · Utilidades comunes de las tablas de modelado.

Responsabilidad única: reunir las piezas que comparten los constructores de
variables (`muestra.py`, `series.py`, `cliente.py`): preguntar al contrato qué
columnas crudas pueden usarse en un momento dado, armar el diccionario de
papeles, calcular historiales que respetan el orden en que la información se
conoce y auditar que ninguna variable llegue después del momento de la
predicción.

Convenciones de la capa:

- Cada constructor devuelve `(tabla, papeles)`. La tabla trae la columna
  `fecha`, el objetivo, las variables y los identificadores; `papeles`
  declara qué es cada cosa.
- El grupo `base` reúne columnas crudas disponibles en el momento de la
  predicción; `ampliado` es `base` más las variables construidas.
- Los constructores recorren todas las fechas, pero solo aplican
  transformaciones deterministas e información estrictamente pasada. Las
  constantes de referencia (tasa previa, nivel habitual) se calculan solo con
  el periodo de entrenamiento.
"""
import numpy as np
import pandas as pd

from proyecto.base.config import CORTE_PRUEBA, CORTE_VALIDACION
from proyecto.datos import esquema
from proyecto.datos.calendario import tabla_calendario
from proyecto.preparacion.integracion import banderas_calendario

# Columnas derivadas que son resultados (no variables). Un constructor solo
# puede usarlas si las pide de forma explícita y su momento lo permite.
RESULTADOS_DERIVADOS = ("rechazada", "recuento_ufc", "recuento_censurado", "no_conforme",
                        "reensayada", "falso_positivo", "dias_entrega", "entrega_tardia")

# El año no se usa como variable: en producción siempre tomaría valores que el
# modelo nunca vio (el futuro), y un árbol no puede extrapolarlo.
NO_VARIABLES = ("anio",)

# Momento de las variables que resumen periodos ya cerrados (retrasos de una
# serie, historia de un cliente en trimestres anteriores): se conocen antes de
# que empiece la unidad que se predice.
MOMENTO_HISTORIA = "maestro"

# Peso de la tasa previa en las tasas suavizadas: equivale a añadir esta
# cantidad de observaciones «promedio» a la historia de cada entidad.
FUERZA_PREVIA = 20.0

_TIPOS_CONTRATO = {c.nombre: c.tipo for t in ("clientes", "muestras") for c in esquema.ESQUEMA[t]}


# ---------------------------------------------------------------------------
# Papeles y columnas crudas
# ---------------------------------------------------------------------------

def _unicos(nombres):
    return list(dict.fromkeys(nombres))


def con_datos(tabla, columna):
    """True si la columna existe y tiene al menos un valor."""
    return columna in tabla.columns and bool(tabla[columna].notna().any())


def variables_crudas(tabla, momento, incluir=(), excluir=(), tabla_esquema="muestras"):
    """Columnas crudas utilizables en `momento`, separadas en (numéricas,
    categóricas).

    Parte de las variables del contrato y de las columnas derivadas conocidas
    hasta `momento`, y se queda con las que existen en `tabla` y traen algún
    dato. Deja fuera identificadores, fechas, textos libres, los resultados
    derivados (salvo los que se pidan en `incluir`) y lo listado en `excluir`.
    """
    candidatas = esquema.columnas(tabla_esquema, papel="variable", hasta=momento)
    candidatas += [d for d in esquema.derivadas(tabla_esquema, hasta=momento)
                   if d not in RESULTADOS_DERIVADOS]
    candidatas += list(incluir)
    numericas, categoricas = [], []
    for columna in _unicos(candidatas):
        if columna in excluir or columna in NO_VARIABLES or not con_datos(tabla, columna):
            continue
        tipo = _TIPOS_CONTRATO.get(columna)
        if tipo is None:   # derivada sin tipo en el contrato: se mira el dato
            if pd.api.types.is_datetime64_any_dtype(tabla[columna]):
                continue
            tipo = "numero" if pd.api.types.is_numeric_dtype(tabla[columna]) else "categoria"
        if tipo == "numero":
            numericas.append(columna)
        elif tipo == "categoria":
            categoricas.append(columna)
    return numericas, categoricas


def armar_papeles(objetivo, momento, unidad, base_numericas, base_categoricas,
                  nuevas_numericas=(), nuevas_categoricas=(), identificadores=(),
                  momentos_derivados=None, referencias=()):
    """Diccionario de papeles de una tabla de modelado (serializable a JSON).

    `nuevas_*` son las variables construidas: entran solo al grupo `ampliado`.
    `momentos_derivados` declara el momento de cada variable que no está en el
    contrato; `referencias` son columnas de pronóstico ingenuo (no variables).
    """
    base = _unicos(list(base_numericas) + list(base_categoricas))
    ampliado = _unicos(base + list(nuevas_numericas) + list(nuevas_categoricas))
    return {
        "objetivo": objetivo,
        "fecha": "fecha",
        "numericas": _unicos(list(base_numericas) + list(nuevas_numericas)),
        "categoricas": _unicos(list(base_categoricas) + list(nuevas_categoricas)),
        "grupos": {"base": base, "ampliado": ampliado},
        "momento": momento,
        "unidad": unidad,
        "identificadores": list(identificadores),
        "momentos_derivados": dict(momentos_derivados or {}),
        "referencias": list(referencias),
    }


def columnas_de_grupo(papeles, grupo="ampliado"):
    """Devuelve (numéricas, categóricas) del grupo de variables pedido."""
    del_grupo = set(papeles["grupos"][grupo])
    return ([c for c in papeles["numericas"] if c in del_grupo],
            [c for c in papeles["categoricas"] if c in del_grupo])


def ordenar_tabla(tabla, papeles, extras=()):
    """Deja la tabla con las columnas en orden fijo (fecha, identificadores,
    objetivo, referencias, variables), ordenada por fecha y con índice nuevo."""
    columnas = ["fecha"] + papeles["identificadores"]
    if papeles["objetivo"]:
        columnas.append(papeles["objetivo"])
    columnas += papeles["referencias"] + list(extras) + papeles["grupos"]["ampliado"]
    columnas = [c for c in _unicos(columnas) if c in tabla.columns]
    return tabla[columnas].sort_values("fecha", kind="stable").reset_index(drop=True)


def auditar_momentos(tabla, papeles, tabla_esquema="muestras"):
    """Auditoría de fuga de información por momento.

    Para cada variable de `papeles` busca su momento (primero el declarado en
    `papeles["momentos_derivados"]`, después el del contrato) y lo compara con
    el momento del análisis. Devuelve la lista de hallazgos; cada uno es un
    diccionario con `columna`, `momento` y `problema`. Lista vacía = sin fugas.
    """
    limite = papeles["momento"]
    declarados = papeles.get("momentos_derivados", {})
    variables = _unicos(papeles["numericas"] + papeles["categoricas"])
    hallazgos = []
    for columna in variables:
        momento = declarados.get(columna) or esquema.momento_de(tabla_esquema, columna)
        if columna not in tabla.columns:
            hallazgos.append({"columna": columna, "momento": momento, "problema": "no está en la tabla"})
        elif momento is None:
            hallazgos.append({"columna": columna, "momento": None, "problema": "sin momento declarado"})
        elif momento not in esquema.MOMENTOS:
            hallazgos.append({"columna": columna, "momento": momento, "problema": "momento desconocido"})
        elif not esquema.es_anterior_o_igual(momento, limite):
            hallazgos.append({"columna": columna, "momento": momento,
                              "problema": f"posterior al momento del análisis ({limite})"})
    prohibidas = set(papeles["identificadores"]) | set(papeles["referencias"]) | {papeles["objetivo"], "fecha"}
    for columna in variables:
        if columna in prohibidas:
            hallazgos.append({"columna": columna, "momento": None,
                              "problema": "identificador, objetivo o referencia usado como variable"})
    sobrantes = set(papeles["grupos"]["base"]) - set(papeles["grupos"]["ampliado"])
    for columna in sorted(sobrantes):
        hallazgos.append({"columna": columna, "momento": None, "problema": "está en base y no en ampliado"})
    return hallazgos


# ---------------------------------------------------------------------------
# Tiempo: periodos de referencia e historiales
# ---------------------------------------------------------------------------

def a_fecha(serie):
    """Serie de fechas con resolución fija (evita diferencias entre versiones
    de pandas al cruzar por fecha)."""
    return pd.to_datetime(serie, errors="coerce").astype("datetime64[ns]")


def es_entrenamiento(fechas):
    """Máscara de las filas del periodo de entrenamiento."""
    return (a_fecha(pd.Series(fechas)) < pd.Timestamp(CORTE_VALIDACION)).to_numpy()


def es_desarrollo(fechas):
    """Máscara de las filas anteriores al periodo de prueba."""
    return (a_fecha(pd.Series(fechas)) < pd.Timestamp(CORTE_PRUEBA)).to_numpy()


def media_de_entrenamiento(valores, fechas):
    """Media de `valores` en el periodo de entrenamiento (NaN si no hay datos).
    Es la única forma permitida de fijar una constante de referencia."""
    valores = pd.to_numeric(pd.Series(np.asarray(valores)), errors="coerce")
    del_periodo = valores[es_entrenamiento(fechas)]
    return float(del_periodo.mean()) if del_periodo.notna().any() else float("nan")


def nivel_habitual(serie, fechas):
    """Nivel más frecuente de una categoría en el periodo de entrenamiento
    (None si no hay datos)."""
    del_periodo = pd.Series(np.asarray(serie, dtype=object))[es_entrenamiento(fechas)].dropna()
    if not len(del_periodo):
        return None
    conteo = del_periodo.value_counts()
    return sorted(conteo[conteo == conteo.max()].index)[0]


def tasa_suavizada(suma, n, previa, fuerza=FUERZA_PREVIA):
    """Tasa con encogimiento hacia la previa: (suma + previa·fuerza) / (n +
    fuerza). Sin historia devuelve la previa."""
    return (np.asarray(suma, dtype=float) + previa * fuerza) / (np.asarray(n, dtype=float) + fuerza)


def _acumulado_hasta(eventos, consultas, llave, momento_consulta):
    """Para cada consulta, número y suma de los eventos de su misma llave
    conocidos estrictamente antes de `momento_consulta`."""
    izquierda = pd.DataFrame({llave: consultas[llave].astype(object)})
    izquierda["_cuando"] = a_fecha(momento_consulta)
    izquierda["_posicion"] = np.arange(len(izquierda))
    validas = izquierda.dropna(subset=[llave, "_cuando"]).sort_values("_cuando", kind="stable")
    resultado = np.zeros((len(izquierda), 2))
    if len(validas) and len(eventos):
        cruce = pd.merge_asof(validas, eventos, left_on="_cuando", right_on="_conocido", by=llave,
                              direction="backward", allow_exact_matches=False)
        resultado[cruce["_posicion"].to_numpy()] = cruce[["_n", "_s"]].fillna(0.0).to_numpy()
    return resultado


def historial_conocido(tabla, llave, fecha_evento, fecha_conocido, valor, ventana_dias=None):
    """Historia de cada fila dentro de su `llave` (cliente, analista...), con
    orden temporal estricto.

    Un resultado anterior cuenta solo si ya se conocía cuando ocurre la fila
    nueva: su `fecha_conocido` debe ser estrictamente anterior a la
    `fecha_evento` de la fila. No basta con que la fila anterior haya ocurrido
    antes. Las filas sin `valor` o sin `fecha_conocido` no aportan historia.

    Devuelve un DataFrame alineado con `tabla` con las columnas `n` y `suma`
    (toda la historia conocida) y, si se indica `ventana_dias`, `n_ventana` y
    `suma_ventana` (resultados conocidos en los últimos `ventana_dias` días).
    Cálculo vectorizado: sumas acumuladas por llave y cruce `merge_asof`.
    """
    eventos = tabla[[llave, fecha_conocido, valor]].dropna()
    eventos = pd.DataFrame({llave: eventos[llave].astype(object).to_numpy(),
                            "_conocido": a_fecha(eventos[fecha_conocido]).to_numpy(),
                            "_valor": eventos[valor].astype(float).to_numpy()})
    eventos[llave] = eventos[llave].astype(object)
    eventos = eventos.sort_values("_conocido", kind="stable")
    eventos["_n"] = eventos.groupby(llave).cumcount() + 1.0
    eventos["_s"] = eventos.groupby(llave)["_valor"].cumsum()
    # Una fila por (llave, fecha): la última trae el acumulado completo del día.
    eventos = eventos.drop_duplicates([llave, "_conocido"], keep="last")[[llave, "_conocido", "_n", "_s"]]

    momento = a_fecha(tabla[fecha_evento])
    total = _acumulado_hasta(eventos, tabla, llave, momento)
    historia = pd.DataFrame({"n": total[:, 0], "suma": total[:, 1]}, index=tabla.index)
    if ventana_dias:
        previo = _acumulado_hasta(eventos, tabla, llave, momento - pd.Timedelta(days=ventana_dias))
        historia["n_ventana"] = total[:, 0] - previo[:, 0]
        historia["suma_ventana"] = total[:, 1] - previo[:, 1]
    return historia


def conteo_previo(fechas_evento, fechas_consulta):
    """Número de eventos con fecha estrictamente anterior a cada fecha de
    consulta (las fechas faltantes de los eventos no cuentan)."""
    eventos = np.sort(a_fecha(pd.Series(fechas_evento)).dropna().to_numpy())
    consulta = a_fecha(pd.Series(fechas_consulta)).to_numpy()
    return np.searchsorted(eventos, consulta, side="left").astype(float)


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------

def calendario_con_marcas(limpias, desde, hasta, margen_dias=40):
    """Una fila por día entre `desde` y `hasta` (más un margen) con las marcas
    de `integracion.banderas_calendario` y las columnas `es_habil` y
    `es_festivo`. Usa el calendario limpio si cubre el rango; si no, lo
    recalcula (el calendario es determinista y se conoce de antemano)."""
    desde = pd.Timestamp(desde) - pd.Timedelta(days=margen_dias)
    hasta = pd.Timestamp(hasta) + pd.Timedelta(days=margen_dias)
    calendario = limpias.get("calendario") if limpias is not None else None
    cubre = (calendario is not None and len(calendario)
             and {"fecha", "es_habil", "es_festivo"} <= set(calendario.columns)
             and a_fecha(calendario["fecha"]).min() <= desde and a_fecha(calendario["fecha"]).max() >= hasta)
    if not cubre:
        calendario = tabla_calendario(desde, hasta)
    calendario = calendario.sort_values("fecha").reset_index(drop=True)
    marcas = banderas_calendario(calendario)
    marcas["fecha"] = a_fecha(marcas["fecha"])
    marcas["es_habil"] = calendario["es_habil"].astype(int).to_numpy()
    marcas["es_festivo"] = calendario["es_festivo"].astype(int).to_numpy()
    marcas["habiles_acumulados"] = marcas["es_habil"].cumsum()
    marcas["festivos_acumulados"] = marcas["es_festivo"].cumsum()
    return marcas


def ciclicas(valores, periodo):
    """Par (seno, coseno) de una variable periódica (mes, día de la semana)."""
    angulo = 2 * np.pi * np.asarray(valores, dtype=float) / periodo
    return np.sin(angulo), np.cos(angulo)


def lunes_de(fechas):
    """Lunes de la semana de cada fecha."""
    fechas = a_fecha(pd.Series(fechas))
    return (fechas - pd.to_timedelta(fechas.dt.dayofweek, unit="D")).dt.normalize()


def periodo_completo(fecha_final_periodo, ultima_fecha, tolerancia_dias=3):
    """True si los datos llegan (con una tolerancia de fin de semana) hasta el
    último día del periodo."""
    return pd.Timestamp(ultima_fecha) >= pd.Timestamp(fecha_final_periodo) - pd.Timedelta(days=tolerancia_dias)


def apilar(ancha, nombre, columna):
    """Pasa una tabla ancha (índice = fecha, una columna por serie) a una
    serie larga indexada por (fecha, `columna`), conservando los faltantes."""
    larga = (ancha.rename_axis(index="fecha", columns=columna).reset_index()
             .melt(id_vars="fecha", var_name=columna, value_name=nombre))
    return larga.set_index(["fecha", columna])[nombre]


def inicio_de_mes(fechas):
    """Primer día del mes de cada fecha."""
    fechas = a_fecha(pd.Series(fechas))
    return (fechas - pd.to_timedelta(fechas.dt.day - 1, unit="D")).dt.normalize()


def inicio_de_trimestre(fechas):
    """Primer día del trimestre calendario de cada fecha."""
    fechas = a_fecha(pd.Series(fechas))
    mes_inicial = 3 * ((fechas.dt.month - 1) // 3) + 1
    return a_fecha(pd.to_datetime({"year": fechas.dt.year, "month": mes_inicial, "day": 1}, errors="coerce"))
