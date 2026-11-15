"""Capa 6 · Catálogo de análisis candidatos.

Responsabilidad única: declarar, en un solo lugar y antes de mirar ningún
resultado, qué análisis podrían hacerse con la base del laboratorio (una
`FichaAnalisis` por candidato), qué columnas necesita cada uno y cómo se
construye su tabla de modelado. El cribado de nivel 1 (`cribado/analisis.py`)
recorre este catálogo y decide con una regla fija cuál se estudia a fondo.

Las fichas de tipo «bayes» son analíticas: aplican el teorema de Bayes a los
controles de calidad y no se criban ni tienen tabla de modelado.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from proyecto.variables import cliente, muestra, series

TIPOS = ["regresion", "clasificacion", "agrupamiento", "bayes"]
TIPOS_SUPERVISADOS = ["regresion", "clasificacion"]
METRICA_AGRUPAMIENTO = "silueta"


@dataclass(frozen=True)
class FichaAnalisis:
    """Un análisis candidato.

    `metrica` es un nombre de `evaluacion.metricas.METRICAS` en los análisis
    supervisados, «silueta» en los de agrupamiento y None en los analíticos.
    `columna_linea_base` es la columna de la tabla de modelado con un
    pronóstico ingenuo que el modelo debe superar (None = estimador `Dummy`).
    `requiere` lista, por tabla limpia, las columnas sin las cuales el
    análisis no puede hacerse. `valor_negocio` va de 1 a 3.
    """
    nombre: str
    titulo: str
    tipo: str
    unidad: str
    objetivo: Optional[str]
    momento: Optional[str]
    metrica: Optional[str]
    decision: str
    valor_negocio: int
    requiere: dict = field(default_factory=dict)
    columna_linea_base: Optional[str] = None
    constructor: Optional[Callable] = None

    def __post_init__(self):
        if self.tipo not in TIPOS or self.valor_negocio not in (1, 2, 3):
            raise ValueError(f"Ficha mal declarada: {self.nombre}")


# Valor de negocio, declarado a priori (antes de cualquier resultado):
#   3 = lo que el laboratorio pidió de forma explícita: prever el volumen de
#       muestras y anticipar no conformidades y falsos positivos.
#   2 = apoyo a la operación diaria: plazos de entrega, carga del día,
#       segmentos de clientes y clientes que dejan de enviar.
#   1 = el resto: útil como descripción, sin una decisión directa detrás.
_RECEPCION = ["fecha_recepcion", "id_cliente"]
_CONTROLES = ["ensayo", "tipo_control", "resultado_control"]

CATALOGO = [
    # ------------------------------ regresión ------------------------------
    FichaAnalisis(
        nombre="volumen_semanal_linea", titulo="Volumen semanal de muestras por línea",
        tipo="regresion", unidad="semana × línea de negocio", objetivo="muestras_semana",
        momento="programacion", metrica="rmse", columna_linea_base="base_media_4",
        decision="Programar analistas, medios de cultivo e incubadoras de la semana siguiente.",
        valor_negocio=3, requiere={"muestras": _RECEPCION},
        constructor=series.volumen_semanal_linea),
    FichaAnalisis(
        nombre="carga_diaria", titulo="Carga diaria del laboratorio",
        tipo="regresion", unidad="día hábil", objetivo="muestras_dia",
        momento="programacion", metrica="rmse", columna_linea_base="base_media_5",
        decision="Ajustar los turnos de recepción y siembra del día siguiente.",
        valor_negocio=2, requiere={"muestras": ["fecha_recepcion"]},
        constructor=series.carga_diaria),
    FichaAnalisis(
        nombre="dias_entrega_recepcion", titulo="Días de entrega del informe",
        tipo="regresion", unidad="muestra aceptada", objetivo="dias_entrega",
        momento="recepcion", metrica="rmse",
        decision="Prometer al cliente una fecha de entrega realista al recibir la muestra.",
        valor_negocio=2, requiere={"muestras": _RECEPCION + ["dias_entrega"]},
        constructor=muestra.dias_entrega_recepcion),
    FichaAnalisis(
        nombre="recuento_log_recepcion", titulo="Recuento microbiano (log UFC)",
        tipo="regresion", unidad="muestra aceptada con recuento", objetivo="recuento_log",
        momento="recepcion", metrica="rmse",
        decision="Anticipar las diluciones necesarias antes de sembrar.",
        valor_negocio=1, requiere={"muestras": _RECEPCION + ["recuento_ufc"]},
        constructor=muestra.recuento_log_recepcion),
    FichaAnalisis(
        nombre="visitas_mensuales_cliente", titulo="Visitas de muestreo por cliente y mes",
        tipo="regresion", unidad="cliente × mes", objetivo="visitas_mes",
        momento="maestro", metrica="rmse", columna_linea_base="base_media_3",
        decision="Reservar técnicos de muestreo en planta para el mes siguiente.",
        valor_negocio=1, requiere={"muestreos": ["id_cliente", "fecha_programada"],
                                   "muestras": _RECEPCION},
        constructor=cliente.visitas_mensuales_cliente),
    # ---------------------------- clasificación ----------------------------
    FichaAnalisis(
        nombre="no_conforme_recepcion", titulo="Resultado no conforme, visto desde la recepción",
        tipo="clasificacion", unidad="muestra aceptada", objetivo="no_conforme",
        momento="recepcion", metrica="roc_auc",
        decision="Priorizar la siembra y avisar pronto al cliente de las muestras de mayor riesgo.",
        valor_negocio=3, requiere={"muestras": _RECEPCION + ["no_conforme"]},
        constructor=muestra.no_conforme_recepcion),
    FichaAnalisis(
        nombre="falso_positivo_lectura", titulo="Falso positivo entre los no conformes",
        tipo="clasificacion", unidad="resultado no conforme reensayado", objetivo="falso_positivo",
        momento="lectura", metrica="roc_auc",
        decision="Decidir qué resultados no conformes conviene reensayar antes de informar.",
        valor_negocio=3, requiere={"muestras": _RECEPCION + ["falso_positivo"]},
        constructor=muestra.falso_positivo_lectura),
    FichaAnalisis(
        nombre="entrega_tardia_recepcion", titulo="Entrega tardía del informe",
        tipo="clasificacion", unidad="muestra aceptada", objetivo="entrega_tardia",
        momento="recepcion", metrica="roc_auc",
        decision="Reordenar la cola o avisar al cliente cuando el plazo prometido está en riesgo.",
        valor_negocio=2, requiere={"muestras": _RECEPCION + ["entrega_tardia"]},
        constructor=muestra.entrega_tardia_recepcion),
    FichaAnalisis(
        nombre="cliente_inactivo_trimestre", titulo="Cliente inactivo en el trimestre siguiente",
        tipo="clasificacion", unidad="cliente × trimestre", objetivo="inactivo",
        momento="maestro", metrica="roc_auc",
        decision="Indicar al área comercial qué clientes contactar antes de que dejen de enviar.",
        valor_negocio=2, requiere={"muestras": _RECEPCION},
        constructor=cliente.cliente_inactivo_trimestre),
    FichaAnalisis(
        nombre="rechazo_llegada", titulo="Muestra rechazada en recepción",
        tipo="clasificacion", unidad="muestra recibida", objetivo="rechazada",
        momento="llegada", metrica="roc_auc",
        decision="Advertir al cliente de los envíos que suelen llegar en condiciones no aceptables.",
        valor_negocio=1, requiere={"muestras": _RECEPCION + ["rechazada"]},
        constructor=muestra.rechazo_llegada),
    # ---------------------------- agrupamiento -----------------------------
    FichaAnalisis(
        nombre="segmentos_clientes", titulo="Segmentos de clientes",
        tipo="agrupamiento", unidad="cliente", objetivo=None, momento="reporte",
        metrica=METRICA_AGRUPAMIENTO,
        decision="Diferenciar la atención comercial y los acuerdos de servicio por tipo de cliente.",
        valor_negocio=2, requiere={"muestras": _RECEPCION + ["ensayo"]},
        constructor=cliente.segmentos_clientes),
    FichaAnalisis(
        nombre="perfiles_muestra", titulo="Perfiles de muestra",
        tipo="agrupamiento", unidad="muestra aceptada", objetivo=None, momento="reporte",
        metrica=METRICA_AGRUPAMIENTO,
        decision="Describir los tipos de muestra que llegan para organizar los flujos de trabajo.",
        valor_negocio=1, requiere={"muestras": ["fecha_recepcion"]},
        constructor=cliente.perfiles_muestra),
    FichaAnalisis(
        nombre="dias_atipicos", titulo="Días atípicos de operación",
        tipo="agrupamiento", unidad="día hábil", objetivo=None, momento="reporte",
        metrica=METRICA_AGRUPAMIENTO,
        decision="Señalar los días fuera de lo común para revisar qué ocurrió en el laboratorio.",
        valor_negocio=1, requiere={"muestras": ["fecha_recepcion"]},
        constructor=cliente.dias_atipicos),
    # ------------------------- analíticos (Bayes) --------------------------
    FichaAnalisis(
        nombre="vpp_por_ensayo", titulo="Valor predictivo positivo por ensayo",
        tipo="bayes", unidad="ensayo", objetivo=None, momento="lectura", metrica=None,
        decision="Saber cuánto confiar en un resultado no conforme según el ensayo que lo produjo.",
        valor_negocio=3, requiere={"controles": _CONTROLES, "muestras": ["ensayo", "no_conforme"]}),
    FichaAnalisis(
        nombre="vpp_por_muestra", titulo="Valor predictivo positivo con previa por muestra",
        tipo="bayes", unidad="muestra aceptada", objetivo=None, momento="lectura", metrica=None,
        decision="Combinar el riesgo previo de la muestra con la fiabilidad del ensayo antes de informar.",
        valor_negocio=3, requiere={"controles": _CONTROLES, "muestras": ["ensayo", "no_conforme"]}),
    FichaAnalisis(
        nombre="posterior_reensayo", titulo="Probabilidad posterior tras el reensayo",
        tipo="bayes", unidad="resultado no conforme", objetivo=None, momento="reensayo", metrica=None,
        decision="Decidir si un reensayo basta para confirmar o descartar un resultado no conforme.",
        valor_negocio=3, requiere={"controles": _CONTROLES, "muestras": ["ensayo", "no_conforme"]}),
]


def ficha_por_nombre(nombre):
    """Ficha del catálogo con ese nombre."""
    for ficha in CATALOGO:
        if ficha.nombre == nombre:
            return ficha
    raise KeyError(f"No hay ningún análisis llamado {nombre!r} en el catálogo.")


def disponible(ficha, tablas):
    """Comprueba que las tablas limpias traen lo que el análisis necesita.

    Devuelve (True, "") o (False, motivo). Una columna requerida debe existir
    y tener algún dato; si además es el objetivo del análisis, debe tomar al
    menos dos valores distintos (no se puede predecir una constante).
    """
    for tabla, columnas in ficha.requiere.items():
        if tabla not in tablas or tablas[tabla] is None or not len(tablas[tabla]):
            return False, f"falta la tabla \"{tabla}\""
        datos = tablas[tabla]
        for columna in columnas:
            if columna not in datos.columns:
                return False, f"falta la columna \"{tabla}.{columna}\""
            if not datos[columna].notna().any():
                return False, f"la columna \"{tabla}.{columna}\" no tiene datos"
            if columna == ficha.objetivo and datos[columna].nunique(dropna=True) < 2:
                return False, f"la columna \"{tabla}.{columna}\" toma un solo valor"
    return True, ""


def construir(ficha, limpias, integradas):
    """Tabla de modelado y papeles de un análisis: `(tabla, papeles)`."""
    if ficha.constructor is None:
        raise ValueError(f"El análisis {ficha.nombre!r} es analítico: no tiene tabla de modelado.")
    return ficha.constructor(limpias, integradas)


def construir_disponibles(limpias, integradas, fichas=None):
    """Construye las tablas de modelado de todos los análisis que pueden
    hacerse con los datos presentes.

    Devuelve `(tablas_modelado, motivos)`: `tablas_modelado[nombre] = (tabla,
    papeles)` para los disponibles y `motivos[nombre]` con la razón de los
    que no lo están. Las fichas analíticas no aparecen en ninguno de los dos.
    """
    tablas_modelado, motivos = {}, {}
    for ficha in (CATALOGO if fichas is None else fichas):
        if ficha.constructor is None:
            continue
        hay, motivo = disponible(ficha, limpias)
        if hay:
            tablas_modelado[ficha.nombre] = construir(ficha, limpias, integradas)
        else:
            motivos[ficha.nombre] = motivo
    return tablas_modelado, motivos


def tabla_catalogo(fichas=None):
    """El catálogo en forma de tabla, para mostrarlo."""
    return pd.DataFrame([
        {"Análisis": f.nombre, "Título": f.titulo, "Tipo": f.tipo, "Unidad": f.unidad,
         "Objetivo": f.objetivo or "", "Momento": f.momento or "", "Métrica": f.metrica or "",
         "Línea base": f.columna_linea_base or ("" if f.tipo not in TIPOS_SUPERVISADOS else "Dummy"),
         "Decisión que apoya": f.decision, "Valor": f.valor_negocio}
        for f in (CATALOGO if fichas is None else fichas)
    ])
