"""Capa 1 · Contrato de datos.

Responsabilidad única: declarar qué tablas y columnas espera el proyecto, con
su tipo, su papel y el **momento** del proceso en que cada dato pasa a
conocerse. Es la única fuente de verdad sobre los nombres de columna: el
resto de capas pregunta aquí qué columnas puede usar en lugar de escribirlas
a mano, de forma que

- una variable no puede colarse en un modelo antes de existir (fuga de
  información bloqueada por estructura), y
- conectar la base real del laboratorio se reduce a traducir sus nombres a
  estos (ver `datos/mapeo.py`), sin tocar las capas siguientes.
"""
from dataclasses import dataclass

import pandas as pd

# Momentos del proceso, en orden. Una predicción hecha en el momento m solo
# puede usar columnas cuyo momento sea anterior o igual a m.
MOMENTOS = [
    "maestro",       # datos del cliente, conocidos de antemano
    "programacion",  # agenda de muestreos en planta
    "toma",          # lo que viaja con la muestra
    "llegada",       # condiciones medidas al llegar al laboratorio
    "recepcion",     # decisión de aceptar o rechazar
    "siembra",       # condiciones del montaje del ensayo
    "lectura",       # lectura de placas y resultado
    "reensayo",      # repetición del ensayo
    "reporte",       # entrega del informe
]

TIPOS = ["id", "fecha", "categoria", "numero", "texto"]
PAPELES = ["id", "fecha", "variable", "resultado", "auxiliar"]


@dataclass(frozen=True)
class Columna:
    """Una columna del contrato."""
    nombre: str
    tipo: str
    papel: str
    momento: str
    descripcion: str
    obligatoria: bool = False


def _c(nombre, tipo, papel, momento, descripcion, obligatoria=False):
    assert tipo in TIPOS and papel in PAPELES and momento in MOMENTOS, nombre
    return Columna(nombre, tipo, papel, momento, descripcion, obligatoria)


ESQUEMA = {
    "clientes": [
        _c("id_cliente", "id", "id", "maestro", "Identificador del cliente", True),
        _c("sector", "categoria", "variable", "maestro", "Línea de negocio del cliente", True),
        _c("zona", "categoria", "variable", "maestro", "Municipio o zona de la planta"),
        _c("tamano", "categoria", "variable", "maestro", "Tamaño de la empresa"),
        _c("tipo_contrato", "categoria", "variable", "maestro", "Contrato marco o servicio por demanda"),
        _c("certificacion_bpm", "categoria", "variable", "maestro", "Cuenta con certificación BPM (sí/no)"),
        _c("plan_muestreo", "categoria", "variable", "maestro", "Frecuencia pactada de muestreo en planta"),
        _c("fecha_vinculacion", "fecha", "fecha", "maestro", "Fecha de vinculación como cliente"),
        _c("distancia_km", "numero", "variable", "maestro", "Distancia de la planta al laboratorio (km)"),
        _c("canal_captacion", "categoria", "variable", "maestro", "Canal por el que llegó el cliente"),
        _c("ejecutivo_cuenta", "categoria", "variable", "maestro", "Ejecutivo comercial asignado"),
    ],
    "muestreos": [
        _c("id_muestreo", "id", "id", "programacion", "Identificador de la visita", True),
        _c("id_cliente", "id", "id", "programacion", "Cliente visitado", True),
        _c("fecha_programada", "fecha", "fecha", "programacion", "Fecha agendada de la visita", True),
        _c("puntos_programados", "numero", "variable", "programacion", "Puntos de muestreo previstos"),
        _c("tecnico", "categoria", "variable", "programacion", "Técnico asignado"),
        _c("fecha_visita", "fecha", "fecha", "toma", "Fecha en que se hizo la visita"),
        _c("duracion_horas", "numero", "variable", "toma", "Duración de la visita (h)"),
    ],
    "muestras": [
        _c("id_muestra", "id", "id", "toma", "Identificador de la muestra", True),
        _c("id_cliente", "id", "id", "toma", "Cliente que envía la muestra", True),
        _c("id_lote_envio", "id", "id", "toma", "Envío en el que llegó la muestra"),
        _c("id_muestreo", "id", "id", "toma", "Visita de muestreo de origen (si aplica)"),
        _c("fecha_toma", "fecha", "fecha", "toma", "Fecha de toma de la muestra"),
        _c("origen", "categoria", "variable", "toma", "Enviada por el cliente o tomada en planta"),
        _c("matriz", "categoria", "variable", "toma", "Tipo de muestra (producto, materia prima, agua, superficie...)"),
        _c("tipo_producto", "categoria", "variable", "toma", "Familia de producto"),
        _c("codigo_producto", "texto", "auxiliar", "toma", "Referencia del producto del cliente"),
        _c("ensayo", "categoria", "variable", "toma", "Ensayo microbiológico solicitado", True),
        _c("actividad_agua", "numero", "variable", "toma", "Actividad de agua declarada (aw)"),
        _c("ph", "numero", "variable", "toma", "pH declarado"),
        _c("conservante", "categoria", "variable", "toma", "El producto tiene sistema conservante (sí/no)"),
        _c("volumen_ml", "numero", "variable", "toma", "Cantidad de muestra (mL o g)"),
        _c("urgencia", "categoria", "variable", "toma", "Prioridad solicitada"),
        _c("fecha_recepcion", "fecha", "fecha", "llegada", "Fecha de llegada al laboratorio", True),
        _c("hora_recepcion", "numero", "variable", "llegada", "Hora de llegada (decimal)"),
        _c("temperatura_transporte_c", "numero", "variable", "llegada", "Temperatura al llegar (°C)"),
        _c("horas_transporte", "numero", "variable", "llegada", "Horas entre la toma y la llegada"),
        _c("envase_integro", "categoria", "variable", "llegada", "El envase llegó íntegro (sí/no)"),
        _c("documentacion_completa", "categoria", "variable", "llegada", "La solicitud llegó completa (sí/no)"),
        _c("recepcionista", "categoria", "variable", "llegada", "Persona que recibe"),
        _c("estado_recepcion", "categoria", "resultado", "recepcion", "Aceptada o rechazada"),
        _c("dias_prometidos", "numero", "variable", "recepcion", "Plazo de entrega prometido (días)"),
        _c("fecha_siembra", "fecha", "fecha", "siembra", "Fecha de montaje del ensayo"),
        _c("analista", "categoria", "variable", "siembra", "Analista que monta el ensayo"),
        _c("turno", "categoria", "variable", "siembra", "Turno de siembra"),
        _c("cabina", "categoria", "variable", "siembra", "Cabina de flujo laminar"),
        _c("incubadora", "categoria", "variable", "siembra", "Incubadora"),
        _c("lote_medio", "texto", "auxiliar", "siembra", "Lote del medio de cultivo"),
        _c("carga_dia", "numero", "variable", "siembra", "Muestras sembradas ese día"),
        _c("fecha_lectura", "fecha", "fecha", "lectura", "Fecha de lectura"),
        _c("control_ambiental_ufc", "numero", "variable", "lectura", "Recuento de la placa de control ambiental de la cabina"),
        _c("recuento_reportado", "texto", "resultado", "lectura", 'Recuento tal como se reporta (admite "<10", "Ausencia"...)'),
        _c("limite_ufc", "numero", "variable", "toma", "Límite de especificación (UFC/g o mL)"),
        _c("resultado", "categoria", "resultado", "lectura", "Conforme o no conforme", True),
        _c("reensayo", "categoria", "resultado", "reensayo", "Se repitió el ensayo (sí/no)"),
        _c("resultado_reensayo", "categoria", "resultado", "reensayo", "Resultado de la repetición"),
        _c("fecha_reporte", "fecha", "fecha", "reporte", "Fecha de entrega del informe"),
    ],
    "controles": [
        _c("id_control", "id", "id", "siembra", "Identificador del control", True),
        _c("fecha", "fecha", "fecha", "siembra", "Fecha del control", True),
        _c("ensayo", "categoria", "variable", "siembra", "Ensayo controlado", True),
        _c("tipo_control", "categoria", "variable", "siembra", "Control positivo o negativo", True),
        _c("analista", "categoria", "variable", "siembra", "Analista"),
        _c("incubadora", "categoria", "variable", "siembra", "Incubadora"),
        _c("lote_medio", "texto", "auxiliar", "siembra", "Lote del medio de cultivo"),
        _c("resultado_control", "categoria", "resultado", "lectura", "Crecimiento o sin crecimiento", True),
    ],
}

TABLAS = list(ESQUEMA)

# Columnas derivadas: las que añaden la limpieza (`preparacion/limpieza.py`) y
# la integración (`preparacion/integracion.py`), con el momento en que pasan a
# conocerse. La tabla integrada se consulta como «muestras»: es la misma tabla
# con los atributos del cliente y las marcas de calendario de la recepción.
DERIVADAS = {
    "clientes": {},
    "muestreos": {},
    "muestras": {
        # --- limpieza ---
        "recuento_ufc": "lectura",          # recuento numérico (cota si está censurado)
        "recuento_censurado": "lectura",    # 1 si se reportó como «<N» o «>N»
        "es_cuantitativo": "toma",          # 1 si el ensayo reporta recuento (propiedad del ensayo)
        "rechazada": "recepcion",           # 1 si no se aceptó en recepción
        "no_conforme": "lectura",           # 1 no conforme, 0 conforme, NaN si no hubo ensayo
        "reensayada": "reensayo",           # 1 si se repitió el ensayo
        "falso_positivo": "reensayo",       # 1 no conforme sin confirmar, 0 confirmado, NaN resto
        "dias_entrega": "reporte",          # días calendario entre recepción y reporte
        "entrega_tardia": "reporte",        # 1 si dias_entrega > dias_prometidos
        # --- integración: atributos del cliente ---
        **{c.nombre: c.momento for c in ESQUEMA["clientes"] if c.papel != "id"},
        "antiguedad_dias": "llegada",       # días desde la vinculación del cliente
        # --- integración: calendario de la fecha de recepción ---
        "anio": "llegada",
        "mes": "llegada",
        "dia_mes": "llegada",
        "dia_semana": "llegada",            # 0 = lunes
        "semana": "llegada",                # lunes de la semana (fecha)
        "dias_habiles_semana": "llegada",
        "es_semana_santa": "llegada",
        "tras_festivo": "llegada",          # primer día hábil después de un festivo
        "cierre_mes": "llegada",            # últimos cinco días hábiles del mes
        "inicio_mes": "llegada",            # primeros cinco días hábiles del mes
    },
    "controles": {},
}


def columnas(tabla, papel=None, tipo=None, hasta=None):
    """Nombres de columna de `tabla` filtrados por papel, tipo y momento
    máximo (`hasta`)."""
    limite = MOMENTOS.index(hasta) if hasta else len(MOMENTOS) - 1
    papeles = [papel] if isinstance(papel, str) else papel
    tipos = [tipo] if isinstance(tipo, str) else tipo
    return [
        c.nombre for c in ESQUEMA[tabla]
        if (papeles is None or c.papel in papeles)
        and (tipos is None or c.tipo in tipos)
        and MOMENTOS.index(c.momento) <= limite
    ]


def momento_de(tabla, columna):
    """Momento en que se conoce `columna` (del contrato o derivada), o None si
    no está declarada."""
    for c in ESQUEMA.get(tabla, []):
        if c.nombre == columna:
            return c.momento
    return DERIVADAS.get(tabla, {}).get(columna)


def derivadas(tabla, hasta=None):
    """Nombres de las columnas derivadas de `tabla` conocidas hasta el momento
    `hasta` (todas si no se indica)."""
    limite = MOMENTOS.index(hasta) if hasta else len(MOMENTOS) - 1
    return [nombre for nombre, momento in DERIVADAS.get(tabla, {}).items()
            if MOMENTOS.index(momento) <= limite]


def es_anterior_o_igual(momento, limite):
    return MOMENTOS.index(momento) <= MOMENTOS.index(limite)


def diccionario():
    """Diccionario de datos en forma de tabla."""
    return pd.DataFrame(
        [{"Tabla": tabla, "Columna": c.nombre, "Tipo": c.tipo, "Papel": c.papel,
          "Momento": c.momento, "Obligatoria": "sí" if c.obligatoria else "no",
          "Descripción": c.descripcion}
         for tabla, cols in ESQUEMA.items() for c in cols]
    )


def cobertura(tablas):
    """Compara un juego de tablas con el contrato: qué columnas están, cuáles
    faltan y cuáles sobran."""
    filas = []
    for tabla, cols in ESQUEMA.items():
        presentes = set(tablas[tabla].columns) if tabla in tablas else set()
        for c in cols:
            filas.append({"Tabla": tabla, "Columna": c.nombre, "Obligatoria": c.obligatoria,
                          "Presente": c.nombre in presentes, "Momento": c.momento})
        for extra in sorted(presentes - {c.nombre for c in cols}):
            filas.append({"Tabla": tabla, "Columna": extra, "Obligatoria": False,
                          "Presente": True, "Momento": DERIVADAS.get(tabla, {}).get(extra, "fuera del contrato")})
    return pd.DataFrame(filas)


def validar(tablas):
    """Lanza un error si falta alguna tabla o columna obligatoria."""
    cob = cobertura(tablas)
    faltan = cob[cob["Obligatoria"] & ~cob["Presente"]]
    if len(faltan):
        detalle = ", ".join(f"{t}.{c}" for t, c in zip(faltan["Tabla"], faltan["Columna"]))
        raise ValueError(f"Faltan columnas obligatorias del contrato: {detalle}")
    return cob
