"""Capa 1 · Muestra pública y diccionario de la exportación real.

Responsabilidad única: producir lo único de la base real que sale del
proyecto en la entrega: unas pocas filas con todos los identificadores y
textos libres reemplazados, y el diccionario de las 36 columnas (tipo,
descripción, papel en la clasificación de no conformidad y momento en que se
conoce). La exportación completa es confidencial y nunca se entrega.

Las filas se eligen de muestras distintas (producto y planta, al menos una no
conforme) con la semilla del proyecto, así que la muestra es siempre la misma.
"""
import pandas as pd

from proyecto.base.config import SEMILLA
from proyecto.datos import rodam

FILAS_PUBLICAS = 5
TEXTO_OMITIDO = "[texto libre omitido]"

# Papeles en la clasificación de no conformidad al ingreso.
OBJETIVO = "objetivo"
PREDICTOR = "predictor o insumo de uno"
IDENTIFICADOR = "identificador"
POSTERIOR = "excluida: se conoce después del resultado"
TEXTO = "excluida: texto libre"
DESCRIPTIVA = "excluida: descriptiva o de control interno"

# (columna, tipo, descripción, papel, momento en que se conoce)
DICCIONARIO = [
    ("id_muestra", "entero", "Identificador único de la muestra (clave de agrupación)", IDENTIFICADOR, "ingreso"),
    ("fecha_ingreso", "fecha y hora", "Registro de la muestra en el sistema (hora de Colombia); de aquí salen el día, "
     "la hora y el orden temporal de la partición", PREDICTOR, "ingreso"),
    ("fecha_inicio", "fecha y hora", "Inicio del análisis; suele ser anterior al ingreso porque el registro llega "
     "después", DESCRIPTIVA, "ingreso"),
    ("estado_muestra", "categoría", "Estado del flujo (con aprobación final, archivado, sin remitir...)", POSTERIOR,
     "reporte"),
    ("concepto_muestra", "booleano", "Dictamen final: verdadero = cumple, falso = no cumple (clase 1 del modelo)",
     OBJETIVO, "reporte"),
    ("tuvo_reanalisis", "booleano", "Bandera de reanálisis del sistema (poco confiable)", POSTERIOR, "lectura"),
    ("origen_muestra", "categoría", "Muestreo en planta, producto terminado, agua, granel, materia prima o material "
     "de envase; define el tipo de lote", PREDICTOR, "ingreso"),
    ("lote_muestra", "texto", "Lote del cliente escrito a mano", TEXTO, "ingreso"),
    ("nombre_complementario", "código", "Punto de muestreo, área o persona (código aleatorio)", TEXTO, "ingreso"),
    ("usa_prefijo_del_producto", "booleano", "Si el nombre de la muestra se arma con el del producto", DESCRIPTIVA,
     "ingreso"),
    ("presentacion", "texto", "Presentación o tamaño de la muestra", TEXTO, "ingreso"),
    ("id_cuadro_analitico", "entero", "Plan de análisis acordado con el cliente", IDENTIFICADOR, "ingreso"),
    ("cuadro_analitico", "texto", "Nombre del cuadro analítico", TEXTO, "ingreso"),
    ("cuadro_complejo", "booleano", "Si el cuadro es de análisis complejo", PREDICTOR, "ingreso"),
    ("estado_cuadro", "categoría", "Estado actual del cuadro (aprobado, archivado, sin remitir)", DESCRIPTIVA,
     "ingreso"),
    ("id_producto", "entero", "Producto al que pertenece el cuadro", IDENTIFICADOR, "ingreso"),
    ("producto", "categoría", "Nombre genérico del producto (44 valores)", PREDICTOR, "ingreso"),
    ("forma_farmaceutica", "categoría", "Forma farmacéutica (13 valores, varias escrituras de no aplica)", PREDICTOR,
     "ingreso"),
    ("sector", "categoría", "Farmacéutico, cosmético, todos, veterinario o alimentos", PREDICTOR, "ingreso"),
    ("cliente", "código", "Cliente dueño del producto (código aleatorio de 5 letras); solo entra como historia de "
     "dictámenes ya conocidos", PREDICTOR, "ingreso"),
    ("id_analisis", "entero", "Análisis de la especificación", IDENTIFICADOR, "ingreso"),
    ("analisis", "categoría", "Nombre del análisis (69 valores); su conteo por muestra es el número de "
     "especificaciones", PREDICTOR, "ingreso"),
    ("id_metodo", "entero", "Método de la especificación", IDENTIFICADOR, "ingreso"),
    ("metodo", "categoría", "Nombre del método (61 valores)", DESCRIPTIVA, "ingreso"),
    ("grupo", "categoría", "Grupo de la especificación dentro del cuadro", DESCRIPTIVA, "ingreso"),
    ("id_especificacion", "entero", "Especificación (análisis, método y límite)", IDENTIFICADOR, "ingreso"),
    ("valor_especificacion", "texto", "Límite o criterio de aceptación (por ejemplo <1000 UFC/g o Ausencia/g)",
     DESCRIPTIVA, "ingreso"),
    ("id_lectura", "entero", "Lectura del laboratorio (nulo = especificación sin leer)", IDENTIFICADOR, "lectura"),
    ("n_lectura", "entero", "Orden de la lectura; más de una indica reanálisis", POSTERIOR, "lectura"),
    ("lectura", "texto", "Resultado escrito (por ejemplo 0 UFC o Ausencia)", POSTERIOR, "lectura"),
    ("lectura_transformada", "texto", "Resultado normalizado con su unidad", POSTERIOR, "lectura"),
    ("fecha_lectura", "fecha y hora", "Momento de la lectura", POSTERIOR, "lectura"),
    ("concepto_lectura", "booleano", "Si la lectura cumple su especificación", POSTERIOR, "lectura"),
    ("observaciones_lectura", "texto", "Observaciones del analista", TEXTO, "lectura"),
    ("analista_lectura", "código", "Analista que leyó (código aleatorio)", POSTERIOR, "lectura"),
    ("lote_de_medio_id", "entero", "Lote del medio de cultivo usado", POSTERIOR, "siembra"),
]

_TEXTOS = ["lote_muestra", "presentacion", "observaciones_lectura", "cuadro_analitico"]
_CODIGOS = {"cliente": "CLIENTE", "nombre_complementario": "PUNTO", "analista_lectura": "ANALISTA"}
_IDS = ["id_muestra", "id_cuadro_analitico", "id_producto", "id_lectura", "lote_de_medio_id"]


def tabla_diccionario():
    """El diccionario como tabla: una fila por columna de la exportación."""
    return pd.DataFrame(DICCIONARIO, columns=["columna", "tipo", "descripcion", "papel", "momento"])


def muestra_anonimizada(filas=None, n=FILAS_PUBLICAS, semilla=SEMILLA):
    """`n` filas de muestras distintas (una lectura original de cada una) con
    los códigos cambiados por rótulos genéricos, los textos libres omitidos y
    los identificadores renumerados."""
    filas = rodam.leer_exportacion() if filas is None else filas
    originales = filas[(filas["n_lectura"] == 1) & filas["concepto_muestra"].notna()]
    una_por_muestra = originales.drop_duplicates("id_muestra")
    planta = una_por_muestra["origen_muestra"].astype(str) == rodam.ORIGEN_PLANTA
    no_conforme = una_por_muestra["concepto_muestra"] == False  # noqa: E712 (booleano con nulos)
    elegidas = pd.concat([
        una_por_muestra[no_conforme & ~planta].sample(1, random_state=semilla),
        una_por_muestra[~no_conforme & ~planta].sample(max(n // 2 - 1, 0), random_state=semilla),
        una_por_muestra[~no_conforme & planta].sample(n - n // 2, random_state=semilla),
    ]).sort_values("fecha_ingreso").reset_index(drop=True)
    publica = elegidas.astype(object).where(elegidas.notna(), None)
    for columna in _TEXTOS:
        publica[columna] = publica[columna].map(lambda v: None if v is None else TEXTO_OMITIDO)
    for columna, rotulo in _CODIGOS.items():
        valores = {v: f"{rotulo}_{i + 1}" for i, v in enumerate(pd.unique(publica[columna].dropna()))}
        publica[columna] = publica[columna].map(valores)
    for columna in _IDS:
        valores = {v: i + 1 for i, v in enumerate(pd.unique(publica[columna].dropna()))}
        publica[columna] = publica[columna].map(valores)
    return publica[[c for c, *_ in DICCIONARIO]]


def escribir(destino):
    """Escribe `muestra_anonimizada.csv` y `diccionario_datos.csv` en `destino`."""
    destino.mkdir(parents=True, exist_ok=True)
    muestra_anonimizada().to_csv(destino / "muestra_anonimizada.csv", index=False)
    tabla_diccionario().to_csv(destino / "diccionario_datos.csv", index=False)
    return destino
