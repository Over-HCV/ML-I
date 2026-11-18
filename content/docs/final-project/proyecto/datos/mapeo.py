"""Capa 1 · Mapeo de la exportación real al contrato.

Responsabilidad única: traducir los nombres de columna y las etiquetas de la
exportación real del laboratorio a los nombres canónicos de `esquema.py`. La
traducción se describe en `configuracion/mapeo_columnas.json`, con esta forma:

    {"<tabla>": {"archivo": "muestras.csv",
                 "columnas": {"<canónico>": "<nombre real>"},
                 "valores": {"<canónico>": {"<valor real>": "<valor canónico>"}}}}

Conectar la base real consiste en editar ese archivo: ninguna otra capa
conoce los nombres de la exportación. Las claves que empiezan por «_» son
comentarios y se ignoran; una columna que el laboratorio no tenga se deja en
`null` (o se borra) y el resto del proyecto la trata como ausente.
"""
import json
import unicodedata

import pandas as pd

from proyecto.base import rutas
from proyecto.datos import esquema

# Etiquetas de las que depende el código (el resto de categorías puede traer
# los valores que quiera). La limpieza reconoce estas etiquetas aunque vengan
# con otra combinación de mayúsculas, tildes o espacios; cualquier otro
# sinónimo («Cumple», «NC», «1/0»...) debe traducirse en la sección «valores».
VALORES_CANONICOS = {
    "clientes": {
        "certificacion_bpm": ["sí", "no"],
    },
    "muestras": {
        "origen": ["enviada por el cliente", "muestreo en planta"],
        "conservante": ["sí", "no", "no aplica"],
        "envase_integro": ["sí", "no"],
        "documentacion_completa": ["sí", "no"],
        "estado_recepcion": ["aceptada", "rechazada"],
        "resultado": ["conforme", "no conforme"],
        "reensayo": ["sí", "no"],
        "resultado_reensayo": ["conforme", "no conforme"],
    },
    "controles": {
        "tipo_control": ["positivo", "negativo"],
        "resultado_control": ["crecimiento", "sin crecimiento"],
    },
}

OPCIONES_POR_DEFECTO = {
    "separador": None,        # None = detectar («,», «;» o tabulador)
    "codificacion": "utf-8",  # por ejemplo «latin-1» en exportaciones de Excel antiguas
    "dia_primero": True,      # fechas escritas como día/mes/año cuando no vienen en formato ISO
    "hoja": 0,                # hoja del libro, solo para archivos de Excel
}

_INSTRUCCIONES = [
    "Plantilla del mapeo entre la exportación real del laboratorio y el contrato de datos.",
    "1. Copie los archivos exportados en datos/reales/ y escriba su nombre en «archivo».",
    "2. En «columnas», reemplace cada valor por el nombre que tiene esa columna en la exportación. "
    "La clave (izquierda) es el nombre canónico y no se cambia. Si el laboratorio no tiene esa "
    "columna, déjela en null: solo las obligatorias son imprescindibles.",
    "3. En «valores», traduzca las etiquetas de la exportación (izquierda) a las canónicas (derecha). "
    "Puede añadir tantas entradas como sinónimos existan; mayúsculas, tildes y espacios no importan.",
    "4. «opciones» es opcional: separador, codificación del archivo y orden día/mes de las fechas.",
    "5. Active la lectura con la variable de entorno PROYECTO_ORIGEN=real (o ORIGEN_DATOS = \"real\" "
    "en proyecto/base/config.py) y ejecute la etapa 01_datos.",
    "Dos tablas pueden leerse del mismo archivo (por ejemplo, clientes dentro de la exportación de muestras).",
]


def clave_etiqueta(texto):
    """Forma de comparación de una etiqueta: minúsculas, sin tildes y con los
    espacios colapsados. Dos grafías con la misma clave son la misma categoría."""
    sin_tildes = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    return " ".join(sin_tildes.lower().split())


def plantilla_mapeo():
    """Mapeo identidad: cada columna canónica se llama igual en la fuente y
    cada etiqueta se traduce a sí misma. Es el punto de partida para describir
    la exportación real."""
    plantilla = {"_instrucciones": list(_INSTRUCCIONES)}
    for tabla in esquema.TABLAS:
        plantilla[tabla] = {
            "archivo": f"{tabla}.csv",
            "_obligatorias": [c.nombre for c in esquema.ESQUEMA[tabla] if c.obligatoria],
            "columnas": {c.nombre: c.nombre for c in esquema.ESQUEMA[tabla]},
            "valores": {columna: {valor: valor for valor in valores}
                        for columna, valores in VALORES_CANONICOS.get(tabla, {}).items()},
            "opciones": dict(OPCIONES_POR_DEFECTO),
        }
    return plantilla


def guardar_plantilla(ruta=None):
    """Escribe la plantilla identidad en `configuracion/mapeo_columnas.json`."""
    ruta = rutas.ARCHIVO_MAPEO if ruta is None else ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(plantilla_mapeo(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ruta


def leer_mapeo(ruta=None):
    """Lee el mapeo y lo devuelve sin las claves de comentario. Si el archivo
    no existe se usa la plantilla identidad."""
    ruta = rutas.ARCHIVO_MAPEO if ruta is None else ruta
    if not ruta.exists():
        crudo = plantilla_mapeo()
    else:
        try:
            crudo = json.loads(ruta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"El archivo de mapeo {ruta} no es un JSON válido (línea {error.lineno}, "
                f"columna {error.colno}): {error.msg}."
            ) from error
    mapeo = {}
    for tabla, descripcion in crudo.items():
        if tabla.startswith("_"):
            continue
        if tabla not in esquema.TABLAS:
            raise ValueError(
                f"El mapeo describe la tabla «{tabla}», que no existe en el contrato. "
                f"Tablas válidas: {', '.join(esquema.TABLAS)}."
            )
        if not isinstance(descripcion, dict):
            raise ValueError(f"La entrada «{tabla}» del mapeo debe ser un objeto con «archivo» y «columnas».")
        conocidas = {c.nombre for c in esquema.ESQUEMA[tabla]}
        columnas = {canonica: real for canonica, real in (descripcion.get("columnas") or {}).items()
                    if not canonica.startswith("_")}
        desconocidas = sorted(set(columnas) - conocidas)
        if desconocidas:
            raise ValueError(
                f"El mapeo de «{tabla}» usa nombres canónicos que no están en el contrato: "
                f"{', '.join(desconocidas)}. Las claves de «columnas» son los nombres del contrato; "
                "el nombre de la exportación va a la derecha."
            )
        mapeo[tabla] = {
            "archivo": descripcion.get("archivo"),
            "columnas": columnas,
            "valores": {columna: dict(traduccion or {})
                        for columna, traduccion in (descripcion.get("valores") or {}).items()
                        if not columna.startswith("_")},
            "opciones": {**OPCIONES_POR_DEFECTO, **(descripcion.get("opciones") or {})},
        }
    return mapeo


def aplicar_mapeo(tablas_fuente, mapeo):
    """Convierte las tablas de la exportación (`{tabla: DataFrame}` con los
    nombres reales) en tablas con nombres y etiquetas canónicos.

    Solo pasan las columnas declaradas en el mapeo que existan en la fuente;
    las demás columnas de la exportación se dejan fuera. Las etiquetas se
    traducen comparando sin distinguir mayúsculas, tildes ni espacios, y las
    que no tienen traducción se conservan tal como vienen."""
    tablas = {}
    for tabla, descripcion in mapeo.items():
        if tabla not in tablas_fuente:
            continue
        fuente = tablas_fuente[tabla]
        columnas = {}
        for canonica, real in descripcion.get("columnas", {}).items():
            if real is None or str(real).strip() == "":
                continue
            if real in fuente.columns:
                columnas[canonica] = fuente[real]
        destino = pd.DataFrame(columnas, index=fuente.index)
        for columna, traduccion in descripcion.get("valores", {}).items():
            if columna not in destino.columns or not traduccion:
                continue
            por_clave = {clave_etiqueta(real): canonico for real, canonico in traduccion.items()}
            destino[columna] = destino[columna].map(
                lambda valor, por_clave=por_clave: (
                    por_clave.get(clave_etiqueta(valor), valor) if isinstance(valor, str) else valor))
        tablas[tabla] = destino.reset_index(drop=True)
    return tablas


def columnas_sin_origen(tablas_fuente, mapeo):
    """Columnas canónicas que el mapeo declara pero cuyo nombre real no está
    en el archivo: `[(tabla, canónica, nombre real)]`. Sirve para explicar por
    qué falta una columna."""
    faltantes = []
    for tabla, descripcion in mapeo.items():
        if tabla not in tablas_fuente:
            continue
        for canonica, real in descripcion.get("columnas", {}).items():
            if real is None or str(real).strip() == "":
                continue
            if real not in tablas_fuente[tabla].columns:
                faltantes.append((tabla, canonica, real))
    return faltantes
