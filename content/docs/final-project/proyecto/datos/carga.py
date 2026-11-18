"""Capa 1 · Carga de las tablas crudas.

Responsabilidad única: entregar las tablas crudas con los nombres del
contrato, vengan de la simulación o de la exportación real del laboratorio,
y guardarlas o leerlas del disco. Aquí no se limpia nada: las tablas crudas
se guardan como CSV (fechas en texto ISO) y se leen como texto, tal como
llegarían de un sistema externo. La verdad oculta de la simulación se guarda
aparte, en `datos/auditoria/`.
"""
import hashlib

import pandas as pd

from proyecto.base import config, rutas
from proyecto.datos import esquema, mapeo as mapeo_columnas
from proyecto.datos.calendario import tabla_calendario

TABLAS_CRUDAS = esquema.TABLAS + ["calendario"]
TABLAS_VERDAD = ["muestras", "clientes", "ensayos"]
ORIGENES = ("simulacion", "real")


# ----------------------------------------------------------------------
# Obtención
# ----------------------------------------------------------------------
def obtener_tablas_crudas(origen=None):
    """Devuelve `(tablas, verdad)`.

    - `origen="simulacion"`: genera la base con la semilla del proyecto;
      `verdad` contiene las variables latentes.
    - `origen="real"`: lee `datos/reales/` a través de
      `configuracion/mapeo_columnas.json` y valida el contrato; `verdad` es
      None.
    """
    origen = config.ORIGEN_DATOS if origen is None else origen
    if origen == "simulacion":
        from proyecto.datos import simulacion  # único lugar que usa el generador
        return simulacion.simular(config.SEMILLA)
    if origen == "real":
        return leer_reales(), None
    raise ValueError(
        f"Origen de datos desconocido: {origen!r}. Use «simulacion» o «real» "
        "(variable de entorno PROYECTO_ORIGEN o ORIGEN_DATOS en proyecto/base/config.py)."
    )


def _leer_archivo(ruta, opciones):
    """Lee un archivo de la exportación como texto, sin interpretar nada."""
    extension = ruta.suffix.lower()
    if extension == ".parquet":
        tabla = pd.read_parquet(ruta)
        return tabla.astype(object).where(tabla.notna(), None)
    if extension in (".xlsx", ".xls"):
        try:
            return pd.read_excel(ruta, sheet_name=opciones.get("hoja") or 0, dtype=str)
        except ImportError as error:
            raise ValueError(
                f"No se puede leer {ruta.name}: falta el lector de Excel en este entorno. "
                "Exporte la tabla a CSV y actualice «archivo» en el mapeo."
            ) from error
    separador = opciones.get("separador")
    try:
        return pd.read_csv(ruta, dtype=str, keep_default_na=False, na_values=[""],
                           sep=separador, engine="python" if separador is None else "c",
                           encoding=opciones.get("codificacion") or "utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"No se puede leer {ruta.name} con la codificación «{opciones.get('codificacion')}». "
            "Indique la correcta en «opciones» → «codificacion» del mapeo (por ejemplo «latin-1»)."
        ) from error


def _fechas_a_iso(serie, dia_primero):
    """Convierte a texto ISO las fechas escritas en otro formato. Lo que no se
    reconoce como fecha se deja intacto, para que la limpieza lo cuente."""
    texto = serie.astype(object).where(serie.notna(), None)
    con_valor = texto.notna()
    if not con_valor.any():
        return serie
    valores = texto[con_valor].astype(str).str.strip()
    fechas = pd.to_datetime(valores, format="ISO8601", errors="coerce")
    resto = fechas.isna()
    if resto.any():
        fechas = fechas.astype("datetime64[ns]")
        fechas[resto] = pd.to_datetime(valores[resto], format="mixed", dayfirst=bool(dia_primero),
                                       errors="coerce").astype("datetime64[ns]")
    salida = texto.copy()
    salida[con_valor] = fechas.dt.strftime("%Y-%m-%d").where(fechas.notna(), valores)
    return salida


def leer_reales(carpeta=None, mapeo=None):
    """Lee la exportación real y la devuelve con la forma del contrato.

    Lanza `ValueError` con la lista completa de lo que falta (archivos,
    tablas y columnas obligatorias) para poder corregir el mapeo de una vez."""
    carpeta = rutas.DIR_REALES if carpeta is None else carpeta
    mapeo = mapeo_columnas.leer_mapeo() if mapeo is None else mapeo
    problemas, fuentes, leidos = [], {}, {}
    for tabla in esquema.TABLAS:
        if tabla not in mapeo:
            problemas.append(f"- Tabla «{tabla}»: no está descrita en el mapeo "
                             f"({rutas.ARCHIVO_MAPEO.name}). Añada una entrada con «archivo» y «columnas».")
            continue
        archivo = mapeo[tabla].get("archivo")
        if not archivo:
            problemas.append(f"- Tabla «{tabla}»: el mapeo no indica «archivo».")
            continue
        ruta = carpeta / archivo
        if not ruta.exists():
            problemas.append(f"- Tabla «{tabla}»: no se encontró el archivo {ruta}.")
            continue
        clave = (archivo, tuple(sorted((k, str(v)) for k, v in mapeo[tabla]["opciones"].items())))
        if clave not in leidos:
            leidos[clave] = _leer_archivo(ruta, mapeo[tabla]["opciones"])
        fuentes[tabla] = leidos[clave]

    tablas = mapeo_columnas.aplicar_mapeo(fuentes, mapeo)
    sin_origen = {(t, c): real for t, c, real in mapeo_columnas.columnas_sin_origen(fuentes, mapeo)}
    for tabla in tablas:
        for columna in esquema.ESQUEMA[tabla]:
            if not columna.obligatoria or columna.nombre in tablas[tabla].columns:
                continue
            real = sin_origen.get((tabla, columna.nombre))
            if real is not None:
                detalle = (f"el mapeo la busca como «{real}», que no existe en {mapeo[tabla]['archivo']} "
                           f"(columnas disponibles: {', '.join(map(str, fuentes[tabla].columns))})")
            else:
                detalle = "el mapeo no le asigna ninguna columna de la exportación"
            problemas.append(f"- Columna obligatoria «{tabla}.{columna.nombre}» "
                             f"({columna.descripcion.lower()}): {detalle}.")
    if problemas:
        raise ValueError(
            "La exportación real no cumple el contrato de datos. Corrija "
            f"{rutas.ARCHIVO_MAPEO} o los archivos de {carpeta}:\n" + "\n".join(problemas)
        )

    for tabla in tablas:
        dia_primero = mapeo[tabla]["opciones"].get("dia_primero", True)
        for columna in esquema.columnas(tabla, tipo="fecha"):
            if columna in tablas[tabla].columns:
                tablas[tabla][columna] = _fechas_a_iso(tablas[tabla][columna], dia_primero)
    # Un mismo archivo puede traer los clientes repetidos en cada muestra.
    tablas["clientes"] = tablas["clientes"].drop_duplicates("id_cliente").reset_index(drop=True)
    tablas["calendario"] = _calendario_para(tablas)
    esquema.validar(tablas)
    return tablas


def _calendario_para(tablas):
    """Calendario que cubre todas las fechas de las tablas reales."""
    fechas = []
    for tabla in esquema.TABLAS:
        for columna in esquema.columnas(tabla, tipo="fecha"):
            if columna in tablas[tabla].columns and columna != "fecha_vinculacion":
                fechas.append(pd.to_datetime(tablas[tabla][columna], errors="coerce").dropna())
    fechas = pd.concat(fechas) if fechas else pd.Series([], dtype="datetime64[ns]")
    if fechas.empty:
        raise ValueError("La exportación real no contiene ninguna fecha legible: revise el formato "
                         "de las columnas de fecha y la opción «dia_primero» del mapeo.")
    return tabla_calendario(fechas.min().normalize(), fechas.max().normalize())


def cobertura_opcionales(tablas):
    """Resumen por tabla de las columnas opcionales del contrato: cuántas
    trae el origen y cuáles faltan (los análisis que las usen las omiten)."""
    cobertura = esquema.cobertura(tablas)
    opcionales = cobertura[~cobertura["Obligatoria"] & (cobertura["Momento"] != "fuera del contrato")]
    filas = []
    for tabla in esquema.TABLAS:
        de_tabla = opcionales[opcionales["Tabla"] == tabla]
        faltan = de_tabla.loc[~de_tabla["Presente"], "Columna"].tolist()
        filas.append({"Tabla": tabla, "Opcionales del contrato": len(de_tabla),
                      "Presentes": int(de_tabla["Presente"].sum()), "Faltan": len(faltan),
                      "Columnas que faltan": ", ".join(faltan) if faltan else "ninguna"})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# Disco
# ----------------------------------------------------------------------
def guardar_crudas(tablas, verdad=None):
    """Guarda las tablas crudas como CSV (fechas en texto ISO) y, si la hay,
    la verdad oculta como parquet en `datos/auditoria/`. Sin `verdad` (datos
    reales) borra la de una simulación anterior para que no se mezclen."""
    rutas.DIR_CRUDOS.mkdir(parents=True, exist_ok=True)
    rutas.DIR_AUDITORIA.mkdir(parents=True, exist_ok=True)
    for nombre in TABLAS_CRUDAS:
        tablas[nombre].to_csv(rutas.DIR_CRUDOS / f"{nombre}.csv", index=False,
                              date_format="%Y-%m-%d", encoding="utf-8")
    for nombre in TABLAS_VERDAD:
        ruta = rutas.DIR_AUDITORIA / f"verdad_{nombre}.parquet"
        if verdad is not None:
            verdad[nombre].to_parquet(ruta, index=False)
        elif ruta.exists():
            ruta.unlink()
    return rutas.DIR_CRUDOS


def leer_crudas():
    """Lee las tablas crudas del disco como texto, sin limpiar ni convertir
    (las celdas vacías quedan como faltantes)."""
    tablas = {}
    for nombre in TABLAS_CRUDAS:
        ruta = rutas.DIR_CRUDOS / f"{nombre}.csv"
        if not ruta.exists():
            raise FileNotFoundError(
                f"Falta {ruta}. Ejecute primero la etapa «01_datos», que genera o carga las tablas crudas."
            )
        tablas[nombre] = pd.read_csv(ruta, dtype=str, keep_default_na=False, na_values=[""],
                                     encoding="utf-8")
    return tablas


def leer_verdad():
    """Verdad oculta de la simulación (`{muestras, clientes, ensayos}`), o
    None si los datos son reales. Solo para las celdas de auditoría."""
    archivos = {nombre: rutas.DIR_AUDITORIA / f"verdad_{nombre}.parquet" for nombre in TABLAS_VERDAD}
    if not all(ruta.exists() for ruta in archivos.values()):
        return None
    return {nombre: pd.read_parquet(ruta) for nombre, ruta in archivos.items()}


def huella(tablas):
    """Huella SHA-256 (12 caracteres) del contenido de las tablas, para dejar
    constancia de con qué datos se obtuvo cada resultado."""
    resumen = hashlib.sha256()
    for nombre in sorted(tablas):
        resumen.update(nombre.encode("utf-8"))
        resumen.update(tablas[nombre].to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"))
    return resumen.hexdigest()[:12]
