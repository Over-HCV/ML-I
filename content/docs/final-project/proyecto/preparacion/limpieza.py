"""Capa 2 · Limpieza de las tablas crudas.

Responsabilidad única: convertir las tablas crudas (texto) en tablas con
tipos correctos y sin errores de registro, y dejar constancia de cada
problema encontrado en una tabla de auditoría.

Todas las reglas son deterministas y están escritas aquí: tipos y fechas,
grafía de las etiquetas, duplicados, valores físicamente imposibles, lectura
del recuento reportado y coherencia entre el recuento y el resultado. No se
aprende nada de la distribución de los datos y no se imputa: un valor que no
se conoce queda como faltante, y la imputación ocurre después, dentro de la
tubería de cada modelo (ajustada solo con entrenamiento). Las reglas no
dependen de nombres o etiquetas de la simulación: una columna opcional que no
venga se omite, y una etiqueta desconocida se conserva.
"""
import re

import numpy as np
import pandas as pd

from proyecto.datos import esquema
from proyecto.datos.mapeo import VALORES_CANONICOS, clave_etiqueta

COLUMNAS_AUDITORIA = ["tabla", "problema", "filas_afectadas", "accion"]

LLAVES = {"clientes": "id_cliente", "muestreos": "id_muestreo", "muestras": "id_muestra",
          "controles": "id_control"}

# Intervalos físicamente posibles (cerrados; None = sin cota). Fuera de ellos
# el valor es un error de registro y se convierte en faltante.
RANGOS_VALIDOS = {
    "clientes": {"distancia_km": (0, 3000)},
    "muestreos": {"puntos_programados": (0, None), "duracion_horas": (0, 24)},
    "muestras": {
        "actividad_agua": (0, 1), "ph": (0, 14), "volumen_ml": (0, None),
        "hora_recepcion": (0, 24), "temperatura_transporte_c": (-30, 60),
        "horas_transporte": (0, 720), "dias_prometidos": (0, 365), "carga_dia": (0, None),
        "control_ambiental_ufc": (0, None), "limite_ufc": (0, None),
    },
    "controles": {},
}

# Pares (antes, después, columna que se anula si el orden no se cumple).
ORDEN_FECHAS = [
    ("fecha_toma", "fecha_recepcion", "fecha_toma"),
    ("fecha_recepcion", "fecha_siembra", "fecha_siembra"),
    ("fecha_siembra", "fecha_lectura", "fecha_lectura"),
    ("fecha_recepcion", "fecha_reporte", "fecha_reporte"),
    ("fecha_lectura", "fecha_reporte", "fecha_reporte"),
]

_NEGATIVOS = ("ausencia", "ausente", "negativ", "no detect", "no se detect", "sin crecimiento")
_POSITIVOS = ("presencia", "presente", "positiv", "detect", "crecimiento")
_NUMERO = re.compile(
    r"^(?P<signo>[<>]?)=?(?P<mantisa>\d+(?:[.,]\d+)?)(?:(?:[x*]10\^?|e\+?)(?P<exponente>\d+))?")
_NUMERO_SIMPLE = re.compile(r"^[<>]?\d+(?:\.\d+)?$")


# ----------------------------------------------------------------------
# Conversión de tipos
# ----------------------------------------------------------------------
def _es_nulo(valor):
    return valor is None or valor is pd.NaT or (isinstance(valor, float) and np.isnan(valor))


def _texto(serie, recortar=True):
    """Texto sin espacios en los extremos; vacío o nulo → faltante. Con
    `recortar=False` devuelve el texto tal como viene (para comparar)."""
    def limpiar(valor):
        if _es_nulo(valor):
            return None
        if not recortar:
            return str(valor)
        valor = str(valor).strip()
        return valor if valor else None
    return pd.Series([limpiar(v) for v in serie.tolist()], index=serie.index, dtype=object)


def normalizar_etiquetas(serie, canonicas=None):
    """Unifica la grafía de una columna categórica.

    Dos valores son la misma etiqueta si coinciden al ignorar mayúsculas,
    tildes y espacios repetidos. Cada grupo se escribe como en `canonicas` si
    alguna de ellas pertenece al grupo y, si no, con la grafía más frecuente
    del grupo (a igualdad, la que tiene tildes y menos mayúsculas). La regla
    solo decide cómo se escribe cada categoría, nunca a qué categoría
    pertenece una fila; las etiquetas que no se parecen a ninguna otra se
    conservan."""
    texto = _texto(serie).map(lambda v: " ".join(v.split()) if isinstance(v, str) else None)
    claves = texto.map(lambda v: clave_etiqueta(v) if isinstance(v, str) else None)
    grafias = {clave_etiqueta(c): c for c in (canonicas or [])}
    conteo = pd.DataFrame({"clave": claves, "grafia": texto}).dropna()
    if len(conteo):
        conteo = conteo.groupby(["clave", "grafia"]).size().reset_index(name="n")
        conteo["tildes"] = [sum(ord(c) > 127 for c in g) for g in conteo["grafia"]]
        conteo["mayusculas"] = [sum(c.isupper() for c in g) for g in conteo["grafia"]]
        conteo = conteo.sort_values(["clave", "n", "tildes", "mayusculas", "grafia"],
                                    ascending=[True, False, False, True, True])
        for clave, grafia in zip(conteo["clave"], conteo["grafia"]):
            grafias.setdefault(clave, grafia)
    return pd.Series([grafias.get(k) if k is not None else None for k in claves.tolist()],
                     index=serie.index, dtype=object)


def _a_numero(serie):
    """Devuelve `(números, filas ilegibles)`. Acepta coma decimal."""
    if pd.api.types.is_bool_dtype(serie):
        return serie.astype(float), 0
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float), 0
    texto = _texto(serie)
    def normalizar(valor):
        if valor is None:
            return None
        valor = valor.replace(" ", "")
        return valor.replace(",", ".") if ("," in valor and "." not in valor) else valor
    numeros = pd.to_numeric(texto.map(normalizar), errors="coerce").astype(float)
    return numeros, int((texto.notna() & numeros.isna()).sum())


def _a_fecha(serie):
    """Devuelve `(fechas sin hora, filas ilegibles)`. Espera texto ISO."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie.astype("datetime64[ns]").dt.normalize(), 0
    texto = _texto(serie)
    fechas = pd.to_datetime(texto, format="ISO8601", errors="coerce").astype("datetime64[ns]").dt.normalize()
    return fechas, int((texto.notna() & fechas.isna()).sum())


def _a_booleano(serie):
    if pd.api.types.is_bool_dtype(serie):
        return serie.astype(bool)
    verdaderos = {"true", "1", "si", "verdadero"}
    return pd.Series([clave_etiqueta(v) in verdaderos if v is not None else False
                      for v in _texto(serie).tolist()], index=serie.index, dtype=bool)


def _interpretar_recuento(texto):
    """Un recuento reportado → (valor, censurado, cualitativo, con texto extra).
    `cualitativo` es 1 (presencia), 0 (ausencia) o NaN (recuento numérico)."""
    clave = clave_etiqueta(texto.replace("≤", "<").replace("≥", ">").replace("×", "x"))
    if any(patron in clave for patron in _NEGATIVOS) or clave in ("nd", "neg", "-"):
        return (np.nan, 0, 0.0, 0)
    if any(patron in clave for patron in _POSITIVOS) or clave in ("pos", "+"):
        return (np.nan, 0, 1.0, 0)
    compacto = clave.replace(" ", "")
    hallado = _NUMERO.match(compacto)
    if hallado is None:
        return (np.nan, 0, np.nan, 0)
    valor = float(hallado.group("mantisa").replace(",", "."))
    if hallado.group("exponente"):
        valor *= 10.0 ** int(hallado.group("exponente"))
    return (valor, int(hallado.group("signo") != ""), np.nan, int(_NUMERO_SIMPLE.match(compacto) is None))


def interpretar_recuento(serie):
    """Lee la columna `recuento_reportado` y devuelve una tabla con
    `valor` (UFC; la cota si el recuento está censurado), `censurado` (0/1),
    `cualitativo` (1 presencia, 0 ausencia, NaN si es numérico), `direccion`
    («<», «>» o vacío), `con_texto` (1 si traía unidad u otro texto) e
    `ilegible` (1 si había texto y no se entendió)."""
    texto = _texto(serie)
    unicos = {valor: _interpretar_recuento(valor) for valor in texto.dropna().unique()}
    vacio = (np.nan, 0, np.nan, 0)
    filas = [unicos[v] if v is not None else vacio for v in texto.tolist()]
    tabla = pd.DataFrame(filas, index=serie.index, columns=["valor", "censurado", "cualitativo", "con_texto"])
    def direccion(valor):
        if valor is None:
            return ""
        inicio = valor.replace("≤", "<").replace("≥", ">").strip()[:1]
        return inicio if inicio in "<>" else ""
    tabla["direccion"] = [direccion(v) if c else "" for v, c in zip(texto.tolist(), tabla["censurado"])]
    tabla["ilegible"] = (texto.notna() & tabla["valor"].isna() & tabla["cualitativo"].isna()).astype(int)
    return tabla


# ----------------------------------------------------------------------
# Limpieza genérica de una tabla del contrato
# ----------------------------------------------------------------------
def _anotar(auditoria, tabla, problema, filas, accion):
    if int(filas) > 0:
        auditoria.append({"tabla": tabla, "problema": problema, "filas_afectadas": int(filas),
                          "accion": accion})


def _limpiar_tabla(nombre, cruda, auditoria):
    tabla = cruda.copy()
    repetidas = tabla.duplicated(keep="first")
    _anotar(auditoria, nombre, "filas duplicadas exactas", repetidas.sum(), "se conserva una copia")
    tabla = tabla.loc[~repetidas].reset_index(drop=True)

    canonicas = VALORES_CANONICOS.get(nombre, {})
    rangos = RANGOS_VALIDOS.get(nombre, {})
    for columna in esquema.ESQUEMA[nombre]:
        if columna.nombre not in tabla.columns:
            continue
        cruda_col = tabla[columna.nombre]
        if columna.tipo == "categoria":
            original = _texto(cruda_col, recortar=False)
            limpia = normalizar_etiquetas(cruda_col, canonicas.get(columna.nombre))
            cambiadas = limpia.notna() & (original != limpia)
            _anotar(auditoria, nombre, f"etiquetas con grafía inconsistente en «{columna.nombre}»",
                    cambiadas.sum(), "se unifican mayúsculas, tildes y espacios")
            if columna.nombre in canonicas:
                ajenas = limpia.notna() & ~limpia.isin(canonicas[columna.nombre])
                _anotar(auditoria, nombre, f"etiqueta no reconocida en «{columna.nombre}»", ajenas.sum(),
                        "se conserva; tradúzcala en «valores» del mapeo si es un sinónimo")
            tabla[columna.nombre] = limpia
        elif columna.tipo == "numero":
            numeros, ilegibles = _a_numero(cruda_col)
            _anotar(auditoria, nombre, f"valor no numérico en «{columna.nombre}»", ilegibles,
                    "se convierte en faltante")
            if columna.nombre in rangos:
                minimo, maximo = rangos[columna.nombre]
                fuera = pd.Series(False, index=numeros.index)
                if minimo is not None:
                    fuera = fuera | (numeros < minimo)
                if maximo is not None:
                    fuera = fuera | (numeros > maximo)
                intervalo = f"[{'' if minimo is None else minimo}, {'∞' if maximo is None else maximo}]"
                _anotar(auditoria, nombre, f"valor imposible en «{columna.nombre}» (fuera de {intervalo})",
                        fuera.sum(), "se convierte en faltante")
                numeros = numeros.mask(fuera)
            tabla[columna.nombre] = numeros
        elif columna.tipo == "fecha":
            fechas, ilegibles = _a_fecha(cruda_col)
            _anotar(auditoria, nombre, f"fecha ilegible en «{columna.nombre}»", ilegibles,
                    "se convierte en faltante")
            tabla[columna.nombre] = fechas
        else:  # id y texto
            tabla[columna.nombre] = _texto(cruda_col)

    llave = LLAVES.get(nombre)
    if llave in tabla.columns:
        sin_llave = tabla[llave].isna()
        _anotar(auditoria, nombre, f"filas sin «{llave}»", sin_llave.sum(), "se eliminan")
        tabla = tabla.loc[~sin_llave]
        repetidas = tabla.duplicated(llave, keep="first")
        _anotar(auditoria, nombre, f"«{llave}» repetido con contenido distinto", repetidas.sum(),
                "se conserva la primera fila")
        tabla = tabla.loc[~repetidas].reset_index(drop=True)

    for antes, despues, anulada in ORDEN_FECHAS:
        if antes in tabla.columns and despues in tabla.columns:
            invertidas = tabla[despues] < tabla[antes]
            _anotar(auditoria, nombre, f"«{despues}» anterior a «{antes}»", invertidas.sum(),
                    f"«{anulada}» se convierte en faltante")
            tabla[anulada] = tabla[anulada].mask(invertidas)
    return tabla


def _limpiar_calendario(cruda):
    cal = cruda.copy()
    cal["fecha"] = _a_fecha(cal["fecha"])[0]
    for columna in ("es_festivo", "es_habil", "es_semana_santa"):
        if columna in cal.columns:
            cal[columna] = _a_booleano(cal[columna])
    if "nombre_festivo" in cal.columns:
        cal["nombre_festivo"] = _texto(cal["nombre_festivo"]).fillna("")
    return cal.dropna(subset=["fecha"]).drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)


# ----------------------------------------------------------------------
# Columnas derivadas de `muestras`
# ----------------------------------------------------------------------
def _derivar_muestras(m, auditoria):
    m = m.copy()
    n = len(m)
    nan = pd.Series(np.nan, index=m.index)

    # Recepción.
    if "estado_recepcion" in m.columns:
        m["rechazada"] = (m["estado_recepcion"] == "rechazada").astype(int)
    else:
        m["rechazada"] = 0
    rechazada = m["rechazada"] == 1

    # Recuento reportado.
    if "recuento_reportado" in m.columns:
        recuento = interpretar_recuento(m["recuento_reportado"])
        _anotar(auditoria, "muestras", "recuento censurado («<N» o «>N»)", recuento["censurado"].sum(),
                "la cota queda en «recuento_ufc» y se marca «recuento_censurado» = 1")
        _anotar(auditoria, "muestras", "recuento con unidad u otro texto", recuento["con_texto"].sum(),
                "se extrae el número")
        _anotar(auditoria, "muestras", "recuento ilegible", recuento["ilegible"].sum(),
                "«recuento_ufc» queda faltante")
    else:
        recuento = pd.DataFrame({"valor": nan, "censurado": 0, "cualitativo": nan, "direccion": ""},
                                index=m.index)
    m["recuento_ufc"] = recuento["valor"].astype(float)
    m["recuento_censurado"] = recuento["censurado"].astype(int)

    # Un ensayo es cuantitativo si tiene límite numérico o si alguna vez
    # reporta un recuento: es una propiedad del ensayo, no de la muestra.
    con_numero = m["recuento_ufc"].notna()
    if "limite_ufc" in m.columns:
        con_numero = con_numero | m["limite_ufc"].notna()
    m["es_cuantitativo"] = (con_numero.astype(float).groupby(m["ensayo"]).transform("max")
                            .fillna(0.0).astype(int))

    # Resultado: etiqueta registrada frente a lo que dice el recuento.
    etiqueta = m["resultado"].map({"no conforme": 1.0, "conforme": 0.0})
    esperado = recuento["cualitativo"].astype(float).copy()
    if "limite_ufc" in m.columns:
        valor, limite = m["recuento_ufc"], m["limite_ufc"]
        exacto = valor.notna() & limite.notna() & (recuento["censurado"] == 0)
        esperado = esperado.mask(exacto, (valor > limite).astype(float))
        bajo = valor.notna() & limite.notna() & (recuento["direccion"] == "<") & (valor <= limite)
        esperado = esperado.mask(bajo, 0.0)
        alto = valor.notna() & limite.notna() & (recuento["direccion"] == ">") & (valor >= limite)
        esperado = esperado.mask(alto, 1.0)
    contradice = etiqueta.notna() & esperado.notna() & (etiqueta != esperado)
    _anotar(auditoria, "muestras", "resultado que contradice el recuento reportado", contradice.sum(),
            "se corrige el resultado según el recuento y el límite")
    deducido = etiqueta.isna() & esperado.notna() & ~rechazada
    _anotar(auditoria, "muestras", "resultado faltante con recuento reportado", deducido.sum(),
            "se deduce el resultado del recuento y el límite")
    no_conforme = etiqueta.mask(contradice | deducido, esperado).mask(rechazada)
    con_resultado_rechazada = rechazada & etiqueta.notna()
    _anotar(auditoria, "muestras", "muestra rechazada con resultado registrado",
            con_resultado_rechazada.sum(), "«no_conforme» queda faltante")
    m["no_conforme"] = no_conforme.astype(float)
    corregir = (contradice | deducido) & ~rechazada
    m["resultado"] = m["resultado"].mask(
        corregir, m["no_conforme"].map({1.0: "no conforme", 0.0: "conforme"}))

    # Reensayo.
    hecho = pd.Series(False, index=m.index)
    if "reensayo" in m.columns:
        hecho = hecho | (m["reensayo"] == "sí")
    if "resultado_reensayo" in m.columns:
        hecho = hecho | m["resultado_reensayo"].notna()
    m["reensayada"] = hecho.astype(int)
    m["falso_positivo"] = np.nan
    if "resultado_reensayo" in m.columns:
        positiva = m["no_conforme"] == 1
        m.loc[positiva & (m["resultado_reensayo"] == "conforme"), "falso_positivo"] = 1.0
        m.loc[positiva & (m["resultado_reensayo"] == "no conforme"), "falso_positivo"] = 0.0

    # Entrega.
    m["dias_entrega"] = np.nan
    m["entrega_tardia"] = np.nan
    if "fecha_reporte" in m.columns:
        m["dias_entrega"] = (m["fecha_reporte"] - m["fecha_recepcion"]).dt.days.astype(float)
        if "dias_prometidos" in m.columns:
            comparable = m["dias_entrega"].notna() & m["dias_prometidos"].notna()
            m.loc[comparable, "entrega_tardia"] = (
                m.loc[comparable, "dias_entrega"] > m.loc[comparable, "dias_prometidos"]).astype(float)
    assert len(m) == n
    return m


# ----------------------------------------------------------------------
# Faltantes e integridad
# ----------------------------------------------------------------------
def _faltantes_esperados(nombre, tabla):
    """{columna: máscara de filas en las que el dato no puede existir} (faltante
    estructural: explicado por el proceso, no por un descuido de registro)."""
    if nombre != "muestras":
        return {}
    falso = pd.Series(False, index=tabla.index)
    rechazada = tabla["rechazada"] == 1 if "rechazada" in tabla.columns else falso
    esperados = {}
    posteriores = set(esquema.columnas("muestras")) - set(esquema.columnas("muestras", hasta="recepcion"))
    for columna in posteriores:
        esperados[columna] = rechazada
    if "reensayada" in tabla.columns:
        esperados["resultado_reensayo"] = rechazada | (tabla["reensayada"] == 0)
    if "origen" in tabla.columns:
        en_planta = tabla["origen"] == "muestreo en planta"
        for columna in ("id_lote_envio", "codigo_producto", "actividad_agua", "ph"):
            esperados[columna] = en_planta
        esperados["id_muestreo"] = ~en_planta
    if "es_cuantitativo" in tabla.columns:
        esperados["limite_ufc"] = tabla["es_cuantitativo"] == 0
    return esperados


def faltantes_por_columna(limpias):
    """Tabla con los faltantes de cada columna del contrato, separando los
    estructurales (el dato no puede existir: muestra rechazada, ensayo sin
    límite, muestra sin producto...) de los no estructurales."""
    filas = []
    for nombre in esquema.TABLAS:
        if nombre not in limpias:
            continue
        tabla = limpias[nombre]
        esperados = _faltantes_esperados(nombre, tabla)
        for columna in esquema.columnas(nombre):
            if columna not in tabla.columns:
                continue
            falta = tabla[columna].isna()
            estructural = falta & esperados.get(columna, pd.Series(False, index=tabla.index))
            filas.append({
                "tabla": nombre, "columna": columna, "momento": esquema.momento_de(nombre, columna),
                "filas": len(tabla), "faltantes": int(falta.sum()),
                "estructurales": int(estructural.sum()),
                "no_estructurales": int((falta & ~estructural).sum()),
            })
    resumen = pd.DataFrame(filas)
    if len(resumen):
        resumen["pct_faltante"] = 100 * resumen["faltantes"] / resumen["filas"].clip(lower=1)
        resumen["pct_no_estructural"] = 100 * resumen["no_estructurales"] / resumen["filas"].clip(lower=1)
    return resumen


def _auditar_faltantes(limpias, auditoria):
    resumen = faltantes_por_columna(limpias)
    for fila in resumen.itertuples(index=False):
        _anotar(auditoria, fila.tabla, f"faltante estructural en «{fila.columna}»", fila.estructurales,
                "se conserva: el dato no puede existir para esa fila")
        _anotar(auditoria, fila.tabla, f"faltante no estructural en «{fila.columna}»",
                fila.no_estructurales, "se conserva como faltante (si la columna se usa, se imputa dentro de la tubería del modelo)")


def _auditar_integridad(limpias, auditoria):
    """Referencias entre tablas que no encuentran su fila."""
    referencias = [("muestras", "id_cliente", "clientes"), ("muestras", "id_muestreo", "muestreos"),
                   ("muestreos", "id_cliente", "clientes")]
    for tabla, columna, destino in referencias:
        if tabla in limpias and destino in limpias and columna in limpias[tabla].columns:
            valores = limpias[tabla][columna]
            huerfanas = valores.notna() & ~valores.isin(limpias[destino][columna])
            _anotar(auditoria, tabla, f"«{columna}» sin fila correspondiente en «{destino}»", huerfanas.sum(),
                    "se conserva; los atributos de la otra tabla quedan faltantes")


# ----------------------------------------------------------------------
# Entrada principal
# ----------------------------------------------------------------------
def limpiar_tablas(crudas):
    """Limpia todas las tablas crudas.

    Devuelve `(limpias, auditoria)`: `limpias` es un diccionario con las
    mismas tablas ya tipadas (y `muestras` con sus columnas derivadas) y
    `auditoria` una tabla con una fila por problema encontrado
    (`tabla`, `problema`, `filas_afectadas`, `accion`)."""
    auditoria, limpias = [], {}
    for nombre in esquema.TABLAS:
        if nombre in crudas:
            limpias[nombre] = _limpiar_tabla(nombre, crudas[nombre], auditoria)
    if "calendario" in crudas:
        limpias["calendario"] = _limpiar_calendario(crudas["calendario"])
    if "muestras" in limpias:
        limpias["muestras"] = _derivar_muestras(limpias["muestras"], auditoria)
    _auditar_integridad(limpias, auditoria)
    _auditar_faltantes(limpias, auditoria)
    return limpias, pd.DataFrame(auditoria, columns=COLUMNAS_AUDITORIA)
