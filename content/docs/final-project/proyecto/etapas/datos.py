"""Capa 8 · Etapa 01, parte A: el conjunto de datos.

Responsabilidad única: armar las piezas de la parte A de la etapa 01
(contrato de datos, tablas crudas validadas, cómo llegan los datos, cómo se
conecta la exportación real y protocolo temporal) y registrar sus cifras para
el informe. Cada función pública es una sección del notebook. Abre la etapa y
la deja abierta para la parte B (`exploracion`), que la cierra.
"""
import html as html_estandar
import json

import pandas as pd

from proyecto.base import config
from proyecto.datos import carga, esquema, mapeo
from proyecto.etapas.comun import Resultado, abrir, cerrar_parte
from proyecto.preparacion import limpieza
from proyecto.reporte import latex
from proyecto.reporte.html import linea_tiempo, nota, pasos, pestanas, tabla_html, tarjetas

ETAPA = "01_datos_exploracion"
FASES_MUESTRAS = {
    "muestras (toma)": ["maestro", "programacion", "toma"],
    "muestras (llegada y recepción)": ["llegada", "recepcion"],
    "muestras (siembra a reporte)": ["siembra", "lectura", "reensayo", "reporte"],
}
SIGNIFICADO_MOMENTOS = {
    "maestro": "datos del cliente", "programacion": "visita agendada", "toma": "muestra y producto",
    "llegada": "transporte", "recepcion": "aceptación", "siembra": "analista y equipos",
    "lectura": "recuento y resultado", "reensayo": "confirmación", "reporte": "informe",
}
PROPOSITO_ETAPAS = {
    "01_datos_exploracion": "Contrato de datos, tablas crudas, auditoría de calidad, limpieza, integración y exploración "
                            "del periodo de desarrollo.",
    "02_base_real": "La exportación real: Llegadas de Poisson, tamaño de los lotes, segmentos de clientes y riesgo de "
                    "no conformidad al ingreso con una sola prueba en 2026.",
    "03_cribado_regresion": "Catálogo de análisis candidatos, prueba rápida uniforme, selección con una regla fijada de "
                            "antemano y estudio a fondo del análisis de regresión elegido.",
    "04_clasificacion_bayes": "Estudio a fondo del análisis de clasificación elegido, umbral por costo y capa bayesiana.",
    "05_agrupamiento": "K-medias y DBSCAN sobre el análisis de agrupamiento elegido.",
    "06_evaluacion_final": "Modelos congelados evaluados una sola vez sobre el bloque de prueba.",
}


def iniciar():
    """Contexto de la etapa (tras `iniciar_etapa`)."""
    c = abrir(ETAPA, "El problema, el contrato de datos, la limpieza y lo que vimos antes de modelar")
    c.autores = nota(config.AUTORES, titulo=config.ROTULO_PROYECTO)
    return c


def contrato(c):
    """Diccionario de datos por tabla, cifras del contrato y línea de tiempo de los momentos."""
    r = Resultado()
    diccionario = esquema.diccionario()
    fichas = {}
    for nombre_tabla in esquema.TABLAS:
        parte = diccionario[diccionario["Tabla"] == nombre_tabla].drop(columns="Tabla")
        if nombre_tabla != "muestras":
            fichas[nombre_tabla] = tabla_html(parte, f'Diccionario de "{nombre_tabla}"', maximo_filas=None)
            continue
        for rotulo, momentos in FASES_MUESTRAS.items():
            fase = parte[parte["Momento"].isin(momentos)]
            if len(fase):
                fichas[rotulo] = tabla_html(fase, f"Diccionario de {rotulo}", maximo_filas=None)
    r.agregar(pestanas(fichas, titulo="Diccionario de datos por tabla"))

    n_columnas = len(diccionario)
    n_obligatorias = int((diccionario["Obligatoria"] == "sí").sum())
    n_tablas = int(diccionario["Tabla"].nunique())
    r.agregar(nota(
        f"Al final el contrato quedó con **{n_columnas} columnas** en {n_tablas} tablas. Solo **{n_obligatorias}** son "
        "obligatorias, o sea que el código corre con una exportación que traiga apenas esas, y las otras "
        f"{n_columnas - n_obligatorias} son opcionales. A esto le sumamos el calendario laboral colombiano (festivos y días "
        "hábiles), que no viene en la base del laboratorio sino que lo calculamos nosotros.",
    ))
    columnas_por_momento = diccionario["Momento"].value_counts()
    r.agregar(linea_tiempo(
        [(f"{int(columnas_por_momento.get(m, 0))} col.", m, SIGNIFICADO_MOMENTOS.get(m, "")) for m in esquema.MOMENTOS],
        titulo="Momentos del proceso y columnas que se conocen en cada uno",
    ))
    latex.registrar_varias([
        ("DatColumnasContrato", n_columnas, "columnas declaradas en el contrato de datos"),
        ("DatColumnasObligatorias", n_obligatorias, "columnas obligatorias del contrato"),
        ("DatColumnasOpcionales", n_columnas - n_obligatorias, "columnas opcionales del contrato"),
        ("DatTablasContrato", n_tablas, "tablas del contrato de datos"),
        ("DatMomentos", len(esquema.MOMENTOS), "momentos del proceso declarados en el contrato"),
    ])
    latex.exportar_tabla(diccionario, "diccionario_datos", "Diccionario de datos completo (contrato)")
    latex.exportar_tabla(columnas_por_momento.reindex(esquema.MOMENTOS).fillna(0).astype(int)
                         .rename_axis("Momento").reset_index(name="Columnas"),
                         "columnas_por_momento", "Columnas del contrato que se conocen en cada momento")
    r.valores.update(columnas=n_columnas, obligatorias=n_obligatorias)
    return r


def tablas_crudas(c):
    """Obtiene las tablas (simulador o exportación real), las valida, las guarda y las resume."""
    r = Resultado()
    tablas, verdad = carga.obtener_tablas_crudas()
    cobertura = esquema.validar(tablas)
    carga.guardar_crudas(tablas, verdad)
    c.tablas, c.verdad, c.huella = tablas, verdad, carga.huella(tablas)
    fechas = pd.to_datetime(tablas["muestras"]["fecha_recepcion"], errors="coerce")
    c.desde, c.hasta = f"{fechas.min():%Y-%m-%d}", f"{fechas.max():%Y-%m-%d}"
    c.clientes_con_muestras = int(tablas["muestras"]["id_cliente"].nunique())
    r.agregar(tarjetas([
        {"titulo": "Muestras", "valor": len(tablas["muestras"]), "detalle": "filas crudas, una por ensayo"},
        {"titulo": "Clientes", "valor": len(tablas["clientes"]),
         "detalle": f"{c.clientes_con_muestras} con alguna muestra", "color": "verde"},
        {"titulo": "Visitas de muestreo", "valor": len(tablas["muestreos"]), "detalle": "muestreos en planta",
         "color": "naranja"},
        {"titulo": "Controles de calidad", "valor": len(tablas["controles"]),
         "detalle": "controles positivos y negativos", "color": "morado"},
        {"titulo": "Días de calendario", "valor": len(tablas["calendario"]), "detalle": f"{c.desde} a {c.hasta}",
         "color": "gris"},
    ], titulo="Tablas crudas obtenidas"))
    r.agregar(nota(
        f"Los datos salieron de **{'la simulación' if config.ORIGEN_DATOS == 'simulacion' else 'la exportación real'}**. "
        "Para poder comprobar después con qué datos salió cada cifra guardamos con los resultados de cada etapa una "
        f"huella del contenido, `{c.huella}`, que son los primeros caracteres de un SHA-256 (un resumen que cambia si "
        "cambia una sola celda).",
    ))

    resumen_obligatorias = (
        cobertura[cobertura["Obligatoria"]]
        .groupby("Tabla", sort=False)["Presente"].agg(["size", "sum"])
        .rename(columns={"size": "Obligatorias del contrato", "sum": "Presentes"})
        .reset_index()
    )
    opcionales = carga.cobertura_opcionales(tablas)
    r.agregar(pestanas({
        "Columnas obligatorias": tabla_html(resumen_obligatorias, "Columnas obligatorias por tabla"),
        "Columnas opcionales": tabla_html(opcionales, "Columnas opcionales que trae el origen"),
    }, titulo="Validación de las tablas contra el contrato"))
    faltan = int(opcionales["Faltan"].sum())
    r.agregar(nota(
        "El origen trae todas las columnas del contrato (obligatorias y opcionales) así que ningún análisis del "
        "catálogo se cae por falta de datos." if faltan == 0 else
        f"Al origen le faltan **{faltan}** columnas opcionales. Los análisis que las necesitan no fallan: En el cribado "
        "del notebook 03 quedan marcados como no disponibles.",
        "hallazgo",
    ))
    filas = {nombre: int(len(t)) for nombre, t in tablas.items()}
    latex.registrar_varias([
        ("DatOrigen", config.ORIGEN_DATOS, "origen de los datos (simulacion o real)"),
        ("DatHuella", c.huella, "huella SHA-256 abreviada de las tablas crudas"),
        ("DatMuestrasCrudas", filas["muestras"], "filas crudas de la tabla de muestras"),
        ("DatClientes", filas["clientes"], "clientes en la tabla de clientes"),
        ("DatClientesConMuestras", c.clientes_con_muestras, "clientes con al menos una muestra"),
        ("DatVisitas", filas["muestreos"], "visitas de muestreo en planta"),
        ("DatControles", filas["controles"], "controles de calidad (positivos y negativos)"),
        ("DatDiasCalendario", filas["calendario"], "días de la tabla de calendario"),
        ("DatDesde", c.desde, "primera fecha de recepción"),
        ("DatHasta", c.hasta, "última fecha de recepción"),
        ("DatFaltanOpcionales", faltan, "columnas opcionales que no trae el origen"),
    ])
    latex.exportar_tabla(opcionales, "cobertura_opcionales", "Columnas opcionales que trae el origen, por tabla")
    latex.exportar_tabla(pd.DataFrame([{"Tabla": n, "Filas": f} for n, f in filas.items()]), "filas_crudas",
                         "Filas de cada tabla cruda")
    r.valores.update(filas=filas, huella=c.huella, faltan_opcionales=faltan)
    return r


def como_llegan(c):
    """Ejemplos de la suciedad de las tablas crudas (no se corrige aquí)."""
    r = Resultado()
    crudas = c.tablas["muestras"]
    ejemplos = {}
    vista = [x for x in ["id_muestra", "id_cliente", "fecha_recepcion", "matriz", "ensayo", "recuento_reportado",
                         "limite_ufc", "resultado"] if x in crudas.columns]
    ejemplos["Primeras filas"] = tabla_html(crudas[vista].head(6), 'Primeras filas de "muestras"')

    columna = next((x for x in ["matriz", "ensayo"] if x in crudas.columns), None)
    if columna is not None:
        grafias = crudas[columna].dropna().astype(str).value_counts().rename_axis("Como viene escrita").reset_index(name="Filas")
        grafias["Categoría"] = grafias["Como viene escrita"].map(mapeo.clave_etiqueta)
        categoria = grafias.groupby("Categoría")["Filas"].size().idxmax()
        ejemplo = grafias[grafias["Categoría"] == categoria].drop(columns="Categoría")
        ejemplo["Como viene escrita"] = '"' + ejemplo["Como viene escrita"] + '"'
        ejemplos["Etiquetas"] = tabla_html(ejemplo.head(10), f'Grafías de una misma categoría en "{columna}"')

    recuentos = crudas["recuento_reportado"].dropna().astype(str)
    no_numericos = recuentos[~recuentos.str.fullmatch(r"\d+(\.\d+)?")]
    ejemplos["Recuentos"] = tabla_html(
        no_numericos.value_counts().head(10).rename_axis("Recuento reportado").reset_index(name="Filas"),
        "Recuentos que no son un número simple (los más frecuentes)")

    imposibles = []
    repetidas = crudas.duplicated(keep="first")
    en_repetidas = 0   # valores imposibles que caen en filas duplicadas (la limpieza los cuenta tras quitarlas)
    for col in [x for x in limpieza.RANGOS_VALIDOS["muestras"] if x in crudas.columns]:
        minimo, maximo = limpieza.RANGOS_VALIDOS["muestras"][col]
        valores = pd.to_numeric(crudas[col], errors="coerce")
        fuera = (valores < minimo) if maximo is None else ((valores < minimo) | (valores > maximo))
        en_repetidas += int((fuera & repetidas).sum())
        if fuera.any():
            imposibles.append({"Columna": col, "Rango posible": f"[{minimo}, {'∞' if maximo is None else maximo}]",
                               "Filas fuera del rango": int(fuera.sum()), "Ejemplo": float(valores[fuera].iloc[0])})
    if imposibles:
        ejemplos["Valores imposibles"] = tabla_html(pd.DataFrame(imposibles), "Valores fuera del rango físicamente posible",
                                                    decimales=2)
    r.agregar(pestanas(ejemplos, titulo="Las tablas tal como llegan (sin limpiar)"))

    duplicadas = int(repetidas.sum())
    n_imposibles = sum(f["Filas fuera del rango"] for f in imposibles)
    proporcion = len(no_numericos) / max(len(recuentos), 1)
    r.agregar(nota(
        f"En las muestras crudas encontramos **{len(no_numericos)}** recuentos que no son un número simple "
        f"({100 * proporcion:.1f} % de los reportados), **{duplicadas}** filas que repiten exactamente a otra y "
        f"**{n_imposibles}** valores fuera de su rango posible"
        + (f". De esos valores {en_repetidas} caen en filas duplicadas, por eso la limpieza (que primero quita las "
           f"duplicadas) cuenta {n_imposibles - en_repetidas}" if en_repetidas else "")
        + ". Aquí no corregimos nada: Eso queda para la parte B, que deja constancia de cada cambio.",
        "advertencia",
    ))
    latex.registrar_varias([
        ("DatRecuentosNoNumericos", len(no_numericos), "recuentos crudos que no son un número simple"),
        ("DatRecuentosNoNumericosPct", 100 * proporcion, "porcentaje de recuentos no numéricos", 1),
        ("DatDuplicadas", duplicadas, "filas crudas de muestras duplicadas exactas"),
        ("DatValoresImposiblesCrudos", n_imposibles,
         "valores fuera de su rango físico en las filas crudas, con las duplicadas incluidas"),
        ("DatValoresImposiblesEnDuplicadas", en_repetidas,
         "de ellos, los que caen en filas duplicadas (la auditoría de limpieza cuenta tras quitarlas)"),
    ])
    r.valores.update(no_numericos=len(no_numericos), duplicadas=duplicadas, imposibles=n_imposibles)
    return r


def exportacion_real(c):
    """Extracto de la plantilla de mapeo y, con datos simulados, las variables latentes que quedan aparte."""
    r = Resultado()
    descripcion = mapeo.leer_mapeo()["muestras"]
    extracto = {"muestras": {
        "archivo": descripcion["archivo"],
        "columnas": dict(list(descripcion["columnas"].items())[:5]),
        "valores": dict(list(descripcion.get("valores", {}).items())[:1]),
    }}
    texto = html_estandar.escape(json.dumps(extracto, ensure_ascii=False, indent=2))
    r.agregar(nota(
        f'<pre style="margin:0;font-size:12.5px;line-height:1.45;white-space:pre-wrap">{texto}</pre>'
        "A la izquierda va el nombre del contrato y a la derecha el nombre (o la etiqueta) que usa la exportación real. "
        "En la plantilla los dos coinciden: Al conectar una exportación se cambia solo la columna derecha.",
        titulo="configuracion/mapeo_columnas.json (extracto)", con_marcado=True,
    ))
    if c.verdad is not None:
        latentes = pd.DataFrame([
            {"Tabla de auditoría": f"verdad_{nombre}", "Filas": len(t),
             "Columnas latentes": ", ".join(x for x in t.columns if not x.startswith("id_"))}
            for nombre, t in c.verdad.items()])
        r.agregar(tabla_html(latentes, "Variables ocultas que el simulador guarda aparte (ningún modelo las lee)"))
        r.agregar(nota(
            "Lo que concluyamos sobre la operación queda pendiente hasta repetir las etapas con la exportación real.",
            "simulacion",
        ))
    else:
        r.agregar(nota("Los datos vienen de la exportación real del laboratorio, leída a través del mapeo de columnas. "
                       "Aquí no hay variables ocultas y lo que decimos del simulador no aplica."))
    return r


def protocolo(c):
    """Bloques temporales del protocolo y mapa de las etapas."""
    r = Resultado()
    r.agregar(linea_tiempo([
        {"fecha": config.FECHA_INICIO, "titulo": "Entrenamiento", "detalle": "ajustar y comparar", "color": "azul"},
        {"fecha": config.CORTE_VALIDACION, "titulo": "Validación", "detalle": "decidir", "color": "naranja"},
        {"fecha": config.CORTE_PRUEBA, "titulo": "Prueba",
         "detalle": f"se abre solo en la etapa {config.ETAPA_PRUEBA_SIMULACION.split('_')[0]}",
         "color": "rojo"},
        {"fecha": config.FECHA_FIN, "titulo": "Fin del periodo", "color": "gris"},
    ], titulo="Bloques temporales del protocolo"))
    r.agregar(pasos([(config.TITULOS_ETAPAS[e], PROPOSITO_ETAPAS.get(e, "")) for e in config.ETAPAS],
                    titulo="Etapas del proyecto", actual=config.ETAPAS.index(ETAPA) + 1))
    latex.registrar_varias([
        ("DatFechaInicio", config.FECHA_INICIO, "inicio del periodo de datos"),
        ("DatFechaFin", config.FECHA_FIN, "fin del periodo de datos"),
        ("DatCorteValidacion", config.CORTE_VALIDACION, "fecha de corte entre entrenamiento y validación"),
        ("DatCortePrueba", config.CORTE_PRUEBA, "fecha de corte entre validación y prueba"),
        ("DatBloquesCV", config.BLOQUES_VALIDACION_CRUZADA, "bloques temporales de la validación cruzada"),
        ("DatCostoFP", config.COSTO_FP, "costo de un falso positivo (alerta innecesaria)"),
        ("DatCostoFN", config.COSTO_FN, "costo de un falso negativo (no conforme sin alerta)"),
        ("DatSemilla", config.SEMILLA, "semilla del proyecto"),
    ])
    return r


def terminar(c):
    """Cierra la parte A. La etapa sigue abierta para la parte B (`exploracion`)."""
    cerrar_parte({
        "origen": config.ORIGEN_DATOS, "huella": c.huella,
        "filas": {nombre: int(len(t)) for nombre, t in c.tablas.items()},
        "clientes_con_muestras": c.clientes_con_muestras, "desde": c.desde, "hasta": c.hasta,
    })
