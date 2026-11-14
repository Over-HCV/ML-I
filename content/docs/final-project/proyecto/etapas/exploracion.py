"""Capa 8 · Etapa 02: limpieza, integración y exploración.

Responsabilidad única: armar las piezas de la etapa 02 (auditoría de la
limpieza, partición temporal y análisis exploratorio del periodo de
desarrollo), guardar las tablas limpias y el artefacto de calidad, y
registrar sus cifras para el informe. Cada función pública es una sección.
"""
import numpy as np
import pandas as pd

from proyecto.base import config, rutas
from proyecto.base.artefactos import guardar_json, guardar_tabla
from proyecto.datos import carga, esquema
from proyecto.etapas.comun import Resultado, abrir, cerrar, aviso_origen, pct
from proyecto.preparacion import integracion, limpieza, particion
from proyecto.reporte import exploracion as graficos
from proyecto.reporte import latex
from proyecto.reporte.html import nota, pasos, pestanas, tabla_html, tarjetas

ETAPA = "02_exploracion"


def iniciar():
    return abrir(ETAPA, "Calidad de los datos, limpieza y qué se ve antes de modelar")


def limpieza_auditada(c):
    """Lee las crudas, limpia con auditoría, integra y guarda las tablas limpias."""
    r = Resultado()
    c.crudas = carga.leer_crudas()
    c.limpias, auditoria = limpieza.limpiar_tablas(c.crudas)
    c.integradas = integracion.integrar_muestras(c.limpias)
    c.auditoria = auditoria.assign(tipo_problema=auditoria["problema"].str.split(" en «").str[0])
    correcciones = c.auditoria[~c.auditoria["tipo_problema"].str.startswith("faltante")]
    c.n_correcciones = int(len(correcciones))
    resumen = (
        correcciones.groupby("tipo_problema", sort=False)
        .agg(reglas=("problema", "size"), filas_afectadas=("filas_afectadas", "sum"), accion=("accion", "first"))
        .reset_index()
        .rename(columns={"tipo_problema": "Tipo de problema", "reglas": "Columnas o reglas",
                         "filas_afectadas": "Filas afectadas", "accion": "Acción"})
    )
    detalle = {"tabla": "Tabla", "problema": "Problema", "filas_afectadas": "Filas afectadas", "accion": "Acción"}
    es_etiqueta = correcciones["tipo_problema"].str.startswith("etiquetas")
    fichas = {"Resumen": tabla_html(resumen, "Correcciones de la limpieza por tipo de problema", barras="Filas afectadas")}
    if es_etiqueta.any():
        fichas["Etiquetas"] = tabla_html(correcciones[es_etiqueta][list(detalle)].rename(columns=detalle),
                                         "Columnas con grafías inconsistentes", maximo_filas=15)
    if (~es_etiqueta).any():
        fichas["Valores, fechas y recuentos"] = tabla_html(
            correcciones[~es_etiqueta][list(detalle)].rename(columns=detalle),
            "Duplicados, valores imposibles, fechas y recuentos", maximo_filas=15)
    r.agregar(pestanas(fichas, titulo="Auditoría de la limpieza"))

    c.filas_crudas = {n: int(len(t)) for n, t in c.crudas.items()}
    c.filas_limpias = {n: int(len(t)) for n, t in c.limpias.items()}
    eliminados = c.filas_crudas["muestras"] - c.filas_limpias["muestras"]
    r.agregar(tarjetas([
        {"titulo": "Muestras crudas", "valor": c.filas_crudas["muestras"], "color": "gris"},
        {"titulo": "Muestras limpias", "valor": c.filas_limpias["muestras"],
         "detalle": f"{eliminados} duplicados eliminados", "color": "azul"},
        {"titulo": "Reglas con hallazgos", "valor": c.n_correcciones, "detalle": "sin contar los faltantes",
         "color": "naranja"},
        {"titulo": "Columnas integradas", "valor": int(c.integradas.shape[1]),
         "detalle": "muestra + cliente + calendario", "color": "verde"},
    ]))
    mayor = resumen.loc[resumen["Filas afectadas"].idxmax()]
    r.agregar(nota(
        f"El problema que más filas toca es «{mayor['Tipo de problema']}» ({int(mayor['Filas afectadas'])} filas), "
        "que no es un error sino la forma habitual de reportar un recuento por debajo (o por encima) del límite de "
        "detección: la cota se conserva como valor y la fila se marca como censurada."
        if mayor["Tipo de problema"].startswith("recuento censurado") else
        f"El problema que más filas toca es «{mayor['Tipo de problema']}» ({int(mayor['Filas afectadas'])} filas).",
        "hallazgo",
    ))
    rutas_limpias = [guardar_tabla(t, rutas.DIR_LIMPIOS / f"{n}.parquet") for n, t in c.limpias.items()]
    guardar_tabla(c.integradas, rutas.DIR_LIMPIOS / "muestras_integradas.parquet")
    misma = len(c.integradas) == len(c.limpias["muestras"])
    r.agregar(nota(
        f"Se guardaron **{len(rutas_limpias)}** tablas limpias y la tabla integrada ({len(c.integradas)} muestras × "
        f"{c.integradas.shape[1]} columnas) en `datos/limpios/`. La integración no elimina ni duplica muestras: la "
        f"tabla integrada tiene {'las mismas filas que' if misma else 'distinto número de filas que'} la tabla limpia "
        "de muestras.",
    ))
    latex.registrar_varias([
        ("ExpMuestrasCrudas", c.filas_crudas["muestras"], "muestras crudas leídas por la limpieza"),
        ("ExpMuestrasLimpias", c.filas_limpias["muestras"], "muestras tras la limpieza"),
        ("ExpDuplicadosEliminados", eliminados, "muestras duplicadas eliminadas"),
        ("ExpReglasConHallazgos", c.n_correcciones, "reglas de limpieza con hallazgos (sin faltantes)"),
        ("ExpColumnasIntegradas", int(c.integradas.shape[1]), "columnas de la tabla integrada por muestra"),
        ("ExpTablasLimpias", len(rutas_limpias), "tablas limpias guardadas"),
        ("ExpProblemaMayor", mayor["Tipo de problema"], "tipo de problema de limpieza que más filas toca"),
        ("ExpProblemaMayorFilas", int(mayor["Filas afectadas"]), "filas que toca el problema de limpieza mayor"),
    ])
    latex.exportar_tabla(resumen, "auditoria_limpieza", "Correcciones de la limpieza por tipo de problema")
    r.valores.update(muestras_limpias=c.filas_limpias["muestras"], correcciones=c.n_correcciones)
    return r


def particion_temporal(c):
    """Filas por bloque temporal y cifras del periodo de desarrollo."""
    r = Resultado()
    bloques = particion.resumen_particion(c.integradas, "fecha_recepcion")
    c.vista_bloques = bloques.assign(
        Desde=pd.to_datetime(bloques["Desde"]).dt.strftime("%Y-%m-%d"),
        Hasta=pd.to_datetime(bloques["Hasta"]).dt.strftime("%Y-%m-%d"),
    )
    r.agregar(tabla_html(c.vista_bloques, "Muestras por bloque temporal", decimales=1, barras="Filas"))
    c.desarrollo = particion.solo_desarrollo(c.integradas, "fecha_recepcion")
    c.limpias_desarrollo = dict(c.limpias)
    c.limpias_desarrollo["muestras"] = particion.solo_desarrollo(c.limpias["muestras"], "fecha_recepcion")
    c.limpias_desarrollo["muestreos"] = particion.solo_desarrollo(c.limpias["muestreos"], "fecha_programada")
    c.limpias_desarrollo["controles"] = particion.solo_desarrollo(c.limpias["controles"], "fecha")
    c.aceptadas = c.desarrollo[c.desarrollo["rechazada"] == 0]
    c.tasa_no_conforme = float(c.aceptadas["no_conforme"].mean())
    tasa_rechazo = float(c.desarrollo["rechazada"].mean())
    clientes = int(c.desarrollo["id_cliente"].nunique())
    r.agregar(tarjetas([
        {"titulo": "Muestras en desarrollo", "valor": int(len(c.desarrollo)),
         "detalle": f"{c.desarrollo['fecha_recepcion'].min():%Y-%m-%d} a {c.desarrollo['fecha_recepcion'].max():%Y-%m-%d}"},
        {"titulo": "Rechazadas en recepción", "valor": 100 * tasa_rechazo, "sufijo": " %", "decimales": 1, "color": "gris"},
        {"titulo": "No conformes", "valor": 100 * c.tasa_no_conforme, "sufijo": " %", "decimales": 1,
         "detalle": "de las muestras aceptadas", "color": "rojo"},
        {"titulo": "Clientes con muestras", "valor": clientes, "color": "verde"},
    ], titulo="Periodo de desarrollo (lo único que se explora)"))
    r.agregar(aviso_origen("Los datos son **simulados**, por lo que los hallazgos de esta etapa describen la base "
                              "simulada y sirven para decidir cómo modelar, no para concluir sobre el laboratorio real."))
    filas = bloques.set_index("Bloque")["Filas"]
    latex.registrar_varias([
        ("ExpFilasEntrenamiento", int(filas["entrenamiento"]), "muestras del bloque de entrenamiento"),
        ("ExpFilasValidacion", int(filas["validacion"]), "muestras del bloque de validación"),
        ("ExpFilasPrueba", int(filas["prueba"]), "muestras del bloque de prueba (solo se cuentan)"),
        ("ExpMuestrasDesarrollo", int(len(c.desarrollo)), "muestras del periodo de desarrollo"),
        ("ExpTasaRechazoPct", 100 * tasa_rechazo, "porcentaje de muestras rechazadas en recepción (desarrollo)", 1),
        ("ExpTasaNoConformePct", 100 * c.tasa_no_conforme, "porcentaje de no conformes entre aceptadas (desarrollo)", 1),
        ("ExpClientesDesarrollo", clientes, "clientes con muestras en el desarrollo"),
    ])
    latex.exportar_tabla(c.vista_bloques, "particion", "Muestras por bloque temporal", decimales={"Porcentaje": 1})
    r.valores.update(desarrollo=int(len(c.desarrollo)), tasa_no_conforme=c.tasa_no_conforme)
    return r


def faltantes(c):
    r = Resultado()
    tabla = limpieza.faltantes_por_columna(c.limpias_desarrollo)
    r.figura(graficos.figura_faltantes(tabla), "02_faltantes")
    con_faltantes = tabla[tabla["faltantes"] > 0]
    c.peor_faltante = tabla.loc[tabla["pct_no_estructural"].idxmax()]
    solo_estructural = int(((con_faltantes["estructurales"] > 0) & (con_faltantes["no_estructurales"] == 0)).sum())
    peor = c.peor_faltante
    r.agregar(nota(
        f"De las {len(tabla)} columnas del contrato, **{len(con_faltantes)}** tienen algún faltante en el periodo de "
        f"desarrollo y en {solo_estructural} de ellas todos los faltantes son estructurales. El mayor porcentaje de "
        f"faltantes no estructurales es el de «{peor['columna']}» (**{peor['pct_no_estructural']:.1f} %** de las "
        f"filas), una columna que se conoce en el momento «{peor['momento']}».",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ExpColumnasConFaltantes", len(con_faltantes), "columnas con algún faltante en el desarrollo"),
        ("ExpColumnasSoloEstructural", solo_estructural, "columnas cuyos faltantes son todos estructurales"),
        ("ExpPeorFaltanteColumna", peor["columna"], "columna con más faltantes no estructurales"),
        ("ExpPeorFaltantePct", float(peor["pct_no_estructural"]), "porcentaje de faltantes no estructurales de esa columna", 1),
    ])
    return r


def volumen(c):
    """Volumen semanal por línea y estacionalidad mensual."""
    r = Resultado()
    d = c.desarrollo
    r.figura(graficos.figura_volumen_semanal(d, c.limpias["calendario"]), "02_volumen_semanal")
    semanal = d.groupby("semana").size()
    completas = semanal.iloc[1:-1] if len(semanal) > 2 else semanal
    por_sector = d["sector"].value_counts(normalize=True)
    cv = completas.std() / completas.mean()
    r.agregar(nota(
        f"En las {len(semanal)} semanas del periodo de desarrollo llegan en promedio **{completas.mean():.0f} muestras "
        f"por semana**, con una semana mínima de {int(completas.min())} y una máxima de {int(completas.max())} "
        f"(coeficiente de variación de {cv:.2f}). La línea con más volumen es «{por_sector.index[0]}» "
        f"({pct(por_sector.iloc[0])} de las muestras) y la de menos es «{por_sector.index[-1]}» ({pct(por_sector.iloc[-1])}).",
        "hallazgo",
    ))
    r.figura(graficos.figura_estacionalidad(d), "02_estacionalidad")
    mensual = d.groupby(["anio", "mes"]).size()
    indice = mensual.groupby(level="mes").mean() / mensual.mean()
    alto, bajo = int(indice.idxmax()), int(indice.idxmin())
    r.agregar(nota(
        f"Sumando todas las líneas, el mes más cargado es **{graficos.MESES[alto - 1]}** (índice estacional "
        f"{indice.max():.2f}, donde 1 es un mes típico) y el más liviano es **{graficos.MESES[bajo - 1]}** "
        f"({indice.min():.2f}). El índice se calcula con {d['anio'].nunique()} años de datos, de forma que cada mes del "
        "año se observa pocas veces y el patrón debe leerse como indicio, no como una estacionalidad confirmada.",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ExpSemanas", len(semanal), "semanas del periodo de desarrollo"),
        ("ExpMuestrasSemana", float(completas.mean()), "muestras por semana en promedio (semanas completas)", 0),
        ("ExpMuestrasSemanaMin", int(completas.min()), "semana completa con menos muestras"),
        ("ExpMuestrasSemanaMax", int(completas.max()), "semana completa con más muestras"),
        ("ExpCVSemanal", float(cv), "coeficiente de variación del volumen semanal", 2),
        ("ExpSectorMayor", por_sector.index[0], "línea de negocio con más volumen"),
        ("ExpSectorMayorPct", 100 * float(por_sector.iloc[0]), "porcentaje de muestras de esa línea", 1),
        ("ExpSectorMenor", por_sector.index[-1], "línea de negocio con menos volumen"),
        ("ExpSectorMenorPct", 100 * float(por_sector.iloc[-1]), "porcentaje de muestras de esa línea", 1),
        ("ExpMesAlto", graficos.MESES[alto - 1], "mes más cargado"),
        ("ExpIndiceMesAlto", float(indice.max()), "índice estacional del mes más cargado", 2),
        ("ExpMesBajo", graficos.MESES[bajo - 1], "mes más liviano"),
        ("ExpIndiceMesBajo", float(indice.min()), "índice estacional del mes más liviano", 2),
    ])
    return r


def tasa_no_conforme(c):
    """Tasa de no conformes por sector, matriz y ensayo."""
    r = Resultado()
    a = c.aceptadas
    columnas = [x for x in ("sector", "matriz", "ensayo") if x in a.columns]
    r.figura(graficos.figura_tasa_por_categoria(a, columnas=tuple(columnas)), "02_tasa_por_categoria")
    frases = []
    for columna in columnas:
        por_nivel = a.groupby(columna)["no_conforme"].agg(["mean", "count"])
        por_nivel = por_nivel[por_nivel["count"] >= graficos.MINIMO_CELDA]
        if len(por_nivel) > 1:
            frases.append(f"por «{columna}» la tasa va de {pct(por_nivel['mean'].min())} ({por_nivel['mean'].idxmin()}) a "
                          f"{pct(por_nivel['mean'].max())} ({por_nivel['mean'].idxmax()})")
    t = c.tasa_no_conforme
    r.agregar(nota(
        f"La tasa global de no conformes entre las muestras aceptadas es **{pct(t)}** (una muestra no conforme por cada "
        f"{(1 - t) / t:.0f} conformes), " + "; ".join(frases) + ".",
        "hallazgo",
    ))
    latex.registrar("ExpConformesPorNoConforme", (1 - t) / t, "conformes por cada no conforme (desarrollo)", 0)
    return r


def distribuciones(c):
    """Variables numéricas de llegada según el resultado, y el transporte."""
    r = Resultado()
    a = c.aceptadas
    numericas = [x for x in esquema.columnas("muestras", tipo="numero", hasta="llegada")
                 if x in a.columns and a[x].notna().sum() > graficos.MINIMO_CELDA]
    r.figura(graficos.figura_distribuciones_por_resultado(a, columnas=numericas), "02_distribuciones_por_resultado")
    medias = a.groupby("no_conforme")[numericas].mean()
    diferencia = ((medias.loc[1] - medias.loc[0]) / a[numericas].std()).sort_values(key=np.abs)
    r.agregar(nota(
        f"La variable que más separa a las dos clases es «{diferencia.index[-1]}»: su media entre las no conformes queda "
        f"{abs(diferencia.iloc[-1]):.2f} desviaciones estándar por {'encima' if diferencia.iloc[-1] > 0 else 'debajo'} "
        f"de la de las conformes. La que menos separa es «{diferencia.index[0]}» ({abs(diferencia.iloc[0]):.2f} "
        "desviaciones). "
        + ("Al quedar todas las diferencias por debajo de media desviación estándar, las distribuciones de las dos clases "
           "se traslapan ampliamente y ninguna variable por sí sola distingue el resultado."
           if diferencia.abs().max() < 0.5 else
           "Una diferencia de ese tamaño ya es visible en los histogramas, aunque las distribuciones se traslapan."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ExpVariableMasSepara", diferencia.index[-1], "variable de llegada que más separa no conformes de conformes"),
        ("ExpDiferenciaEstandarMax", abs(float(diferencia.iloc[-1])), "diferencia de medias en desviaciones estándar", 2),
    ])
    if {"temperatura_transporte_c", "horas_transporte"} <= set(a.columns):
        r.figura(graficos.figura_transporte(a), "02_transporte")
        frases = []
        for columna, nombre, macro in (("temperatura_transporte_c", "temperatura de transporte", "Temperatura"),
                                       ("horas_transporte", "duración del transporte", "Horas")):
            q1, q3 = a[columna].quantile([0.25, 0.75])
            baja = float(a.loc[a[columna] <= q1, "no_conforme"].mean())
            alta = float(a.loc[a[columna] >= q3, "no_conforme"].mean())
            frases.append(f"en el cuarto de muestras con mayor {nombre} (desde {q3:.1f}) la tasa de no conformes es "
                          f"{pct(alta)}, frente a {pct(baja)} en el cuarto con menor {nombre} (hasta {q1:.1f})")
            latex.registrar_varias([
                (f"Exp{macro}CuartoAltoPct", 100 * alta, f"no conformes (%) en el cuarto con mayor {nombre}", 1),
                (f"Exp{macro}CuartoBajoPct", 100 * baja, f"no conformes (%) en el cuarto con menor {nombre}", 1),
            ])
        r.agregar(nota("Comparando los extremos de cada variable: " + "; y ".join(frases) + ".", "hallazgo"))
    return r


def entrega(c):
    r = Resultado()
    a = c.aceptadas
    r.figura(graficos.figura_dias_entrega(a), "02_dias_entrega")
    con_entrega = a.dropna(subset=["dias_entrega"])
    c.tasa_tardia = float(con_entrega["entrega_tardia"].mean())
    por_ensayo = con_entrega.groupby("ensayo")["entrega_tardia"].agg(["mean", "count"])
    por_ensayo = por_ensayo[por_ensayo["count"] >= graficos.MINIMO_CELDA]
    mediana, p90 = con_entrega["dias_entrega"].median(), con_entrega["dias_entrega"].quantile(0.9)
    r.agregar(nota(
        f"La mediana de entrega es de **{mediana:.0f} días** calendario (el 90 % de los informes sale en {p90:.0f} días "
        f"o menos) y el **{pct(c.tasa_tardia)}** de los informes sale tarde. El ensayo con más entregas tardías es "
        f"«{por_ensayo['mean'].idxmax()}» ({pct(por_ensayo['mean'].max())}) y el que menos "
        f"«{por_ensayo['mean'].idxmin()}» ({pct(por_ensayo['mean'].min())}).",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ExpEntregaMediana", float(mediana), "mediana de días de entrega", 0),
        ("ExpEntregaPNoventa", float(p90), "percentil 90 de días de entrega", 0),
        ("ExpEntregaTardiaPct", 100 * c.tasa_tardia, "porcentaje de informes entregados tarde", 1),
        ("ExpEnsayoMasTardio", por_ensayo["mean"].idxmax(), "ensayo con más entregas tardías"),
        ("ExpEnsayoMasTardioPct", 100 * float(por_ensayo["mean"].max()), "entregas tardías (%) de ese ensayo", 1),
        ("ExpEnsayoMenosTardio", por_ensayo["mean"].idxmin(), "ensayo con menos entregas tardías"),
        ("ExpEnsayoMenosTardioPct", 100 * float(por_ensayo["mean"].min()), "entregas tardías (%) de ese ensayo", 1),
    ])
    return r


def estabilidad(c):
    r = Resultado()
    a = c.aceptadas
    r.figura(graficos.figura_tasa_mensual(a), "02_tasa_mensual")
    c.tasa_mes = a.groupby(a["fecha_recepcion"].dt.to_period("M"))["no_conforme"].mean()
    por_anio = a.groupby("anio")["no_conforme"].mean()
    primero, ultimo = int(por_anio.index.min()), int(por_anio.index.max())
    cambio = float(por_anio.loc[ultimo] - por_anio.loc[primero])
    m = c.tasa_mes
    r.agregar(nota(
        f"La tasa mensual oscila entre {pct(m.min())} ({m.idxmin()}) y {pct(m.max())} ({m.idxmax()}). Por años, pasa de "
        f"{pct(por_anio.loc[primero])} en {primero} a {pct(por_anio.loc[ultimo])} en {ultimo} "
        f"({'sube' if cambio > 0 else 'baja'} {abs(100 * cambio):.1f} puntos porcentuales). Parte de esa oscilación es azar "
        "de muestreo (la banda de la figura lo indica) y parte puede ser deriva real, y con esta figura no se pueden "
        "separar; por precaución, la validación de los modelos respeta el orden temporal, ya que mezclar meses al azar "
        "podría hacer parecer el problema más fácil de lo que es.",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ExpTasaMensualMinPct", 100 * float(m.min()), "tasa mensual mínima de no conformes (%)", 1),
        ("ExpTasaMensualMaxPct", 100 * float(m.max()), "tasa mensual máxima de no conformes (%)", 1),
        ("ExpPrimerAnio", primero, "primer año del desarrollo"),
        ("ExpUltimoAnio", ultimo, "último año del desarrollo"),
        ("ExpTasaPrimerAnioPct", 100 * float(por_anio.loc[primero]), "no conformes (%) en el primer año", 1),
        ("ExpTasaUltimoAnioPct", 100 * float(por_anio.loc[ultimo]), "no conformes (%) en el último año", 1),
    ])
    return r


def correlaciones(c):
    r = Resultado()
    a = c.aceptadas
    columnas = [
        x for x in dict.fromkeys(esquema.columnas("muestras", tipo="numero")
                                 + ["distancia_km", "antiguedad_dias", "recuento_ufc", "dias_entrega", "no_conforme",
                                    "entrega_tardia"])
        if x in a.columns and a[x].notna().sum() > graficos.MINIMO_CELDA and a[x].nunique() > 1
    ]
    r.figura(graficos.figura_correlaciones(a, columnas=columnas), "02_correlaciones")
    corr = a[columnas].astype(float).corr(method="spearman")
    resultados = [x for x in ("recuento_ufc", "dias_entrega", "no_conforme", "entrega_tardia") if x in columnas]
    entradas = [x for x in columnas if x not in resultados]
    pares = corr.loc[entradas, entradas].where(np.triu(np.ones((len(entradas), len(entradas)), dtype=bool), k=1)).stack()
    par = pares.abs().idxmax()
    con_objetivo = corr.loc[entradas, "no_conforme"]
    mayor = con_objetivo.abs().idxmax()
    r.agregar(nota(
        f"Entre las variables de entrada, el par más correlacionado es «{par[0]}» y «{par[1]}» (ρ = {pares.loc[par]:.2f}). "
        f"Con el resultado no conforme, la correlación más alta en valor absoluto es la de «{mayor}» "
        f"(ρ = {con_objetivo.loc[mayor]:.2f}), "
        + ("una asociación débil: la señal, si la hay, no está en una sola variable numérica tomada por separado."
           if con_objetivo.abs().max() < 0.3 else "una asociación apreciable para una sola variable."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ExpParCorrelacionado", f"{par[0]} y {par[1]}", "par de variables de entrada más correlacionado"),
        ("ExpRhoPar", float(pares.loc[par]), "correlación de Spearman de ese par", 2),
        ("ExpVariableCorrObjetivo", mayor, "variable más correlacionada con no conforme"),
        ("ExpRhoObjetivo", float(con_objetivo.loc[mayor]), "su correlación de Spearman con no conforme", 2),
    ])
    return r


def concentracion(c):
    r = Resultado()
    r.figura(graficos.figura_volumen_clientes(c.desarrollo), "02_volumen_clientes")
    por_cliente = c.desarrollo["id_cliente"].value_counts()
    acumulado = por_cliente.cumsum() / por_cliente.sum()
    c.clientes_mitad = int((acumulado < 0.5).sum()) + 1
    decimo = max(int(round(0.1 * len(por_cliente))), 1)
    r.agregar(nota(
        f"De los {len(por_cliente)} clientes con muestras, los **{c.clientes_mitad}** más grandes "
        f"({pct(c.clientes_mitad / len(por_cliente))}) aportan la mitad del volumen y el 10 % más grande aporta el "
        f"{pct(float(acumulado.iloc[decimo - 1]))}. En el otro extremo, la mitad de los clientes tiene "
        f"{int(por_cliente.median())} muestras o menos en todo el periodo.",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("ExpClientesMitad", c.clientes_mitad, "clientes más grandes que aportan la mitad del volumen"),
        ("ExpClientesMitadPct", 100 * c.clientes_mitad / len(por_cliente), "esos clientes como porcentaje del total", 1),
        ("ExpDecimoClientesPct", 100 * float(acumulado.iloc[decimo - 1]), "volumen (%) del 10 % de clientes más grande", 1),
        ("ExpMedianaMuestrasCliente", int(por_cliente.median()), "mediana de muestras por cliente"),
    ])
    return r


def implicaciones(c):
    """Lista de implicaciones para el modelado y artefacto de calidad."""
    r = Resultado()
    d, peor, m = c.desarrollo, c.peor_faltante, c.tasa_mes
    categoricas = [x for x in esquema.columnas("muestras", tipo="categoria") + ["id_cliente", "codigo_producto"]
                   if x in d.columns]
    cardinalidad = d[categoricas].nunique().sort_values(ascending=False)
    r.agregar(pasos([
        ("Clases desbalanceadas",
         f"Solo el {pct(c.tasa_no_conforme)} de las muestras aceptadas es no conforme: la exactitud no sirve como "
         "métrica, se usan ROC-AUC, precisión, recall y un umbral elegido por costo."),
        ("Categorías con muchos niveles",
         f"«{cardinalidad.index[0]}» tiene {int(cardinalidad.iloc[0])} niveles y «{cardinalidad.index[1]}» "
         f"{int(cardinalidad.iloc[1])}: no se codifican tal cual, se resumen con historiales o se agrupan los niveles raros."),
        ("Faltantes dentro de la tubería",
         f"Hay faltantes no estructurales de hasta {peor['pct_no_estructural']:.1f} % («{peor['columna']}»): la "
         "imputación se ajusta solo con entrenamiento, dentro de la tubería, y se añade un indicador de faltante."),
        ("Orden temporal",
         f"La tasa mensual de no conformes va de {pct(m.min())} a {pct(m.max())}: la validación cruzada usa bloques de "
         "ventana creciente, nunca mezcla al azar."),
        ("Clientes de cola pesada",
         f"{c.clientes_mitad} clientes aportan la mitad del volumen: las variables de historia por cliente se suavizan "
         "hacia la tasa global para que los clientes con pocas muestras no produzcan tasas extremas."),
    ], titulo="Implicaciones para las etapas de modelado"))
    latex.registrar_varias([
        ("ExpCategoriaMasNiveles", cardinalidad.index[0], "categórica con más niveles"),
        ("ExpNivelesCategoria", int(cardinalidad.iloc[0]), "niveles de esa categórica"),
    ])
    guardar_json({
        "origen": config.ORIGEN_DATOS,
        "huella": carga.huella(c.crudas),
        "filas_crudas": c.filas_crudas,
        "filas_limpias": c.filas_limpias,
        "columnas_integradas": int(c.integradas.shape[1]),
        "auditoria": c.auditoria.drop(columns="tipo_problema").to_dict("records"),
        "particion": c.vista_bloques.to_dict("records"),
        "desarrollo": {
            "muestras": int(len(d)),
            "tasa_no_conforme": c.tasa_no_conforme,
            "tasa_rechazo": float(d["rechazada"].mean()),
            "tasa_entrega_tardia": c.tasa_tardia,
        },
    }, rutas.DIR_ARTEFACTOS / "02_calidad.json")
    return r


def terminar(c):
    cerrar({
        "muestras_limpias": c.filas_limpias["muestras"],
        "muestras_desarrollo": int(len(c.desarrollo)),
        "tasa_no_conforme_desarrollo": round(c.tasa_no_conforme, 4),
        "correcciones_auditadas": c.n_correcciones,
    })
