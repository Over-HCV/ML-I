"""Capa 8 · Etapa 08: datos reales.

Responsabilidad única: armar las piezas de la etapa 08 sobre la exportación
real del laboratorio, solo con fechas de desarrollo (anteriores al corte de
prueba): la base y su granularidad, las llegadas de muestras frente a las de
lotes con la prueba de Poisson, el tamaño de los lotes por tipo y las
campañas, los segmentos de clientes, un clasificador rápido del riesgo de no
conformidad al ingreso y la comparación con el mecanismo de la simulación.
Guarda el artefacto de la etapa y registra sus cifras (prefijo `Real`).

Si la exportación no está (por ejemplo en Colab, porque el archivo es
confidencial), cada sección devuelve una nota que lo explica y la etapa se
cierra sin registrar cifras.
"""
import time

import numpy as np
import pandas as pd

from proyecto.base import config, rutas
from proyecto.base.artefactos import guardar_json
from proyecto.datos import rodam
from proyecto.etapas.comun import Resultado, abrir, cerrar, pct
from proyecto.evaluacion import metricas, poisson
from proyecto.modelos import agrupamiento
from proyecto.modelos.tuberias import construir_tuberia
from proyecto.reporte import graficos_agrupamiento, graficos_reales, latex
from proyecto.reporte import segmentos as lectura_segmentos
from proyecto.reporte.html import nota, pasos, tabla_html, tarjetas
from proyecto.variables import cliente_real
from proyecto.variables.comunes import es_desarrollo

ETAPA = "08_datos_reales"
# Reglas fijadas antes de mirar resultados.
NIVEL = 0.05                                  # valor p de la prueba de dispersión
MAXIMO_LOTES_SEMANA = 6                       # última clase de las figuras de lotes por semana
MAXIMO_MUESTRAS_SEMANA = 30                   # última clase de las figuras de muestras por semana
CORTES_TAMANO = [1, 2, 3, 5, 10, 20, 50]      # clases del tamaño de lote
VALORES_K = list(range(2, 8))                 # pocos clientes: k de 2 a 7
TOLERANCIA_SILUETA = 0.02
SEMILLAS_ESTABILIDAD = [config.SEMILLA + i for i in range(1, 6)]
CUANTILES_EPS = np.linspace(0.60, 0.98, 8)
RUIDO_MAXIMO, RUIDO_RESPALDO = 0.15, 0.10
ASIMETRIA_LOG = 1.0
UMBRAL_NOMBRE, VARIABLES_NOMBRE = 0.5, 2
MODELOS_RIESGO = {"logistica": "Regresión logística", "arbol": "Árbol de decisión"}
# Lo que hace la simulación (`proyecto/datos/simulacion.py`, función `_envios`), escrito a mano porque fuera de
# `proyecto/datos/` no se importa la simulación: tamaño = 1 + Poisson(m − 1) con m, la media del cliente, de
# 1.8 a 3.0 según el segmento (antes del factor por tamaño de cliente y del ruido lognormal).
LOTE_MEDIO_SIMULACION = (1.8, 3.0)
TIPOS = {"producto": "producto", "planta": "muestreo en planta"}
ORDINALES = ["Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis", "Siete"]
SIN_DATOS = ("La exportación real no está en `datos/reales/`. El archivo es confidencial así que no viaja con el "
             "proyecto (ni a Colab ni al repositorio) y esta sección se queda sin cifras. Con el archivo en esa "
             "carpeta la etapa se vuelve a ejecutar completa.")


def _sin_datos():
    return Resultado().agregar(nota(SIN_DATOS, "limite", titulo="Sin la exportación real"))


def iniciar():
    c = abrir(ETAPA, "Las llegadas de muestras del laboratorio real: Lotes de Poisson y lotes de tamaño variable")
    c.inicio = time.perf_counter()
    c.hay = rodam.hay_exportacion()
    c.resumen = {"hay_exportacion": c.hay}
    if not c.hay:
        return c
    filas = rodam.leer_exportacion()
    c.filas_total = len(filas)
    filas = filas[es_desarrollo(filas["fecha_ingreso"])]   # lo de 2026 en adelante no se mira
    c.filas = len(filas)
    muestras = cliente_real.del_desarrollo(rodam.asignar_lotes(rodam.tabla_por_muestra(filas)))
    c.muestras = muestras
    c.lotes = rodam.lotes(muestras)
    c.campanas = rodam.campanas(muestras)
    return c


# ----------------------------------------------------------------------
# 1. La base
# ----------------------------------------------------------------------
def la_base(c):
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    m, lotes = c.muestras, c.lotes
    desde, hasta = m["fecha_ingreso"].min(), m["fecha_ingreso"].max()
    planta = float((m["tipo_lote"] == "planta").mean())
    tasa = float(m["no_conforme"].astype(float).mean())
    r.agregar(tarjetas([
        {"titulo": "Filas de la exportación", "valor": c.filas, "decimales": 0,
         "detalle": "una por muestra, especificación y lectura"},
        {"titulo": "Muestras", "valor": len(m), "decimales": 0, "color": "morado",
         "detalle": f"de {m['cliente'].nunique()} clientes"},
        {"titulo": "Lotes", "valor": len(lotes), "decimales": 0, "color": "verde",
         "detalle": "cliente, día y tipo"},
        {"titulo": "No conformidad", "valor": 100 * tasa, "decimales": 1, "sufijo": " %", "color": "rojo",
         "detalle": "muestras con dictamen no cumple"},
    ]))
    filas = []
    for particion, rotulo in (("entrenamiento", "Entrenamiento"), ("validacion", "Validación")):
        parte = m[m["particion"] == particion]
        filas.append({"Partición": rotulo, "Desde": parte["dia"].min(), "Hasta": parte["dia"].max(),
                      "Muestras": len(parte), "Lotes": parte["id_lote"].nunique(),
                      "Clientes": parte["cliente"].nunique(),
                      "Muestreo en planta": float((parte["tipo_lote"] == "planta").mean()),
                      "No conformidad": float(parte["no_conforme"].astype(float).mean())})
    t = pd.DataFrame(filas)
    r.agregar(tabla_html(t, "Muestras de desarrollo por partición", decimales=3))
    origen = m["origen_muestra"].value_counts(normalize=True)
    r.agregar(nota(
        f"La exportación trae {c.filas} filas del desarrollo (de {desde:%Y-%m-%d} a {hasta:%Y-%m-%d}) que se vuelven "
        f"{len(m)} muestras al juntar sus especificaciones y lecturas. El {pct(planta)} de las muestras viene de un "
        f"muestreo en planta y el resto llega como producto (el origen más común después de planta es "
        f"\"{origen.index[1].lower()}\" con {pct(origen.iloc[1])}). Las fechas desde el corte de prueba "
        f"({config.CORTE_PRUEBA}) no entran en ninguna cuenta de esta etapa.", "info"))
    c.base = {"muestras": len(m), "lotes": len(lotes), "clientes": int(m["cliente"].nunique()), "planta": planta,
              "no_conformidad": tasa}
    latex.registrar_varias([
        ("RealFilas", c.filas, "filas de desarrollo de la exportación real"),
        ("RealMuestras", len(m), "muestras de desarrollo de la exportación real"),
        ("RealLotes", len(lotes), "lotes de desarrollo (cliente, día y tipo)"),
        ("RealClientes", int(m["cliente"].nunique()), "clientes con muestras en el desarrollo"),
        ("RealDesde", f"{desde:%Y-%m-%d}", "primer ingreso del desarrollo"),
        ("RealHasta", f"{hasta:%Y-%m-%d}", "último ingreso del desarrollo"),
        ("RealPlantaPct", 100 * planta, "muestras de muestreo en planta (%)", 1),
        ("RealNoConformePct", 100 * tasa, "no conformidad de las muestras de desarrollo (%)", 1),
    ])
    latex.exportar_tabla(t, "real_particiones", "Muestras de desarrollo por partición", decimales=3)
    return r


# ----------------------------------------------------------------------
# 2. Llegadas: muestras frente a lotes
# ----------------------------------------------------------------------
def _resumen_dispersion(ancha):
    d = poisson.dispersion_por_fila(ancha).dropna(subset=["indice"])
    pruebas = [poisson.prueba_dispersion(fila.dropna()) for _, fila in ancha.loc[d.index].iterrows()]
    d["p_sobre"] = [p["p_sobredispersion"] for p in pruebas]
    d["p_sub"] = [p["p_subdispersion"] for p in pruebas]
    d["binomial_negativa"] = [poisson.ajustar_conteos(fila.dropna())["mejor"] == "binomial negativa"
                              for _, fila in ancha.loc[d.index].iterrows()]
    return d


def llegadas(c):
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    por_tipo = cliente_real.dispersion_por_tipo(c.muestras, c.lotes)
    compuesto = cliente_real.indice_compuesto_por_cliente(c.lotes)
    filas, c.llegadas = [], {}
    for tipo, (por_muestras, por_lotes) in por_tipo.items():
        dm, dl = _resumen_dispersion(por_muestras), _resumen_dispersion(por_lotes)
        predicho = compuesto[compuesto["tipo_lote"] == tipo].set_index("cliente")["indice"].reindex(dm.index)
        c.llegadas[tipo] = {"muestras": dm, "lotes": dl}
        filas.append({"Tipo": TIPOS[tipo], "Clientes": len(dl),
                      "Índice muestras (mediana)": dm["indice"].median(),
                      "Índice muestras (máximo)": dm["indice"].max(),
                      "Índice lotes (mediana)": dl["indice"].median(),
                      "Muestras sobredispersas": float((dm["p_sobre"] < NIVEL).mean()),
                      "Lotes sobredispersos": float((dl["p_sobre"] < NIVEL).mean()),
                      "Lotes subdispersos": float((dl["p_sub"] < NIVEL).mean()),
                      "Índice compuesto predicho (mediana)": float(predicho.median())})
        r.figura(graficos_reales.figura_llegadas(
            poisson.frecuencias_mezcla_poisson(por_lotes, MAXIMO_LOTES_SEMANA),
            poisson.frecuencias_mezcla_poisson(por_muestras, MAXIMO_MUESTRAS_SEMANA),
            dl["indice"].median(), dm["indice"].median(), TIPOS[tipo]), f"08_llegadas_{tipo}")
    t = pd.DataFrame(filas)
    c.tabla_llegadas = t
    r.agregar(tabla_html(t, f"Índice de dispersión semanal por cliente (clientes con al menos "
                            f"{cliente_real.MINIMO_LOTES} lotes del tipo) y proporción de clientes con valor p menor "
                            f"que {NIVEL}", decimales=2))
    # La figura que resume todo: cada cliente con sus muestras y sus lotes de todos los tipos.
    desde, hasta = cliente_real.semanas_de_desarrollo(c.muestras)
    cuantos = c.lotes.groupby("cliente").size()
    clientes = cuantos[cuantos >= cliente_real.MINIMO_LOTES].index
    todas_m = poisson.dispersion_por_fila(cliente_real.conteos_semanales(c.muestras, desde, hasta, clientes))
    todas_l = poisson.dispersion_por_fila(cliente_real.conteos_semanales(c.lotes, desde, hasta, clientes))
    r.figura(graficos_reales.figura_dispersion_clientes(todas_m, todas_l), "08_dispersion_clientes")
    p, q = t.set_index("Tipo").loc[TIPOS["producto"]], t.set_index("Tipo").loc[TIPOS["planta"]]
    r.agregar(nota(
        f"En producto el índice de las muestras por semana tiene mediana {p['Índice muestras (mediana)']:.2f} y el de "
        f"los lotes {p['Índice lotes (mediana)']:.2f}. En planta la diferencia es mucho mayor: "
        f"{q['Índice muestras (mediana)']:.2f} para las muestras y {q['Índice lotes (mediana)']:.2f} para los lotes. "
        f"La prueba de dispersión marca sobredispersión en las muestras del {pct(p['Muestras sobredispersas'], 0)} de los "
        f"clientes de producto y del {pct(q['Muestras sobredispersas'], 0)} de los de planta, mientras que en los lotes "
        f"la marca solo en el {pct(p['Lotes sobredispersos'], 0)} y el {pct(q['Lotes sobredispersos'], 0)}. O sea que contar "
        "lotes deja a cada cliente cerca de una Poisson y contar muestras no. Lo que no esperábamos es que en el "
        f"{pct(p['Lotes subdispersos'], 0)} de los clientes de producto los lotes salen más regulares que una Poisson "
        "(índice por debajo de 1 con valor p menor que el nivel), como si despacharan según un plan de producción.",
        "hallazgo"))
    r.agregar(nota(
        "Si los lotes llegan como Poisson y cada uno trae X muestras, el índice de las muestras debería ser "
        f"E[X²]/E[X]. Con los tamaños de lote de cada cliente esa cuenta da una mediana de "
        f"{p['Índice compuesto predicho (mediana)']:.2f} en producto y {q['Índice compuesto predicho (mediana)']:.2f} "
        "en planta, del mismo orden que lo observado. La sobredispersión de las muestras la explica el tamaño "
        "variable de los lotes y no hace falta otro mecanismo para entenderla.", "info"))
    c.dispersion_todos = {"muestras": float(todas_m["indice"].median()), "lotes": float(todas_l["indice"].median()),
                          "clientes": int(len(clientes)), "maximo_muestras": float(todas_m["indice"].max())}
    latex.registrar_varias([
        ("RealNivel", NIVEL, "nivel de la prueba de dispersión", 2),
        ("RealMinimoLotes", cliente_real.MINIMO_LOTES, "lotes mínimos de un cliente para medir su dispersión"),
        ("RealClientesDispersion", int(len(clientes)), "clientes con lotes suficientes (todos los tipos)"),
        ("RealDispMuestrasTodos", c.dispersion_todos["muestras"], "mediana del índice de muestras por semana", 2),
        ("RealDispLotesTodos", c.dispersion_todos["lotes"], "mediana del índice de lotes por semana", 2),
        ("RealDispMuestrasMax", c.dispersion_todos["maximo_muestras"], "máximo índice de muestras por semana", 0),
        ("RealClientesProducto", int(p["Clientes"]), "clientes con lotes de producto suficientes"),
        ("RealClientesPlanta", int(q["Clientes"]), "clientes con lotes de planta suficientes"),
        ("RealDispMuestrasProducto", float(p["Índice muestras (mediana)"]), "índice de muestras de producto", 2),
        ("RealDispLotesProducto", float(p["Índice lotes (mediana)"]), "índice de lotes de producto", 2),
        ("RealDispMuestrasPlanta", float(q["Índice muestras (mediana)"]), "índice de muestras de planta", 2),
        ("RealDispLotesPlanta", float(q["Índice lotes (mediana)"]), "índice de lotes de planta", 2),
        ("RealCompuestoProducto", float(p["Índice compuesto predicho (mediana)"]), "índice compuesto en producto", 2),
        ("RealCompuestoPlanta", float(q["Índice compuesto predicho (mediana)"]), "índice compuesto en planta", 2),
        ("RealSobreMuestrasProductoPct", 100 * float(p["Muestras sobredispersas"]), "clientes con muestras "
         "sobredispersas en producto (%)", 0),
        ("RealSobreLotesProductoPct", 100 * float(p["Lotes sobredispersos"]), "clientes con lotes sobredispersos "
         "en producto (%)", 0),
        ("RealSobreMuestrasPlantaPct", 100 * float(q["Muestras sobredispersas"]), "clientes con muestras "
         "sobredispersas en planta (%)", 0),
        ("RealSobreLotesPlantaPct", 100 * float(q["Lotes sobredispersos"]), "clientes con lotes sobredispersos en "
         "planta (%)", 0),
    ])
    latex.exportar_tabla(t, "real_dispersion", "Índice de dispersión semanal por cliente y tipo de lote", decimales=2)
    return r


# ----------------------------------------------------------------------
# 3. Tamaño de los lotes y campañas
# ----------------------------------------------------------------------
def lotes(c):
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    por_tipo, filas, frecuencias, c.ajustes = {}, [], [], {}
    for tipo in rodam.TIPOS_LOTE:
        tamanos = c.lotes.loc[c.lotes["tipo_lote"] == tipo, "n_muestras"].to_numpy()
        ajustes, mejor = poisson.ajustar_tamano_lote(tamanos)
        por_tipo[TIPOS[tipo]] = (tamanos, ajustes, poisson.probabilidad_tamano)
        nb = ajustes.set_index("modelo").loc["1 + binomial negativa"]
        geo = ajustes.set_index("modelo").loc["1 + geométrica"]
        c.ajustes[tipo] = {"lotes": len(tamanos), "media": float(tamanos.mean()), "mediana": float(np.median(tamanos)),
                           "maximo": int(tamanos.max()), "mejor": mejor, "r": float(nb["r"]),
                           "delta_geometrica": float(geo["delta_aic"]),
                           "ajustes": ajustes.to_dict("records")}
        for _, a in ajustes.iterrows():
            filas.append({"Tipo": TIPOS[tipo], "Modelo": a["modelo"], "Parámetros": int(a["parametros"]),
                          "Media de X menos uno": a["media_extra"], "r": a["r"], "AIC": a["aic"],
                          "Diferencia de AIC": a["delta_aic"]})
        f = poisson.frecuencias_tamano(tamanos, ajustes, CORTES_TAMANO)
        frecuencias.append(f.assign(Tipo=TIPOS[tipo]))
    r.figura(graficos_reales.figura_tamano_lote(por_tipo), "08_tamano_lote")
    t = pd.DataFrame(filas)
    r.agregar(tabla_html(t, "Modelos del tamaño de lote X: X menos uno es Poisson, geométrico o binomial negativo",
                         resaltar=(t["Diferencia de AIC"] == 0).to_numpy(), decimales=2))
    f = pd.concat(frecuencias)[["Tipo", "clase", "observada", *poisson.MODELOS_TAMANO]].rename(
        columns={"clase": "Muestras en el lote", "observada": "Observada"})
    r.agregar(tabla_html(f, "Proporción de lotes por tamaño: Observada y esperada por cada modelo", decimales=3,
                         maximo_filas=20))
    p, q = c.ajustes["producto"], c.ajustes["planta"]
    r.agregar(nota(
        f"Los lotes de producto son pequeños (media {p['media']:.1f}, mediana {p['mediana']:.0f}) y casi geométricos: "
        f"La binomial negativa gana con r = {p['r']:.2f} pero la geométrica (que es el caso r = 1) queda a solo "
        f"{p['delta_geometrica']:.1f} de AIC. Los de planta son otra cosa: Media {q['media']:.1f}, mediana "
        f"{q['mediana']:.0f} y un máximo de {q['maximo']} muestras en un solo día. Ahí la binomial negativa "
        f"(r = {q['r']:.2f}) le saca {q['delta_geometrica']:.0f} de AIC a la geométrica. En los dos tipos la forma "
        "que usa la simulación (uno más una Poisson) es la peor por mucho porque no deja que el tamaño varíe más que "
        "su media.", "hallazgo"))
    camp = c.campanas
    un_dia = float((camp["dias"] == 1).mean())
    r.agregar(tarjetas([
        {"titulo": "Campañas de planta", "valor": len(camp), "decimales": 0,
         "detalle": f"días de planta a menos de {rodam.SEPARACION_CAMPANA_DIAS + 1} días entre sí"},
        {"titulo": "De un solo día", "valor": 100 * un_dia, "decimales": 0, "sufijo": " %", "color": "verde"},
        {"titulo": "Días por campaña", "valor": float(camp["dias"].mean()), "decimales": 1, "color": "morado",
         "detalle": f"máximo {int(camp['dias'].max())}"},
        {"titulo": "Muestras por campaña", "valor": float(camp["n_muestras"].median()), "decimales": 0,
         "color": "naranja", "detalle": f"mediana (media {camp['n_muestras'].mean():.0f})"},
    ], titulo="Campañas de muestreo en planta"))
    c.campanas_resumen = {"campanas": len(camp), "un_dia": un_dia, "dias_media": float(camp["dias"].mean()),
                          "muestras_mediana": float(camp["n_muestras"].median())}
    latex.registrar_varias([
        ("RealLotesProducto", p["lotes"], "lotes de producto"),
        ("RealLotesPlanta", q["lotes"], "lotes de planta"),
        ("RealLoteMediaProducto", p["media"], "tamaño medio de los lotes de producto", 1),
        ("RealLoteMedianaProducto", p["mediana"], "tamaño mediano de los lotes de producto", 0),
        ("RealLoteMaxProducto", p["maximo"], "lote de producto más grande"),
        ("RealLoteMediaPlanta", q["media"], "tamaño medio de los lotes de planta", 1),
        ("RealLoteMedianaPlanta", q["mediana"], "tamaño mediano de los lotes de planta", 0),
        ("RealLoteMaxPlanta", q["maximo"], "lote de planta más grande"),
        ("RealLoteMejorProducto", p["mejor"], "mejor modelo del tamaño de lote en producto"),
        ("RealLoteMejorPlanta", q["mejor"], "mejor modelo del tamaño de lote en planta"),
        ("RealLoteRProducto", p["r"], "r de la binomial negativa en producto", 2),
        ("RealLoteRPlanta", q["r"], "r de la binomial negativa en planta", 2),
        ("RealLoteDeltaGeoProducto", p["delta_geometrica"], "AIC de la geométrica sobre el mejor (producto)", 1),
        ("RealLoteDeltaGeoPlanta", q["delta_geometrica"], "AIC de la geométrica sobre el mejor (planta)", 0),
        ("RealCampanas", len(camp), "campañas de muestreo en planta"),
        ("RealCampanaUnDiaPct", 100 * un_dia, "campañas de un solo día (%)", 0),
        ("RealCampanaMuestrasMediana", c.campanas_resumen["muestras_mediana"], "muestras por campaña (mediana)", 0),
    ])
    latex.exportar_tabla(t, "real_tamano_lote", "Modelos del tamaño de lote por tipo", decimales=2)
    return r


# ----------------------------------------------------------------------
# 4. Segmentos de clientes
# ----------------------------------------------------------------------
def segmentos(c):
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    tabla = cliente_real.tabla_clientes(c.muestras, c.lotes)
    a = agrupamiento.analizar_agrupamiento(
        tabla, cliente_real.VARIABLES_CLIENTE, [], valores_k=VALORES_K, tolerancia=TOLERANCIA_SILUETA,
        semillas=SEMILLAS_ESTABILIDAD, cuantiles_eps=CUANTILES_EPS, ruido_maximo=RUIDO_MAXIMO,
        ruido_respaldo=RUIDO_RESPALDO, asimetria=ASIMETRIA_LOG)
    nombres, descripciones, _ = lectura_segmentos.nombrar_grupos(a["perfil_z"], UMBRAL_NOMBRE, VARIABLES_NOMBRE)
    c.segmentos = {"k": a["k"], "silueta": a["silueta_k"], "nombres": nombres, "clientes": len(tabla)}
    r.agregar(tarjetas([
        {"titulo": "Clientes", "valor": len(tabla), "decimales": 0,
         "detalle": f"con al menos {cliente_real.MINIMO_MUESTRAS} muestras"},
        {"titulo": "Variables", "valor": len(cliente_real.VARIABLES_CLIENTE), "decimales": 0, "color": "morado",
         "detalle": f"{len(a['con_log'])} con log1p"},
        {"titulo": "Grupos (k)", "valor": a["k"], "decimales": 0, "color": "verde",
         "detalle": f"silueta {a['silueta_k']:.3f}"},
        {"titulo": "Estabilidad mínima", "valor": float(a["estabilidad"].min()), "decimales": 2, "color": "naranja",
         "detalle": "índice de Rand ajustado entre semillas"},
    ]))
    r.figura(graficos_agrupamiento.figura_codo_silueta(a["tabla_k"], a["k"], "K-means sobre los clientes reales"),
             "08_codo_silueta")
    r.figura(graficos_agrupamiento.figura_pca_grupos(a["pca"], a["etiquetas_km"], a["pca_varianza"], nombres,
                                                     "Clientes reales en dos componentes principales"),
             "08_segmentos_pca")
    seg = lectura_segmentos.tabla_segmentos(tabla, a["etiquetas_km"], nombres, peso="muestras")
    medias = tabla.assign(grupo=a["etiquetas_km"]).groupby("grupo")[cliente_real.VARIABLES_CLIENTE].median()
    medias.index = [nombres[g] for g in medias.index]
    medias.columns = [lectura_segmentos.etiqueta(x) for x in medias.columns]
    seg["Variables que lo distinguen"] = [descripciones[g] for g in nombres]
    r.agregar(tabla_html(seg, "Segmentos de clientes reales", decimales=3))
    r.agregar(tabla_html(medias.reset_index(names="Segmento"), "Mediana de cada variable por segmento", decimales=2))
    mayor = seg.loc[seg["Proporción de muestras"].idxmax()]
    r.agregar(nota(
        f"La regla de la silueta eligió {a['k']} grupos ({a['silueta_k']:.3f}). El segmento con más muestras es "
        f"\"{mayor['Segmento']}\": {int(mayor['Clientes'])} clientes que traen el {pct(mayor['Proporción de muestras'])} "
        "de las muestras. Los nombres salen de las variables en que cada grupo más se aparta de la media y no de "
        "una lectura a ojo. Con tan pocos clientes los grupos los tomamos como una descripción y no como tipos fijos.",
        "hallazgo"))
    latex.registrar_varias([
        ("RealSegClientes", len(tabla), "clientes en el agrupamiento real"),
        ("RealSegMinimoMuestras", cliente_real.MINIMO_MUESTRAS, "muestras mínimas para entrar al agrupamiento"),
        ("RealSegK", int(a["k"]), "número de segmentos de clientes reales"),
        ("RealSegSilueta", float(a["silueta_k"]), "silueta de los segmentos reales", 3),
        ("RealSegMayor", mayor["Segmento"], "segmento real con más muestras"),
        ("RealSegMayorMuestrasPct", 100 * float(mayor["Proporción de muestras"]), "sus muestras (%)", 1),
    ])
    for grupo, nombre in list(nombres.items())[: len(ORDINALES)]:
        latex.registrar(f"RealSegNombre{ORDINALES[grupo]}", nombre, f"nombre del segmento real {grupo + 1}")
    latex.exportar_tabla(seg.drop(columns=["Variables que lo distinguen"]), "real_segmentos",
                         "Segmentos de clientes reales", decimales=3)
    return r


# ----------------------------------------------------------------------
# 5. Riesgo de no conformidad al ingreso
# ----------------------------------------------------------------------
def riesgo(c):
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    tabla, previa = cliente_real.variables_de_ingreso(c.muestras, c.lotes)
    ajuste = tabla[tabla["particion"] == "entrenamiento"]
    valida = tabla[tabla["particion"] == "validacion"]
    y_val = valida["no_conforme"].to_numpy()
    filas = [{"Modelo": "Línea base (tasa de entrenamiento)", "ROC-AUC": metricas.roc_auc(y_val, np.full(len(y_val), previa)),
              "Precisión media": metricas.precision_media(y_val, np.full(len(y_val), previa))},
             {"Modelo": "Solo la historia del cliente", "ROC-AUC": metricas.roc_auc(y_val, valida["tasa_cliente"]),
              "Precisión media": metricas.precision_media(y_val, valida["tasa_cliente"])}]
    for clave, rotulo in MODELOS_RIESGO.items():
        modelo = construir_tuberia("clasificacion", clave, cliente_real.NUMERICAS_INGRESO,
                                   cliente_real.CATEGORICAS_INGRESO)
        modelo.fit(ajuste, ajuste["no_conforme"])
        proba = modelo.predict_proba(valida)[:, 1]
        filas.append({"Modelo": rotulo, "ROC-AUC": metricas.roc_auc(y_val, proba),
                      "Precisión media": metricas.precision_media(y_val, proba)})
    t = pd.DataFrame(filas)
    mejor = t.iloc[2:].sort_values("ROC-AUC").iloc[-1]
    c.riesgo = {"previa": previa, "entrenamiento": len(ajuste), "validacion": len(valida),
                "prevalencia_validacion": float(y_val.mean()), "tabla": t.to_dict("records"), "mejor": mejor["Modelo"]}
    r.agregar(tarjetas([
        {"titulo": "Entrenamiento", "valor": len(ajuste), "decimales": 0, "detalle": "muestras con dictamen"},
        {"titulo": "Validación", "valor": len(valida), "decimales": 0, "color": "morado",
         "detalle": f"no conformidad {pct(y_val.mean())}"},
        {"titulo": f"ROC-AUC ({mejor['Modelo'].lower()})", "valor": float(mejor["ROC-AUC"]), "decimales": 3,
         "color": "verde", "detalle": "0.5 es no ordenar"},
        {"titulo": "Precisión media", "valor": float(mejor["Precisión media"]), "decimales": 3, "color": "naranja",
         "detalle": f"la línea base da {t.loc[0, 'Precisión media']:.3f}"},
    ]))
    r.agregar(tabla_html(t, "Riesgo de no conformidad al ingreso: Ajuste en entrenamiento y lectura en validación",
                         resaltar=(t["Modelo"] == mejor["Modelo"]).to_numpy(), decimales=3))
    historia, logistica, arbol = t.loc[1], t.loc[2], t.loc[3]
    r.agregar(nota(
        f"La línea base que le da a todas las muestras la tasa de entrenamiento ({pct(previa)}) no ordena nada "
        f"(ROC-AUC {t.loc[0, 'ROC-AUC']:.3f}) y su precisión media es la prevalencia de validación "
        f"({t.loc[0, 'Precisión media']:.3f}). La regresión logística llega a un ROC-AUC de "
        f"{logistica['ROC-AUC']:.3f} con precisión media {logistica['Precisión media']:.3f} y el árbol a "
        f"{arbol['ROC-AUC']:.3f} con {arbol['Precisión media']:.3f}. Solo con la tasa histórica del cliente se logra "
        f"{historia['ROC-AUC']:.3f} y {historia['Precisión media']:.3f}, casi lo mismo que los modelos, así que buena "
        "parte de lo que se puede anticipar al ingreso es quién manda la muestra. Es una primera mirada sin ajuste de "
        "hiperparámetros ni elección de umbral (eso lo hicimos con datos simulados en la etapa 05) y sirve para saber "
        "que sí hay señal en la base real aunque es modesta.", "hallazgo"))
    latex.registrar_varias([
        ("RealRiesgoEntrenamiento", len(ajuste), "muestras de entrenamiento del clasificador real"),
        ("RealRiesgoValidacion", len(valida), "muestras de validación del clasificador real"),
        ("RealRiesgoPrevalenciaPct", 100 * float(y_val.mean()), "no conformidad en validación (%)", 1),
        ("RealRiesgoAPBase", float(t.loc[0, "Precisión media"]), "precisión media de la línea base", 3),
        ("RealRiesgoAUCHistoria", float(historia["ROC-AUC"]), "ROC-AUC de la tasa histórica del cliente", 3),
        ("RealRiesgoAPHistoria", float(historia["Precisión media"]), "precisión media de la tasa histórica", 3),
        ("RealRiesgoAUCLogistica", float(t.loc[2, "ROC-AUC"]), "ROC-AUC de la regresión logística", 3),
        ("RealRiesgoAPLogistica", float(t.loc[2, "Precisión media"]), "precisión media de la regresión logística", 3),
        ("RealRiesgoAUCArbol", float(t.loc[3, "ROC-AUC"]), "ROC-AUC del árbol de decisión", 3),
        ("RealRiesgoAPArbol", float(t.loc[3, "Precisión media"]), "precisión media del árbol de decisión", 3),
    ])
    latex.exportar_tabla(t, "real_riesgo", "Clasificador de riesgo al ingreso en validación", decimales=3)
    return r


# ----------------------------------------------------------------------
# 6. Comparación con la simulación
# ----------------------------------------------------------------------
def _indice_simulacion(media_lote):
    """E[X²]/E[X] para X = 1 + Poisson(m − 1): (Var + E²)/E con Var = m − 1 y E = m."""
    return ((media_lote - 1) + media_lote ** 2) / media_lote


def comparacion_simulacion(c):
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    p, q = c.ajustes["producto"], c.ajustes["planta"]
    llegadas = c.tabla_llegadas.set_index("Tipo")
    bajo, alto = (_indice_simulacion(m) for m in LOTE_MEDIO_SIMULACION)
    t = pd.DataFrame([
        {"Pieza": "Número de lotes", "Simulación": "Poisson por cliente y día con tasa que cambia por semana; los "
                                                    "clientes industriales con un número casi fijo por semana",
         "Exportación real": f"Cerca de Poisson: Índice semanal {llegadas.iloc[0]['Índice lotes (mediana)']:.2f} "
                             f"(producto) y {llegadas.iloc[1]['Índice lotes (mediana)']:.2f} (planta)"},
        {"Pieza": "Tamaño del lote", "Simulación": f"1 + Poisson con tamaño medio de {LOTE_MEDIO_SIMULACION[0]} a "
                                                   f"{LOTE_MEDIO_SIMULACION[1]} según el segmento",
         "Exportación real": f"1 + binomial negativa: Media {p['media']:.1f} (producto) y {q['media']:.1f} (planta), "
                             f"máximo {q['maximo']}"},
        {"Pieza": "Tipos de lote", "Simulación": "Uno solo",
         "Exportación real": f"Producto y planta ({pct(c.base['planta'])} de las muestras en planta)"},
        {"Pieza": "Índice de las muestras", "Simulación": f"E[X²]/E[X] de {bajo:.1f} a {alto:.1f}",
         "Exportación real": f"Mediana {c.dispersion_todos['muestras']:.2f}, máximo "
                             f"{c.dispersion_todos['maximo_muestras']:.0f}"},
    ])
    r.agregar(tabla_html(t, "Mecanismo de llegada: Simulación frente a exportación real"))
    r.agregar(nota(
        "La simulación ya generaba las muestras como una Poisson compuesta (lotes de Poisson con varias muestras "
        "cada uno) así que la estructura era la correcta. Lo que no reproduce es el tamaño de los lotes: Con uno más "
        f"una Poisson la varianza del tamaño no pasa de su media y el índice de las muestras queda entre {bajo:.1f} y "
        f"{alto:.1f}, lejos de la cola de planta. Para acercarla bastaría con separar producto y planta y sacar el "
        "tamaño de una binomial negativa con los parámetros de la sección 3. Eso lo dejamos propuesto y no lo "
        "cambiamos porque las etapas 01 a 07 ya están cerradas sobre la simulación actual.", "info"))
    latex.registrar_varias([
        ("RealSimIndiceBajo", bajo, "índice de muestras de la simulación con el lote medio menor", 1),
        ("RealSimIndiceAlto", alto, "índice de muestras de la simulación con el lote medio mayor", 1),
    ])
    latex.exportar_tabla(t, "real_simulacion", "Mecanismo de llegada en la simulación y en la exportación real",
                         alineacion=r"l>{\raggedright\arraybackslash}p{5.5cm}>{\raggedright\arraybackslash}p{5.5cm}")
    return r


# ----------------------------------------------------------------------
# 7. Cierre
# ----------------------------------------------------------------------
def cierre(c):
    r = Resultado()
    if c.hay:
        r.agregar(pasos([
            ("Llegadas", "Los lotes por cliente y semana se comportan casi como Poisson y las muestras no: Son una "
                         "Poisson compuesta."),
            ("Tamaño de lote", f"Producto con lotes pequeños ({c.ajustes['producto']['mejor']}, casi geométrica) y "
                               f"planta con cola larga ({c.ajustes['planta']['mejor']})."),
            ("Segmentos", f"{c.segmentos['k']} grupos de clientes: "
                          + "; ".join(f'"{n}"' for n in c.segmentos["nombres"].values()) + "."),
            ("Riesgo al ingreso", f"{c.riesgo['mejor']} con ROC-AUC "
                                  f"{max(f['ROC-AUC'] for f in c.riesgo['tabla'][2:]):.3f} en validación."),
        ], titulo="Qué deja la etapa"))
        c.resumen.update({
            "base": c.base, "dispersion_todos": c.dispersion_todos,
            "llegadas": c.tabla_llegadas.to_dict("records"),
            "tamano_lote": c.ajustes, "campanas": c.campanas_resumen,
            "segmentos": {"k": c.segmentos["k"], "silueta": c.segmentos["silueta"], "clientes": c.segmentos["clientes"],
                          "nombres": list(c.segmentos["nombres"].values())},
            "riesgo": c.riesgo,
        })
    else:
        r.agregar(nota(SIN_DATOS, "limite", titulo="Sin la exportación real"))
    c.resumen["segundos"] = time.perf_counter() - c.inicio
    guardar_json(c.resumen, rutas.DIR_ARTEFACTOS / f"{ETAPA}.json")
    cerrar({"hay_exportacion": c.hay, "segundos": c.resumen["segundos"]})
    return r
