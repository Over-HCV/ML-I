"""Capa 8 · Etapa 08: datos reales.

Responsabilidad única: armar las piezas de la etapa 08 sobre la exportación
real del laboratorio, con fechas de desarrollo (anteriores al corte de
prueba) en todo menos la última sección: la base y su granularidad, las llegadas de muestras frente a las de
lotes con la prueba de Poisson, el tamaño de los lotes por tipo y las
campañas, los segmentos de clientes, el clasificador del riesgo de no
conformidad al ingreso (referencia, regla, regresión logística y árbol con
tres configuraciones cada uno, umbral por costo, cálculo ilustrativo e
interpretación) y la comparación con el mecanismo de la simulación. Al final
congela la decisión del clasificador y abre una sola vez el bloque de 2026
(la prueba), con la lectura anotada en el registro de la custodia. Guarda el
artefacto de la etapa y registra sus cifras (prefijo `Real`).

Si la exportación no está (por ejemplo en Colab, porque el archivo es
confidencial), cada sección devuelve una nota que lo explica y la etapa se
cierra sin registrar cifras.
"""
import hashlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from proyecto.base import config, rutas
from proyecto.base.artefactos import guardar_json, guardar_modelo, leer_modelo
from proyecto.datos import rodam
from proyecto.etapas.comun import Resultado, abrir, cerrar, enumerar, pct
from proyecto.evaluacion import metricas, poisson, umbral
from proyecto.evaluacion.remuestreo import intervalo_bootstrap
from proyecto.modelos import agrupamiento, registro, tuberias
from proyecto.modelos.tuberias import construir_tuberia
from proyecto.reporte import graficos_agrupamiento, graficos_reales, latex
from proyecto.reporte import graficos_clasificacion as graficos_clf
from proyecto.reporte import segmentos as lectura_segmentos
from proyecto.reporte.html import nota, pasos, tabla_html, tarjetas
from proyecto.variables import cliente_real

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
VALORES_C = [0.01, 0.1, 1.0]                  # regularización de la logística (C menor, más fuerte)
PROFUNDIDADES = [3, 5, 8]                     # profundidad máxima del árbol
MINIMO_HOJA = 50                              # muestras mínimas por hoja del árbol
UMBRALES_RIESGO = [round(float(t), 2) for t in np.arange(0.01, 0.951, 0.01)]
REPLICAS_BOOTSTRAP = 1000
DECIMALES_RIESGO = {"Umbral": 2, "Precisión": 3, "Recall": 3, "F1": 3, "ROC-AUC": 3, "PR-AUC": 3}
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
    filas = rodam.leer_desarrollo()   # lo de 2026 en adelante no se mira hasta la prueba
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
def _composicion(parte, rotulo, clientes_previos=None):
    """Tamaño y composición de una partición de la tabla de modelado."""
    fila = {"Partición": rotulo, "Desde": parte["dia"].min(), "Hasta": parte["dia"].max(),
            "Muestras": len(parte), "Clientes": parte["cliente"].nunique(),
            "Muestreo en planta": float((parte["tipo_lote"] == "planta").mean()),
            "Sin historia del cliente": float((parte["historia_cliente"] == 0).mean()),
            "No conformidad": float(parte["no_conforme"].mean())}
    if clientes_previos is not None:
        fila["Clientes nuevos"] = int((~pd.Index(parte["cliente"].unique()).isin(clientes_previos)).sum())
    return fila


def _candidatos():
    """Las configuraciones que comparamos, fijadas antes de ver resultados."""
    lista = [{"modelo": "base", "familia": "Referencia", "configuracion": "probabilidad de entrenamiento"},
             {"modelo": "regla", "familia": "Regla simple", "configuracion": "tasa histórica del cliente"}]
    lista += [{"modelo": "logistica", "familia": MODELOS_RIESGO["logistica"], "configuracion": f"C = {valor:g}",
               "hiper": {"C": valor}} for valor in VALORES_C]
    lista += [{"modelo": "arbol", "familia": MODELOS_RIESGO["arbol"], "configuracion": f"profundidad {valor}",
               "hiper": {"max_depth": valor, "min_samples_leaf": MINIMO_HOJA}} for valor in PROFUNDIDADES]
    for candidato in lista:
        candidato["nombre"] = f"{candidato['familia']} ({candidato['configuracion']})"
    return lista


def _tuberia(candidato):
    """Tubería sin ajustar de un candidato (la regla no se ajusta: es una columna)."""
    if candidato["modelo"] == "regla":
        return None
    if candidato["modelo"] == "base":
        return Pipeline([("modelo", registro.linea_base("clasificacion"))])
    escalado = "ninguno" if candidato["modelo"] == "arbol" else "estandar"
    return construir_tuberia("clasificacion", candidato["modelo"], cliente_real.NUMERICAS_INGRESO,
                             cliente_real.CATEGORICAS_INGRESO, escalado=escalado, hiper=candidato["hiper"])


def _probabilidad(candidato, modelo, tabla):
    if candidato["modelo"] == "regla":
        return tabla["tasa_cliente"].to_numpy(dtype=float)
    return metricas.prediccion_de(modelo, tabla)


def _fila_metricas(y, proba, umbral_corte):
    m = metricas.metricas_clf(y, proba, umbral_corte)
    return {"Umbral": umbral_corte, "Precisión": m["Precisión"], "Recall": m["Recall"], "F1": m["F1"],
            "ROC-AUC": m["ROC-AUC"], "PR-AUC": metricas.precision_media(y, proba)}


def _semanas(tabla):
    return int(tabla["dia"].dt.to_period("W-SUN").nunique())


def _legible(nombre):
    """`sector_Farmacéutico` → `sector = Farmacéutico`; las numéricas quedan igual."""
    for columna in cliente_real.CATEGORICAS_INGRESO:
        if nombre.startswith(columna + "_"):
            nivel = nombre[len(columna) + 1:].replace("infrequent_sklearn", "otros (poco frecuentes)")
            return f"{columna} = {nivel}"
    return nombre


def riesgo_protocolo(c):
    """Tabla de modelado, particiones y su composición."""
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    c.tabla_riesgo, c.previa = cliente_real.variables_de_ingreso(c.muestras)
    c.ajuste = c.tabla_riesgo[c.tabla_riesgo["particion"] == "entrenamiento"]
    c.valida = c.tabla_riesgo[c.tabla_riesgo["particion"] == "validacion"]
    c.y_ent, c.y_val = c.ajuste["no_conforme"].to_numpy(), c.valida["no_conforme"].to_numpy()
    t = pd.DataFrame([_composicion(c.ajuste, "Entrenamiento"), _composicion(c.valida, "Validación")])
    sin_dictamen = len(c.muestras) - len(c.tabla_riesgo)
    proporcion = len(c.ajuste) / len(c.tabla_riesgo)
    r.agregar(tarjetas([
        {"titulo": "Entrenamiento", "valor": len(c.ajuste), "decimales": 0,
         "detalle": f"{pct(proporcion, 0)} del desarrollo"},
        {"titulo": "Validación", "valor": len(c.valida), "decimales": 0, "color": "morado",
         "detalle": f"{pct(1 - proporcion, 0)} del desarrollo"},
        {"titulo": "No conformidad", "valor": 100 * c.y_val.mean(), "decimales": 1, "sufijo": " %", "color": "rojo",
         "detalle": f"en validación ({pct(c.y_ent.mean())} en entrenamiento)"},
        {"titulo": "Variables al ingreso", "valor": len(cliente_real.NUMERICAS_INGRESO)
         + len(cliente_real.CATEGORICAS_INGRESO), "decimales": 0, "color": "verde",
         "detalle": f"{len(cliente_real.NUMERICAS_INGRESO)} numéricas y {len(cliente_real.CATEGORICAS_INGRESO)} "
                    "categóricas"},
    ]))
    r.agregar(tabla_html(t, "Particiones del clasificador: Tamaño y composición (muestras con dictamen)",
                         decimales=3))
    r.agregar(nota(
        f"Dejamos por fuera {sin_dictamen} muestras sin dictamen porque no tienen clase. Entrenamiento queda con el "
        f"{pct(proporcion, 0)} de las muestras de desarrollo y validación con el {pct(1 - proporcion, 0)}, que es lo "
        "que dan los mismos cortes por fecha del resto del proyecto. La clase 1 es la muestra no conforme y es rara "
        f"({pct(c.y_ent.mean())} en entrenamiento) así que una lista de alertas al azar acertaría más o menos esa "
        "fracción de las veces. Ese es el piso contra el que medimos.", "info"))
    latex.registrar_varias([
        ("RealRiesgoEntrenamiento", len(c.ajuste), "muestras de entrenamiento del clasificador real"),
        ("RealRiesgoValidacion", len(c.valida), "muestras de validación del clasificador real"),
        ("RealRiesgoSinDictamen", sin_dictamen, "muestras de desarrollo sin dictamen"),
        ("RealRiesgoEntrenamientoPct", 100 * proporcion, "entrenamiento sobre el desarrollo (%)", 0),
        ("RealRiesgoValidacionPct", 100 * (1 - proporcion), "validación sobre el desarrollo (%)", 0),
        ("RealRiesgoPrevalenciaEntPct", 100 * float(c.y_ent.mean()), "no conformidad en entrenamiento (%)", 1),
        ("RealRiesgoPrevalenciaPct", 100 * float(c.y_val.mean()), "no conformidad en validación (%)", 1),
        ("RealRiesgoVariables", len(cliente_real.NUMERICAS_INGRESO) + len(cliente_real.CATEGORICAS_INGRESO),
         "variables conocidas al ingreso"),
    ])
    latex.exportar_tabla(t, "real_riesgo_particiones", "Particiones del clasificador real", decimales=3)
    return r


def riesgo_modelos(c):
    """Referencia, regla y las seis configuraciones: Entrenamiento frente a validación y elección por PR-AUC."""
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    filas, resumen, c.ajustados = [], [], {}
    for candidato in _candidatos():
        modelo = _tuberia(candidato)
        if modelo is not None:
            modelo.fit(c.ajuste, c.y_ent)
        p_ent, p_val = _probabilidad(candidato, modelo, c.ajuste), _probabilidad(candidato, modelo, c.valida)
        corte = float(umbral.elegir_umbral(umbral.tabla_umbrales(c.y_val, p_val, UMBRALES_RIESGO))["Umbral"])
        ent, val = _fila_metricas(c.y_ent, p_ent, corte), _fila_metricas(c.y_val, p_val, corte)
        c.ajustados[candidato["nombre"]] = {**candidato, "tuberia": modelo, "p_val": p_val, "umbral": corte}
        for rotulo, m in (("Entrenamiento", ent), ("Validación", val)):
            filas.append({"Modelo": candidato["familia"], "Configuración": candidato["configuracion"],
                          "Partición": rotulo, **m})
        resumen.append({"nombre": candidato["nombre"], "modelo": candidato["modelo"], "ap_ent": ent["PR-AUC"],
                        "ap_val": val["PR-AUC"], "auc_ent": ent["ROC-AUC"], "auc_val": val["ROC-AUC"]})
    t = pd.DataFrame(filas)
    c.resumen_riesgo = s = pd.DataFrame(resumen).set_index("nombre")
    modelos = s[s["modelo"].isin(["logistica", "arbol"])].reset_index()
    c.elegido = str(metricas.seleccionar(modelos, [("ap_val", False)])["nombre"])
    c.mejor_logistica = str(metricas.seleccionar(modelos[modelos["modelo"] == "logistica"], [("ap_val", False)])["nombre"])
    c.mejor_arbol = str(metricas.seleccionar(modelos[modelos["modelo"] == "arbol"], [("ap_val", False)])["nombre"])
    elegido, regla, base = s.loc[c.elegido], s[s["modelo"] == "regla"].iloc[0], s[s["modelo"] == "base"].iloc[0]
    log, arb = s.loc[c.mejor_logistica], s.loc[c.mejor_arbol]
    profundo = s[s["modelo"] == "arbol"].iloc[-1]
    rango_c = (s.loc[s["modelo"] == "logistica", "ap_val"].min(), s.loc[s["modelo"] == "logistica", "ap_val"].max())
    marca = ((t["Modelo"] + " (" + t["Configuración"] + ")") == c.elegido) & (t["Partición"] == "Validación")
    r.agregar(tabla_html(t, "Las ocho configuraciones en entrenamiento y en validación (umbral de menor costo en "
                            "validación para cada una)", resaltar=marca.to_numpy(), columnas_resaltadas="PR-AUC",
                         decimales=DECIMALES_RIESGO, maximo_filas=20))
    curvas = {"Regla simple": (c.y_val, c.ajustados[regla.name]["p_val"]),
              c.mejor_logistica: (c.y_val, c.ajustados[c.mejor_logistica]["p_val"]),
              c.mejor_arbol: (c.y_val, c.ajustados[c.mejor_arbol]["p_val"])}
    r.figura(graficos_reales.figura_precision_recall(curvas, float(c.y_val.mean()),
                                                     "Precisión y recall en validación"), "08_precision_recall")
    gana_regla = elegido["ap_val"] - regla["ap_val"]
    r.agregar(nota(
        f"El elegido por PR-AUC en validación es **{c.elegido}** con {elegido['ap_val']:.3f}. La referencia, que le "
        f"da a todas la misma probabilidad, se queda en {base['ap_val']:.3f} (la prevalencia) y la regla de la tasa "
        f"histórica llega a {regla['ap_val']:.3f}. O sea que el modelo "
        + (f"le gana a la regla por apenas {gana_regla:.3f} de PR-AUC" if gana_regla > 0 else
           f"no le gana a la regla (le faltan {-gana_regla:.3f} de PR-AUC)")
        + " y casi todo lo que se puede anticipar al ingreso ya está en quién manda la muestra. "
        + (f"Entre las regresiones logísticas casi no cambia nada con C (de {rango_c[0]:.3f} a {rango_c[1]:.3f}). "
           if rango_c[1] - rango_c[0] < 0.01 else
           f"En las regresiones logísticas C sí importa (de {rango_c[0]:.3f} a {rango_c[1]:.3f}). ")
        + f"En los árboles el sobreajuste se ve claro: El de profundidad {PROFUNDIDADES[-1]} sube a "
          f"{profundo['ap_ent']:.3f} de PR-AUC en entrenamiento pero en validación se queda en {profundo['ap_val']:.3f}. "
        + (f"El mejor árbol ({arb['ap_val']:.3f}) queda por debajo de la mejor logística ({log['ap_val']:.3f}) porque "
           "divide el riesgo en pocos escalones en vez de sumar señales pequeñas. "
           if arb["ap_val"] < log["ap_val"] else
           f"El mejor árbol ({arb['ap_val']:.3f}) alcanza a la mejor logística ({log['ap_val']:.3f}). ")
        + f"Para el árbol usamos hojas de al menos {MINIMO_HOJA} muestras porque "
        f"con la prevalencia de entrenamiento una hoja así espera {MINIMO_HOJA * c.previa:.1f} no conformes y con hojas "
        "más chicas la fracción de cada hoja saldría de cero o un caso.",
        "hallazgo"))
    latex.registrar_varias([
        ("RealRiesgoMinimoHoja", MINIMO_HOJA, "mínimo de muestras por hoja del árbol"),
        ("RealRiesgoHojaEsperados", MINIMO_HOJA * c.previa, "no conformes esperados en una hoja mínima", 1),
        ("RealRiesgoAPBase", float(base["ap_val"]), "PR-AUC de la referencia en validación", 3),
        ("RealRiesgoAUCHistoria", float(regla["auc_val"]), "ROC-AUC de la tasa histórica del cliente", 3),
        ("RealRiesgoAPHistoria", float(regla["ap_val"]), "PR-AUC de la tasa histórica en validación", 3),
        ("RealRiesgoLogistica", c.ajustados[c.mejor_logistica]["configuracion"], "mejor configuración logística"),
        ("RealRiesgoAUCLogistica", float(log["auc_val"]), "ROC-AUC de la mejor regresión logística", 3),
        ("RealRiesgoAPLogistica", float(log["ap_val"]), "PR-AUC de la mejor regresión logística", 3),
        ("RealRiesgoAPLogisticaEnt", float(log["ap_ent"]), "PR-AUC de la mejor logística en entrenamiento", 3),
        ("RealRiesgoArbol", c.ajustados[c.mejor_arbol]["configuracion"], "mejor configuración del árbol"),
        ("RealRiesgoAUCArbol", float(arb["auc_val"]), "ROC-AUC del mejor árbol", 3),
        ("RealRiesgoAPArbol", float(arb["ap_val"]), "PR-AUC del mejor árbol en validación", 3),
        ("RealRiesgoAPArbolProfundoEnt", float(profundo["ap_ent"]), "PR-AUC del árbol más profundo en entrenamiento", 3),
        ("RealRiesgoAPArbolProfundoVal", float(profundo["ap_val"]), "PR-AUC del árbol más profundo en validación", 3),
        ("RealRiesgoElegido", c.elegido, "configuración elegida por PR-AUC en validación"),
        ("RealRiesgoAPElegido", float(elegido["ap_val"]), "PR-AUC del elegido en validación", 3),
        ("RealRiesgoAUCElegido", float(elegido["auc_val"]), "ROC-AUC del elegido en validación", 3),
    ])
    latex.exportar_tabla(t, "real_configuraciones", "Configuraciones del clasificador real en entrenamiento y "
                         "validación", decimales=DECIMALES_RIESGO)
    return r


def riesgo_umbral(c):
    """Umbral por costo del elegido en validación y su matriz de confusión."""
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    elegido = c.ajustados[c.elegido]
    t = umbral.tabla_umbrales(c.y_val, elegido["p_val"], UMBRALES_RIESGO)
    t["Alertas por semana"] = t["Alertas"] / _semanas(c.valida)
    f = umbral.elegir_umbral(t)
    c.umbral = u = float(f["Umbral"])
    c.fila_umbral = f
    teorico = umbral.umbral_teorico()
    positivos, negativos = int(c.y_val.sum()), int(len(c.y_val) - c.y_val.sum())
    c.costo_nada, c.costo_todo = config.COSTO_FN * positivos, config.COSTO_FP * negativos
    r.figura(graficos_clf.figura_umbrales(t, u, teorico), "08_umbrales")
    r.figura(graficos_clf.figura_confusion(int(f["TN"]), int(f["FP"]), int(f["FN"]), int(f["TP"]),
                                           f"Validación con umbral {u:.2f} (etiquetas 0 y 1)",
                                           clases=("0 conforme", "1 no conforme"),
                                           acciones=("0 sin alerta", "1 alerta")), "08_confusion_validacion")
    mejor_trivial = min(c.costo_nada, c.costo_todo)
    r.agregar(nota(
        f"Supusimos que dejar pasar una no conforme cuesta {config.COSTO_FN} veces lo que una falsa alarma. Con esa "
        f"relación el umbral que minimiza el costo sería {teorico:.3f} si las probabilidades estuvieran bien "
        f"calibradas pero en validación el de menor costo fue **{u:.2f}**. Con él se levantan {int(f['Alertas'])} "
        f"alertas ({f['Alertas por semana']:.1f} por semana) y "
        f"aciertan el {pct(f['Precisión'], 0)}. De los {positivos} no conformes de validación alerta "
        f"{int(f['TP'])} (recall {f['Recall']:.3f}) así que la mayoría sigue pasando sin aviso. El costo baja de "
        f"{mejor_trivial} (no alertar nunca) a {int(f['Costo'])}"
        + (f", un ahorro del {pct(1 - f['Costo'] / mejor_trivial, 0)}." if f["Costo"] < mejor_trivial else
           " así que con estos costos el modelo no ahorra nada frente a no alertar.")
        + " Eso sí, como el umbral lo elegimos en estos mismos datos este costo es algo optimista.",
        "hallazgo"))
    latex.registrar_varias([
        ("RealUmbral", u, "umbral elegido por costo en validación", 2),
        ("RealUmbralTeorico", teorico, "umbral teórico por costo", 3),
        ("RealUmbralCosto", int(f["Costo"]), "costo en validación con el umbral elegido"),
        ("RealUmbralCostoNada", c.costo_nada, "costo en validación de no alertar nunca"),
        ("RealUmbralCostoTodo", c.costo_todo, "costo en validación de alertar todas"),
        ("RealUmbralAlertas", int(f["Alertas"]), "alertas en validación con el umbral elegido"),
        ("RealUmbralAlertasSemana", float(f["Alertas por semana"]), "alertas por semana en validación", 1),
        ("RealUmbralPrecision", float(f["Precisión"]), "precisión en validación con el umbral", 3),
        ("RealUmbralRecall", float(f["Recall"]), "recall en validación con el umbral", 3),
        ("RealUmbralFUno", float(f["F1"]), "F1 en validación con el umbral", 3),
        ("RealUmbralTP", int(f["TP"]), "verdaderos positivos en validación"),
        ("RealUmbralFP", int(f["FP"]), "falsos positivos en validación"),
        ("RealUmbralFN", int(f["FN"]), "falsos negativos en validación"),
        ("RealUmbralTN", int(f["TN"]), "verdaderos negativos en validación"),
        ("RealCostoFP", config.COSTO_FP, "costo supuesto de un falso positivo"),
        ("RealCostoFN", config.COSTO_FN, "costo supuesto de un falso negativo"),
    ])
    return r


def riesgo_calculo(c):
    """Cálculo ilustrativo: La probabilidad logística término a término y el primer corte del árbol con su Gini."""
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    logistica = c.ajustados[c.mejor_logistica]
    tuberia = logistica["tuberia"]
    modelo = tuberia.named_steps["modelo"]
    # La muestra: La no conforme de validación a la que la logística le dio más riesgo.
    positivas = np.flatnonzero(c.y_val == 1)
    i = positivas[np.argmax(logistica["p_val"][positivas])]
    fila = c.valida.iloc[[i]]
    x = tuberia.named_steps["preparar"].transform(fila)[0]
    nombres = list(tuberia.named_steps["preparar"].get_feature_names_out())
    activas = np.flatnonzero(x != 0)
    calculo = pd.DataFrame({
        "Variable": [_legible(nombres[j]) for j in activas],
        "Valor original": [f"{fila[nombres[j]].iat[0]:g}" if nombres[j] in cliente_real.NUMERICAS_INGRESO else "1"
                           for j in activas],
        "Valor en el modelo": x[activas], "Coeficiente": modelo.coef_[0][activas],
        "Aporte": modelo.coef_[0][activas] * x[activas]})
    calculo = calculo.reindex(calculo["Aporte"].abs().sort_values(ascending=False).index)
    intercepto = float(modelo.intercept_[0])
    logit = intercepto + float(calculo["Aporte"].sum())
    proba = 1 / (1 + np.exp(-logit))
    if not np.isclose(proba, logistica["p_val"][i]):
        raise RuntimeError("El cálculo a mano no coincide con predict_proba.")
    filas_calc = pd.concat([pd.DataFrame([{"Variable": "Intercepto", "Valor original": "", "Valor en el modelo": 1.0,
                                           "Coeficiente": intercepto, "Aporte": intercepto}]), calculo])
    r.agregar(tabla_html(filas_calc, f"Regresión logística ({logistica['configuracion']}): Los términos de una muestra "
                                     "no conforme de validación", decimales=3))
    r.agregar(nota(
        f"Sumando el intercepto ({intercepto:.3f}) y los {len(calculo)} aportes da un logit de z = {logit:.3f}. La "
        f"probabilidad es 1 / (1 + e elevado a {-logit:.3f}) = **{proba:.3f}**, la misma que entrega el modelo. "
        f"Comparada con el umbral elegido ({c.umbral:.2f}) la muestra "
        + ("sí recibe alerta." if proba >= c.umbral else "no recibe alerta.")
        + f" El aporte más grande es el de \"{calculo['Variable'].iat[0]}\" ({calculo['Aporte'].iat[0]:.3f}). Las "
          "numéricas entran estandarizadas (menos la media de entrenamiento y entre su desviación) y las categóricas "
          "como indicadoras que valen 1 en el nivel de la muestra.", "info"))
    # El primer corte del árbol con su Gini.
    arbol_info = c.ajustados[c.mejor_arbol]
    arbol = arbol_info["tuberia"].named_steps["modelo"]
    nombres_arbol = tuberias.nombres_variables(arbol_info["tuberia"])
    e = arbol.tree_
    izq, der = e.children_left[0], e.children_right[0]
    variable, corte = nombres_arbol[e.feature[0]], float(e.threshold[0])
    binaria = variable not in cliente_real.NUMERICAS_INGRESO
    condicion = (f"{_legible(variable)} es 0", f"{_legible(variable)} es 1") if binaria else \
        (f"{variable} ≤ {corte:g}", f"{variable} > {corte:g}")
    nodos = []
    for nodo, rotulo in ((0, "Raíz (todas)"), (izq, condicion[0]), (der, condicion[1])):
        valores = np.asarray(e.value[nodo][0], dtype=float)
        fraccion = float(valores[-1] / valores.sum())
        nodos.append({"Nodo": rotulo, "Muestras": int(e.n_node_samples[nodo]), "Fracción no conforme": fraccion,
                      "Gini a mano": 2 * fraccion * (1 - fraccion), "Gini del árbol": float(e.impurity[nodo])})
    n = pd.DataFrame(nodos)
    despues = float((n["Muestras"].iloc[1:] * n["Gini del árbol"].iloc[1:]).sum() / n["Muestras"].iat[0])
    antes = float(n["Gini del árbol"].iat[0])
    r.agregar(tabla_html(n, f"Primer corte del árbol ({arbol_info['configuracion']}) en entrenamiento", decimales=4))
    r.agregar(nota(
        f"El árbol empieza separando por \"{condicion[1] if binaria else variable}\". Antes del corte el Gini de la "
        f"raíz es {antes:.4f} y después, promediando los dos hijos por su número de muestras, queda en {despues:.4f}. "
        f"La reducción es de {antes - despues:.4f}, pequeña porque la clase 1 es tan rara que casi cualquier nodo ya es "
        "casi puro. Aun así es el corte que más la reduce entre todas las variables y todos los puntos de corte.",
        "info"))
    c.calculo = {"logit": logit, "proba": proba, "intercepto": intercepto, "variable_arbol": variable,
                 "gini_antes": antes, "gini_despues": despues}
    latex.registrar_varias([
        ("RealCalcIntercepto", intercepto, "intercepto de la logística", 3),
        ("RealCalcTerminos", len(calculo), "variables activas en el cálculo ilustrativo"),
        ("RealCalcLogit", logit, "logit de la muestra del cálculo ilustrativo", 3),
        ("RealCalcProba", proba, "probabilidad de la muestra del cálculo ilustrativo", 3),
        ("RealCalcMayor", calculo["Variable"].iat[0], "variable con el mayor aporte en el cálculo"),
        ("RealCalcMayorAporte", float(calculo["Aporte"].iat[0]), "su aporte al logit", 3),
        ("RealArbolPrimerCorte", condicion[1] if binaria else f"{variable} > {corte:g}", "primer corte del árbol"),
        ("RealArbolGiniAntes", antes, "Gini de la raíz del árbol", 4),
        ("RealArbolGiniDespues", despues, "Gini ponderado tras el primer corte", 4),
        ("RealArbolGiniReduccion", antes - despues, "reducción del Gini en el primer corte", 4),
    ])
    latex.exportar_tabla(filas_calc, "real_calculo_logistica", "Cálculo ilustrativo de la regresión logística",
                         decimales=3)
    return r


def riesgo_interpretacion(c):
    """Razones de odds de la logística y reglas del árbol."""
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    tuberia = c.ajustados[c.mejor_logistica]["tuberia"]
    coef = pd.Series(tuberia.named_steps["modelo"].coef_[0], index=[_legible(n) for n in
                                                                      tuberias.nombres_variables(tuberia)])
    principales = coef.reindex(coef.abs().sort_values(ascending=False).index).head(8)
    odds = pd.DataFrame({"Variable": principales.index, "Coeficiente": principales.to_numpy(),
                         "Razón de odds": np.exp(principales.to_numpy())})
    r.agregar(tabla_html(odds, "Las ocho variables con mayor peso en la regresión logística (entrenamiento)",
                         decimales=3))
    arbol_info = c.ajustados[c.mejor_arbol]
    nombres = tuberias.nombres_variables(arbol_info["tuberia"])
    binarias = [n for n in nombres if n not in cliente_real.NUMERICAS_INGRESO]
    reglas = graficos_clf.reglas_arbol(arbol_info["tuberia"].named_steps["modelo"], [_legible(n) for n in nombres],
                                       binarias=[_legible(n) for n in binarias], maximo=5)
    r.agregar(tabla_html(reglas, f"Las cinco hojas de mayor riesgo del árbol ({arbol_info['configuracion']}, "
                                 "entrenamiento)", decimales=3, barras="Fracción de positivos"))
    sube, baja = principales[principales > 0], principales[principales < 0]
    primera = reglas.iloc[0]
    r.agregar(nota(
        "En la logística suben el riesgo "
        + (enumerar(f'"{n}" (razón de odds {np.exp(v):.2f})' for n, v in sube.head(3).items()) or "ninguna")
        + " y lo bajan " + (enumerar(f'"{n}" ({np.exp(v):.2f})' for n, v in baja.head(3).items()) or "ninguna")
        + ". En las numéricas la razón de odds es por cada desviación estándar porque entran estandarizadas. La hoja "
          f"de mayor riesgo del árbol junta {int(primera['Filas'])} muestras de entrenamiento con "
          f"{primera['Fracción de positivos']:.3f} de no conformes, {primera['Fracción de positivos'] / c.previa:.1f} "
          "veces la tasa general. Eso sí, todo esto son asociaciones dentro de cada modelo y no efectos causales: Que "
          "un sector tenga más riesgo no quiere decir que el sector lo cause (puede ser el tipo de producto o el "
          "cliente que hay detrás) y con variables correlacionadas el peso se reparte de forma inestable.",
        "hallazgo"))
    if len(sube):
        latex.registrar_varias([("RealOddsSube", sube.index[0], "variable que más sube el riesgo"),
                                ("RealOddsSubeValor", float(np.exp(sube.iloc[0])), "su razón de odds", 2)])
    if len(baja):
        latex.registrar_varias([("RealOddsBaja", baja.index[0], "variable que más baja el riesgo"),
                                ("RealOddsBajaValor", float(np.exp(baja.iloc[0])), "su razón de odds", 2)])
    latex.registrar_varias([
        ("RealHojaFilas", int(primera["Filas"]), "muestras en la hoja de mayor riesgo"),
        ("RealHojaFraccion", float(primera["Fracción de positivos"]), "fracción de no conformes en esa hoja", 3),
        ("RealHojaVeces", float(primera["Fracción de positivos"] / c.previa), "veces la tasa general", 1),
    ])
    latex.exportar_tabla(odds, "real_odds", "Variables con mayor peso en la logística real", decimales=3)
    latex.exportar_tabla(reglas, "real_reglas_arbol", "Hojas de mayor riesgo del árbol real", decimales=3)
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
# 7. La prueba de 2026: Se congela y se abre una sola vez
# ----------------------------------------------------------------------
def _congelar(c):
    """Guarda el modelo elegido y la decisión completa antes de leer la prueba."""
    elegido = c.ajustados[c.elegido]
    ruta = guardar_modelo(elegido["tuberia"], rutas.DIR_ARTEFACTOS / "modelos" / "riesgo_real.joblib")
    huella = hashlib.sha256(ruta.read_bytes()).hexdigest()
    decision = {
        "congelado": datetime.now().isoformat(timespec="seconds"),
        "modelo": elegido["familia"], "configuracion": elegido["configuracion"],
        "hiperparametros": elegido.get("hiper", {}),
        "predictores": {"numericas": cliente_real.NUMERICAS_INGRESO, "categoricas": cliente_real.CATEGORICAS_INGRESO},
        "transformaciones": "imputación por la mediana, estandarización (solo en la logística), one-hot con niveles "
                            "raros agrupados; todo ajustado con entrenamiento dentro de la tubería",
        "ajustado_con": f"entrenamiento (ingreso antes de {config.CORTE_VALIDACION})",
        "umbral": c.umbral, "costos": {"FP": config.COSTO_FP, "FN": config.COSTO_FN},
        "metrica_principal": "PR-AUC",
        "validacion": {"pr_auc": float(c.resumen_riesgo.loc[c.elegido, "ap_val"]),
                       "roc_auc": float(c.resumen_riesgo.loc[c.elegido, "auc_val"])},
        "referencias": {"regla": "tasa histórica del cliente",
                        "umbral_regla": c.ajustados[_nombre_regla(c)]["umbral"]},
        "tasa_entrenamiento": c.previa,
        "archivo_modelo": str(ruta.relative_to(rutas.DIR_ARTEFACTOS)), "huella_sha256": huella,
    }
    guardar_json(decision, rutas.DIR_ARTEFACTOS / f"{ETAPA[:2]}_congelado.json")
    return decision, ruta


def _nombre_regla(c):
    return next(n for n, a in c.ajustados.items() if a["modelo"] == "regla")


def _metricas_prueba(y, proba, corte):
    m = _fila_metricas(y, proba, corte)
    _, ap_inf, ap_sup = intervalo_bootstrap(metricas.precision_media, y, proba, n=REPLICAS_BOOTSTRAP)
    _, auc_inf, auc_sup = intervalo_bootstrap(metricas.roc_auc, y, proba, n=REPLICAS_BOOTSTRAP)
    return {**m, "PR-AUC inf": ap_inf, "PR-AUC sup": ap_sup, "ROC-AUC inf": auc_inf, "ROC-AUC sup": auc_sup}


def prueba_real(c):
    """Congela la decisión, abre el bloque de 2026 una sola vez y lo compara con la prevalencia."""
    if not c.hay:
        return _sin_datos()
    r = Resultado()
    decision, ruta = _congelar(c)
    r.agregar(tarjetas([
        {"titulo": "Modelo congelado", "valor": decision["validacion"]["pr_auc"], "decimales": 3,
         "prefijo": "PR-AUC ", "detalle": f"{decision['modelo']} ({decision['configuracion']}) en validación"},
        {"titulo": "Umbral", "valor": decision["umbral"], "decimales": 2, "color": "naranja",
         "detalle": f"costo {config.COSTO_FP} FP + {config.COSTO_FN} FN"},
        {"titulo": "Predictores", "valor": len(cliente_real.NUMERICAS_INGRESO) + len(cliente_real.CATEGORICAS_INGRESO),
         "decimales": 0, "color": "morado", "detalle": "conocidos al ingreso"},
        {"titulo": "Huella", "valor": decision["huella_sha256"][:12], "color": "verde",
         "detalle": "SHA-256 del modelo guardado"},
    ], titulo="La decisión que congelamos antes de mirar 2026"))
    # Una sola lectura, registrada en `lecturas_prueba.log`.
    todas = cliente_real.con_particion(rodam.asignar_lotes(rodam.tabla_por_muestra(rodam.leer_con_prueba())))
    tabla, previa = cliente_real.variables_de_ingreso(todas)
    if not np.isclose(previa, c.previa):
        raise RuntimeError("La tasa de entrenamiento cambió al abrir la prueba.")
    prueba = tabla[tabla["particion"] == "prueba"]
    y = prueba["no_conforme"].to_numpy()
    modelo = leer_modelo(ruta)
    if hashlib.sha256(ruta.read_bytes()).hexdigest() != decision["huella_sha256"]:
        raise RuntimeError("El modelo cambió después de congelarlo.")
    prevalencia = float(y.mean())
    regla = c.ajustados[_nombre_regla(c)]
    base = next(a for a in c.ajustados.values() if a["modelo"] == "base")
    probas = {"Referencia (probabilidad de entrenamiento)": (metricas.prediccion_de(base["tuberia"], prueba),
                                                            base["umbral"]),
              "Regla simple (tasa histórica del cliente)": (prueba["tasa_cliente"].to_numpy(dtype=float),
                                                            regla["umbral"]),
              f"{decision['modelo']} ({decision['configuracion']})": (metricas.prediccion_de(modelo, prueba),
                                                                      decision["umbral"])}
    filas = [{"Modelo": nombre, **_metricas_prueba(y, p, corte)} for nombre, (p, corte) in probas.items()]
    t = pd.DataFrame(filas)
    final = t.iloc[-1]
    p_modelo = probas[final["Modelo"]][0]
    clientes_dev = pd.Index(c.tabla_riesgo["cliente"].unique())
    nuevos = ~prueba["cliente"].isin(clientes_dev).to_numpy()
    comp = pd.DataFrame([_composicion(c.ajuste, "Entrenamiento"), _composicion(c.valida, "Validación"),
                         _composicion(prueba, "Prueba", clientes_dev)])
    r.agregar(tabla_html(comp, "Las tres particiones (la prueba con los clientes que no aparecen en el desarrollo)",
                         decimales={"Clientes nuevos": 0, "Muestreo en planta": 3, "Sin historia del cliente": 3,
                                    "No conformidad": 3}))
    m = metricas.metricas_clf(y, p_modelo, decision["umbral"])
    ap_conocidos = metricas.precision_media(y[~nuevos], p_modelo[~nuevos])
    ap_nuevos = metricas.precision_media(y[nuevos], p_modelo[nuevos]) if nuevos.any() else float("nan")
    vista = t[["Modelo", "Umbral", "Precisión", "Recall", "F1", "ROC-AUC"]].assign(
        **{"IC ROC-AUC": [f"{a:.3f} a {b:.3f}" for a, b in zip(t["ROC-AUC inf"], t["ROC-AUC sup"])],
           "PR-AUC": t["PR-AUC"],
           "IC PR-AUC": [f"{a:.3f} a {b:.3f}" for a, b in zip(t["PR-AUC inf"], t["PR-AUC sup"])]})
    r.agregar(tabla_html(vista, f"Prueba de 2026 ({len(y)} muestras, prevalencia {prevalencia:.3f}): Evaluada una "
                                "sola vez con lo congelado e intervalos bootstrap del 95 %",
                         resaltar=(vista.index == len(vista) - 1), columnas_resaltadas="PR-AUC",
                         decimales=DECIMALES_RIESGO))
    r.figura(graficos_clf.figura_confusion(m["TN"], m["FP"], m["FN"], m["TP"],
                                           f"Prueba con umbral {decision['umbral']:.2f} (etiquetas 0 y 1)",
                                           clases=("0 conforme", "1 no conforme"),
                                           acciones=("0 sin alerta", "1 alerta")), "08_confusion_prueba")
    regla_t = t.iloc[1]
    val_ap, val_auc = decision["validacion"]["pr_auc"], decision["validacion"]["roc_auc"]
    val_prev = float(c.y_val.mean())
    veces_val, veces_prueba = val_ap / val_prev, float(final["PR-AUC"]) / prevalencia
    costo_modelo, costo_nada = int(m["Costo"]), config.COSTO_FN * int(y.sum())
    sin_historia = float((prueba["historia_cliente"] == 0).mean())
    se_pisan = final["PR-AUC inf"] <= regla_t["PR-AUC sup"] and regla_t["PR-AUC inf"] <= final["PR-AUC sup"]
    r.agregar(nota(
        f"En 2026 la no conformidad {'bajó' if prevalencia < val_prev else 'subió'} a {pct(prevalencia)} (en validación "
        f"era {pct(val_prev)}) y {int(comp['Clientes nuevos'].iat[2])} de los {int(comp['Clientes'].iat[2])} clientes "
        f"de la prueba no aparecen en el desarrollo. Sus muestras son el {pct(nuevos.mean(), 0)} de la prueba aunque solo "
        f"el {pct(sin_historia)} llega sin ningún dictamen previo porque los clientes nuevos acumulan historia rápido.\n\n"
        f"El modelo congelado tiene un PR-AUC de **{final['PR-AUC']:.3f}** (intervalo de {final['PR-AUC inf']:.3f} a "
        f"{final['PR-AUC sup']:.3f}) frente a {val_ap:.3f} en validación. A primera vista empeoró pero el PR-AUC depende "
        f"de la prevalencia, que es lo que daría ordenar al azar. Medido en veces la prevalencia pasó de "
        f"{veces_val:.1f} en validación a {veces_prueba:.1f} en la prueba y el ROC-AUC pasó de {val_auc:.3f} a "
        f"{final['ROC-AUC']:.3f}, o sea que el orden del riesgo aguantó el cambio. Entre clientes ya conocidos el "
        f"PR-AUC es {ap_conocidos:.3f} y entre los nuevos {ap_nuevos:.3f}. La regla de la tasa histórica queda en "
        f"{regla_t['PR-AUC']:.3f} ({regla_t['PR-AUC inf']:.3f} a {regla_t['PR-AUC sup']:.3f})"
        + (" así que los intervalos se pisan y no podemos decir que el modelo le gane a la regla."
           if se_pisan else
           (" y el modelo le gana con intervalos que no se tocan." if final["PR-AUC"] > regla_t["PR-AUC"] else
            " y en la prueba la regla le gana al modelo."))
        + f"\n\nLo que no aguantó fue el umbral. Con {decision['umbral']:.2f} el modelo alerta {m['TP']} de los "
          f"{int(y.sum())} no conformes (recall {m['Recall']:.3f}) con precisión {m['Precisión']:.3f}. El costo es "
          f"{costo_modelo} frente a {costo_nada} de no alertar nunca"
        + (" así que en 2026 la regla de alerta no ahorra nada. Las probabilidades quedaron calibradas para un mundo con "
           "el doble de no conformidad y casi ninguna pasa el umbral." if costo_modelo >= costo_nada else
           f", un ahorro pequeño del {pct(1 - costo_modelo / costo_nada, 0)}."),
        "hallazgo" if not se_pisan and final["PR-AUC"] > regla_t["PR-AUC"] else "advertencia"))
    r.agregar(nota(
        "Esta es la única vez que leímos el bloque de 2026. La lectura quedó anotada en el registro de la prueba y la "
        "decisión que evaluamos es la que guardamos antes de abrirlo (con la huella del modelo para comprobar que no "
        "cambió). Si quisiéramos ajustar algo con lo que vimos aquí necesitaríamos un periodo nuevo para medirlo.",
        "limite"))
    c.prueba = {"muestras": len(y), "prevalencia": prevalencia, "tabla": t.to_dict("records"),
                "clientes_nuevos": int(comp["Clientes nuevos"].iat[2]), "huella": decision["huella_sha256"]}
    latex.registrar_varias([
        ("RealCongeladoModelo", f"{decision['modelo']} ({decision['configuracion']})", "modelo congelado"),
        ("RealCongeladoHuella", decision["huella_sha256"][:12], "primeros caracteres de la huella del modelo"),
        ("RealPruebaMuestras", len(y), "muestras con dictamen en la prueba de 2026"),
        ("RealPruebaClientes", int(comp["Clientes"].iat[2]), "clientes en la prueba"),
        ("RealPruebaClientesNuevos", int(comp["Clientes nuevos"].iat[2]), "clientes de la prueba sin desarrollo"),
        ("RealPruebaNuevasPct", 100 * float(nuevos.mean()), "muestras de prueba de clientes nuevos (%)", 0),
        ("RealPruebaPrevalenciaPct", 100 * prevalencia, "no conformidad en la prueba (%)", 1),
        ("RealPruebaPrevalencia", prevalencia, "prevalencia de la prueba", 3),
        ("RealPruebaAP", float(final["PR-AUC"]), "PR-AUC del modelo en la prueba", 3),
        ("RealPruebaAPInf", float(final["PR-AUC inf"]), "límite inferior del PR-AUC en la prueba", 3),
        ("RealPruebaAPSup", float(final["PR-AUC sup"]), "límite superior del PR-AUC en la prueba", 3),
        ("RealPruebaAUC", float(final["ROC-AUC"]), "ROC-AUC del modelo en la prueba", 3),
        ("RealPruebaAUCInf", float(final["ROC-AUC inf"]), "límite inferior del ROC-AUC en la prueba", 3),
        ("RealPruebaAUCSup", float(final["ROC-AUC sup"]), "límite superior del ROC-AUC en la prueba", 3),
        ("RealPruebaVeces", float(final["PR-AUC"] / prevalencia), "PR-AUC sobre la prevalencia", 1),
        ("RealPruebaAPRegla", float(regla_t["PR-AUC"]), "PR-AUC de la regla en la prueba", 3),
        ("RealPruebaAPReglaInf", float(regla_t["PR-AUC inf"]), "límite inferior del PR-AUC de la regla", 3),
        ("RealPruebaAPReglaSup", float(regla_t["PR-AUC sup"]), "límite superior del PR-AUC de la regla", 3),
        ("RealPruebaAPConocidos", ap_conocidos, "PR-AUC en clientes conocidos", 3),
        ("RealPruebaAPNuevos", ap_nuevos, "PR-AUC en clientes nuevos", 3),
        ("RealPruebaPrecision", m["Precisión"], "precisión en la prueba con el umbral fijo", 3),
        ("RealPruebaRecall", m["Recall"], "recall en la prueba con el umbral fijo", 3),
        ("RealPruebaFUno", m["F1"], "F1 en la prueba con el umbral fijo", 3),
        ("RealPruebaTP", m["TP"], "verdaderos positivos en la prueba"),
        ("RealPruebaFP", m["FP"], "falsos positivos en la prueba"),
        ("RealPruebaFN", m["FN"], "falsos negativos en la prueba"),
        ("RealPruebaTN", m["TN"], "verdaderos negativos en la prueba"),
        ("RealPruebaPositivos", int(y.sum()), "no conformes en la prueba"),
        ("RealPruebaSinHistoriaPct", 100 * sin_historia, "muestras de prueba sin dictamen previo del cliente (%)", 1),
        ("RealValidacionVeces", veces_val, "PR-AUC de validación sobre su prevalencia", 1),
        ("RealPruebaCosto", costo_modelo, "costo en la prueba con el umbral fijo"),
        ("RealPruebaCostoNada", costo_nada, "costo en la prueba de no alertar nunca"),
    ])
    latex.exportar_tabla(vista, "real_prueba", "Prueba de 2026 con lo congelado", decimales=DECIMALES_RIESGO)
    latex.exportar_tabla(comp, "real_riesgo_particiones", "Particiones del clasificador real", decimales=3)
    return r


# ----------------------------------------------------------------------
# 8. Cierre
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
            ("Riesgo al ingreso", f"{c.elegido} con PR-AUC "
                                  f"{c.resumen_riesgo.loc[c.elegido, 'ap_val']:.3f} en validación y umbral "
                                  f"{c.umbral:.2f} por costo."),
            ("Prueba de 2026", f"PR-AUC {c.prueba['tabla'][-1]['PR-AUC']:.3f} con prevalencia "
                               f"{c.prueba['prevalencia']:.3f}, evaluada una sola vez."),
        ], titulo="Qué deja la etapa"))
        c.resumen.update({
            "base": c.base, "dispersion_todos": c.dispersion_todos,
            "llegadas": c.tabla_llegadas.to_dict("records"),
            "tamano_lote": c.ajustes, "campanas": c.campanas_resumen,
            "segmentos": {"k": c.segmentos["k"], "silueta": c.segmentos["silueta"], "clientes": c.segmentos["clientes"],
                          "nombres": list(c.segmentos["nombres"].values())},
            "riesgo": {"previa": c.previa, "entrenamiento": len(c.ajuste), "validacion": len(c.valida),
                       "elegido": c.elegido, "umbral": c.umbral,
                       "configuraciones": c.resumen_riesgo.reset_index().to_dict("records")},
            "prueba": c.prueba,
        })
    else:
        r.agregar(nota(SIN_DATOS, "limite", titulo="Sin la exportación real"))
    c.resumen["segundos"] = time.perf_counter() - c.inicio
    guardar_json(c.resumen, rutas.DIR_ARTEFACTOS / f"{ETAPA}.json")
    cerrar({"hay_exportacion": c.hay, "segundos": c.resumen["segundos"]})
    return r
