"""Capa 8 · Etapa 05: agrupamiento.

Responsabilidad única: armar las piezas de la etapa 05 (reglas fijadas de
antemano, preparación y escalado, K-means, DBSCAN, ruido, comparación de
particiones, perfiles con nombre, segmentos de clientes, auditoría con la
simulación y comparación con la silueta del cribado), guardar el artefacto de
la etapa y registrar sus cifras para el informe. Las funciones de sección
reciben el análisis (`c.sel` o `c.comp`) para servir a los dos análisis.
"""
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from proyecto.base import config, rutas
from proyecto.base.artefactos import guardar_json, leer_json, leer_modelado
from proyecto.cribado import analisis as analisis_cribado
from proyecto.cribado import catalogo
from proyecto.datos import carga
from proyecto.etapas import presentacion
from proyecto.etapas.comun import ES_SIMULACION, Resultado, abrir, cerrar, enumerar, lista_nombres
from proyecto.modelos import agrupamiento
from proyecto.reporte import graficos_agrupamiento as graficos
from proyecto.reporte import latex, segmentos
from proyecto.reporte.html import barras_animadas, nota, pasos, tabla_html, tarjetas
from proyecto.reporte.segmentos import formato_ari
from proyecto.variables.comunes import columnas_de_grupo

ETAPA = "05_agrupamiento"
# Reglas fijadas antes de mirar resultados.
VALORES_K = list(range(2, 11))
TOLERANCIA_SILUETA = 0.02          # diferencia de silueta que se discute (sin cambiar la regla)
SEMILLAS_ESTABILIDAD = [config.SEMILLA + i for i in range(1, 6)]
RUIDO_MAXIMO = 0.15                # DBSCAN: proporción de ruido admitida para una solución con grupos
RUIDO_RESPALDO = 0.10              # DBSCAN sin dos grupos: ruido admitido para leerlo como detector de atípicos
CUANTILES_EPS = np.linspace(0.60, 0.98, 8)
ASIMETRIA_LOG = 1.0                # log1p a variables de conteo (≥ 0, máximo > 1) con asimetría mayor
UMBRAL_NOMBRE = 0.5                # desviaciones estándar para que una variable entre en el nombre del grupo
VARIABLES_NOMBRE = 2
FILAS_RUIDO = 8


def _prefijo(a):
    """Prefijo de macro según el análisis: "AgrSel" para el seleccionado y "AgrComp" para el complementario."""
    return "AgrSel" if a["papel"] == "seleccionado" else "AgrComp"


def iniciar():
    c = abrir(ETAPA, "Grupos sin etiqueta: Cómo se eligen, qué tan estables son y qué significan")
    c.inicio = time.perf_counter()
    c.cribado = leer_json(rutas.DIR_ARTEFACTOS / "03_cribado.json")
    c.seleccionado = c.cribado["seleccion"]["agrupamiento"]
    c.complementario = c.cribado.get("agrupamiento_complementario")
    if c.seleccionado is None:
        raise RuntimeError("La etapa 03 no eligió ningún análisis de agrupamiento: No hay nada que agrupar.")
    if c.complementario == c.seleccionado:
        c.complementario = None
    c.sel, c.comp, c.auditoria = None, None, None
    return c


def reglas(c):
    r = Resultado()
    ficha = catalogo.ficha_por_nombre(c.seleccionado)
    lineas = [("Análisis seleccionado", f"\"{c.seleccionado}\" ({ficha.titulo}), el que eligió la regla de la etapa 03.")]
    if c.complementario:
        comp = catalogo.ficha_por_nombre(c.complementario)
        lineas.append(("Análisis complementario", f"\"{c.complementario}\" ({comp.titulo}), por fuera de la regla que "
                                                  "fijamos de antemano. Lo incluimos porque al laboratorio le interesa "
                                                  "entender a sus clientes."))
    r.agregar(pasos([
        *lineas,
        ("Preparación", "Las variables de los papeles que guardamos en la etapa 03. Las de conteo (no negativas y con "
                        f"máximo mayor que 1) con asimetría mayor que {ASIMETRIA_LOG} pasan por log1p y después todo se "
                        "estandariza."),
        ("K-means", f"k de {VALORES_K[0]} a {VALORES_K[-1]} y gana el k de mayor silueta. Si gana k = 2 y algún k ≥ 3 queda a "
                    f"menos de {TOLERANCIA_SILUETA} de silueta lo discutimos pero la regla se mantiene. La estabilidad se "
                    f"mide con {len(SEMILLAS_ESTABILIDAD)} semillas más."),
        ("DBSCAN", "min_samples en {d, 2d} (d = número de variables) y eps en cuantiles de la curva de k-distancias. Gana "
                   "la mayor silueta entre las soluciones con dos o más grupos y ruido de a lo sumo "
                   f"{100 * RUIDO_MAXIMO:.0f} %. Si ninguna califica, DBSCAN se usa para detectar casos atípicos con el "
                   f"menor eps cuyo ruido no pase del {100 * RUIDO_RESPALDO:.0f} %."),
        ("Nombres de los grupos", f"Las {VARIABLES_NOMBRE} variables cuya media en el grupo más se aleja de la media "
                                  f"general, si se alejan al menos {UMBRAL_NOMBRE} desviaciones estándar (DE)."),
    ], titulo="Reglas de la etapa"))
    latex.registrar_varias([
        ("AgrSeleccionado", c.seleccionado, "análisis de agrupamiento seleccionado por la regla"),
        ("AgrComplementario", c.complementario or "ninguno", "análisis de agrupamiento complementario"),
        ("AgrKMin", VALORES_K[0], "menor k explorado en K-means"),
        ("AgrKMax", VALORES_K[-1], "mayor k explorado en K-means"),
        ("AgrToleranciaSilueta", TOLERANCIA_SILUETA, "diferencia de silueta que se discute", 2),
        ("AgrSemillasEstabilidad", len(SEMILLAS_ESTABILIDAD), "semillas adicionales para la estabilidad"),
        ("AgrRuidoMaximoPct", 100 * RUIDO_MAXIMO, "ruido máximo (%) de una solución DBSCAN con grupos", 0),
        ("AgrRuidoRespaldoPct", 100 * RUIDO_RESPALDO, "ruido máximo (%) de DBSCAN como detector de atípicos", 0),
        ("AgrUmbralNombre", UMBRAL_NOMBRE, "desviaciones estándar para nombrar un grupo", 1),
    ])
    return r


def _preparar(nombre, papel):
    papeles = leer_json(rutas.DIR_MODELADO / f"{nombre}.json")
    tabla = leer_modelado(nombre).reset_index(drop=True)
    numericas, categoricas = columnas_de_grupo(papeles, "ampliado")
    a = agrupamiento.analizar_agrupamiento(
        tabla, numericas, categoricas, valores_k=VALORES_K, tolerancia=TOLERANCIA_SILUETA,
        semillas=SEMILLAS_ESTABILIDAD, cuantiles_eps=CUANTILES_EPS, ruido_maximo=RUIDO_MAXIMO,
        ruido_respaldo=RUIDO_RESPALDO, asimetria=ASIMETRIA_LOG)
    nombres, descripciones, sugerencias = segmentos.nombrar_grupos(a["perfil_z"], UMBRAL_NOMBRE, VARIABLES_NOMBRE)
    a.update({"nombre": nombre, "papel": papel, "papeles": papeles, "nombres": nombres, "descripciones": descripciones,
              "sugerencias": sugerencias})
    return a


def preparacion(c, papel="seleccionado"):
    """Prepara el análisis (`papel` = "seleccionado" o "complementario"): Tarjetas, distribución de la varianza
    sin escalar y lectura del escalado y de log1p."""
    r = Resultado()
    if papel == "complementario" and not c.complementario:
        return r.agregar(nota("La etapa 03 no dejó un análisis complementario de agrupamiento.", "limite"))
    a = _preparar(c.seleccionado if papel == "seleccionado" else c.complementario, papel)
    setattr(c, "sel" if papel == "seleccionado" else "comp", a)
    r.agregar(tarjetas([
        {"titulo": "Filas", "valor": len(a["tabla"]), "decimales": 0, "detalle": a["papeles"]["unidad"]},
        {"titulo": "Variables", "valor": a["matriz"].shape[1], "decimales": 0, "color": "morado",
         "detalle": f"{len(a['con_log'])} con log1p antes de escalar"},
        {"titulo": "Hopkins (matriz escalada)", "valor": a["hopkins"], "decimales": 2, "color": "verde",
         "detalle": "0.5 es sin estructura y cerca de 1 hay grupos"},
    ]))
    r.agregar(barras_animadas({segmentos.etiqueta(x): 100 * v for x, v in a["reparto"].items()},
                              "Parte de la varianza total que aporta cada variable sin escalar", formato="{:.1f} %",
                              resaltar=segmentos.etiqueta(a["dominante"]), color="naranja"))
    con_log = lista_nombres(segmentos.etiqueta(x) for x in a["con_log"])
    if papel == "seleccionado":
        r.agregar(nota(
            f"Sin escalar \"{segmentos.etiqueta(a['dominante'])}\" aporta el **{100 * a['reparto'].iloc[0]:.1f} %** de "
            "la varianza total así que decidiría casi sola las distancias. Para comprobarlo ajustamos K-means con el mismo "
            "k sobre las variables sin escalar y el índice de Rand ajustado contra la solución escalada dio "
            f"{formato_ari(a['ari_sin_escalar'])}. O sea que sale otra partición. "
            + (f"Antes de escalar pasamos por log1p {con_log}, variables de conteo con una cola larga a la derecha. Sin la "
               "transformación unos pocos valores extremos serían los que mandan en la distancia."
               if a["con_log"] else "Ninguna variable cumplió la regla de log1p."),
            "info"))
    else:
        r.agregar(nota(
            f"Sin escalar \"{segmentos.etiqueta(a['dominante'])}\" aporta el **{100 * a['reparto'].iloc[0]:.1f} %** de "
            "la varianza total y el índice de Rand ajustado entre K-means sin escalar y la solución escalada es de "
            f"{formato_ari(a['ari_sin_escalar'])}. " + (f"Pasaron por log1p {con_log}. " if a["con_log"] else "")
            + "El volumen mensual ya llega en escala logarítmica desde la etapa 03 y las proporciones no se transforman.",
            "info"))
    p = _prefijo(a)
    latex.registrar_varias([
        (f"{p}Filas", len(a["tabla"]), f"filas del análisis {papel}"),
        (f"{p}Variables", int(a["matriz"].shape[1]), f"variables de la matriz del análisis {papel}"),
        (f"{p}ConLog", len(a["con_log"]), f"variables con log1p en el análisis {papel}"),
        (f"{p}Hopkins", float(a["hopkins"]), f"Hopkins de la matriz escalada ({papel})", 3),
        (f"{p}Dominante", segmentos.etiqueta(a["dominante"]), f"variable que domina la varianza sin escalar ({papel})"),
        (f"{p}DominantePct", 100 * float(a["reparto"].iloc[0]), f"varianza (%) que aporta sin escalar ({papel})", 1),
        (f"{p}ARISinEscalar", float(a["ari_sin_escalar"]), f"ARI entre K-means sin escalar y escalado ({papel})", 3),
    ])
    return r


def kmeans(c, papel="seleccionado"):
    r = Resultado()
    a = c.sel if papel == "seleccionado" else c.comp
    if a is None:
        return r
    r.figura(graficos.figura_codo_silueta(a["tabla_k"], a["k"], f'K-means en "{a["nombre"]}"'),
             f"05_{a['nombre']}_codo_silueta")
    siluetas = a["tabla_k"].set_index("k")["silueta"]
    texto = (f"La silueta más alta sale con **k = {a['k']}** ({a['silueta_k']:.3f}) y por la regla ese es el número de "
             f"grupos de K-means. En toda la exploración la silueta va de {siluetas.min():.3f} a {siluetas.max():.3f}, ")
    texto += ("valores bajos que hablan de grupos con fronteras difusas más que de nubes bien separadas."
              if siluetas.max() < 0.25 else "o sea que los grupos tienen una separación razonable.")
    if a["k_alternativo"] is not None:
        alt = a["k_alternativo"]
        texto += (f"\n\nCon k = {alt} la silueta queda a menos de {TOLERANCIA_SILUETA} ({siluetas[alt]:.3f}). Esa "
                  "solución daría una lectura más fina a cambio de grupos algo menos separados, pero nos quedamos con la "
                  "regla.")
    estab = a["estabilidad"]
    texto += (f"\n\nAl repetir el ajuste con {len(estab)} semillas distintas el índice de Rand ajustado contra la solución "
              f"de referencia queda entre {formato_ari(estab.min())} y {formato_ari(estab.max())} (1 es la misma "
              "partición). ")
    texto += ("O sea que los grupos no dependen del punto de partida." if estab.min() >= 0.9 else
              "O sea que una parte de las asignaciones cambia con el punto de partida y los grupos hay que leerlos con "
              "cautela.")
    r.agregar(nota(texto, "hallazgo"))
    p = _prefijo(a)
    latex.registrar_varias([
        (f"{p}K", int(a["k"]), f"k elegido por silueta ({papel})"),
        (f"{p}Silueta", float(a["silueta_k"]), f"silueta de K-means con el k elegido ({papel})", 3),
        (f"{p}SiluetaMin", float(siluetas.min()), f"silueta mínima de la exploración ({papel})", 3),
        (f"{p}SiluetaMax", float(siluetas.max()), f"silueta máxima de la exploración ({papel})", 3),
        (f"{p}EstabilidadMin", float(estab.min()), f"ARI mínimo entre semillas ({papel})", 3),
        (f"{p}EstabilidadMax", float(estab.max()), f"ARI máximo entre semillas ({papel})", 3),
    ])
    latex.exportar_tabla(a["tabla_k"], f"agr_{'sel' if papel == 'seleccionado' else 'comp'}_kmeans",
                         f"Inercia y silueta por k ({a['nombre']})", decimales=3)
    return r


def dbscan(c, papel="seleccionado"):
    """Curva de k-distancias, rejilla, elección y casos de ruido."""
    r = Resultado()
    a = c.sel if papel == "seleccionado" else c.comp
    if a is None:
        return r
    tabla_db, fila, nombre = a["tabla_db"], a["fila_db"], a["nombre"]
    min_muestras = int(fila["min_muestras"]) if fila is not None else min(a["curvas"])
    eps = float(fila["eps"]) if fila is not None else float(np.median(a["rejillas"][min_muestras]))
    r.figura(graficos.figura_k_distancia(a["curvas"][min_muestras], min_muestras, eps, a["rejillas"][min_muestras]),
             f"05_{nombre}_k_distancia")
    vista = tabla_db.rename(columns={"min_muestras": "min_samples", "grupos": "Grupos", "proporcion_ruido": "Ruido",
                                     "silueta": "Silueta (sin ruido)"})
    resaltar = None
    if fila is not None:
        resaltar = ((tabla_db["eps"] == fila["eps"]) & (tabla_db["min_muestras"] == fila["min_muestras"])).to_numpy()
    r.agregar(tabla_html(vista, f'Rejilla de DBSCAN para "{nombre}"', resaltar=resaltar,
                         decimales={"eps": 2, "Ruido": 3, "Silueta (sin ruido)": 3}, maximo_filas=40))
    con_grupos = int((tabla_db["grupos"] >= 2).sum())
    if a["criterio_db"] == "regla":
        texto = (f"La regla elige **eps = {fila['eps']:.2f}** con **min_samples = {int(fila['min_muestras'])}**, que deja "
                 f"{int(fila['grupos'])} grupos, {100 * fila['proporcion_ruido']:.1f} % de ruido y una silueta de "
                 f"{fila['silueta']:.3f} (calculada sin los puntos de ruido). De las {len(tabla_db)} combinaciones "
                 f"{con_grupos} dan dos o más grupos.")
        tipo = "hallazgo"
    elif a["criterio_db"] == "respaldo":
        texto = (f"Ninguna combinación da dos o más grupos con a lo sumo {100 * RUIDO_MAXIMO:.0f} % de ruido así que "
                 "DBSCAN no encuentra grupos separados por zonas de baja densidad: Ve una sola masa densa. Por el respaldo "
                 f"que fijamos lo usamos para detectar casos atípicos, con **eps = {fila['eps']:.2f}** y "
                 f"**min_samples = {int(fila['min_muestras'])}**, que deja {int(fila['grupos'])} grupo y "
                 f"{100 * fila['proporcion_ruido']:.1f} % de los casos como ruido.")
        tipo = "advertencia"
    else:
        texto = "Ninguna combinación de la rejilla cumple la regla ni el respaldo así que DBSCAN no se usa en este análisis."
        tipo = "limite"
    r.agregar(nota(texto, tipo))
    p = _prefijo(a)
    latex.registrar_varias([
        (f"{p}DBCriterio", a["criterio_db"], f"criterio con que se eligió DBSCAN ({papel})"),
        (f"{p}DBCombinaciones", len(tabla_db), f"combinaciones de la rejilla de DBSCAN ({papel})"),
        (f"{p}DBConGrupos", con_grupos, f"combinaciones de DBSCAN con dos o más grupos ({papel})"),
    ])
    if fila is not None:
        latex.registrar_varias([
            (f"{p}DBEps", float(fila["eps"]), f"eps elegido de DBSCAN ({papel})", 2),
            (f"{p}DBMinMuestras", int(fila["min_muestras"]), f"min_samples elegido de DBSCAN ({papel})"),
            (f"{p}DBGrupos", int(fila["grupos"]), f"grupos de DBSCAN ({papel})"),
            (f"{p}DBRuidoPct", 100 * float(fila["proporcion_ruido"]), f"casos de ruido (%) de DBSCAN ({papel})", 1),
            (f"{p}DBSilueta", float(fila["silueta"]), f"silueta sin ruido de DBSCAN ({papel})", 3),
        ])

    # Casos de ruido, con columnas descriptivas y la variable en que más se apartan.
    etiquetas_db = a["etiquetas_db"]
    if etiquetas_db is not None and (etiquetas_db == -1).any():
        extra = ["matriz", "ensayo"] if papel == "seleccionado" else ["muestras"]
        identificadores = [x for x in [*a["papeles"].get("identificadores", []), *extra] if x in a["tabla"].columns]
        vista_r, a["ruido_variables"] = segmentos.vista_ruido(a["tabla"], etiquetas_db, a["numericas"], identificadores,
                                                              FILAS_RUIDO)
        r.agregar(tabla_html(vista_r, f'Casos que DBSCAN marca como ruido en "{nombre}" (los más apartados)',
                             decimales=2))
        if papel == "seleccionado":
            frec = a["ruido_variables"]
            r.agregar(nota(
                f"Entre los {int(frec.sum())} casos de ruido la variable en que más se apartan suele ser "
                f"\"{frec.index[0]}\" ({int(frec.iloc[0])} casos), seguida de \"{frec.index[1]}\" "
                f"({int(frec.iloc[1])} casos)." if len(frec) > 1 else
                f"Los {int(frec.sum())} casos de ruido se apartan sobre todo en \"{frec.index[0]}\".", "info"))
    return r


def comparar_particiones(c, papel="seleccionado"):
    """Proyección PCA de K-means y DBSCAN e índice de Rand ajustado entre ambas."""
    r = Resultado()
    a = c.sel if papel == "seleccionado" else c.comp
    if a is None:
        return r
    soluciones = {f"K-means (k = {a['k']})": (a["etiquetas_km"], a["nombres"])}
    if a["etiquetas_db"] is not None:
        soluciones["DBSCAN"] = (a["etiquetas_db"], None)
    if papel == "seleccionado" and a["etiquetas_db"] is None:
        r.figura(graficos.figura_pca_grupos(a["pca"], a["etiquetas_km"], a["pca_varianza"], a["nombres"]),
                 f"05_{a['nombre']}_pca")
    else:
        r.figura(graficos.figura_pca_comparada(a["pca"], soluciones, a["pca_varianza"]), f"05_{a['nombre']}_pca")
    varianza = 100 * a["pca_varianza"].sum()
    if papel == "seleccionado":
        texto = (f"Las dos componentes explican el {varianza:.1f} % de la varianza de la matriz escalada, "
                 + ("o sea que el dibujo es una sombra parcial de los grupos. Que se vean encimados en el plano no quiere "
                    f"decir que se encimen en las {a['matriz'].shape[1]} dimensiones." if varianza < 60 else
                    "así que el dibujo representa bien la estructura."))
        if a["ari_km_db"] is not None:
            grupos = int(a["fila_db"]["grupos"])
            texto += (f"\n\nEl índice de Rand ajustado entre K-means y DBSCAN es de {formato_ari(a['ari_km_db'])} con todos "
                      f"los puntos (el ruido como un grupo más) y de {formato_ari(a['ari_km_db_sin_ruido'])} sin los puntos "
                      f"de ruido. K-means divide todas las muestras en {a['k']} grupos redondeados aunque la frontera pase "
                      "por zonas densas. DBSCAN en cambio solo separa donde la densidad cae (aquí deja "
                      f"{grupos} {'grupo' if grupos == 1 else 'grupos'}) y aparta los casos aislados.")
    else:
        texto = f"Las dos componentes explican el {varianza:.1f} % de la varianza de la matriz escalada."
        if a["ari_km_db"] is not None:
            texto += (f" El índice de Rand ajustado entre K-means y DBSCAN es de {formato_ari(a['ari_km_db'])}"
                      + (" porque DBSCAN no separa grupos y solo aparta a los clientes atípicos. Las dos técnicas "
                         "responden preguntas distintas: Qué tipos de cliente hay y qué clientes no se parecen a "
                         "nadie." if a["criterio_db"] == "respaldo" else
                         f" (y de {formato_ari(a['ari_km_db_sin_ruido'])} sin el ruido)."))
    r.agregar(nota(texto, "hallazgo"))
    p = _prefijo(a)
    latex.registrar(f"{p}PCAVarianzaPct", varianza, f"varianza (%) que explican las dos componentes ({papel})", 1)
    if a["ari_km_db"] is not None:
        latex.registrar(f"{p}ARIKmDb", float(a["ari_km_db"]), f"ARI entre K-means y DBSCAN con ruido ({papel})", 3)
        if a.get("ari_km_db_sin_ruido") is not None:
            latex.registrar(f"{p}ARIKmDbSinRuido", float(a["ari_km_db_sin_ruido"]),
                            f"ARI entre K-means y DBSCAN sin ruido ({papel})", 3)
    return r


def perfiles(c, papel="seleccionado"):
    """Perfil estandarizado con nombres, mapa de calor, medias y lectura (sugerencias)."""
    r = Resultado()
    a = c.sel if papel == "seleccionado" else c.comp
    if a is None:
        return r
    extras = ("matriz", "ensayo", "sector") if papel == "seleccionado" else ()
    tabla, numericas = a["tabla"], a["numericas"]
    columnas = [*numericas, *[x for x in extras if x in tabla.columns]]
    perfil, _ = agrupamiento.perfil_estandarizado(tabla, a["etiquetas_km"], columnas)
    a["perfil"] = perfil
    nombres, descripciones = a["nombres"], a["descripciones"]
    filas_calor = a["perfil_z"].rename(index=nombres)
    if a["etiquetas_db"] is not None and (a["etiquetas_db"] == -1).any():
        _, z_db = agrupamiento.perfil_estandarizado(tabla, a["etiquetas_db"], numericas)
        filas_calor = pd.concat([filas_calor, z_db.loc[[-1]].rename(index={-1: "Ruido de DBSCAN"})])
    r.figura(graficos.figura_perfil_calor(filas_calor, segmentos.ETIQUETAS, f"Perfil de los grupos de K-means en "
                                          f'"{a["nombre"]}"'), f"05_{a['nombre']}_perfiles")
    resumen = pd.DataFrame({"Grupo": [nombres[g] for g in perfil.index], "n": perfil["n"].to_numpy(),
                            "Proporción": perfil["proporcion"].to_numpy(),
                            "Variables que lo distinguen": [descripciones[g] for g in perfil.index]})
    for extra in extras:
        if extra in perfil.columns:
            resumen[f"{extra.capitalize()} más frecuente"] = perfil[extra].to_numpy()
    r.agregar(tabla_html(resumen, f'Grupos de K-means en "{a["nombre"]}", con el nombre que sale de su perfil',
                         decimales={"Proporción": 3}))
    medias = perfil[numericas].rename(index=nombres).rename(columns=segmentos.etiqueta).T.reset_index(names="Variable")
    r.agregar(tabla_html(medias, "Medias de cada grupo en las variables originales (sin escalar ni transformar)",
                         decimales=2))
    corto = "sel" if papel == "seleccionado" else "comp"
    latex.exportar_tabla(resumen, f"agr_{corto}_grupos", f"Grupos de K-means en {a['nombre']} con su nombre",
                         decimales={"Proporción": 3})
    latex.exportar_tabla(medias, f"agr_{corto}_medias", f"Medias por grupo en {a['nombre']} (variables originales)",
                         decimales=2)
    if papel == "seleccionado":
        lineas = [(nombres[g], f"{int(perfil.loc[g, 'n'])} muestras ({100 * perfil.loc[g, 'proporcion']:.1f} %). "
                               f"Lo distinguen: {descripciones[g][:1].upper() + descripciones[g][1:]}."
                               + (f" Se podría {a['sugerencias'][g][0]}." if a["sugerencias"][g] else ""))
                  for g in perfil.index]
        r.agregar(pasos(lineas, titulo="Lectura de los perfiles de muestra (sugerencias y no hechos comprobados)"))
        latex.registrar("AgrSelGrupos", ", ".join(nombres[g] for g in perfil.index), "nombres de los perfiles de muestra")
        return r

    t = segmentos.tabla_segmentos(tabla, a["etiquetas_km"], nombres)
    a["segmentos_tabla"] = t
    r.agregar(tabla_html(t, "Tamaño y desempeño de cada segmento (tasas ponderadas por el número de muestras)",
                         decimales=3, barras="Proporción de muestras" if "Proporción de muestras" in t.columns else None))
    latex.exportar_tabla(t, "agr_comp_segmentos", "Tamaño y desempeño de cada segmento de clientes", decimales=3)
    lineas = []
    for g, fila in zip(perfil.index, t.itertuples(index=False)):
        detalle = f"{fila.Clientes} clientes. Lo distinguen: {descripciones[g][:1].upper() + descripciones[g][1:]}."
        if a["sugerencias"][g]:
            detalle += " El laboratorio podría " + enumerar(a["sugerencias"][g][:2]) + "."
        lineas.append((nombres[g], detalle))
    r.agregar(pasos(lineas, titulo="Lectura comercial de cada segmento (sugerencias y no hechos comprobados)"))
    latex.registrar("AgrCompGrupos", ", ".join(nombres[g] for g in perfil.index), "nombres de los segmentos de clientes")
    if "Proporción de muestras" in t.columns:
        mayor = t.loc[t["Proporción de muestras"].idxmax()]
        texto = (f"El segmento \"{mayor['Segmento']}\" reúne el {100 * mayor['Proporción de clientes']:.1f} % de los "
                 f"clientes y el {100 * mayor['Proporción de muestras']:.1f} % de las muestras")
        latex.registrar_varias([
            ("AgrCompSegmentoMayor", mayor["Segmento"], "segmento de clientes con más muestras"),
            ("AgrCompSegmentoMayorClientesPct", 100 * float(mayor["Proporción de clientes"]), "sus clientes (%)", 1),
            ("AgrCompSegmentoMayorMuestrasPct", 100 * float(mayor["Proporción de muestras"]), "sus muestras (%)", 1),
        ])
        if "No conformidad" in t.columns:
            peor, mejor = t.loc[t["No conformidad"].idxmax()], t.loc[t["No conformidad"].idxmin()]
            texto += (f". La no conformidad ponderada va de {mejor['No conformidad']:.3f} en "
                      f"\"{mejor['Segmento']}\" a {peor['No conformidad']:.3f} en \"{peor['Segmento']}\"")
        r.agregar(nota(texto + ". Las sugerencias de la lista salen de una tabla de reglas que escribimos de antemano "
                       "(variable que distingue al segmento → acción posible) y habría que contrastarlas con el área "
                       "comercial antes de aplicarlas.", "hallazgo"))
    return r


def auditoria(c):
    """Con datos simulados, compara los segmentos de clientes con el segmento latente."""
    r = Resultado()
    a = c.comp
    if a is None:
        return r
    verdad = carga.leer_verdad()
    if verdad is None or "segmento_latente" not in verdad["clientes"].columns:
        return r.agregar(nota("No hay verdad oculta (datos reales): La auditoría de los segmentos no aplica.", "simulacion"))
    latente = verdad["clientes"][["id_cliente", "segmento_latente"]]
    auditados = a["tabla"][["id_cliente"]].assign(
        segmento=[a["nombres"][g] for g in a["etiquetas_km"]],
        dbscan=a["etiquetas_db"] if a["etiquetas_db"] is not None else -2,
    ).merge(latente, on="id_cliente", how="inner")
    contingencia, ari_km, pureza = agrupamiento.acuerdo_con_referencia(auditados["segmento_latente"], auditados["segmento"])
    ari_db = (agrupamiento.acuerdo_con_referencia(auditados["segmento_latente"], auditados["dbscan"])[1]
              if a["etiquetas_db"] is not None else None)
    c.auditoria = au = {"clientes": int(len(auditados)), "segmentos_latentes": int(auditados["segmento_latente"].nunique()),
                        "ari_kmeans": ari_km, "ari_dbscan": ari_db, "pureza": pureza,
                        "contingencia": contingencia.to_dict()}
    r.agregar(tabla_html(contingencia.reset_index(names="Segmento latente (simulación)"),
                         "Clientes por segmento latente de la simulación y segmento encontrado por K-means", decimales=0))
    r.agregar(nota(
        f"Con {au['clientes']} clientes y {au['segmentos_latentes']} segmentos latentes el índice de Rand ajustado entre el "
        f"segmento latente y el de K-means es de **{formato_ari(ari_km)}**"
        + (f" (con DBSCAN es {formato_ari(ari_db)})" if ari_db is not None else "")
        + f". La pureza, o sea la proporción de clientes que caen en el segmento latente que más se repite en su grupo, es "
        f"de {pureza:.3f}. Esta comprobación existe solo porque los datos son simulados. Que la recuperación sea parcial es "
        "lo esperable por cómo está hecha la simulación: Los segmentos latentes generan comportamientos con mucho ruido "
        f"que se enciman en las variables observables. Además K-means con la regla de la silueta eligió {a['k']} grupos y "
        "no el número de segmentos latentes.",
        "simulacion"))
    latex.registrar_varias([
        ("AgrAudClientes", au["clientes"], "clientes auditados contra el segmento latente"),
        ("AgrAudSegmentosLatentes", au["segmentos_latentes"], "segmentos latentes de la simulación"),
        ("AgrAudARIKmeans", float(ari_km), "ARI entre el segmento latente y K-means", 3),
        ("AgrAudPureza", float(pureza), "pureza de los segmentos de K-means", 3),
    ])
    if ari_db is not None:
        latex.registrar("AgrAudARIDbscan", float(ari_db), "ARI entre el segmento latente y DBSCAN", 3)
    return r


def silueta_cribado(c):
    """Separa el efecto de las filas y del preprocesamiento entre la silueta del cribado y la de esta etapa."""
    r = Resultado()
    filas_cribado = {f["analisis"]: f for f in c.cribado["tabla_cribado"]}
    filas = []
    for a in [x for x in (c.sel, c.comp) if x]:
        fila = filas_cribado[a["nombre"]]
        sin_log = agrupamiento.explorar_kmeans(
            agrupamiento.preparar_matrices(a["tabla"], a["numericas"], a["categoricas"], asimetria=np.inf)[0], VALORES_K)
        mejor = sin_log.loc[sin_log["silueta"].idxmax()]
        puntaje = analisis_cribado.aplicar_regla({**fila, "silueta": a["silueta_k"], "hopkins": a["hopkins"],
                                                  "filas": len(a["tabla"])})[2]
        filas.append({
            "Análisis": a["nombre"], "Filas (cribado)": int(fila["filas"]), "Filas (etapa 05)": len(a["tabla"]),
            "k (cribado)": int(fila["mejor_k"]), "Silueta (cribado)": float(fila["silueta"]),
            "k (filas 05, sin log1p)": int(mejor["k"]), "Silueta (filas 05, sin log1p)": float(mejor["silueta"]),
            "k (etapa 05)": a["k"], "Silueta (etapa 05)": a["silueta_k"],
            "Puntaje de la regla (cribado)": float(fila["puntaje_seleccion"]),
            "Puntaje con la silueta de la etapa 05": puntaje,
        })
    c.comparacion_cribado = t = pd.DataFrame(filas)
    r.agregar(tabla_html(t, "Silueta y puntaje de la regla de cribado con el preprocesamiento de cada etapa",
                         resaltar=(t["Análisis"] == c.seleccionado).to_numpy(), decimales=3))
    s = t.set_index("Análisis").loc[c.seleccionado]
    efecto_filas = s["Silueta (filas 05, sin log1p)"] - s["Silueta (cribado)"]
    efecto_log = s["Silueta (etapa 05)"] - s["Silueta (filas 05, sin log1p)"]
    n_log = len(c.sel["con_log"])
    texto = (f"En \"{c.seleccionado}\" la silueta pasa de {s['Silueta (cribado)']:.3f} en el cribado "
             f"({s['Filas (cribado)']} filas de entrenamiento, k = {s['k (cribado)']}) a {s['Silueta (etapa 05)']:.3f} en "
             f"esta etapa ({s['Filas (etapa 05)']} filas de desarrollo, k = {s['k (etapa 05)']}), una diferencia de "
             f"{s['Silueta (etapa 05)'] - s['Silueta (cribado)']:+.3f}. Si cambiamos solo las filas (sin log1p) la "
             f"silueta se mueve {efecto_filas:+.3f} y al aplicar después log1p a {n_log} "
             f"{'variable' if n_log == 1 else 'variables'} se mueve {efecto_log:+.3f}, ")
    texto += ("o sea que la mayor parte de la diferencia la explica el preprocesamiento y no las filas."
              if abs(efecto_log) > abs(efecto_filas) else
              "o sea que la mayor parte de la diferencia la explican las filas y no el preprocesamiento.")
    if s["k (filas 05, sin log1p)"] != s["k (etapa 05)"]:
        texto += (f" La transformación cambia además el número de grupos que gana por silueta, de k = "
                  f"{s['k (filas 05, sin log1p)']} sin log1p a k = {s['k (etapa 05)']} con ella. Es decir que las dos "
                  "siluetas describen particiones distintas y no la misma partición medida dos veces.")
    latex.registrar_varias([
        ("AgrSiluetaCribado", float(s["Silueta (cribado)"]), "silueta del seleccionado en el cribado", 3),
        ("AgrSiluetaSinLog", float(s["Silueta (filas 05, sin log1p)"]), "silueta con filas de la etapa 05 sin log1p", 3),
        ("AgrEfectoFilas", float(efecto_filas), "cambio de silueta por las filas", 3),
        ("AgrEfectoLog", float(efecto_log), "cambio de silueta por log1p", 3),
        ("AgrPuntajeSelEtapa", float(s["Puntaje con la silueta de la etapa 05"]),
         "puntaje de la regla del seleccionado con la silueta de la etapa 05", 3),
    ])
    if c.comp:
        k = t.set_index("Análisis").loc[c.complementario]
        supera = k["Puntaje con la silueta de la etapa 05"] > s["Puntaje con la silueta de la etapa 05"]
        texto += (f"\n\nCon la silueta de esta etapa el puntaje de la regla (la silueta por el peso de valor de negocio) "
                  f"sería de {s['Puntaje con la silueta de la etapa 05']:.3f} para \"{c.seleccionado}\" y de "
                  f"{k['Puntaje con la silueta de la etapa 05']:.3f} para \"{c.complementario}\". ")
        texto += ("Es decir que con el preprocesamiento de la etapa 05 el análisis complementario habría quedado por "
                  "encima del seleccionado. Aun así mantenemos la selección porque la regla del cribado la fijamos antes "
                  "de ver los resultados y cambiarla después sería acomodar la regla al resultado. Además el análisis "
                  "complementario ya lo desarrollamos completo en la sección 3 así que no se pierde nada."
                  if supera else
                  "Es decir que también con el preprocesamiento de la etapa 05 el análisis seleccionado se queda con el "
                  "primer lugar y la selección del cribado se mantiene, como de todos modos habría pasado porque la regla "
                  "la fijamos antes de ver los resultados.")
        texto += (" La comparación cubre solo estos dos análisis y los demás candidatos de agrupamiento del cribado no se "
                  "recalculan aquí.")
        latex.registrar_varias([
            ("AgrPuntajeCompEtapa", float(k["Puntaje con la silueta de la etapa 05"]),
             "puntaje de la regla del complementario con la silueta de la etapa 05", 3),
            ("AgrComplementarioSupera", bool(supera), "el complementario habría superado al seleccionado"),
        ])
    r.agregar(nota(texto, "hallazgo"))
    latex.exportar_tabla(t, "agr_silueta_cribado", "Silueta y puntaje del cribado con el preprocesamiento de cada etapa",
                         decimales=3)
    return r


def sintesis(c):
    r = Resultado()
    sel, comp, au = c.sel, c.comp, c.auditoria
    puntos = [(f'"{c.seleccionado}"',
               f"K-means con k = {sel['k']} (silueta {sel['silueta_k']:.3f}, estabilidad mínima "
               f"{formato_ari(sel['estabilidad'].min())}). "
               + (f"DBSCAN con eps = {sel['fila_db']['eps']:.2f} y {100 * sel['fila_db']['proporcion_ruido']:.1f} % de ruido"
                  if sel["fila_db"] is not None else "DBSCAN sin solución válida")
               + f". Grupos: {lista_nombres(sel['nombres'].values())}.")]
    if comp:
        puntos.append((f'"{c.complementario}"',
                       f"K-means con k = {comp['k']} (silueta {comp['silueta_k']:.3f}, estabilidad mínima "
                       f"{formato_ari(comp['estabilidad'].min())}). "
                       + (f"DBSCAN para detectar atípicos ({100 * comp['fila_db']['proporcion_ruido']:.1f} % de clientes "
                          "apartados)" if comp["criterio_db"] == "respaldo" else
                          f"DBSCAN con {int(comp['fila_db']['grupos'])} grupos" if comp["fila_db"] is not None else
                          "DBSCAN sin solución válida")
                       + (f". Recupera el segmento latente con índice de Rand ajustado {formato_ari(au['ari_kmeans'])}"
                          if au else "") + "."))
    puntos.append(("Uso posible en la operación",
                   "Los segmentos de clientes podrían servir como variable de entrada de modelos como los de la etapa 04 "
                   "o para diferenciar el servicio. El ruido de DBSCAN serviría como lista de casos para revisar a mano. Las "
                   "dos cosas quedan como trabajo futuro y no las probamos aquí."))
    r.agregar(pasos(puntos, titulo="Qué deja la etapa"))
    r.agregar(nota(
        "Las siluetas de esta etapa son modestas así que los grupos son más bien regiones de un continuo que tipos "
        "nítidos. Además los datos son simulados: Los perfiles describen el mecanismo de la simulación y no al laboratorio "
        "real. Las sugerencias para la operación solo se podrían evaluar con la exportación real.",
        "simulacion" if ES_SIMULACION else "limite"))
    c.segundos = time.perf_counter() - c.inicio
    resumen = {"seleccionado": agrupamiento.resumen_para_artefacto(sel), "complementario": None,
               "segundos_etapa": c.segundos,
               "reglas": {"valores_k": VALORES_K, "tolerancia_silueta": TOLERANCIA_SILUETA,
                          "semillas": SEMILLAS_ESTABILIDAD, "ruido_maximo": RUIDO_MAXIMO, "ruido_respaldo": RUIDO_RESPALDO,
                          "asimetria_log": ASIMETRIA_LOG, "umbral_nombre": UMBRAL_NOMBRE}}
    if comp:
        resumen["complementario"] = agrupamiento.resumen_para_artefacto(comp)
        if "segmentos_tabla" in comp:
            resumen["complementario"]["segmentos_operacion"] = comp["segmentos_tabla"].to_dict("records")
        resumen["complementario"]["auditoria"] = au
    resumen["comparacion_cribado"] = c.comparacion_cribado.to_dict("records")
    guardar_json(resumen, rutas.DIR_ARTEFACTOS / "05_agrupamiento.json")
    return r


def datos_paneles(c):
    """Muestra fija de la matriz del análisis seleccionado con su proyección en dos componentes
    (`presentacion.datos_kmeans`, el mismo cálculo de la presentación) y la exploración de K-means de esta etapa:
    Lo que necesitan la animación y el panel de K-means."""
    arreglos, meta = presentacion.datos_kmeans(c.sel["matriz"], c.sel["k"])
    return SimpleNamespace(**arreglos, **meta, interactivo={"kmeans_exploracion": c.sel["tabla_k"].to_dict("records")})


def terminar(c):
    cerrar({"seleccionado": c.seleccionado, "k": c.sel["k"], "silueta": c.sel["silueta_k"],
            "complementario": c.complementario, "k_complementario": c.comp["k"] if c.comp else None,
            "ari_auditoria": c.auditoria["ari_kmeans"] if c.auditoria else None, "segundos": c.segundos})
