"""Capa 8 · Etapa 03: cribado de análisis.

Responsabilidad única: armar las piezas de la etapa 03 (catálogo de
candidatos, regla de selección, tablas de modelado auditadas, prueba rápida,
regla simple, agrupamiento y selección), guardar las tablas de modelado y el
artefacto del cribado, y registrar sus cifras para el informe.
"""
import numpy as np
import pandas as pd

from proyecto.base import config, rutas
from proyecto.base.artefactos import guardar_json, guardar_modelado, leer_tabla
from proyecto.cribado import analisis, catalogo
from proyecto.etapas.comun import Resultado, abrir, cerrar, lista_nombres, aviso_origen
from proyecto.preparacion import particion
from proyecto.reporte import graficos_cribado, latex
from proyecto.reporte.html import comparacion, embudo, nota, pasos, pestanas, tabla_html, tarjetas
from proyecto.variables.comunes import auditar_momentos

ETAPA = "03_cribado_analisis"
TIPOS_CRIBABLES = [*catalogo.TIPOS_SUPERVISADOS, "agrupamiento"]
NOMBRES_TIPO = {"regresion": "Regresión", "clasificacion": "Clasificación", "agrupamiento": "Agrupamiento",
                "bayes": "Bayes"}
NOMBRES_COLUMNAS = {
    "analisis": "Análisis", "valor_negocio": "Valor", "filas": "Filas", "tasa_positivos": "Tasa de positivos",
    "mejor_modelo": "Mejor modelo", "habilidad_media": "Habilidad media", "habilidad_minima": "Habilidad mínima",
    "variable_regla": "Mejor variable sola", "proporcion_regla": "Proporción de la regla", "hopkins": "Hopkins",
    "mejor_k": "Mejor k", "silueta": "Silueta", "elegible": "Elegible", "observacion": "Observación",
    "puntaje_seleccion": "Puntaje",
}
COLUMNAS_POR_TIPO = {
    "regresion": ["analisis", "valor_negocio", "filas", "mejor_modelo", "habilidad_media", "habilidad_minima",
                  "variable_regla", "proporcion_regla", "elegible", "observacion", "puntaje_seleccion"],
    "clasificacion": ["analisis", "valor_negocio", "filas", "tasa_positivos", "mejor_modelo", "habilidad_media",
                      "habilidad_minima", "variable_regla", "proporcion_regla", "elegible", "observacion",
                      "puntaje_seleccion"],
    "agrupamiento": ["analisis", "valor_negocio", "filas", "hopkins", "mejor_k", "silueta", "elegible",
                     "observacion", "puntaje_seleccion"],
}
MACRO_TIPO = {"regresion": "Reg", "clasificacion": "Clf", "agrupamiento": "Agr", "bayes": "Bayes"}


def iniciar():
    return abrir(ETAPA, "Muchos candidatos, una prueba uniforme y una regla fija")


def catalogo_y_regla(c):
    """Catálogo de candidatos por tipo, regla de selección y línea base de cada supervisado."""
    r = Resultado()
    c.fichas = catalogo.tabla_catalogo()
    f = c.fichas
    r.agregar(pestanas({
        NOMBRES_TIPO[t]: tabla_html(f[f["Tipo"] == t].drop(columns="Tipo"), f"Candidatos de {NOMBRES_TIPO[t].lower()}",
                                    decimales=0)
        for t in catalogo.TIPOS if (f["Tipo"] == t).any()}, titulo="Catálogo de análisis candidatos"))
    r.agregar(tarjetas([
        {"titulo": "Análisis candidatos", "valor": int(len(f)), "decimales": 0},
        *[{"titulo": NOMBRES_TIPO[t], "valor": int((f["Tipo"] == t).sum()), "decimales": 0, "color": color}
          for t, color in zip(catalogo.TIPOS, ("azul", "rojo", "verde", "morado"))],
    ]))
    c.constantes_regla = {
        "MINIMO_FILAS": analisis.MINIMO_FILAS, "MINIMO_POSITIVOS": analisis.MINIMO_POSITIVOS,
        "UMBRAL_REGLA_SIMPLE": analisis.UMBRAL_REGLA_SIMPLE, "UMBRAL_HOPKINS": analisis.UMBRAL_HOPKINS,
        "MINIMO_FILAS_AGRUPAMIENTO": analisis.MINIMO_FILAS_AGRUPAMIENTO,
        "PROFUNDIDAD_ARBOL": analisis.PROFUNDIDAD_ARBOL, "VALORES_K": list(analisis.VALORES_K),
        "GRUPO_VARIABLES": analisis.GRUPO_VARIABLES,
        "BLOQUES_VALIDACION_CRUZADA": config.BLOQUES_VALIDACION_CRUZADA,
    }
    r.agregar(pasos([
        ("Tamaño mínimo",
         f"Un candidato supervisado necesita al menos {analisis.MINIMO_FILAS} filas de entrenamiento y, si es de "
         f"clasificación, al menos {analisis.MINIMO_POSITIVOS} casos positivos; uno de agrupamiento, al menos "
         f"{analisis.MINIMO_FILAS_AGRUPAMIENTO} filas."),
        ("Ganancia en todos los bloques",
         f"La habilidad (ganancia sobre la línea base) debe ser mayor que cero en cada uno de los "
         f"{config.BLOQUES_VALIDACION_CRUZADA} bloques temporales, no solo en promedio."),
        ("No basta una regla simple",
         f"Si un árbol que ve una sola variable logra {100 * analisis.UMBRAL_REGLA_SIMPLE:.0f} % o más de la habilidad del "
         "mejor modelo rápido, el problema se resuelve con una regla y no necesita un modelo."),
        ("Tendencia a agruparse",
         f"Un candidato de agrupamiento debe tener un estadístico de Hopkins mayor que {analisis.UMBRAL_HOPKINS}, "
         "el valor que corresponde a datos sin estructura de grupos."),
        ("Puntaje",
         "Entre los elegibles de cada tipo gana el de mayor puntaje: habilidad media × valor de negocio / 3 en los "
         "supervisados y silueta × valor de negocio / 3 en los de agrupamiento (empates: mayor valor, después nombre)."),
    ], titulo="Regla de selección (constantes leídas de cribado/analisis.py)"))
    supervisadas = f[f["Tipo"].isin(catalogo.TIPOS_SUPERVISADOS)]
    r.agregar(tabla_html(supervisadas[["Análisis", "Tipo", "Unidad", "Métrica", "Línea base"]],
                         "Línea base de cada candidato supervisado", decimales=0,
                         resaltar=(supervisadas["Línea base"] != "Dummy").to_numpy()))
    latex.registrar_varias([
        ("CriCandidatos", int(len(f)), "análisis candidatos del catálogo"),
        *[(f"CriCandidatos{MACRO_TIPO[t]}", int((f["Tipo"] == t).sum()), f"candidatos de tipo {t}") for t in catalogo.TIPOS],
        ("CriMinimoFilas", analisis.MINIMO_FILAS, "filas mínimas de entrenamiento de un supervisado"),
        ("CriMinimoPositivos", analisis.MINIMO_POSITIVOS, "positivos mínimos de un clasificador"),
        ("CriMinimoFilasAgrupamiento", analisis.MINIMO_FILAS_AGRUPAMIENTO, "filas mínimas de un agrupamiento"),
        ("CriUmbralReglaSimplePct", 100 * analisis.UMBRAL_REGLA_SIMPLE, "umbral (%) de la regla simple", 0),
        ("CriUmbralHopkins", analisis.UMBRAL_HOPKINS, "umbral del estadístico de Hopkins", 2),
        ("CriProfundidadArbol", analisis.PROFUNDIDAD_ARBOL, "profundidad del árbol rápido del cribado"),
    ])
    latex.exportar_tabla(f, "catalogo", "Catálogo de análisis candidatos", decimales=0)
    return r


def tablas_modelado(c):
    """Construye, audita (fugas por momento), parte y guarda cada tabla de modelado."""
    r = Resultado()
    limpias = {n: leer_tabla(rutas.DIR_LIMPIOS / f"{n}.parquet")
               for n in ("clientes", "muestreos", "muestras", "controles", "calendario")}
    integradas = leer_tabla(rutas.DIR_LIMPIOS / "muestras_integradas.parquet")
    c.tablas_modelado, c.motivos = catalogo.construir_disponibles(limpias, integradas)
    filas = []
    for nombre, (tabla, papeles) in c.tablas_modelado.items():
        hallazgos = auditar_momentos(tabla, papeles)
        desarrollo, reservada = particion.separar(tabla, papeles["fecha"])
        guardar_modelado(nombre, desarrollo, reservada)
        guardar_json(papeles, rutas.DIR_MODELADO / f"{nombre}.json")
        filas.append({
            "Análisis": nombre, "Tipo": catalogo.ficha_por_nombre(nombre).tipo, "Unidad": papeles["unidad"],
            "Momento": papeles["momento"], "Variables": len(papeles["numericas"]) + len(papeles["categoricas"]),
            "Filas de desarrollo": len(desarrollo), "Filas de prueba": len(reservada),
            "Hallazgos de fuga": len(hallazgos), "Auditoría": "superada" if not hallazgos else "con hallazgos",
        })
    c.resumen_tablas = pd.DataFrame(filas)
    t, motivos = c.resumen_tablas, c.motivos
    r.agregar(tabla_html(t, "Tablas de modelado construidas y auditadas", decimales=0, barras="Filas de desarrollo",
                         resaltar=(t["Hallazgos de fuga"] > 0).to_numpy()))
    c.n_superadas = int((t["Hallazgos de fuga"] == 0).sum())
    n = len(c.tablas_modelado)
    con_tabla = int((c.fichas["Tipo"] != "bayes").sum())
    r.agregar(tarjetas([
        {"titulo": "Tablas construidas", "valor": n, "decimales": 0, "detalle": f"de {con_tabla} análisis con tabla"},
        {"titulo": "Auditorías superadas", "valor": c.n_superadas, "decimales": 0,
         "color": "verde" if c.n_superadas == n else "rojo"},
        {"titulo": "No disponibles", "valor": len(motivos), "decimales": 0, "color": "gris",
         "detalle": "faltan columnas en la exportación" if motivos else "todas las columnas requeridas están"},
    ]))
    r.agregar(nota(
        (f"Las **{n}** tablas superan la auditoría de fuga: ninguna variable se conoce después del momento de la "
         "predicción y ningún identificador u objetivo entra como variable." if c.n_superadas == n else
         f"Hay hallazgos de fuga en {lista_nombres(t.loc[t['Hallazgos de fuga'] > 0, 'Análisis'])}, que deben "
         "corregirse en `proyecto/variables/` antes de seguir.")
        + (f" Los análisis no disponibles son {lista_nombres(motivos)} ({'; '.join(motivos.values())})." if motivos else ""),
        "hallazgo" if c.n_superadas == n else "advertencia",
    ))
    sin_reserva = t.loc[t["Filas de prueba"] == 0, "Análisis"].tolist()
    r.agregar(nota(
        f"Las tablas de agrupamiento ({lista_nombres(sin_reserva)}) se construyen solo con el periodo de desarrollo, "
        "porque describen la base tal como se conoce hoy y no predicen nada, de forma que su bloque de prueba queda "
        "vacío." if sin_reserva else "Todas las tablas tienen filas en el bloque de prueba.",
        "info",
    ))
    latex.registrar_varias([
        ("CriTablasConstruidas", n, "tablas de modelado construidas"),
        ("CriAnalisisConTabla", con_tabla, "análisis del catálogo que llevan tabla de modelado"),
        ("CriAuditoriasSuperadas", c.n_superadas, "tablas que superan la auditoría de fugas"),
        ("CriNoDisponibles", len(motivos), "análisis no disponibles por columnas faltantes"),
    ])
    latex.exportar_tabla(t, "tablas_modelado", "Tablas de modelado construidas y auditadas", decimales=0)
    return r


def prueba_rapida(c):
    """Cribado de todos los candidatos, tabla por tipo y figura de habilidad por bloque."""
    r = Resultado()
    c.tabla_cribado = analisis.cribar(catalogo.CATALOGO, c.tablas_modelado, c.motivos)
    c.seleccion = analisis.seleccionar_analisis(c.tabla_cribado)
    tc = c.tabla_cribado
    c.cribados = tc[tc["estado"] == analisis.ESTADO_CRIBADO]
    c.supervisados = c.cribados[c.cribados["tipo"].isin(catalogo.TIPOS_SUPERVISADOS)]
    c.de_agrupamiento = c.cribados[c.cribados["tipo"] == "agrupamiento"]
    fichas = {}
    for tipo in TIPOS_CRIBABLES:
        grupo = c.cribados[c.cribados["tipo"] == tipo].sort_values("puntaje_seleccion", ascending=False)
        if len(grupo):
            vista = grupo[COLUMNAS_POR_TIPO[tipo]].assign(
                elegible=grupo["elegible"].map({True: "sí", False: "no"}), filas=grupo["filas"].astype(int),
            ).rename(columns=NOMBRES_COLUMNAS).reset_index(drop=True)
            fichas[NOMBRES_TIPO[tipo]] = tabla_html(
                vista, f"Cribado de {NOMBRES_TIPO[tipo].lower()} (periodo de entrenamiento)", decimales=3,
                resaltar=(grupo["analisis"] == c.seleccion.get(tipo)).to_numpy(), columnas_resaltadas="Puntaje")
            latex.exportar_tabla(vista, f"cribado_{tipo}", f"Cribado de {NOMBRES_TIPO[tipo].lower()} (entrenamiento)",
                                 decimales=3)
    no_cribados = tc[tc["estado"] != analisis.ESTADO_CRIBADO]
    if len(no_cribados):
        fichas["Sin cribar"] = tabla_html(
            no_cribados[["analisis", "tipo", "estado", "motivo"]].rename(
                columns={"analisis": "Análisis", "tipo": "Tipo", "estado": "Estado", "motivo": "Motivo"}),
            "Análisis que no pasan por la prueba rápida")
    r.agregar(pestanas(fichas, titulo="Resultados de la prueba rápida (la fila resaltada es la elegida)"))
    segundos = float(tc["segundos"].sum())
    r.agregar(nota(f"La prueba completa de los {len(c.cribados)} candidatos cribados tarda **{segundos:.1f} s**, lo cual "
                   "confirma que el filtro es barato frente al modelado a fondo que vendrá después."))

    sup = c.supervisados
    r.figura(graficos_cribado.figura_habilidad(tc, analisis.NOTA_REGLA_SIMPLE), "03_habilidad")
    estables = sup[sup["habilidad_minima"] > 0]
    variable = sup.assign(
        rango=[float(np.ptp(graficos_cribado.habilidades_por_bloque(f))) for _, f in sup.iterrows()]
    ).sort_values("rango").iloc[-1]
    por_bloque = graficos_cribado.habilidades_por_bloque(variable)
    r.agregar(nota(
        (f"Los **{len(sup)}** candidatos supervisados superan a su línea base en todos los bloques temporales"
         if len(estables) == len(sup) else
         f"De los {len(sup)} candidatos supervisados, **{len(estables)}** superan a su línea base en todos los bloques "
         "temporales")
        + (f" y {len(sup) - len(estables)} no lo logran en al menos uno." if len(estables) < len(sup) else ".")
        + f" El candidato con la habilidad más inestable entre bloques es «{variable['analisis']}», cuya habilidad va de "
          f"{por_bloque.min():.3f} a {por_bloque.max():.3f} según el bloque, "
        + ("de forma que su media esconde que en algunos periodos apenas aporta." if por_bloque.min() < 0.1 else
           "de forma que su media resume periodos con ganancias bastante distintas."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("CriCribados", len(c.cribados), "candidatos que pasan por la prueba rápida"),
        ("CriSupervisados", len(sup), "candidatos supervisados cribados"),
        ("CriSupervisadosEstables", len(estables), "supervisados que superan a su base en todos los bloques"),
        ("CriSegundosPrueba", segundos, "segundos de la prueba rápida completa", 1),
        ("CriMasInestable", variable["analisis"], "candidato con la habilidad más inestable entre bloques"),
        ("CriMasInestableMin", float(por_bloque.min()), "su habilidad mínima por bloque", 3),
        ("CriMasInestableMax", float(por_bloque.max()), "su habilidad máxima por bloque", 3),
    ])
    return r


def regla_simple(c):
    """Candidatos descartados porque una sola variable basta, y el caso de mayor habilidad bruta."""
    r = Resultado()
    sup = c.supervisados
    r.figura(graficos_cribado.figura_regla_simple(c.tabla_cribado, analisis.UMBRAL_REGLA_SIMPLE), "03_regla_simple")
    por_regla = sup[sup["observacion"] == analisis.NOTA_REGLA_SIMPLE].sort_values("habilidad_media", ascending=False)
    frases = [f"«{f['analisis']}» (con «{f['variable_regla']}» se logra el {100 * f['proporcion_regla']:.0f} % de la "
              "habilidad)" for _, f in por_regla.iterrows()]
    supera = por_regla[por_regla["proporcion_regla"] > 1]
    r.agregar(nota(
        (f"La regla simple descarta {len(por_regla)} candidato{'s' if len(por_regla) != 1 else ''}: "
         + "; ".join(frases) + "."
         + (f" En {lista_nombres(supera['analisis'])} la variable sola incluso supera al modelo rápido completo, señal "
            "de que, en un árbol tan poco profundo, las demás variables no añaden información útil." if len(supera) else ""))
        if len(por_regla) else
        f"Ningún candidato alcanza el umbral de {100 * analisis.UMBRAL_REGLA_SIMPLE:.0f} %: en todos, el modelo combina "
        "información de varias variables.",
        "hallazgo",
    ))
    latex.registrar("CriDescartadosReglaSimple", len(por_regla), "candidatos descartados por la regla simple")
    clasif = sup[sup["tipo"] == "clasificacion"]
    mas_habil = clasif.sort_values("habilidad_media", ascending=False).iloc[0] if len(clasif) else None
    if mas_habil is not None and not bool(mas_habil["elegible"]):
        r.agregar(nota(
            f"El caso más llamativo es «{mas_habil['analisis']}», el candidato de clasificación con la mayor habilidad bruta "
            f"(**{mas_habil['habilidad_media']:.3f}**, un AUC de {(mas_habil['habilidad_media'] + 1) / 2:.3f}), que queda "
            f"fuera porque {mas_habil['observacion']}: un árbol que solo mira «{mas_habil['variable_regla']}» consigue el "
            f"{100 * mas_habil['proporcion_regla']:.0f} % de esa habilidad. "
            + ("La recomendación para esa pregunta es, entonces, una regla operativa sobre esa variable y no un modelo, y la "
               "pregunta de fondo (qué tan probable es que un no conforme sea falso) se responde de forma analítica con las "
               "fichas de Bayes de la etapa 05." if mas_habil["observacion"] == analisis.NOTA_REGLA_SIMPLE else
               "Ese criterio de la regla pesa más que la habilidad bruta."),
            "hallazgo", titulo="La mayor habilidad no siempre gana"))
    elif mas_habil is not None:
        r.agregar(nota(f"El candidato de clasificación con más habilidad bruta, «{mas_habil['analisis']}», también es "
                       "elegible.", "hallazgo"))
    if mas_habil is not None:
        latex.registrar_varias([
            ("CriClfMasHabil", mas_habil["analisis"], "clasificador con la mayor habilidad bruta en el cribado"),
            ("CriClfMasHabilHabilidad", float(mas_habil["habilidad_media"]), "su habilidad media (2 AUC - 1)", 3),
            ("CriClfMasHabilProporcion", 100 * float(mas_habil["proporcion_regla"]),
             "porcentaje de su habilidad que logra la mejor variable sola", 0),
        ])
    return r


def candidatos_agrupamiento(c):
    r = Resultado()
    g = c.de_agrupamiento
    r.figura(graficos_cribado.figura_agrupamiento(c.tabla_cribado, analisis.UMBRAL_HOPKINS, c.seleccion["agrupamiento"]),
             "03_agrupamiento")
    if len(g):
        mejor = g.sort_values("silueta", ascending=False).iloc[0]
        sin_tendencia = g[~(g["hopkins"] > analisis.UMBRAL_HOPKINS)]
        r.agregar(nota(
            f"Los {len(g)} candidatos tienen un Hopkins entre {g['hopkins'].min():.2f} y {g['hopkins'].max():.2f}"
            + (f", todos por encima del umbral de {analisis.UMBRAL_HOPKINS}, de forma que todos muestran alguna tendencia "
               "a agruparse." if not len(sin_tendencia) else
               f"; {lista_nombres(sin_tendencia['analisis'])} no supera{'n' if len(sin_tendencia) > 1 else ''} el umbral "
               f"de {analisis.UMBRAL_HOPKINS}.")
            + f" La mejor silueta es la de «{mejor['analisis']}» ({mejor['silueta']:.3f} con k = {int(mejor['mejor_k'])}), "
            + ("un valor moderado: hay estructura, pero los grupos se traslapan." if mejor["silueta"] < 0.5 else
               "un valor que indica grupos razonablemente separados."),
            "hallazgo",
        ))
        latex.registrar_varias([
            ("CriAgrCandidatos", len(g), "candidatos de agrupamiento cribados"),
            ("CriHopkinsMin", float(g["hopkins"].min()), "Hopkins mínimo entre candidatos de agrupamiento", 2),
            ("CriHopkinsMax", float(g["hopkins"].max()), "Hopkins máximo entre candidatos de agrupamiento", 2),
            ("CriMejorSiluetaAnalisis", mejor["analisis"], "candidato de agrupamiento con la mejor silueta"),
            ("CriMejorSilueta", float(mejor["silueta"]), "mejor silueta del cribado", 3),
            ("CriMejorSiluetaK", int(mejor["mejor_k"]), "k de la mejor silueta del cribado"),
        ])
    return r


def seleccion(c):
    """Embudo, mejores puntajes por tipo y lectura de lo que decidió la regla."""
    r = Resultado()
    tc, cr, sel = c.tabla_cribado, c.cribados, c.seleccion
    c.n_elegibles = int(cr["elegible"].astype(bool).sum())
    n_elegidos = sum(v is not None for v in sel.values())
    n_bayes = int((tc["tipo"] == "bayes").sum())
    r.agregar(embudo([
        ("Análisis candidatos", len(tc), f"{n_bayes} analíticos (Bayes) pasan directo a la etapa 05"),
        ("Cribados con su tabla de modelado", len(cr),
         f"{len(c.motivos)} no disponibles; prueba rápida en el periodo de entrenamiento"),
        ("Elegibles según la regla", c.n_elegibles, "mayor puntaje por tipo"),
        ("Elegidos para el modelado a fondo", n_elegidos),
    ], titulo="Embudo de cribado"))
    por_tipo = {}
    for tipo in TIPOS_CRIBABLES:
        grupo = cr[cr["tipo"] == tipo].sort_values("puntaje_seleccion", ascending=False).head(4)
        opciones = []
        for _, fila in grupo.iterrows():
            cifra = (f"silueta {fila['silueta']:.3f} (k = {int(fila['mejor_k'])}), Hopkins {fila['hopkins']:.2f}"
                     if tipo == "agrupamiento" else
                     f"habilidad media {fila['habilidad_media']:.3f}, mínima {fila['habilidad_minima']:.3f}")
            opciones.append({
                "titulo": fila["analisis"], "valor": float(fila["puntaje_seleccion"]), "decimales": 3,
                "etiqueta": "puntaje", "puntos": [cifra, f"valor de negocio {int(fila['valor_negocio'])}",
                                                  "elegible" if bool(fila["elegible"]) else f"no elegible: {fila['observacion']}"],
            })
        if opciones:
            por_tipo[NOMBRES_TIPO[tipo]] = comparacion(opciones, elegida=sel.get(tipo))
    r.agregar(pestanas(por_tipo, titulo="Mejores puntajes por tipo"))

    frases = []
    regresion = c.supervisados[c.supervisados["tipo"] == "regresion"]
    if sel["regresion"] is not None and len(regresion):
        ganadora = regresion.set_index("analisis").loc[sel["regresion"]]
        valiosa = regresion.sort_values(["valor_negocio", "puntaje_seleccion"], ascending=False).iloc[0]
        if valiosa["analisis"] != sel["regresion"]:
            frases.append(
                f"En regresión, «{valiosa['analisis']}» tiene el mayor valor de negocio declarado "
                f"({int(valiosa['valor_negocio'])} frente a {int(ganadora['valor_negocio'])}), pero su habilidad media "
                f"({valiosa['habilidad_media']:.3f}) es tan inferior a la de «{sel['regresion']}» "
                f"({ganadora['habilidad_media']:.3f}) que su puntaje ({valiosa['puntaje_seleccion']:.3f}) queda por "
                f"debajo ({ganadora['puntaje_seleccion']:.3f})"
                + (f"; además, su habilidad mínima por bloque es apenas {valiosa['habilidad_minima']:.3f}."
                   if valiosa["habilidad_minima"] < 0.1 else "."))
        else:
            frases.append(f"En regresión gana «{sel['regresion']}», que además es el de mayor valor de negocio.")
    if sel["clasificacion"] is not None:
        g = cr.set_index("analisis").loc[sel["clasificacion"]]
        frases.append(f"En clasificación gana «{sel['clasificacion']}» (habilidad {g['habilidad_media']:.3f}, AUC "
                      f"{(g['habilidad_media'] + 1) / 2:.3f}, valor {int(g['valor_negocio'])}).")
    if sel["agrupamiento"] is not None:
        g = cr.set_index("analisis").loc[sel["agrupamiento"]]
        frases.append(f"En agrupamiento gana «{sel['agrupamiento']}» (silueta {g['silueta']:.3f} × valor "
                      f"{int(g['valor_negocio'])} / 3 = {g['puntaje_seleccion']:.3f}).")
    for tipo in TIPOS_CRIBABLES:
        if sel[tipo] is None:
            frases.append(f"En {NOMBRES_TIPO[tipo].lower()} ningún candidato es elegible, por lo que no se elige ninguno.")
    r.agregar(nota("\n\n".join(frases), "hallazgo", titulo="Qué decidió la regla"))
    r.agregar(aviso_origen(
        "Los datos son **simulados**: el resultado del cribado describe la base simulada y muestra que el procedimiento "
        "funciona, pero con la exportación real del laboratorio la regla, que no cambia, podría elegir otros análisis."))
    latex.registrar_varias([
        ("CriElegibles", c.n_elegibles, "candidatos elegibles según la regla"),
        ("CriElegidos", n_elegidos, "análisis elegidos para el modelado a fondo"),
        ("CriBayes", n_bayes, "análisis bayesianos (no se criban)"),
    ])
    for tipo in TIPOS_CRIBABLES:
        prefijo = f"CriSeleccion{MACRO_TIPO[tipo]}"
        latex.registrar(prefijo, sel[tipo] or "ninguno", f"análisis de {tipo} elegido por la regla")
        if sel[tipo] is not None:
            g = cr.set_index("analisis").loc[sel[tipo]]
            latex.registrar(f"{prefijo}Puntaje", float(g["puntaje_seleccion"]), f"puntaje del {tipo} elegido", 3)
            latex.registrar(f"{prefijo}Valor", int(g["valor_negocio"]), f"valor de negocio del {tipo} elegido")
            if tipo == "agrupamiento":
                latex.registrar(f"{prefijo}Silueta", float(g["silueta"]), "silueta del agrupamiento elegido", 3)
            else:
                latex.registrar(f"{prefijo}Habilidad", float(g["habilidad_media"]), f"habilidad media del {tipo} elegido", 3)
    if sel["clasificacion"] is not None:
        g = cr.set_index("analisis").loc[sel["clasificacion"]]
        latex.registrar("CriSeleccionClfAUC", (float(g["habilidad_media"]) + 1) / 2, "AUC del clasificador elegido", 3)
    return r


def consecuencias(c):
    """Qué modela cada etapa siguiente y artefacto del cribado."""
    r = Resultado()
    sel, g = c.seleccion, c.de_agrupamiento
    otros = g[g["analisis"] != sel["agrupamiento"]]
    c.complementario = (str(otros.sort_values(["valor_negocio", "puntaje_seleccion"], ascending=False).iloc[0]["analisis"])
                        if len(otros) else None)
    bayes = [f for f in catalogo.CATALOGO if f.tipo == "bayes"]

    def decision(nombre):
        return catalogo.ficha_por_nombre(nombre).decision

    comp = c.complementario
    r.agregar(pasos([
        ("04 · Regresión", f"Se modela a fondo «{sel['regresion']}»: {decision(sel['regresion'])}" if sel["regresion"]
         else "No hay análisis de regresión elegible: la etapa solo documenta el cribado."),
        ("05 · Clasificación y Bayes",
         (f"Se modela a fondo «{sel['clasificacion']}»: {decision(sel['clasificacion'])} " if sel["clasificacion"]
          else "No hay análisis de clasificación elegible. ")
         + f"Además se desarrollan los {len(bayes)} análisis bayesianos, que son analíticos y no se criban: "
         + lista_nombres(f.nombre for f in bayes) + "."),
        ("06 · Agrupamiento",
         (f"Se modela a fondo «{sel['agrupamiento']}», elegido por la regla. " if sel["agrupamiento"]
          else "Ningún candidato de agrupamiento es elegible. ")
         + (f"Como análisis **complementario y fuera de la regla** se incluye «{comp}», el candidato no elegido con mayor "
            f"valor de negocio declarado ({int(g.set_index('analisis').loc[comp, 'valor_negocio'])}), porque apoya una "
            f"decisión sobre los clientes: {decision(comp)}" if comp else "")),
    ], titulo="Qué modela cada etapa"))
    latex.registrar("CriAgrComplementario", comp or "ninguno", "análisis de agrupamiento complementario (fuera de la regla)")
    guardar_json({
        "tabla_cribado": c.tabla_cribado.to_dict("records"),
        "seleccion": sel,
        "agrupamiento_complementario": comp,
        "bayes": [f.nombre for f in bayes],
        "constantes_regla": c.constantes_regla,
        "tablas_modelado": c.resumen_tablas.to_dict("records"),
        "no_disponibles": c.motivos,
    }, rutas.DIR_ARTEFACTOS / "03_cribado.json")
    return r


def terminar(c):
    cerrar({
        "candidatos": int(len(c.tabla_cribado)), "cribados": int(len(c.cribados)), "elegibles": c.n_elegibles,
        "seleccion": c.seleccion, "agrupamiento_complementario": c.complementario,
        "auditorias_superadas": c.n_superadas,
    })
