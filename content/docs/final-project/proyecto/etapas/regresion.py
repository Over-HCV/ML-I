"""Capa 8 · Etapa 03, parte B: regresión.

Responsabilidad única: armar las piezas de la parte B de la etapa 03
(referencias ingenuas, pasada didáctica con los modelos del curso, cribado de
la tubería con un L16 de Taguchi, afinamiento, evaluación única en
validación, congelamiento y lectura operativa), guardar el modelo y el
artefacto de la regresión, y registrar sus cifras para el informe. Continúa
la etapa que abrió la parte A (`cribado`) y la cierra.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from proyecto.base import config, rutas
from proyecto.base.artefactos import guardar_json, guardar_modelo, leer_json, leer_modelado
from proyecto.cribado import arreglos, catalogo, corridas
from proyecto.etapas import taguchi
from proyecto.etapas.comun import Resultado, cerrar, continuar, legible, enumerar
from proyecto.evaluacion import metricas, validacion
from proyecto.evaluacion.remuestreo import intervalo_bootstrap
from proyecto.modelos import registro, tuberias
from proyecto.reporte import graficos_modelado as graficos
from proyecto.reporte import latex
from proyecto.reporte.html import comparacion, nota, pasos, pestanas, tabla_html, tarjetas
from proyecto.variables.comunes import columnas_de_grupo

ETAPA = "03_cribado_regresion"
TAREA = "regresion"
NOMBRE_DUMMY = "Dummy (media)"
COLUMNAS_L16 = {"modelo": "A", "escalado": "c", "grupo": "d", "objetivo_log": "abd", "seleccion": "acd",
                "codificacion": "bcd"}
ORDEN_SENCILLEZ = ["lineal", "arbol", "knn", "mlp"]
TOLERANCIA_EMPATE = 0.01
FRACCION_TOLERANCIA = 0.20
TRES_METRICAS = (("MAE", "mae"), ("RMSE", "rmse"), ("R2", "r2"))


def _media_desv(valores):
    valores = np.asarray(valores, dtype=float)
    return float(np.mean(valores)), float(np.std(valores, ddof=1))


def iniciar():
    """Lee el análisis elegido en la parte A y parte el desarrollo en entrenamiento y validación."""
    # Los avisos de no convergencia del perceptrón se cuentan en las corridas del diseño (columna `avisos`).
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    c = continuar(ETAPA)
    c.inicio = time.perf_counter()
    c.nombre = leer_json(rutas.DIR_ARTEFACTOS / "03_cribado.json")["seleccion"][TAREA]
    if c.nombre is None:
        raise RuntimeError("El cribado no eligió ningún análisis de regresión: no hay nada que modelar.")
    c.ficha = catalogo.ficha_por_nombre(c.nombre)
    c.papeles = leer_json(rutas.DIR_MODELADO / f"{c.nombre}.json")
    c.objetivo, c.fecha, c.referencias = c.papeles["objetivo"], c.papeles["fecha"], list(c.papeles["referencias"])
    c.desarrollo = leer_modelado(c.nombre)
    c.entrenamiento, c.validacion = validacion.separar_desarrollo(c.desarrollo, c.fecha)
    c.numericas, c.categoricas = columnas_de_grupo(c.papeles, "ampliado")
    columnas = c.numericas + c.categoricas
    c.X_ent, c.y_ent = c.entrenamiento[columnas], c.entrenamiento[c.objetivo].to_numpy(dtype=float)
    c.X_val, c.y_val = c.validacion[columnas], c.validacion[c.objetivo].to_numpy(dtype=float)
    c.bloques = validacion.bloques_temporales(c.entrenamiento[c.fecha])
    c.etiquetas_bloque = [f"Bloque {i + 1}" for i in range(len(c.bloques))]
    return c


def planteamiento(c):
    r = Resultado()
    r.agregar(pasos([
        ("Qué predecimos", f"{c.ficha.titulo}: El objetivo es `{c.objetivo}`, con una fila por {c.papeles['unidad']}."),
        ("En qué momento", f"Con lo que se sabe en el momento \"{c.papeles['momento']}\", es decir la víspera: Solo "
                           "entran rezagos (el valor de días anteriores), medias móviles, el calendario y lo que ya está "
                           "agendado."),
        ("Qué decisión apoya", f"{c.ficha.decision} De la misma cifra depende cuántas placas, medios de cultivo y "
                               "espacio de incubadora preparar."),
        ("Contra qué lo comparamos", "Contra las referencias ingenuas que ya trae la tabla ("
         + ", ".join(f'"{x}"' for x in c.referencias) + ") y contra un `Dummy` que predice la media del entrenamiento."),
    ], titulo="Planteamiento"))
    r.agregar(tarjetas([
        {"titulo": "Días de entrenamiento", "valor": len(c.entrenamiento), "decimales": 0,
         "detalle": f"antes de {config.CORTE_VALIDACION}"},
        {"titulo": "Días de validación", "valor": len(c.validacion), "decimales": 0, "color": "naranja",
         "detalle": f"desde {config.CORTE_VALIDACION} hasta antes de {config.CORTE_PRUEBA}"},
        {"titulo": "Bloques temporales", "valor": len(c.bloques), "decimales": 0, "color": "morado",
         "detalle": "ventana creciente, solo en entrenamiento"},
        {"titulo": "Media diaria (entrenamiento)", "valor": float(np.mean(c.y_ent)), "decimales": 1, "color": "verde",
         "detalle": f"rango observado {np.min(c.y_ent):.0f} a {np.max(c.y_ent):.0f}"},
    ]))
    latex.registrar_varias([
        ("RegAnalisis", c.nombre, "análisis de regresión modelado a fondo"),
        ("RegObjetivo", c.objetivo, "columna objetivo de la regresión"),
        ("RegMomento", c.papeles["momento"], "momento de la predicción de la regresión"),
        ("RegDiasEntrenamiento", len(c.entrenamiento), "días de entrenamiento"),
        ("RegDiasValidacion", len(c.validacion), "días de validación"),
        ("RegBloques", len(c.bloques), "bloques temporales de la validación cruzada"),
        ("RegMediaEntrenamiento", float(np.mean(c.y_ent)), "muestras por día en promedio (entrenamiento)", 1),
        ("RegMinimoEntrenamiento", float(np.min(c.y_ent)), "mínimo diario observado (entrenamiento)", 0),
        ("RegMaximoEntrenamiento", float(np.max(c.y_ent)), "máximo diario observado (entrenamiento)", 0),
    ])
    return r


def referencias(c):
    """RMSE por bloque de las referencias ingenuas y del Dummy; elige la mejor referencia."""
    r = Resultado()
    c.filas_ref = {}
    for ref in c.referencias:
        por_bloque = [metricas.metricas_reg(c.y_ent[v], c.entrenamiento[ref].to_numpy(dtype=float)[v])
                      for _, v in c.bloques]
        c.filas_ref[ref] = {m: np.array([b[m] for b in por_bloque]) for m in ("MAE", "RMSE", "R2")}
    c.filas_ref[NOMBRE_DUMMY] = {m: validacion.puntuar_por_bloques(registro.linea_base(TAREA), c.X_ent, c.y_ent,
                                                                   c.bloques, clave) for m, clave in TRES_METRICAS}
    c.rmse_referencias = pd.DataFrame({n: v["RMSE"] for n, v in c.filas_ref.items()}, index=c.etiquetas_bloque).T
    candidatas = c.referencias or [NOMBRE_DUMMY]
    c.mejor_ref = str(c.rmse_referencias.loc[candidatas].mean(axis=1).idxmin())
    c.rmse_ref_bloques = c.rmse_referencias.loc[c.mejor_ref].to_numpy(dtype=float)
    rr, mejor, bloques = c.rmse_referencias, c.mejor_ref, c.rmse_ref_bloques
    r.agregar(tabla_html(rr.reset_index(names="Referencia"), "RMSE de las referencias en cada bloque temporal (muestras "
                         "por día)", decimales=2, resaltar=(rr.index == mejor)))
    peor = str(rr.mean(axis=1).idxmax())
    dummy = rr.loc[NOMBRE_DUMMY].mean()
    r.agregar(nota(
        f"La mejor referencia ingenua es \"{mejor}\", con un RMSE medio de **{bloques.mean():.2f}** muestras por día "
        f"(entre {bloques.min():.2f} y {bloques.max():.2f} según el bloque). Contra ella medimos la habilidad en el resto "
        "de la regresión. "
        + (f"El `Dummy`, que ignora el pasado reciente, queda en {dummy:.2f}"
           + (", peor que las reglas ingenuas, o sea que el nivel reciente de la serie ya lleva información."
              if dummy > bloques.mean() else
              ", mejor que las reglas ingenuas, señal de que la serie fluctúa alrededor de un nivel casi fijo.")
           if NOMBRE_DUMMY != mejor else "")
        + (f" La referencia más débil es \"{peor}\"." if peor not in (mejor, NOMBRE_DUMMY) else ""),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("RegMejorReferencia", mejor, "mejor referencia ingenua (base de la habilidad)"),
        ("RegRMSEReferencia", float(bloques.mean()), "RMSE medio por bloque de la mejor referencia", 2),
        ("RegRMSEReferenciaMin", float(bloques.min()), "RMSE mínimo por bloque de la mejor referencia", 2),
        ("RegRMSEReferenciaMax", float(bloques.max()), "RMSE máximo por bloque de la mejor referencia", 2),
        ("RegRMSEDummy", float(dummy), "RMSE medio por bloque del Dummy (media)", 2),
    ])
    latex.exportar_tabla(rr.reset_index(names="Referencia"), "reg_referencias", "RMSE de las referencias por bloque",
                         decimales=2)
    return r


def recta_simple(c):
    """Regresión lineal simple con la variable más correlacionada, en unidades originales."""
    r = Resultado()
    corr = c.entrenamiento[c.numericas].corrwith(c.entrenamiento[c.objetivo]).dropna()
    c.variable_simple = v = str(corr.abs().idxmax())
    c.recta = tuberias.construir_tuberia(TAREA, "lineal", [v], [], escalado="ninguno")
    c.recta.fit(c.X_ent, c.y_ent)
    b1 = float(c.recta.named_steps["modelo"].coef_[0])
    b0 = float(c.recta.named_steps["modelo"].intercept_)
    x = c.entrenamiento[v].to_numpy(dtype=float)
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    x_p10, x_p90 = (float(q) for q in np.nanpercentile(x, [10, 90]))
    r2 = c.recta.score(c.X_ent, c.y_ent)
    r.figura(graficos.figura_recta_simple(x, c.y_ent, b1, b0, v, c.objetivo), "03_recta_simple")
    r.agregar(nota(
        f"La variable más correlacionada con el objetivo es \"{v}\" (r = {corr[v]:.3f}) y la recta ajustada es "
        f"ŷ = {b0:.2f} + {b1:.3f} x. La pendiente dice que dentro del rango observado ({x_min:.1f} a {x_max:.1f}) cada "
        f"unidad más de \"{v}\" va con {b1:+.3f} muestras en el día que predecimos. Sola, la variable explica un R² de "
        f"{r2:.3f} en el entrenamiento (el R² es la fracción de la variación del objetivo que el modelo reproduce)."
        + (f"\n\nEl intercepto ({b0:.2f}) no lo interpretamos solo porque corresponde a x = 0, un valor que queda fuera "
           "del rango observado. Lo que sí tiene respaldo es la predicción entre el percentil 10 y el 90 de x, que va de "
           f"{b0 + b1 * x_p10:.1f} a {b0 + b1 * x_p90:.1f} muestras." if not x_min <= 0 <= x_max else
           f"\n\nComo x = 0 sí está dentro del rango observado, el intercepto ({b0:.2f}) se lee como la carga esperada en un "
           "día con ese valor de la variable."),
        "hallazgo", titulo="Lectura de la recta",
    ))
    latex.registrar_varias([
        ("RegVariableSimple", v, "variable de la regresión lineal simple"),
        ("RegCorrSimple", float(corr[v]), "correlación de Pearson de esa variable con el objetivo", 3),
        ("RegInterceptoSimple", b0, "intercepto de la recta simple", 2),
        ("RegPendienteSimple", b1, "pendiente de la recta simple", 3),
        ("RegRDosSimple", float(r2), "R cuadrado de la recta simple en entrenamiento", 3),
    ])
    return r


def poda(c):
    """Ruta de poda por costo-complejidad del árbol; elige el alfa con menor RMSE medio."""
    r = Resultado()
    c.arbol_completo = tuberias.construir_tuberia(TAREA, "arbol", c.numericas, c.categoricas,
                                                  hiper={"max_depth": None, "min_samples_leaf": 5})
    filas = []
    for alfa in tuberias.alfas_de_poda(c.arbol_completo, c.X_ent, c.y_ent):
        tuberia = clone(c.arbol_completo).set_params(modelo__ccp_alpha=float(alfa))
        rmse = validacion.puntuar_por_bloques(tuberia, c.X_ent, c.y_ent, c.bloques, "rmse")
        hojas = tuberia.fit(c.X_ent, c.y_ent).named_steps["modelo"].get_n_leaves()
        filas.append({"ccp_alpha": float(alfa), "Hojas": int(hojas), "RMSE medio": rmse.mean(),
                      "Desv. RMSE": rmse.std(ddof=1)})
    t = pd.DataFrame(filas)
    fila = t.loc[t["RMSE medio"].idxmin()]
    c.alfa = float(fila["ccp_alpha"])
    r.agregar(tabla_html(t, "Ruta de poda por costo-complejidad del árbol de regresión", decimales=3,
                         resaltar=(t["ccp_alpha"] == c.alfa).to_numpy(), columnas_resaltadas="RMSE medio"))
    r.agregar(nota(
        f"Sin poda el árbol tiene {int(t.iloc[0]['Hojas'])} hojas y un RMSE medio de {t.iloc[0]['RMSE medio']:.2f}. El alfa "
        f"elegido ({c.alfa:.3f}) lo deja en {int(fila['Hojas'])} hojas con RMSE {fila['RMSE medio']:.2f}"
        + (", o sea que podar mejora el error además de simplificar el árbol." if c.alfa > 0 else
           ". Es decir que en este caso ningún nivel de poda mejora al árbol completo."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("RegHojasSinPoda", int(t.iloc[0]["Hojas"]), "hojas del árbol de regresión sin poda"),
        ("RegRMSESinPoda", float(t.iloc[0]["RMSE medio"]), "RMSE medio del árbol sin poda", 2),
        ("RegAlfaPoda", c.alfa, "ccp_alpha elegido para el árbol didáctico", 3),
        ("RegHojasPodado", int(fila["Hojas"]), "hojas del árbol podado"),
        ("RegRMSEPodado", float(fila["RMSE medio"]), "RMSE medio del árbol podado", 2),
    ])
    latex.exportar_tabla(t, "reg_poda", "Ruta de poda del árbol de regresión", decimales=3)
    return r


def _fila_resumen(c, nombre, errores, tipo):
    mae_m, _ = _media_desv(errores["MAE"])
    rmse_m, rmse_d = _media_desv(errores["RMSE"])
    r2_m, r2_d = _media_desv(errores["R2"])
    habilidades = 1 - errores["RMSE"] / c.rmse_ref_bloques
    return {"Modelo": nombre, "Tipo": tipo, "MAE": mae_m, "RMSE": rmse_m, "± RMSE": rmse_d, "R²": r2_m,
            "± R²": r2_d, "Habilidad": float(habilidades.mean()), "Habilidad mínima": float(habilidades.min())}


def didactica(c):
    """Los modelos del curso con su configuración de partida, en los mismos bloques."""
    r = Resultado()
    c.didacticos = {
        f"Lineal simple ({c.variable_simple})": c.recta,
        "Lineal múltiple": tuberias.construir_tuberia(TAREA, "lineal", c.numericas, c.categoricas),
        "KNN": tuberias.construir_tuberia(TAREA, "knn", c.numericas, c.categoricas),
        "Árbol podado": clone(c.arbol_completo).set_params(modelo__ccp_alpha=c.alfa),
        "Perceptrón multicapa": tuberias.construir_tuberia(TAREA, "mlp", c.numericas, c.categoricas),
    }
    errores = {n: {m: validacion.puntuar_por_bloques(e, c.X_ent, c.y_ent, c.bloques, clave) for m, clave in TRES_METRICAS}
               for n, e in c.didacticos.items()}
    c.tabla_didactica = t = pd.DataFrame([_fila_resumen(c, n, e, "referencia") for n, e in c.filas_ref.items()]
                                         + [_fila_resumen(c, n, e, "modelo") for n, e in errores.items()])
    mejor = t[t["Tipo"] == "modelo"].sort_values("RMSE").iloc[0]
    r.agregar(tabla_html(t, f"Pasada didáctica en los {len(c.bloques)} bloques (media y desviación entre bloques, "
                            f"habilidad frente a \"{c.mejor_ref}\")", decimales=3,
                         resaltar=(t["Modelo"] == mejor["Modelo"]).to_numpy(), columnas_resaltadas="RMSE"))
    rmse = pd.DataFrame({n: e["RMSE"] for n, e in errores.items()}, index=c.etiquetas_bloque).T
    r.figura(graficos.figura_por_bloque(pd.concat([rmse, c.rmse_referencias.loc[[c.mejor_ref]]]), "RMSE (muestras por día)",
                                        resaltar=[mejor["Modelo"], c.mejor_ref]), "03_didactico_bloques")
    por_modelo = t.set_index("Modelo")
    rmse_mlp, rmse_lineal = por_modelo.loc["Perceptrón multicapa", "RMSE"], por_modelo.loc["Lineal múltiple", "RMSE"]
    positivos = t[(t["Tipo"] == "modelo") & (t["Habilidad mínima"] > 0)]
    r.agregar(nota(
        f"El mejor modelo de la pasada es \"{mejor['Modelo']}\", con RMSE medio **{mejor['RMSE']:.2f}** y una habilidad "
        f"de **{mejor['Habilidad']:.3f}** frente a \"{c.mejor_ref}\" (la mínima por bloque es "
        f"{mejor['Habilidad mínima']:.3f}). O sea que elimina cerca del {100 * mejor['Habilidad']:.0f} % del error de la "
        "referencia. "
        + (f"Los {len(c.didacticos)} modelos" if len(positivos) == len(c.didacticos) else
           f"{len(positivos)} de los {len(c.didacticos)} modelos")
        + " le ganan a la referencia en todos los bloques.\n\n"
        + (f"El perceptrón ({rmse_mlp:.2f}) no mejora a la regresión lineal múltiple ({rmse_lineal:.2f}), lo que sugiere "
           "que con estas variables la relación es casi lineal y la red no encuentra estructura adicional que justifique "
           "su costo." if rmse_mlp >= rmse_lineal else
           f"El perceptrón ({rmse_mlp:.2f}) mejora a la regresión lineal múltiple ({rmse_lineal:.2f}) en "
           f"{100 * (1 - rmse_mlp / rmse_lineal):.1f} % así que hay algo de no linealidad que la red aprovecha."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("RegMejorDidactico", mejor["Modelo"], "mejor modelo de la pasada didáctica"),
        ("RegRMSEMejorDidactico", float(mejor["RMSE"]), "su RMSE medio por bloque", 2),
        ("RegHabilidadMejorDidactico", float(mejor["Habilidad"]), "su habilidad media frente a la referencia", 3),
        ("RegHabilidadMinMejorDidactico", float(mejor["Habilidad mínima"]), "su habilidad mínima por bloque", 3),
        ("RegDidacticosEstables", len(positivos), "modelos didácticos que superan a la referencia en todos los bloques"),
        ("RegDidacticos", len(c.didacticos), "modelos de la pasada didáctica"),
        ("RegRMSEMLPDidactico", float(rmse_mlp), "RMSE medio del perceptrón didáctico", 2),
        ("RegRMSELinealDidactico", float(rmse_lineal), "RMSE medio de la regresión lineal múltiple didáctica", 2),
    ])
    latex.exportar_tabla(t, "reg_didactica", "Pasada didáctica de regresión (media y desviación entre bloques)",
                         decimales=3)
    return r


def coeficientes(c):
    """Coeficientes estandarizados de la regresión lineal múltiple."""
    r = Resultado()
    ajustada = clone(c.didacticos["Lineal múltiple"]).fit(c.X_ent, c.y_ent)
    coef = pd.Series(ajustada.named_steps["modelo"].coef_, index=tuberias.nombres_variables(ajustada))
    r.figura(graficos.figura_coeficientes(coef, "Coeficientes estandarizados de la regresión múltiple"), "03_coeficientes")
    principales = coef.reindex(coef.abs().sort_values(ascending=False).index).head(3)
    corr = c.entrenamiento[c.numericas].corr().abs().where(~np.eye(len(c.numericas), dtype=bool))
    par = corr.stack().idxmax()
    r.agregar(nota(
        "Los tres coeficientes más grandes son " + enumerar(f'"{n}" ({v:+.2f})' for n, v in principales.items())
        + f". Eso sí, varias variables son casi redundantes (la correlación más alta, entre \"{par[0]}\" y "
          f"\"{par[1]}\", es {corr.loc[par]:.2f}) así que el peso se distribuye entre ellas de forma inestable. Los "
          "coeficientes describen el modelo ajustado y no el efecto causal de cada variable.",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("RegCoefPrincipal", principales.index[0], "variable con el mayor coeficiente estandarizado"),
        ("RegCoefPrincipalValor", float(principales.iloc[0]), "valor de ese coeficiente", 2),
        ("RegParRedundante", f"{par[0]} y {par[1]}", "par de variables numéricas más correlacionado"),
        ("RegCorrParRedundante", float(corr.loc[par]), "su correlación absoluta", 2),
    ])
    latex.exportar_tabla(principales.rename_axis("Variable").reset_index(name="Coeficiente"), "reg_coeficientes",
                         "Tres coeficientes estandarizados de mayor tamaño (regresión múltiple)", decimales=3)
    return r


def diseno(c):
    """Factores del L16, diseño con niveles reales y tabla de confusiones."""
    r = Resultado()
    c.columnas_todas = c.numericas + c.categoricas
    c.k_seleccion = max(1, round(len(c.columnas_todas) / 2))
    c.discretas = taguchi.discretas(c.entrenamiento, c.numericas)
    c.factores = {
        "modelo": list(registro.MODELOS[TAREA]), "escalado": ["estandar", "minmax"], "grupo": ["base", "ampliado"],
        "objetivo_log": [False, True], "seleccion": [None, c.k_seleccion], "codificacion": ["ordinal", "one_hot"],
    }
    c.nombres_factor = list(c.factores)
    c.construir = taguchi.constructor(TAREA, c.papeles, c.discretas)
    arreglo, c.diseno_l16 = taguchi.diseno_l16(c.factores, COLUMNAS_L16)
    ortogonal = arreglos.verificar_ortogonalidad(arreglo)
    r.agregar(pasos([
        ("modelo (columna A, 4 niveles)", ", ".join(registro.NOMBRES_LEGIBLES[m] for m in c.factores["modelo"][:-1])
         + " o " + registro.NOMBRES_LEGIBLES[c.factores["modelo"][-1]] + "."),
        ("escalado (c)", "Estandarizar (restar la media y dividir entre la desviación estándar) o llevar al intervalo "
                         "[0, 1] las variables numéricas."),
        ("grupo (d)", f"Grupo base ({len(c.papeles['grupos']['base'])} variables) o ampliado ({len(c.columnas_todas)})."),
        ("objetivo_log (abd)", "Ajustar sobre log(1 + y) y devolver la predicción en la escala original, o no hacerlo."),
        ("seleccion (acd)", f"Sin selección o conservar las {c.k_seleccion} variables con mayor estadístico F (la mitad del "
                            "grupo ampliado). En el grupo base, que tiene menos, se conservan todas."),
        ("codificacion (bcd)", "Cómo entran las variables discretas (" + ", ".join(f'"{x}"' for x in c.discretas)
         + "): Como un entero ordenado o como categorías one-hot (una columna de ceros y unos por categoría)."),
    ], titulo=f"Factores del L16 (ortogonalidad verificada: {'Sí' if ortogonal else 'No'})"))
    confusiones = arreglos.confusiones(COLUMNAS_L16)
    vista = taguchi.vista_confusiones(confusiones)
    r.agregar(pestanas({
        "Diseño con niveles reales": tabla_html(taguchi.vista_niveles(c.diseno_l16, c.nombres_factor)
                                                .reset_index(names="Corrida"), "Diseño L16 con los niveles reales",
                                                decimales=0, maximo_filas=16),
        "Confusiones": tabla_html(vista, "Dónde cae cada interacción de dos factores", decimales=0,
                                  resaltar=(vista["Confundida con"] == "libre").to_numpy()),
    }, titulo="El diseño que corrimos"))
    libres = taguchi.interacciones_libres(confusiones)
    mezcladas = confusiones[confusiones["confundida_con"].map(len) > 0]
    frecuencia = pd.Series([f for lista in mezcladas["confundida_con"] for f in lista]).value_counts()
    r.agregar(nota(
        f"De las {len(confusiones)} interacciones de dos factores {len(libres)} quedan libres ("
        + ", ".join(x.replace(" × ", " con ") for x in libres)
        + ") y las demás se mezclan con algún efecto principal. El factor que más interacciones absorbe es "
          f"\"{frecuencia.index[0]}\" ({int(frecuencia.iloc[0])}). La asignación protege a propósito la interacción "
          "entre modelo y escalado, la única que esperábamos grande de antemano porque el escalado cambia las distancias "
          "de KNN y el entrenamiento del perceptrón pero no afecta al árbol ni a la regresión lineal. Las demás "
          "interacciones las suponemos pequeñas frente a los efectos principales y ese supuesto lo pone a prueba la "
          "corrida de confirmación: Si el modelo aditivo acierta, las interacciones que ignoramos no pesaban.",
        "info", titulo="Qué se confunde y por qué lo aceptamos",
    ))
    latex.registrar_varias([
        ("RegFactores", len(c.factores), "factores del L16 de regresión"),
        ("RegCorridasCompleto", int(np.prod([len(v) for v in c.factores.values()])), "corridas del factorial completo"),
        ("RegKSeleccion", c.k_seleccion, "variables que conserva la selección por F"),
        ("RegInteracciones", len(confusiones), "interacciones de dos factores"),
        ("RegInteraccionesLibres", len(libres), "interacciones que quedan libres en el L16"),
    ])
    latex.exportar_tabla(taguchi.vista_niveles(c.diseno_l16, c.nombres_factor).reset_index(names="Corrida"),
                         "reg_l16_diseno", "Diseño L16 de regresión con los niveles reales", decimales=0)
    return r


def corridas_l16(c):
    """Ejecuta las 16 corridas, efectos principales, tabla de respuesta y óptimo aditivo."""
    r = Resultado()
    c.resultados_l16 = res = corridas.ejecutar_diseno(c.diseno_l16, c.construir, c.X_ent, c.y_ent, c.bloques, "rmse")
    c.segundos_l16 = float(res["segundos"].sum())
    columnas_bloque = [x for x in res.columns if x.startswith("bloque_")]
    vista = taguchi.vista_niveles(res[c.nombres_factor + columnas_bloque + ["media", "desv", "sn", "segundos"]],
                                  c.nombres_factor)
    c.mejor_corrida = mejor = res["sn"].idxmax()
    tabla_corridas = tabla_html(vista.reset_index(names="Corrida"), "Resultados del L16: RMSE por bloque, media, "
                                "desviación y S/N (dB)", decimales=2, resaltar=(res.index == mejor),
                                columnas_resaltadas="sn", maximo_filas=16)
    fallidas = res[res["error"] != ""]
    r.agregar(nota(
        f"Las 16 corridas tardaron **{c.segundos_l16:.1f} s** en total"
        + (f" y {len(fallidas)} fallaron ({'; '.join(fallidas['error'])})." if len(fallidas) else ", sin fallos.")
        + f" La mejor corrida observada es la {mejor} ("
        + ", ".join(f"{f} = {legible(res.loc[mejor, f])}" for f in c.nombres_factor)
        + f"), con S/N {res.loc[mejor, 'sn']:.2f} dB y RMSE medio {res.loc[mejor, 'media']:.2f}.",
        "hallazgo" if not len(fallidas) else "advertencia",
    ))
    c.efectos = corridas.efectos_principales(res, c.nombres_factor)
    c.respuesta = corridas.tabla_respuesta(res, c.nombres_factor)
    c.niveles_optimo, c.sn_predicha = corridas.predecir_optimo(res, c.nombres_factor)
    media_sn = float(res["sn"].mean())
    r.figura(graficos.figura_efectos_principales(c.efectos, media_sn, optimo=c.niveles_optimo), "03_efectos_principales")
    vr = taguchi.respuesta_legible(c.respuesta, c.nombres_factor)
    vr_vista = vr.rename(columns={"mejor_nivel": "Mejor nivel", "media_mejor": "S/N mejor", "peor_nivel": "Peor nivel",
                                  "media_peor": "S/N peor", "delta": "Delta", "rango": "Rango"})
    r.agregar(pestanas({"Tabla de respuesta": tabla_html(vr_vista, "Tabla de respuesta de Taguchi (S/N en dB)",
                                                         decimales=2, barras="Delta"),
                        "Las 16 corridas": tabla_corridas}, titulo="Detalle del L16"))

    interaccion = res.pivot_table(index="modelo", columns="escalado", values="sn", aggfunc="mean")
    efecto_escalado = (interaccion["estandar"] - interaccion["minmax"]).abs()
    primero, segundo = vr.iloc[0], vr.iloc[1]
    indiferentes = c.respuesta.loc[c.respuesta["delta"] < 0.1 * c.respuesta["delta"].max(), "delta"]
    ef = c.efectos
    sn_log = ef.loc[(ef["factor"] == "objetivo_log") & (ef["nivel"] == True), "media"].iloc[0]  # noqa: E712
    sn_sin_log = ef.loc[(ef["factor"] == "objetivo_log") & (ef["nivel"] == False), "media"].iloc[0]  # noqa: E712
    r.agregar(nota(
        f"El factor que más pesa es \"{primero['Factor']}\" (delta de {primero['delta']:.2f} dB, mejor nivel "
        f"\"{primero['mejor_nivel']}\"), seguido de \"{segundo['Factor']}\" ({segundo['delta']:.2f} dB, mejor nivel "
        f"\"{segundo['mejor_nivel']}\"). "
        + (f"Llevar el objetivo a logaritmo {'ayuda' if c.niveles_optimo['objetivo_log'] else 'no ayuda'}: La S/N media es "
           f"{sn_log:.2f} dB con logaritmo y {sn_sin_log:.2f} sin él. ")
        + f"Como la interacción entre modelo y escalado quedó libre podemos leerla por separado: Cambiar el escalado mueve "
          f"la S/N {efecto_escalado.max():.2f} dB en \"{efecto_escalado.idxmax()}\" y solo {efecto_escalado.min():.2f} dB "
          f"en \"{efecto_escalado.idxmin()}\". "
        + (("Los factores " if len(indiferentes) > 1 else "El factor ") + enumerar(
            f'"{f}" (delta {d:.2f} dB)' for f, d in indiferentes.items())
           + (" son prácticamente indiferentes" if len(indiferentes) > 1 else " es prácticamente indiferente")
           + " en esta tabla (menos del 10 % del delta mayor) así que su nivel óptimo lo deciden diferencias sin peso "
             "práctico." if len(indiferentes) else "")
        + "\n\nEl óptimo que predice el modelo aditivo es "
        + ", ".join(f"{f} = {legible(v)}" for f, v in c.niveles_optimo.items())
        + f", con S/N predicha de **{c.sn_predicha:.2f} dB**.",
        "hallazgo",
    ))
    latex.registrar_varias([
        ("RegSegundosLSixteen", c.segundos_l16, "segundos de las 16 corridas del L16", 1),
        ("RegMejorCorrida", int(mejor), "número de la mejor corrida observada del L16"),
        ("RegSNMejorCorrida", float(res.loc[mejor, "sn"]), "S/N (dB) de la mejor corrida", 2),
        ("RegRMSEMejorCorrida", float(res.loc[mejor, "media"]), "RMSE medio de la mejor corrida", 2),
        ("RegFactorPrincipal", primero["Factor"], "factor más influyente (mayor delta)"),
        ("RegDeltaPrincipal", float(primero["delta"]), "delta (dB) del factor más influyente", 2),
        ("RegFactorSegundo", segundo["Factor"], "segundo factor más influyente"),
        ("RegDeltaSegundo", float(segundo["delta"]), "delta (dB) del segundo factor", 2),
        ("RegSNPredicha", c.sn_predicha, "S/N (dB) predicha para el óptimo aditivo", 2),
        ("RegOptimo", ", ".join(f"{f} = {legible(v)}" for f, v in c.niveles_optimo.items()), "óptimo aditivo del L16"),
    ])
    latex.exportar_tabla(vista.reset_index(names="Corrida"), "reg_l16_resultados", "Resultados del L16 de regresión",
                         decimales=2)
    latex.exportar_tabla(vr_vista, "reg_respuesta", "Tabla de respuesta de Taguchi (regresión, S/N en dB)", decimales=2)
    return r


def confirmacion(c):
    """Corrida de confirmación del óptimo y, si cabe en el tiempo, factorial completo."""
    r = Resultado()
    res = c.resultados_l16
    c.conf = conf = taguchi.confirmar(c.niveles_optimo, c.construir, c.X_ent, c.y_ent, c.bloques, "rmse", res,
                                      c.efectos, c.factores)
    obs, tol = conf["sn_observada"], conf["tolerancia"]
    diferencia = obs - c.sn_predicha
    c.coincide = abs(diferencia) <= tol
    mejor = c.mejor_corrida
    r.agregar(comparacion([
        {"titulo": "S/N predicha", "valor": c.sn_predicha, "etiqueta": "dB", "puntos": ["modelo aditivo del L16"]},
        {"titulo": "S/N observada", "valor": obs, "etiqueta": "dB",
         "puntos": [f"RMSE medio {conf['fila']['media']:.2f}", f"desviación entre bloques {conf['fila']['desv']:.2f}"]},
        {"titulo": "Mejor corrida del L16", "valor": float(res.loc[mejor, "sn"]), "etiqueta": "dB",
         "puntos": [f"corrida {mejor}", f"RMSE medio {res.loc[mejor, 'media']:.2f}"]},
    ], titulo="Confirmación del óptimo", decimales=2,
        elegida="S/N observada" if obs >= res["sn"].max() else "Mejor corrida del L16", sello="Mayor S/N"))
    r.agregar(nota(
        f"La S/N observada ({obs:.2f} dB) difiere de la predicha en {diferencia:+.2f} dB, "
        + (f"dentro de la tolerancia de ±{tol:.2f} dB. O sea que el modelo aditivo describe bien la tubería y las "
           "interacciones que el L16 dejó confundidas no parecen importar." if c.coincide else
           f"fuera de la tolerancia de ±{tol:.2f} dB, lo que indica que alguna interacción confundida pesa y que el óptimo "
           "aditivo hay que tomarlo con cautela.")
        + (" Además la confirmación le gana a la mejor de las 16 corridas así que el diseño encontró una combinación "
           "mejor que todas las que llegó a probar." if obs > res["sn"].max() else
           " Con todo, la confirmación no le gana a la mejor corrida observada del L16."),
        "hallazgo" if c.coincide else "advertencia", titulo="Predicho frente a observado",
    ))
    latex.registrar_varias([
        ("RegSNObservada", obs, "S/N (dB) observada en la corrida de confirmación", 2),
        ("RegToleranciaSN", tol, "tolerancia (dB) de la confirmación", 2),
        ("RegConfirmaAditivo", bool(c.coincide), "la confirmación cae dentro de la tolerancia"),
        ("RegRMSEConfirmacion", float(conf["fila"]["media"]), "RMSE medio de la corrida de confirmación", 2),
    ])

    c.resumen_completo, detalle = taguchi.factorial_completo(
        c.factores, c.segundos_l16, len(c.diseno_l16), c.construir, c.X_ent, c.y_ent, c.bloques, "rmse", c.respuesta,
        c.niveles_optimo)
    rc = c.resumen_completo
    if detalle is not None:
        completo, respuesta_completo = detalle["resultados"], detalle["respuesta"]
        mejor_completo = completo.loc[completo["sn"].idxmax()]
        puesto = int((completo["sn"] > obs).sum()) + 1
        r.figura(graficos.figura_deltas(pd.DataFrame({
            "L16 (16 corridas)": c.respuesta["delta"],
            f"Factorial completo ({rc['corridas']} corridas)": respuesta_completo.loc[c.respuesta.index, "delta"]}),
            "Influencia de cada factor según el diseño"), "03_l16_contra_completo")
        rc.update({"mejor_sn_completo": float(mejor_completo["sn"]), "puesto_confirmacion": puesto,
                   "deltas": respuesta_completo["delta"].to_dict()})
        rho, s_comp, iguales = rc["correlacion_rangos"], rc["segundos"], rc["factores_mismo_optimo"]
        r.agregar(nota(
            f"Para comprobar el atajo corrimos también el factorial completo, que tardó {s_comp:.1f} s frente a "
            f"{c.segundos_l16:.1f} s del L16 (el {100 * (1 - c.segundos_l16 / s_comp):.0f} % menos). La correlación de "
            f"rangos de Spearman entre los deltas de los dos diseños es **{rho:.2f}**"
            + (f" así que con solo 16 corridas ordenamos los factores casi igual que con las {rc['corridas']}"
               if rho >= 0.8 else
               " así que el L16 ordena los factores de forma solo parcialmente parecida al diseño completo")
            + f", y los dos coinciden en el mejor nivel de {len(iguales)} de los {len(c.nombres_factor)} factores. La "
              f"combinación de confirmación queda en el puesto {puesto} de las {rc['corridas']} (la mejor tiene S/N "
              f"{mejor_completo['sn']:.2f} dB frente a {obs:.2f}).",
            "hallazgo",
        ))
        latex.registrar_varias([
            ("RegSegundosCompleto", s_comp, "segundos del factorial completo", 1),
            ("RegRhoCompleto", rho, "correlación de Spearman entre deltas del L16 y del factorial completo", 2),
            ("RegFactoresMismoOptimo", len(iguales), "factores con el mismo mejor nivel en ambos diseños"),
            ("RegPuestoConfirmacion", puesto, "puesto de la confirmación entre las combinaciones del factorial"),
            ("RegSNMejorCompleto", float(mejor_completo["sn"]), "mejor S/N (dB) del factorial completo", 2),
        ])
    else:
        r.agregar(nota(
            f"Con {c.segundos_l16:.1f} s para el L16 el factorial completo tardaría unos {rc['segundos_estimados']:.0f} s, "
            f"por encima del límite de {taguchi.LIMITE_SEGUNDOS_COMPLETO:.0f} s que fijamos para esta parte. Por eso no lo "
            "corrimos y la comparación queda pendiente.",
            "limite",
        ))
    latex.registrar("RegCompletoEjecutado", rc["ejecutado"], "se ejecutó el factorial completo")
    return r


def afinamiento(c):
    """Rejilla de hiperparámetros de la mejor configuración y de la segunda familia."""
    r = Resultado()
    c.configuracion, origen, c.candidatos = taguchi.configuracion_y_candidatos(
        c.conf, c.niveles_optimo, c.resultados_l16, c.efectos, c.nombres_factor)
    c.busquedas, c.tablas_rejilla, c.segundos_rejilla = taguchi.buscar_rejillas(
        TAREA, c.candidatos, c.construir, c.X_ent, c.y_ent, c.bloques, "rmse")
    r.agregar(pestanas({
        registro.NOMBRES_LEGIBLES[m]: tabla_html(
            t.drop(columns=[x for x in t.columns if x.startswith("bloque_")]).head(8),
            f"Mejores combinaciones de la rejilla ({len(t)} en total, {c.segundos_rejilla[m]:.1f} s)", decimales=4)
        for m, t in c.tablas_rejilla.items()}, titulo="Búsqueda en rejilla sobre los bloques temporales"))
    r.agregar(nota(
        f"La configuración de partida sale de la {origen} ("
        + ", ".join(f"{f} = {legible(v)}" for f, v in c.configuracion.items() if f != "modelo") + "). "
        + " ".join(
            f"En \"{registro.NOMBRES_LEGIBLES[m]}\" la rejilla elige "
            + (", ".join(f"{k} = {legible(v)}" for k, v in taguchi.hiper_de(b).items())
               or "la única configuración (no tiene hiperparámetros)")
            + f", con RMSE medio de {c.tablas_rejilla[m].iloc[0]['media']:.2f} en los bloques."
            for m, b in c.busquedas.items()),
        "hallazgo",
    ))
    latex.registrar("RegOrigenConfiguracion", origen, "de dónde sale la configuración que pasa al afinamiento")
    for m in c.busquedas:
        clave = {"lineal": "Lineal", "arbol": "Arbol", "knn": "KNN", "mlp": "MLP"}[m]
        latex.registrar(f"RegRejilla{clave}RMSE", float(c.tablas_rejilla[m].iloc[0]["media"]),
                        f"RMSE medio por bloque de la mejor combinación de la rejilla de {m}", 2)
    return r


def evaluacion_validacion(c):
    """Evaluación única de los candidatos en validación, con intervalos bootstrap, y elección final."""
    r = Resultado()
    c.pred_val = {m: metricas.prediccion_de(b.best_estimator_, c.X_val) for m, b in c.busquedas.items()}
    ref_val = {ref: c.validacion[ref].to_numpy(dtype=float) for ref in c.referencias}
    ref_val[NOMBRE_DUMMY] = np.full(len(c.y_val), float(np.mean(c.y_ent)))
    c.base_val = ref_val[c.mejor_ref]

    def habilidad_rmse(y, pred, base):
        return 1 - metricas.rmse(y, pred) / metricas.rmse(y, base)

    filas, c.intervalos = [], {}
    for nombre, pred in {**{registro.NOMBRES_LEGIBLES[m]: p for m, p in c.pred_val.items()}, **ref_val}.items():
        fila = {"Predictor": nombre, **metricas.metricas_reg(c.y_val, pred)}
        fila["Habilidad"] = habilidad_rmse(c.y_val, pred, c.base_val)
        filas.append(fila)
    for m, pred in c.pred_val.items():
        c.intervalos[m] = {"rmse": intervalo_bootstrap(metricas.rmse, c.y_val, pred),
                           "habilidad": intervalo_bootstrap(habilidad_rmse, c.y_val, pred, c.base_val)}
    c.tabla_val = t = pd.DataFrame(filas)
    c.rmse_val = {m: metricas.rmse(c.y_val, p) for m, p in c.pred_val.items()}
    menor = min(c.rmse_val, key=c.rmse_val.get)
    empatados = [m for m in c.rmse_val if c.rmse_val[m] <= c.rmse_val[menor] * (1 + TOLERANCIA_EMPATE)]
    c.final = final = min(empatados, key=ORDEN_SENCILLEZ.index)
    for m in c.pred_val:
        fila = t["Predictor"] == registro.NOMBRES_LEGIBLES[m]
        t.loc[fila, "RMSE inf."] = c.intervalos[m]["rmse"][1]
        t.loc[fila, "RMSE sup."] = c.intervalos[m]["rmse"][2]
        t.loc[fila, "Habilidad inf."] = c.intervalos[m]["habilidad"][1]
        t.loc[fila, "Habilidad sup."] = c.intervalos[m]["habilidad"][2]
    r.agregar(tabla_html(t, f"Bloque de validación ({len(c.y_val)} días, habilidad frente a \"{c.mejor_ref}\" e "
                            "intervalos bootstrap al 95 %)", decimales=3,
                         resaltar=(t["Predictor"] == registro.NOMBRES_LEGIBLES[final]).to_numpy(),
                         columnas_resaltadas="RMSE"))
    h, inf_h, sup_h = c.intervalos[final]["habilidad"]
    otro = next(m for m in c.pred_val if m != final)
    r.agregar(nota(
        f"Según la regla el candidato final es \"{registro.NOMBRES_LEGIBLES[final]}\", con RMSE de validación "
        f"**{c.rmse_val[final]:.2f}** frente a {c.rmse_val[otro]:.2f} de \"{registro.NOMBRES_LEGIBLES[otro]}\""
        + (f" (la diferencia cae dentro del {100 * TOLERANCIA_EMPATE:.0f} % y preferimos el más sencillo)."
           if final != menor else ".")
        + f" Su habilidad frente a \"{c.mejor_ref}\" es **{h:.3f}**, con un intervalo bootstrap de {inf_h:.3f} a "
          f"{sup_h:.3f}"
        + (", que no incluye el cero: La mejora sobre la referencia no se explica por el azar de qué días cayeron en la "
           "validación." if inf_h > 0 else
           ", que incluye el cero: Con solo estos días de validación no podemos descartar que el modelo no mejore a la "
           "referencia.")
        + " Eso sí, el intervalo remuestrea días sueltos e ignora que los días seguidos se parecen así que es algo más "
          "estrecho de lo que daría un remuestreo por bloques.",
        "hallazgo" if inf_h > 0 else "advertencia",
    ))
    fila_final = t.set_index("Predictor").loc[registro.NOMBRES_LEGIBLES[final]]
    rmse_inf, rmse_sup = c.intervalos[final]["rmse"][1], c.intervalos[final]["rmse"][2]
    latex.registrar_varias([
        ("RegModeloFinal", registro.NOMBRES_LEGIBLES[final], "modelo de regresión congelado"),
        ("RegModeloOtro", registro.NOMBRES_LEGIBLES[otro], "otro candidato evaluado en validación"),
        ("RegRMSEValidacion", c.rmse_val[final], "RMSE del modelo final en validación", 2),
        ("RegRMSEValidacionInf", rmse_inf, "límite inferior del IC 95 % del RMSE en validación", 2),
        ("RegRMSEValidacionSup", rmse_sup, "límite superior del IC 95 % del RMSE en validación", 2),
        ("RegRMSEValidacionOtro", c.rmse_val[otro], "RMSE del otro candidato en validación", 2),
        ("RegMAEValidacion", float(fila_final["MAE"]), "MAE del modelo final en validación", 2),
        ("RegRDosValidacion", float(fila_final["R2"]), "R cuadrado del modelo final en validación", 3),
        ("RegHabilidadValidacion", h, "habilidad del modelo final en validación", 3),
        ("RegHabilidadValidacionInf", inf_h, "límite inferior del IC 95 % de la habilidad en validación", 3),
        ("RegHabilidadValidacionSup", sup_h, "límite superior del IC 95 % de la habilidad en validación", 3),
        ("RegPorEmpate", bool(final != menor), "el final se eligió por la regla de empate"),
        ("RegToleranciaEmpatePct", 100 * TOLERANCIA_EMPATE, "tolerancia de empate (%) de la regla de elección", 0),
    ])
    latex.exportar_tabla(t, "reg_validacion", "Bloque de validación de la regresión con intervalos bootstrap", decimales=3)
    return r


def diagnostico(c):
    """Serie y residuos del modelo final en validación."""
    r = Resultado()
    pred = c.pred_val[c.final]
    nombre = registro.NOMBRES_LEGIBLES[c.final]
    r.figura(graficos.figura_serie_prediccion(c.validacion[c.fecha], c.y_val, {nombre: pred, c.mejor_ref: c.base_val},
                                              unidad="muestras por día"), "03_validacion_serie")
    r.figura(graficos.figura_residuos(c.validacion[c.fecha], c.y_val, pred, unidad="muestras por día"), "03_residuos")
    residuo = c.y_val - pred
    mitad = len(residuo) // 2
    corr = float(np.corrcoef(pred, np.abs(residuo))[0, 1])
    r.agregar(nota(
        f"El residuo medio (lo observado menos lo predicho) en validación es {residuo.mean():+.2f} muestras por día "
        + ("(el modelo tiende a quedarse corto)" if residuo.mean() > 0 else "(el modelo tiende a pasarse)")
        + f", con {residuo[:mitad].mean():+.2f} en la primera mitad del periodo y {residuo[mitad:].mean():+.2f} en la "
          "segunda. "
        + (f"El tamaño del error crece con el valor predicho (correlación de {corr:.2f} entre la predicción y el error "
           "absoluto), que es lo esperable en un conteo: Los días de más carga son también los más variables."
           if corr > 0.2 else
           f"El tamaño del error casi no depende del valor predicho (correlación de {corr:.2f}) así que la dispersión es "
           "parecida en días de poca y de mucha carga."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("RegResiduoMedioValidacion", float(residuo.mean()), "residuo medio (observado - predicho) en validación", 2),
        ("RegCorrResiduo", corr, "correlación entre la predicción y el error absoluto en validación", 2),
    ])
    return r


def congelar(c):
    """Reajusta el modelo elegido con todo el desarrollo y lo guarda."""
    r = Resultado()
    X_des = c.desarrollo[c.numericas + c.categoricas]
    y_des = c.desarrollo[c.objetivo].to_numpy(dtype=float)
    c.modelo_final = clone(c.busquedas[c.final].best_estimator_).fit(X_des, y_des)
    guardar_modelo(c.modelo_final, rutas.DIR_ARTEFACTOS / "modelos" / "regresion.joblib")
    c.configuracion_final = {**c.candidatos[c.final]}
    c.variables_finales = list(tuberias.nombres_variables(c.modelo_final))
    h, inf_h, sup_h = c.intervalos[c.final]["habilidad"]
    cfg = c.configuracion_final
    r.agregar(tarjetas([
        {"titulo": "Modelo congelado", "valor": c.rmse_val[c.final], "decimales": 2, "prefijo": "RMSE ",
         "detalle": registro.NOMBRES_LEGIBLES[c.final]},
        {"titulo": "Habilidad en validación", "valor": h, "decimales": 3, "color": "verde",
         "detalle": f"intervalo al 95 % de {inf_h:.3f} a {sup_h:.3f}"},
        {"titulo": "Variables que ve el modelo", "valor": len(c.variables_finales), "decimales": 0, "color": "morado",
         "detalle": f"grupo {cfg['grupo']}, selección {legible(cfg['seleccion'])}"},
        {"titulo": "Filas del reajuste", "valor": len(c.desarrollo), "decimales": 0, "color": "naranja",
         "detalle": "entrenamiento + validación"},
    ]))
    hiper = taguchi.hiper_de(c.busquedas[c.final])
    c.taguchi = taguchi.taguchi_para_artefacto(c.factores, COLUMNAS_L16, c.respuesta, c.niveles_optimo, c.sn_predicha,
                                               c.conf, c.coincide, c.segundos_l16, c.resumen_completo,
                                               efectos=c.efectos, resultados=c.resultados_l16)
    latex.registrar_varias([
        ("RegVariablesModelo", len(c.variables_finales), "variables que ve el modelo congelado (tras codificar)"),
        ("RegFilasReajuste", len(c.desarrollo), "filas del reajuste final (entrenamiento + validación)"),
        ("RegConfiguracionFinal", ", ".join(f"{f} = {legible(v)}" for f, v in cfg.items()), "configuración congelada"),
        ("RegHiperparametros", ", ".join(f"{k} = {legible(v)}" for k, v in hiper.items()) or "ninguno",
         "hiperparámetros congelados"),
    ])
    return r


def operacion(c):
    """Error en muestras por día y días fuera de una tolerancia práctica; artefacto de la regresión."""
    r = Resultado()
    pred = c.pred_val[c.final]
    nombre = registro.NOMBRES_LEGIBLES[c.final]
    media = float(np.mean(c.y_val))
    tolerancia = FRACCION_TOLERANCIA * media
    mae_final, mae_ref = metricas.mae(c.y_val, pred), metricas.mae(c.y_val, c.base_val)
    fuera_final = float(np.mean(np.abs(c.y_val - pred) > tolerancia))
    fuera_ref = float(np.mean(np.abs(c.y_val - c.base_val) > tolerancia))
    r.agregar(comparacion([
        {"titulo": c.mejor_ref, "valor": 100 * fuera_ref, "etiqueta": "% de días fuera de tolerancia", "decimales": 1,
         "sufijo": " %", "puntos": [f"error absoluto medio {mae_ref:.1f} muestras"]},
        {"titulo": nombre, "valor": 100 * fuera_final, "decimales": 1, "sufijo": " %",
         "etiqueta": "% de días fuera de tolerancia", "puntos": [f"error absoluto medio {mae_final:.1f} muestras"]},
    ], titulo=f"Días con error mayor que {tolerancia:.1f} muestras ({100 * FRACCION_TOLERANCIA:.0f} % de la media diaria)",
        elegida=nombre if fuera_final <= fuera_ref else c.mejor_ref, sello="Menos días fuera"))
    r.agregar(nota(
        f"En validación el laboratorio recibió en promedio {media:.1f} muestras por día hábil y el modelo se equivoca en "
        f"{mae_final:.1f} muestras en un día típico (el {100 * mae_final / media:.0f} % de la media), frente a "
        f"{mae_ref:.1f} de \"{c.mejor_ref}\". "
        + (f"El error pasa la tolerancia de {tolerancia:.1f} muestras en el {100 * fuera_final:.0f} % de los días, contra "
           f"el {100 * fuera_ref:.0f} % de la referencia. O sea que el modelo reduce los días mal planificados pero no los "
           "elimina: Serviría para fijar la dotación base del día siguiente, con un margen que cubra la variación diaria."
           if fuera_final < fuera_ref else
           f"El error pasa la tolerancia de {tolerancia:.1f} muestras en el {100 * fuera_final:.0f} % de los días sin "
           f"mejorar el {100 * fuera_ref:.0f} % de la referencia así que en la práctica la ganancia en RMSE no se traduce "
           "en menos días mal planificados."),
        "hallazgo",
    ))
    latex.registrar_varias([
        ("RegMediaValidacion", media, "muestras por día hábil en promedio (validación)", 1),
        ("RegToleranciaOperativa", tolerancia, "tolerancia práctica en muestras por día", 1),
        ("RegToleranciaOperativaPct", 100 * FRACCION_TOLERANCIA, "tolerancia práctica (% de la media diaria)", 0),
        ("RegMAEOperacion", mae_final, "error absoluto medio del modelo en validación (muestras)", 1),
        ("RegMAEReferencia", mae_ref, "error absoluto medio de la referencia en validación (muestras)", 1),
        ("RegDiasFueraPct", 100 * fuera_final, "días de validación fuera de tolerancia (%), modelo", 1),
        ("RegDiasFueraReferenciaPct", 100 * fuera_ref, "días de validación fuera de tolerancia (%), referencia", 1),
    ])
    c.operacion = {"media_validacion": media, "mae": mae_final, "mae_referencia": mae_ref, "tolerancia": tolerancia,
                   "dias_fuera": fuera_final, "dias_fuera_referencia": fuera_ref}
    c.segundos = time.perf_counter() - c.inicio
    cfg = c.configuracion_final
    entrada = columnas_de_grupo(c.papeles, cfg["grupo"])
    guardar_json({
        "analisis": c.nombre, "objetivo": c.objetivo, "momento": c.papeles["momento"],
        "variables_entrada": [*entrada[0], *entrada[1]], "variables_modelo": c.variables_finales,
        "discretas": c.discretas, "modelo": c.final,
        "configuracion": {f: (None if v is None else v) for f, v in cfg.items()},
        "hiperparametros": taguchi.hiper_de(c.busquedas[c.final]),
        "regla_eleccion": f"menor RMSE de validación; empate (< {100 * TOLERANCIA_EMPATE:.0f} %) → más sencillo "
                          f"({' < '.join(ORDEN_SENCILLEZ)})",
        "reajuste": "entrenamiento + validación con hiperparámetros congelados",
        "referencia_habilidad": c.mejor_ref,
        "validacion": c.tabla_val.to_dict("records"),
        "intervalos": {m: {k: list(v) for k, v in d.items()} for m, d in c.intervalos.items()},
        "referencias_bloques": c.rmse_referencias.to_dict("index"),
        "didactica": c.tabla_didactica.to_dict("records"),
        "taguchi": c.taguchi,
        "rejilla": {m: {"hiper": taguchi.hiper_de(b), "rmse_bloques": float(c.tablas_rejilla[m].iloc[0]["media"]),
                        "segundos": c.segundos_rejilla[m]} for m, b in c.busquedas.items()},
        "operacion": c.operacion,
        "segundos_etapa": c.segundos,
    }, rutas.DIR_ARTEFACTOS / "03_regresion.json")
    return r


def terminar(c):
    cerrar({"analisis": c.nombre, "modelo": c.final, "rmse_validacion": c.rmse_val[c.final],
            "habilidad_validacion": c.intervalos[c.final]["habilidad"][0], "segundos": c.segundos})
