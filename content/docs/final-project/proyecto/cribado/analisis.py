"""Capa 6 · Cribado de análisis (nivel 1).

Responsabilidad única: someter cada análisis candidato del catálogo a la
misma prueba rápida y barata, reunir los resultados en una tabla y aplicar la
regla de selección que decide cuál se estudia a fondo en cada tipo.

La prueba usa solo el periodo de entrenamiento (fechas anteriores a
`config.CORTE_VALIDACION`) y bloques temporales de ventana creciente:

- Supervisados: línea base, modelo lineal o logístico y árbol poco profundo,
  sin afinar y con el grupo de variables `ampliado`. Se mide la habilidad
  (ganancia sobre la línea base) en cada bloque, y cuánto de ella logra un
  árbol que solo ve la mejor variable («basta una regla simple»).
- Agrupamiento: estadístico de Hopkins y mejor silueta de K-means.

REGLA DE SELECCIÓN (fijada antes de mirar resultados; no se ajusta después)

Un análisis supervisado es elegible si cumple todo lo siguiente:

1. tiene al menos `MINIMO_FILAS` filas de entrenamiento;
2. si es de clasificación, al menos `MINIMO_POSITIVOS` casos positivos;
3. su habilidad es mayor que cero en todos los bloques (habilidad mínima > 0);
4. una sola variable logra menos de `UMBRAL_REGLA_SIMPLE` de esa habilidad; si
   logra más, el análisis se marca «basta una regla simple» y no pasa al
   estudio a fondo (no necesita un modelo).

Su puntaje es habilidad media × valor de negocio / 3. Un análisis de
agrupamiento es elegible si tiene al menos `MINIMO_FILAS_AGRUPAMIENTO` filas
y un estadístico de Hopkins mayor que `UMBRAL_HOPKINS`; su puntaje es silueta
× valor de negocio / 3. En cada tipo se elige el elegible de mayor puntaje;
los empates se resuelven por mayor valor de negocio y después por nombre. Si
en un tipo no hay ningún elegible, no se elige ninguno.
"""
import time

import numpy as np
import pandas as pd

from proyecto.base.config import SEMILLA
from proyecto.cribado.catalogo import TIPOS_SUPERVISADOS
from proyecto.evaluacion import metricas, validacion
from proyecto.modelos import agrupamiento, registro, tuberias
from proyecto.preparacion import particion

MINIMO_FILAS = 300
MINIMO_POSITIVOS = 100
UMBRAL_REGLA_SIMPLE = 0.90
UMBRAL_HOPKINS = 0.5
MINIMO_FILAS_AGRUPAMIENTO = 50

PROFUNDIDAD_ARBOL = 4          # árbol «poco profundo» de la prueba rápida
MAXIMO_FILAS_REGLA = 5000      # filas de ajuste por bloque en la prueba de una sola variable
VALORES_K = tuple(range(2, 9))
GRUPO_VARIABLES = "ampliado"

ESTADO_CRIBADO, ESTADO_NO_DISPONIBLE, ESTADO_ANALITICO = "cribado", "no disponible", "analítico"
NOTA_REGLA_SIMPLE = "basta una regla simple"

COLUMNAS = [
    "analisis", "titulo", "tipo", "valor_negocio", "estado", "motivo", "filas", "variables",
    "positivos", "tasa_positivos", "media_objetivo", "desv_objetivo", "metrica", "linea_base", "bloques",
    "puntajes_base", "puntajes_lineal", "puntajes_arbol", "habilidades_lineal", "habilidades_arbol",
    "mejor_modelo", "habilidad_media", "habilidad_minima", "variable_regla", "habilidad_regla",
    "proporcion_regla", "hopkins", "mejor_k", "silueta", "elegible", "observacion", "puntaje_seleccion",
    "segundos",
]


# ---------------------------------------------------------------------------
# Datos de la prueba
# ---------------------------------------------------------------------------

def solo_entrenamiento(tabla, columna_fecha="fecha"):
    """Filas del periodo de entrenamiento, ordenadas por fecha."""
    etiquetas = particion.etiquetar_particion(tabla[columna_fecha]).to_numpy()
    return (tabla.loc[etiquetas == particion.ENTRENAMIENTO]
            .sort_values(columna_fecha, kind="stable").reset_index(drop=True))


def _columnas(papeles, tabla):
    del_grupo = set(papeles["grupos"][GRUPO_VARIABLES])
    numericas = [c for c in papeles["numericas"] if c in del_grupo and c in tabla.columns]
    categoricas = [c for c in papeles["categoricas"] if c in del_grupo and c in tabla.columns]
    return numericas, categoricas


def _nombre_lineal(tarea):
    return "logistica" if tarea == "clasificacion" else "lineal"


# ---------------------------------------------------------------------------
# Prueba de una sola variable
# ---------------------------------------------------------------------------

def _matriz_una_variable(valores, y, entrenar, validar, es_categorica):
    """Matrices de ajuste y validación de una sola variable, sin tubería:
    una categoría se reemplaza por la media del objetivo en las filas de
    ajuste (el orden que mejor aprovecha un árbol); un número faltante, por
    la mediana de ajuste más una columna indicadora."""
    if es_categorica:
        texto = pd.Series(valores, dtype=object).where(pd.notna(valores), "faltante").astype(str)
        medias = pd.Series(y[entrenar]).groupby(texto.iloc[entrenar].to_numpy()).mean()
        codigo = texto.map(medias).fillna(float(np.mean(y[entrenar]))).to_numpy(dtype=float)
        return codigo[entrenar, None], codigo[validar, None]
    numero = pd.to_numeric(pd.Series(valores), errors="coerce").to_numpy(dtype=float)
    falta = np.isnan(numero)
    mediana = np.nanmedian(numero[entrenar]) if (~falta[entrenar]).any() else 0.0
    matriz = np.column_stack([np.where(falta, mediana, numero), falta.astype(float)])
    return matriz[entrenar], matriz[validar]


def regla_simple(tarea, X, y, bloques, predicciones_base, numericas, categoricas):
    """Habilidad media, por variable, de un árbol poco profundo que solo ve
    esa variable. Devuelve una Serie ordenada de mayor a menor.

    Para que sea rápido, cada ajuste usa como mucho `MAXIMO_FILAS_REGLA` filas
    (tomadas al azar con semilla); la validación usa todas las del bloque.
    """
    generador = np.random.default_rng(SEMILLA)
    submuestras = [np.sort(generador.choice(entrenar, MAXIMO_FILAS_REGLA, replace=False))
                   if len(entrenar) > MAXIMO_FILAS_REGLA else entrenar for entrenar, _ in bloques]
    habilidades = {}
    for columna in list(numericas) + list(categoricas):
        valores = X[columna].to_numpy()
        por_bloque = []
        for (_, validar), entrenar, base in zip(bloques, submuestras, predicciones_base):
            ajuste, prueba = _matriz_una_variable(valores, y, entrenar, validar, columna in categoricas)
            arbol = registro.crear_modelo(tarea, "arbol", max_depth=PROFUNDIDAD_ARBOL).fit(ajuste, y[entrenar])
            por_bloque.append(metricas.habilidad(tarea, y[validar], metricas.prediccion_de(arbol, prueba), base))
        habilidades[columna] = float(np.mean(por_bloque))
    return pd.Series(habilidades, dtype=float).sort_values(ascending=False, kind="stable")


# ---------------------------------------------------------------------------
# Prueba rápida por tipo de análisis
# ---------------------------------------------------------------------------

def cribar_supervisado(ficha, tabla, papeles):
    """Prueba rápida de un análisis de regresión o clasificación. Devuelve un
    diccionario con las medidas de la fila de cribado."""
    tarea = ficha.tipo
    datos = solo_entrenamiento(tabla)
    numericas, categoricas = _columnas(papeles, datos)
    fila = {"filas": len(datos), "variables": len(numericas) + len(categoricas), "metrica": ficha.metrica,
            "linea_base": ficha.columna_linea_base or "Dummy"}
    y = datos[ficha.objetivo].to_numpy(dtype=float) if len(datos) else np.array([])
    if tarea == "clasificacion":
        fila["positivos"] = int(y.sum())
        fila["tasa_positivos"] = float(y.mean()) if len(y) else float("nan")
    else:
        fila["media_objetivo"] = float(y.mean()) if len(y) else float("nan")
        fila["desv_objetivo"] = float(y.std(ddof=1)) if len(y) > 1 else float("nan")
    if len(datos) < MINIMO_FILAS or (tarea == "clasificacion" and len(np.unique(y)) < 2):
        return fila    # sin datos suficientes no se ajusta nada; la regla lo descarta

    X = datos[numericas + categoricas]
    bloques = validacion.bloques_temporales(datos["fecha"])
    funcion, _ = metricas.resolver_metrica(ficha.metrica)
    referencia = (datos[ficha.columna_linea_base].to_numpy(dtype=float)
                  if ficha.columna_linea_base else None)
    modelos = {
        "lineal": tuberias.construir_tuberia(tarea, _nombre_lineal(tarea), numericas, categoricas),
        "arbol": tuberias.construir_tuberia(tarea, "arbol", numericas, categoricas,
                                            hiper={"max_depth": PROFUNDIDAD_ARBOL}),
    }
    puntajes = {nombre: [] for nombre in ["base", *modelos]}
    habilidades = {nombre: [] for nombre in modelos}
    predicciones_base = []
    for entrenar, validar in bloques:
        if referencia is not None:
            base = referencia[validar]
        else:
            ingenuo = registro.linea_base(tarea).fit(X.iloc[entrenar], y[entrenar])
            base = metricas.prediccion_de(ingenuo, X.iloc[validar])
        predicciones_base.append(base)
        puntajes["base"].append(float(funcion(y[validar], base)))
        for nombre, modelo in modelos.items():
            ajustado = modelo.fit(X.iloc[entrenar], y[entrenar])
            prediccion = metricas.prediccion_de(ajustado, X.iloc[validar])
            puntajes[nombre].append(float(funcion(y[validar], prediccion)))
            habilidades[nombre].append(float(metricas.habilidad(tarea, y[validar], prediccion, base)))

    medias = {nombre: float(np.mean(valores)) for nombre, valores in habilidades.items()}
    mejor = max(medias, key=lambda nombre: (medias[nombre], nombre == "lineal"))
    simple = regla_simple(tarea, X, y, bloques, predicciones_base, numericas, categoricas)
    fila.update({
        "bloques": len(bloques),
        "puntajes_base": puntajes["base"], "puntajes_lineal": puntajes["lineal"],
        "puntajes_arbol": puntajes["arbol"],
        "habilidades_lineal": habilidades["lineal"], "habilidades_arbol": habilidades["arbol"],
        "mejor_modelo": _nombre_lineal(tarea) if mejor == "lineal" else "arbol",
        "habilidad_media": medias[mejor], "habilidad_minima": float(np.min(habilidades[mejor])),
        "variable_regla": simple.index[0], "habilidad_regla": float(simple.iloc[0]),
        "proporcion_regla": (float(max(simple.iloc[0], 0.0) / medias[mejor])
                             if medias[mejor] > 0 else float("nan")),
    })
    return fila


def cribar_agrupamiento(ficha, tabla, papeles):
    """Prueba rápida de un análisis de agrupamiento: tendencia a agruparse
    (Hopkins) y mejor silueta de K-means sobre las variables estandarizadas."""
    datos = solo_entrenamiento(tabla)
    numericas, categoricas = _columnas(papeles, datos)
    fila = {"filas": len(datos), "variables": len(numericas) + len(categoricas), "metrica": ficha.metrica}
    if len(datos) < MINIMO_FILAS_AGRUPAMIENTO:
        return fila
    matriz = np.asarray(tuberias.preprocesador(numericas, categoricas)
                        .fit_transform(datos[numericas + categoricas]), dtype=float)
    exploracion = agrupamiento.explorar_kmeans(matriz, [k for k in VALORES_K if k < len(datos)])
    mejor = exploracion.loc[exploracion["silueta"].idxmax()]
    fila.update({"hopkins": float(agrupamiento.estadistico_hopkins(matriz, SEMILLA)),
                 "mejor_k": int(mejor["k"]), "silueta": float(mejor["silueta"])})
    return fila


# ---------------------------------------------------------------------------
# Regla de selección
# ---------------------------------------------------------------------------

def aplicar_regla(fila):
    """Aplica la regla de selección a una fila de cribado. Devuelve
    (elegible, observación, puntaje)."""
    if fila.get("estado") != ESTADO_CRIBADO:
        return False, "", float("nan")
    valor = fila["valor_negocio"] / 3.0
    filas = fila.get("filas") or 0
    if fila["tipo"] == "agrupamiento":
        silueta, hopkins = fila.get("silueta"), fila.get("hopkins")
        if filas < MINIMO_FILAS_AGRUPAMIENTO or silueta is None or pd.isna(silueta):
            return False, f"menos de {MINIMO_FILAS_AGRUPAMIENTO} filas", float("nan")
        puntaje = float(silueta) * valor
        if not hopkins > UMBRAL_HOPKINS:
            return False, f"sin tendencia a agruparse (Hopkins ≤ {UMBRAL_HOPKINS})", puntaje
        return True, "", puntaje

    media, minima = fila.get("habilidad_media"), fila.get("habilidad_minima")
    puntaje = float(media) * valor if media is not None and not pd.isna(media) else float("nan")
    if filas < MINIMO_FILAS:
        return False, f"menos de {MINIMO_FILAS} filas de entrenamiento", puntaje
    if fila["tipo"] == "clasificacion" and (fila.get("positivos") or 0) < MINIMO_POSITIVOS:
        return False, f"menos de {MINIMO_POSITIVOS} casos positivos", puntaje
    if minima is None or pd.isna(minima) or not minima > 0:
        return False, "no supera a la línea base en todos los bloques", puntaje
    proporcion = fila.get("proporcion_regla")
    if proporcion is not None and not pd.isna(proporcion) and proporcion >= UMBRAL_REGLA_SIMPLE:
        return False, NOTA_REGLA_SIMPLE, puntaje
    return True, "", puntaje


def cribar(fichas, tablas_modelado, motivos=None):
    """Tabla de cribado: una fila por análisis candidato.

    `tablas_modelado[nombre] = (tabla, papeles)` para los análisis que
    pudieron construirse; los que no están ahí quedan como «no disponible»
    con el motivo de `motivos[nombre]`, y los de tipo «bayes» como
    «analítico». Solo se usan las filas del periodo de entrenamiento.
    """
    motivos = motivos or {}
    filas = []
    for ficha in fichas:
        inicio = time.perf_counter()
        fila = {"analisis": ficha.nombre, "titulo": ficha.titulo, "tipo": ficha.tipo,
                "valor_negocio": ficha.valor_negocio, "motivo": ""}
        if ficha.tipo == "bayes":
            fila.update({"estado": ESTADO_ANALITICO, "motivo": "cálculo analítico: no se criba"})
        elif ficha.nombre not in tablas_modelado:
            fila.update({"estado": ESTADO_NO_DISPONIBLE,
                         "motivo": motivos.get(ficha.nombre, "no se construyó su tabla de modelado")})
        else:
            tabla, papeles = tablas_modelado[ficha.nombre]
            fila["estado"] = ESTADO_CRIBADO
            prueba = cribar_supervisado if ficha.tipo in TIPOS_SUPERVISADOS else cribar_agrupamiento
            fila.update(prueba(ficha, tabla, papeles))
        fila["elegible"], fila["observacion"], fila["puntaje_seleccion"] = aplicar_regla(fila)
        fila["segundos"] = time.perf_counter() - inicio
        filas.append(fila)
    return pd.DataFrame(filas).reindex(columns=COLUMNAS)


def seleccionar_analisis(tabla_cribado):
    """Aplica la regla fijada de antemano: por cada tipo, el análisis elegible
    de mayor puntaje (empates: mayor valor de negocio, después nombre).
    Devuelve un diccionario tipo → nombre (None si el tipo no tiene ninguno
    elegible)."""
    elegidos = {}
    cribados = tabla_cribado[tabla_cribado["estado"] == ESTADO_CRIBADO]
    for tipo in [*TIPOS_SUPERVISADOS, "agrupamiento"]:
        candidatos = cribados[(cribados["tipo"] == tipo) & cribados["elegible"].astype(bool)].copy()
        if not len(candidatos):
            elegidos[tipo] = None
            continue
        candidatos["_puntaje"] = candidatos["puntaje_seleccion"].astype(float).round(10)
        orden = candidatos.sort_values(["_puntaje", "valor_negocio", "analisis"],
                                       ascending=[False, False, True], kind="stable")
        elegidos[tipo] = str(orden.iloc[0]["analisis"])
    return elegidos


def tabla_resumen(tabla_cribado):
    """Versión corta de la tabla de cribado, para mostrarla."""
    columnas = ["analisis", "tipo", "valor_negocio", "estado", "filas", "tasa_positivos", "mejor_modelo",
                "habilidad_media", "habilidad_minima", "variable_regla", "proporcion_regla", "hopkins",
                "mejor_k", "silueta", "elegible", "observacion", "puntaje_seleccion", "segundos"]
    return tabla_cribado[columnas].copy()
