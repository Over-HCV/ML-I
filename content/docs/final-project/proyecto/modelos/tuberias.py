"""Capa 4 · Tuberías de preparación y modelo.

Responsabilidad única: armar, para cualquier tabla y lista de columnas, un
estimador de scikit-learn que contiene **todo** lo que se aprende de los
datos (imputación, escalado, codificación, selección de variables,
sobremuestreo y modelo). Como todo vive dentro de la tubería, la validación
cruzada lo reajusta en cada bloque de entrenamiento y nada de la validación
se filtra hacia el ajuste.

No conoce las tablas del laboratorio: recibe los nombres de las columnas
numéricas y categóricas.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from proyecto.base.config import SEMILLA
from proyecto.modelos.registro import crear_modelo

ESCALADOS = ["estandar", "minmax", "ninguno"]
CODIFICACIONES = ["one_hot", "ordinal"]
TAREAS = ["clasificacion", "regresion"]

ETIQUETA_FALTANTE = "faltante"
# Un nivel categórico con menos apariciones que esto en el entrenamiento se
# agrupa con los demás niveles raros.
FRECUENCIA_MINIMA = 20
# Tope del exponente al deshacer el logaritmo del objetivo (evita desbordes
# numéricos si un modelo extrapola muy lejos).
_TOPE_LOG = 700.0


def _a_texto(X):
    """Lleva las columnas categóricas a texto y marca los faltantes con una
    etiqueta propia, de modo que «no informado» sea un nivel más. Unifica
    además los tipos (texto, número, booleano) que puedan venir mezclados."""
    tabla = pd.DataFrame(X)
    faltan = tabla.isna()
    return tabla.astype(str).mask(faltan, ETIQUETA_FALTANTE).astype(object)


def _log(y):
    return np.log1p(y)


def _deshacer_log(z):
    return np.expm1(np.minimum(z, _TOPE_LOG))


def _puntaje_f_clasificacion(X, y):
    """F de ANOVA por variable; las columnas constantes quedan con NaN (que
    `SelectKBest` trata como el peor puntaje) sin emitir avisos."""
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        return f_classif(X, y)


def _puntaje_f_regresion(X, y):
    """F de la regresión simple de `y` sobre cada variable, sin avisos por
    columnas constantes."""
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        return f_regression(X, y)


def _vacio(valor):
    return valor is None or (isinstance(valor, float) and np.isnan(valor))


def preprocesador(numericas, categoricas, escalado="estandar", codificacion="one_hot"):
    """Transformador de columnas que deja la tabla lista para un modelo.

    - Numéricas: imputación por la mediana con una columna indicadora por
      cada variable que tuvo faltantes, y después el escalado pedido
      (`"estandar"`, `"minmax"` o `"ninguno"`).
    - Categóricas: los faltantes pasan a ser el nivel «faltante» y luego se
      codifican con variables indicadoras (`"one_hot"`) o con un entero por
      nivel (`"ordinal"`). Los niveles con menos de `FRECUENCIA_MINIMA`
      apariciones se agrupan, y un nivel nunca visto en el ajuste no produce
      error: queda en ceros (one-hot) o en −1 (ordinal).

    Solo usa las columnas listadas; el resto de la tabla se ignora.
    """
    if escalado not in ESCALADOS:
        raise ValueError(f"Escalado desconocido: {escalado!r}. Opciones: {ESCALADOS}")
    if codificacion not in CODIFICACIONES:
        raise ValueError(f"Codificación desconocida: {codificacion!r}. Opciones: {CODIFICACIONES}")
    numericas, categoricas = list(numericas), list(categoricas)
    if not numericas and not categoricas:
        raise ValueError("Se necesita al menos una columna numérica o categórica.")

    escalador = {"estandar": StandardScaler(), "minmax": MinMaxScaler(), "ninguno": "passthrough"}[escalado]
    rama_numerica = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("escalar", escalador),
    ])
    if codificacion == "one_hot":
        codificador = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", min_frequency=FRECUENCIA_MINIMA
        )
    else:
        codificador = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=FRECUENCIA_MINIMA
        )
    rama_categorica = Pipeline([
        ("texto", FunctionTransformer(_a_texto, feature_names_out="one-to-one")),
        ("codificar", codificador),
    ])

    ramas = []
    if numericas:
        ramas.append(("numericas", rama_numerica, numericas))
    if categoricas:
        ramas.append(("categoricas", rama_categorica, categoricas))
    return ColumnTransformer(ramas, remainder="drop", verbose_feature_names_out=False)


def construir_tuberia(tarea, modelo, numericas, categoricas, escalado="estandar",
                      codificacion="one_hot", seleccion=None, desbalance=None,
                      objetivo_log=False, hiper=None):
    """Arma el estimador completo (sin ajustar) para una tarea y un modelo.

    Pasos, en este orden y con estos nombres:

    1. `preparar`: `preprocesador(numericas, categoricas, escalado, codificacion)`.
    2. `seleccionar` (si `seleccion=k`): conserva las k variables con mayor
       estadístico F frente al objetivo (si k supera las disponibles, todas).
    3. `remuestrear` (si `desbalance="sobremuestreo"`, solo clasificación):
       repite al azar filas de la clase minoritaria hasta igualar las clases.
       Se usa la tubería de imbalanced-learn, que aplica el remuestreo
       únicamente al ajustar: las filas de validación nunca se remuestrean.
       Atención: al cambiar la proporción de clases, las probabilidades que
       entrega el modelo dejan de estar calibradas con la prevalencia real.
    4. `modelo`: `crear_modelo(tarea, modelo, **hiper)`.

    Con `objetivo_log=True` (solo regresión) la tubería se envuelve en un
    `TransformedTargetRegressor` que ajusta sobre log(1 + y) y devuelve las
    predicciones en la escala original; exige y ≥ 0.
    """
    if tarea not in TAREAS:
        raise ValueError(f"Tarea desconocida: {tarea!r}. Opciones: {TAREAS}")
    # Un nivel «sin esta opción» puede llegar como NaN si el diseño pasó por
    # un archivo; se trata igual que None.
    seleccion = None if _vacio(seleccion) else seleccion
    desbalance = None if _vacio(desbalance) else desbalance
    if desbalance not in (None, "sobremuestreo"):
        raise ValueError(f"Opción de desbalance desconocida: {desbalance!r}")
    if desbalance and tarea != "clasificacion":
        raise ValueError("El sobremuestreo solo aplica a clasificación.")
    if objetivo_log and tarea != "regresion":
        raise ValueError("El logaritmo del objetivo solo aplica a regresión.")

    pasos = [("preparar", preprocesador(numericas, categoricas, escalado, codificacion))]
    if seleccion is not None:
        puntaje = _puntaje_f_clasificacion if tarea == "clasificacion" else _puntaje_f_regresion
        pasos.append(("seleccionar", SelectKBest(score_func=puntaje, k=int(seleccion))))
    if desbalance == "sobremuestreo":
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.pipeline import Pipeline as TuberiaConRemuestreo

        pasos.append(("remuestrear", RandomOverSampler(random_state=SEMILLA)))
        clase_tuberia = TuberiaConRemuestreo
    else:
        clase_tuberia = Pipeline
    pasos.append(("modelo", crear_modelo(tarea, modelo, **(hiper or {}))))
    tuberia = clase_tuberia(pasos)

    if objetivo_log:
        return TransformedTargetRegressor(
            regressor=tuberia, func=_log, inverse_func=_deshacer_log, check_inverse=False
        )
    return tuberia


def extraer_tuberia(estimador):
    """Devuelve la tubería interna, tanto si el estimador es la tubería como
    si viene envuelto por el logaritmo del objetivo. Sirve con estimadores
    ajustados y sin ajustar."""
    if isinstance(estimador, TransformedTargetRegressor):
        return getattr(estimador, "regressor_", estimador.regressor)
    return estimador


def nombres_variables(estimador):
    """Nombres de las columnas que recibe el modelo de una tubería ajustada
    (después de preparar y, si existe, de seleccionar)."""
    tuberia = extraer_tuberia(estimador)
    nombres = np.asarray(tuberia.named_steps["preparar"].get_feature_names_out(), dtype=object)
    if "seleccionar" in tuberia.named_steps:
        nombres = nombres[tuberia.named_steps["seleccionar"].get_support()]
    return list(nombres)


def alfas_de_poda(tuberia, X, y, cantidad=6):
    """Valores candidatos de `ccp_alpha` para un árbol, tomados de su ruta de
    poda por costo-complejidad.

    Prepara `X` con los pasos previos de la tubería (ajustados con estas
    mismas filas, que deben ser solo de entrenamiento), calcula la ruta de
    poda del árbol sin podar y devuelve 0 más `cantidad − 1` valores
    repartidos por cuantiles de los alfas efectivos (excluido el último, que
    deja el árbol en la raíz). Así la rejilla queda en la escala correcta
    del objetivo, cosa que una lista fija no garantiza en regresión.
    """
    from sklearn.base import clone

    envuelta = isinstance(tuberia, TransformedTargetRegressor)
    interna = clone(extraer_tuberia(tuberia))
    y = np.asarray(y, dtype=float if envuelta else None)
    if envuelta:
        y = _log(y)
    previos = [(nombre, paso) for nombre, paso in interna.steps[:-1] if nombre != "remuestrear"]
    matriz = Pipeline(previos).fit_transform(X, y)
    arbol = interna.steps[-1][1]
    if not hasattr(arbol, "cost_complexity_pruning_path"):
        raise ValueError("El paso `modelo` de la tubería no es un árbol de decisión.")
    arbol.set_params(ccp_alpha=0.0)
    alfas = np.unique(arbol.cost_complexity_pruning_path(matriz, y).ccp_alphas[:-1])
    alfas = alfas[alfas > 0]
    if len(alfas) == 0:
        return [0.0]
    cuantiles = np.quantile(alfas, np.linspace(0.5, 1.0, max(cantidad - 1, 1)))
    return [0.0] + sorted({float(a) for a in cuantiles})
