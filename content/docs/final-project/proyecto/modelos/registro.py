"""Capa 4 · Registro de modelos.

Responsabilidad única: decir qué modelos existen para cada tarea, crearlos
con valores de partida razonables y sembrados, y entregar la rejilla de
hiperparámetros con la que se refina cada uno. Todos son de scikit-learn.

Los valores de partida sirven para el cribado (donde los modelos se comparan
sin afinar); el afinamiento se hace después con `rejilla` y validación
cruzada ordenada en el tiempo.
"""
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from proyecto.base.config import SEMILLA

MODELOS = {
    "clasificacion": ["knn", "logistica", "arbol", "mlp"],
    "regresion": ["knn", "lineal", "arbol", "mlp"],
}

NOMBRES_LEGIBLES = {
    "knn": "K vecinos más cercanos",
    "logistica": "Regresión logística",
    "lineal": "Regresión lineal",
    "arbol": "Árbol de decisión",
    "mlp": "Perceptrón multicapa",
    "base": "Línea base",
}

# Valores de partida de cada modelo (se pueden sobrescribir con **hiper).
_PARTIDA = {
    "knn": {"n_neighbors": 15, "weights": "uniform"},
    "logistica": {"C": 1.0, "max_iter": 500},
    "lineal": {},
    "arbol": {"max_depth": 6, "min_samples_leaf": 20, "ccp_alpha": 0.0, "random_state": SEMILLA},
    # Parada temprana: la red aparta una fracción del entrenamiento (al azar,
    # no por fecha) y se detiene cuando deja de mejorar en ella. Con el tope
    # de iteraciones, el ajuste tarda segundos.
    "mlp": {
        "hidden_layer_sizes": (32,), "alpha": 1e-3, "early_stopping": True,
        "validation_fraction": 0.15, "n_iter_no_change": 8, "max_iter": 150,
        "random_state": SEMILLA,
    },
}

_CLASES = {
    ("clasificacion", "knn"): KNeighborsClassifier,
    ("clasificacion", "logistica"): LogisticRegression,
    ("clasificacion", "arbol"): DecisionTreeClassifier,
    ("clasificacion", "mlp"): MLPClassifier,
    ("regresion", "knn"): KNeighborsRegressor,
    ("regresion", "lineal"): LinearRegression,
    ("regresion", "arbol"): DecisionTreeRegressor,
    ("regresion", "mlp"): MLPRegressor,
}

# Rejillas de refinamiento (nombres de hiperparámetro sin prefijo). Son cortas
# a propósito: la de KNN es la más cara porque predecir exige comparar cada
# fila de validación con todo el entrenamiento.
_REJILLAS = {
    "knn": {"n_neighbors": [5, 15, 35, 75, 151], "weights": ["uniform", "distance"]},
    "logistica": {"C": [0.01, 0.1, 1.0, 10.0]},
    "lineal": {},
    "arbol": {
        "ccp_alpha": [0.0, 0.0001, 0.001],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [5, 50],
    },
    "mlp": {"hidden_layer_sizes": [(16,), (64,), (32, 16)], "alpha": [1e-4, 1e-2]},
}


def _validar(tarea, nombre):
    if tarea not in MODELOS:
        raise ValueError(f"Tarea desconocida: {tarea!r}. Opciones: {list(MODELOS)}")
    if nombre not in MODELOS[tarea]:
        raise ValueError(f"Modelo desconocido para {tarea}: {nombre!r}. Opciones: {MODELOS[tarea]}")


def crear_modelo(tarea, nombre, **hiper):
    """Crea el modelo `nombre` para la `tarea`, con los valores de partida del
    registro y los hiperparámetros que se pasen por encima.

    El perceptrón de regresión se entrega dentro de un
    `TransformedTargetRegressor` que estandariza el objetivo (media y
    desviación aprendidas solo con las filas de ajuste): sin eso la red
    converge mal cuando el objetivo está en unidades grandes.
    """
    _validar(tarea, nombre)
    parametros = {**_PARTIDA[nombre], **hiper}
    modelo = _CLASES[(tarea, nombre)](**parametros)
    if (tarea, nombre) == ("regresion", "mlp"):
        return TransformedTargetRegressor(regressor=modelo, transformer=StandardScaler())
    return modelo


def prefijo_modelo(tarea, nombre, objetivo_log=False):
    """Prefijo con el que `GridSearchCV` llega a los hiperparámetros del
    modelo dentro de la tubería de `construir_tuberia` (paso `modelo`; un
    nivel más si el objetivo va en logaritmo o el modelo viene envuelto)."""
    _validar(tarea, nombre)
    prefijo = "regressor__modelo__" if objetivo_log else "modelo__"
    if (tarea, nombre) == ("regresion", "mlp"):
        prefijo += "regressor__"
    return prefijo


def rejilla(tarea, nombre, objetivo_log=False):
    """Rejilla de hiperparámetros del modelo, con los prefijos listos para
    `GridSearchCV` sobre la tubería de `construir_tuberia`.

    - knn: número de vecinos y ponderación.
    - logistica: C (inverso de la regularización).
    - arbol: poda por costo-complejidad (`ccp_alpha`), profundidad y tamaño
      mínimo de hoja. En regresión `ccp_alpha` depende de la escala del
      objetivo, así que la rejilla fija solo trae 0; los valores útiles se
      obtienen con `tuberias.alfas_de_poda`.
    - mlp: tamaño de las capas ocultas y regularización `alpha`.
    - lineal: no tiene hiperparámetros (rejilla vacía).
    """
    prefijo = prefijo_modelo(tarea, nombre, objetivo_log)
    valores = {clave: list(lista) for clave, lista in _REJILLAS[nombre].items()}
    if (tarea, nombre) == ("regresion", "arbol"):
        valores["ccp_alpha"] = [0.0]
    return {prefijo + clave: lista for clave, lista in valores.items()}


def linea_base(tarea):
    """Estimador de referencia que ignora las variables: la proporción de la
    clase positiva (clasificación) o la media del objetivo (regresión)."""
    if tarea == "clasificacion":
        return DummyClassifier(strategy="prior")
    if tarea == "regresion":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Tarea desconocida: {tarea!r}")
