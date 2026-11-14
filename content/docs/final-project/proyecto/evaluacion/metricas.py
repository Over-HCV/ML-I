"""Capa 5 · Métricas de evaluación.

Responsabilidad única: calcular, siempre de la misma forma, las métricas de
regresión y de clasificación que usan todas las etapas, y ofrecer el registro
`METRICAS` (nombre → función y sentido) con el que la validación cruzada y el
cribado piden una métrica por su nombre.

Convención: en clasificación la «predicción» es siempre la probabilidad de la
clase 1, y la etiqueta se obtiene con la regla explícita ŷ = 1[p ≥ t]. En
regresión la predicción es el valor numérico.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from proyecto.base.config import COSTO_FN, COSTO_FP


# ----------------------------------------------------------------------
# Regresión
# ----------------------------------------------------------------------
def rmse(y, pred):
    """Raíz del error cuadrático medio."""
    return float(mean_squared_error(y, pred) ** 0.5)


def mae(y, pred):
    """Error absoluto medio."""
    return float(mean_absolute_error(y, pred))


def r2(y, pred):
    """Coeficiente de determinación."""
    return float(r2_score(y, pred))


def metricas_reg(y, pred):
    """MAE, RMSE y R² de un vector de predicciones continuas."""
    return {"MAE": mae(y, pred), "RMSE": rmse(y, pred), "R2": r2(y, pred)}


# ----------------------------------------------------------------------
# Clasificación
# ----------------------------------------------------------------------
def predecir_umbral(proba, t):
    """Regla explícita ŷ = 1[p ≥ t] (evita los desempates internos de
    `predict`)."""
    return (np.asarray(proba, dtype=float) >= t).astype(int)


def _dos_clases(y):
    return len(np.unique(np.asarray(y))) == 2


def roc_auc(y, proba):
    """Área bajo la curva ROC. Devuelve NaN si `y` trae una sola clase."""
    return float(roc_auc_score(y, proba)) if _dos_clases(y) else float("nan")


def precision_media(y, proba):
    """Área bajo la curva precisión-recall (precisión media). NaN si `y`
    trae una sola clase."""
    return float(average_precision_score(y, proba)) if _dos_clases(y) else float("nan")


def brier(y, proba):
    """Puntaje de Brier: error cuadrático medio de la probabilidad."""
    return float(brier_score_loss(y, proba))


def perdida_log(y, proba):
    """Pérdida logarítmica (entropía cruzada)."""
    return float(log_loss(y, proba, labels=[0, 1]))


def f1(y, proba, t=0.5):
    """F1 de la regla ŷ = 1[p ≥ t]."""
    return float(f1_score(y, predecir_umbral(proba, t), zero_division=0))


def costo(y, proba, t=0.5):
    """Costo total COSTO_FP·FP + COSTO_FN·FN de la regla ŷ = 1[p ≥ t]."""
    y = np.asarray(y)
    pred = predecir_umbral(proba, t)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return COSTO_FP * fp + COSTO_FN * fn


def metricas_clf(y, proba, t=0.5):
    """Precisión, recall, F1, ROC-AUC, matriz de confusión y costo para un
    vector de probabilidades de la clase 1 y un umbral `t`.

    El costo es COSTO_FP·FP + COSTO_FN·FN, con los pesos declarados en
    `base/config.py`.
    """
    y = np.asarray(y)
    pred = predecir_umbral(proba, t)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Umbral": t,
        "Precisión": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "ROC-AUC": roc_auc(y, proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Costo": int(COSTO_FP * fp + COSTO_FN * fn),
    }


# ----------------------------------------------------------------------
# Registro de métricas y utilidades comunes
# ----------------------------------------------------------------------
# nombre → (función(y, predicción), sentido). El sentido dice si conviene un
# valor mayor o menor.
METRICAS = {
    "rmse": (rmse, "menor"),
    "mae": (mae, "menor"),
    "r2": (r2, "mayor"),
    "roc_auc": (roc_auc, "mayor"),
    "precision_media": (precision_media, "mayor"),
    "brier": (brier, "menor"),
    "perdida_log": (perdida_log, "menor"),
    "f1": (f1, "mayor"),
}


def resolver_metrica(metrica, sentido=None):
    """Devuelve el par (función, sentido) de una métrica dada por su nombre
    en `METRICAS`, como par (función, sentido) o como función suelta (en ese
    caso hay que indicar `sentido`)."""
    if isinstance(metrica, str):
        if metrica not in METRICAS:
            raise ValueError(f"Métrica desconocida: {metrica!r}. Disponibles: {sorted(METRICAS)}")
        funcion, sentido_registrado = METRICAS[metrica]
        return funcion, sentido or sentido_registrado
    if isinstance(metrica, tuple):
        funcion, sentido_registrado = metrica
        sentido = sentido or sentido_registrado
    else:
        funcion = metrica
    if sentido not in ("mayor", "menor"):
        raise ValueError("Para una métrica propia hay que indicar el sentido: 'mayor' o 'menor'.")
    return funcion, sentido


def prediccion_de(estimador, X):
    """Predicción con la convención del proyecto: probabilidad de la clase 1
    si el estimador es un clasificador, valor numérico si es un regresor."""
    if hasattr(estimador, "predict_proba"):
        proba = np.asarray(estimador.predict_proba(X))
        clases = list(getattr(estimador, "classes_", [0, 1]))
        if proba.shape[1] == 1:  # el entrenamiento solo tenía una clase
            return np.full(len(proba), 1.0 if clases[0] == 1 else 0.0)
        return proba[:, clases.index(1) if 1 in clases else -1]
    return np.asarray(estimador.predict(X), dtype=float)


def habilidad(tarea, y, pred_modelo, pred_base=None):
    """Ganancia del modelo sobre la línea base, en una escala común a los dos
    tipos de problema (0 = no aporta nada, 1 = perfecto).

    - Regresión: 1 − RMSE_modelo / RMSE_base.
    - Clasificación: 2·AUC − 1 (la línea base, que asigna a todos la misma
      probabilidad, tiene AUC 0.5, por lo que `pred_base` no se usa).
    """
    if tarea == "regresion":
        if pred_base is None:
            raise ValueError("En regresión se necesita la predicción de la línea base.")
        base = rmse(y, pred_base)
        return float("nan") if base == 0 else 1.0 - rmse(y, pred_modelo) / base
    if tarea == "clasificacion":
        return 2.0 * roc_auc(y, pred_modelo) - 1.0
    raise ValueError(f"Tarea desconocida: {tarea!r}")


def seleccionar(tabla, orden):
    """Devuelve la fila ganadora de `tabla` aplicando criterios lexicográficos.

    `orden` es una lista de pares (columna, ascendente). Los valores reales se
    redondean a 10 decimales solo para comparar, de modo que diferencias de
    máquina no rompan un empate numérico; el ordenamiento es estable, así que
    ante un empate total gana la primera fila en el orden original.
    """
    claves = pd.DataFrame(index=tabla.index)
    for columna, _ in orden:
        valores = tabla[columna]
        claves[columna] = (
            valores.astype(float).round(10) if pd.api.types.is_float_dtype(valores) else valores
        )
    ganador = claves.sort_values(
        [c for c, _ in orden], ascending=[a for _, a in orden], kind="mergesort"
    ).index[0]
    return tabla.loc[ganador]
