"""Capa 5 · Umbral de decisión por costo.

Responsabilidad única: recorrer umbrales de decisión para un vector de
probabilidades y elegir el de menor costo, mostrando siempre al lado la
carga de alertas (cuántas se levantan y qué fracción acierta).
"""
import numpy as np
import pandas as pd

from proyecto.base.config import COSTO_FN, COSTO_FP
from proyecto.evaluacion.metricas import metricas_clf, seleccionar

UMBRALES = [round(float(t), 2) for t in np.arange(0.05, 0.951, 0.05)]

COLUMNAS_UMBRALES = [
    "Umbral", "Alertas", "Proporción de alertas", "Precisión", "Recall", "F1",
    "TN", "FP", "FN", "TP", "Costo",
]


def umbral_teorico():
    """Umbral que minimiza el costo esperado cuando las probabilidades están
    bien calibradas: t* = COSTO_FP / (COSTO_FP + COSTO_FN)."""
    return COSTO_FP / (COSTO_FP + COSTO_FN)


def tabla_umbrales(y, proba, umbrales=None):
    """Una fila por umbral con la regla ŷ = 1[p ≥ t]: alertas levantadas
    (TP + FP), proporción de casos alertados, precisión (fracción de alertas
    que acierta), recall, F1, matriz de confusión y costo
    COSTO_FP·FP + COSTO_FN·FN. Por defecto recorre 0.05, 0.10, …, 0.95."""
    umbrales = UMBRALES if umbrales is None else list(umbrales)
    tabla = pd.DataFrame([metricas_clf(y, proba, t) for t in umbrales])
    tabla["Alertas"] = tabla["TP"] + tabla["FP"]
    tabla["Proporción de alertas"] = tabla["Alertas"] / len(np.asarray(y))
    return tabla[COLUMNAS_UMBRALES]


def elegir_umbral(tabla):
    """Fila de `tabla_umbrales` con el menor costo; en caso de empate, el
    umbral más alto (el que levanta menos alertas)."""
    return seleccionar(tabla, [("Costo", True), ("Umbral", False)])
