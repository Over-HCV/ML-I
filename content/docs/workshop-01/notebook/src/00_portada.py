# %% [markdown]
# # Taller 1 — KNN y Métricas de Clasificación
# **Aprendizaje Automático 1 — 2026-2**
#
# Componente práctico-computacional (ejercicios 6 a 10), desarrollado
# sobre el dataset **Give Me Some Credit** (Kaggle, 2011).
#
# El notebook está pensado para ejecutarse de principio a fin sin pasos
# manuales: la única dependencia externa es el archivo `cs-training.csv`,
# que se busca primero en el directorio local y, si no está presente, se
# descarga automáticamente desde un repositorio público de GitHub que
# replica el dataset original de la competencia.

# %%
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# %matplotlib inline

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = False

RANDOM_STATE = 42


def calcular_metricas(y_true, y_pred):
    """Calcula Accuracy, Precision, Recall y F1 para un vector de predicciones."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
