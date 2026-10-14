# %% [markdown]
# # Taller 2 — Modelos de aprendizaje supervisado
# **Aprendizaje Automático 1 — 2026-2** · Profesor: Daniel Rambaut · Modalidad: grupal
#
# **Estudiantes:** Tania Fernanda Mora Cubillos · Over Haider Castrillón Valencia
#
# Componente práctico-computacional (ejercicios 6 a 10) sobre dos conjuntos
# reales incluidos en scikit-learn, por lo que el notebook se ejecuta de
# principio a fin sin descargas externas:
#
# - **Caso A** (`load_diabetes`, `scaled=False`): regresión de la progresión
#   de la diabetes un año después de la medición inicial.
# - **Caso B** (`load_breast_cancer`): clasificación de tejido, recodificado
#   como **1 = maligno** y **0 = benigno**.
#
# **Protocolo.** Partición fija 60/20/20 (entrenamiento, validación, prueba)
# con `random_state=42`, estratificada en el caso B. Los modelos se ajustan
# con entrenamiento, las decisiones se toman con validación y la prueba se
# consulta una única vez, en el ejercicio 10, sobre modelos y umbrales ya
# congelados.

# %%
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# %matplotlib inline

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = False

RANDOM_STATE = 42
COLORES = {"azul": "#4C72B0", "rojo": "#C44E52", "verde": "#55A868", "gris": "#8C8C8C"}

# Rutas relativas a notebook/ (build.py ejecuta con ese directorio de trabajo).
FIG_DIR = Path("../latex/figuras")
MACROS_PATH = Path("../latex/generado/resultados.tex")

pd.Series(
    {
        "Python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit-learn": sklearn.__version__,
    },
    name="Versión",
).to_frame()

# %% [markdown]
# ### Funciones auxiliares
#
# Se centralizan aquí las métricas y utilidades que usan varios ejercicios,
# para que todas las tablas se calculen exactamente de la misma forma.

# %%
def rmse(y, y_pred):
    return mean_squared_error(y, y_pred) ** 0.5


def metricas_reg(y, y_pred):
    """MAE, RMSE y R² de un vector de predicciones continuas."""
    return {"MAE": mean_absolute_error(y, y_pred), "RMSE": rmse(y, y_pred), "R2": r2_score(y, y_pred)}


def predecir_umbral(proba, t):
    """Regla explícita ŷ = 1[p ≥ t] (evita los desempates internos de predict)."""
    return (np.asarray(proba) >= t).astype(int)


def metricas_clf(y, proba, t=0.5):
    """Precisión, recall, F1, ROC-AUC, FP, FN y costo L = FP + 5FN para un
    vector de probabilidades de la clase 1 (maligno) y un umbral t."""
    y_pred = predecir_umbral(proba, t)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "Umbral": t,
        "Precisión": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, proba),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Costo L": fp + 5 * fn,
    }


def seleccionar(tabla, orden):
    """Devuelve la fila ganadora de `tabla` aplicando criterios lexicográficos.

    `orden` es una lista de pares (columna, ascendente). Los valores reales se
    redondean a 10 decimales solo para comparar, de modo que diferencias de
    máquina no rompan un empate numérico; el ordenamiento es estable, así que
    ante un empate total gana la primera fila en el orden original de la
    cuadrícula.
    """
    claves = pd.DataFrame(index=tabla.index)
    for col, _ in orden:
        v = tabla[col]
        claves[col] = v.astype(float).round(10) if pd.api.types.is_float_dtype(v) else v
    ganador = claves.sort_values(
        [c for c, _ in orden], ascending=[a for _, a in orden], kind="mergesort"
    ).index[0]
    return tabla.loc[ganador]


def guardar_fig(fig, nombre):
    """Guarda la figura en latex/figuras/ para que el PDF use la misma imagen."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{nombre}.png", dpi=200, bbox_inches="tight")


RESULTADOS = {}


def registrar(nombre, valor, decimales=4):
    """Registra una cifra para el LaTeX como macro \\<nombre>. Los nombres
    deben ser solo letras (restricción de los comandos de LaTeX)."""
    if not re.fullmatch(r"[A-Za-z]+", nombre):
        raise ValueError(f"Nombre de macro inválido para LaTeX: {nombre!r}")
    if isinstance(valor, (int, np.integer)):
        texto = str(int(valor))
    elif isinstance(valor, (float, np.floating)):
        texto = f"{valor:.{decimales}f}"
    else:
        texto = str(valor)
    RESULTADOS[nombre] = texto
    return valor


def escribir_macros():
    """Escribe latex/generado/resultados.tex con todas las cifras registradas."""
    MACROS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lineas = [
        "% Archivo generado automáticamente por el notebook (no editar a mano).",
        *(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(RESULTADOS.items())),
    ]
    MACROS_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"{len(RESULTADOS)} macros escritas en {MACROS_PATH}")

# %% [markdown]
# ### Carga y partición de los datos
#
# Código inicial del enunciado: primero se reserva el 20 % para prueba y
# luego el 25 % del 80 % restante para validación, lo que deja
# aproximadamente 60/20/20.

# %%
def separar(X, y, clasificacion=False):
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE,
        stratify=y if clasificacion else None,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=RANDOM_STATE,
        stratify=y_dev if clasificacion else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


diabetes = load_diabetes(as_frame=True, scaled=False)
reg = separar(diabetes.data, diabetes.target)
Xr_train, Xr_val, Xr_test, yr_train, yr_val, yr_test = reg

cancer = load_breast_cancer(as_frame=True)
y_maligno = (cancer.target == 0).astype(int)
clf = separar(cancer.data, y_maligno, clasificacion=True)
Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test = clf

tamanos = pd.DataFrame(
    {
        "Entrenamiento": [len(Xr_train), len(Xc_train)],
        "Validación": [len(Xr_val), len(Xc_val)],
        "Prueba": [len(Xr_test), len(Xc_test)],
    },
    index=["Caso A (diabetes)", "Caso B (cáncer de mama)"],
)
print("Tabla 0.1 — Tamaño de las particiones")
tamanos
