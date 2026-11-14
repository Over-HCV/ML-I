"""Capa 7 · Figuras de la clasificación y del análisis bayesiano.

Responsabilidad única: dibujar las figuras de la etapa de clasificación y
Bayes a partir de datos ya calculados (curvas ROC, diagramas de fiabilidad,
recorrido de umbrales, matriz de confusión, valor predictivo positivo por
ensayo y por muestra, tasas por tramo) y resumir en una tabla las reglas de
un árbol ajustado. Cada función recibe datos en memoria, no lee archivos y
devuelve la figura de matplotlib (o la tabla).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

from proyecto.base.config import COLORES, PALETA
from proyecto.reporte.graficos_modelado import _ejes


def figura_roc(curvas, titulo="Curvas ROC"):
    """Curva ROC de cada candidato. `curvas` es `{nombre: (y, proba)}`; el
    AUC de cada curva va en la leyenda."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for (nombre, (y, proba)), color in zip(curvas.items(), PALETA):
        tfp, tvp, _ = roc_curve(y, proba)
        ax.plot(tfp, tvp, color=color, label=f"{nombre} (AUC {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORES["gris"], linewidth=1, label="Azar (AUC 0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Tasa de falsas alarmas (1 − especificidad)")
    ax.set_ylabel("Tasa de aciertos (recall)")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    _ejes(ax, rejilla="both")
    fig.tight_layout()
    return fig


def figura_fiabilidad(curvas, n_tramos=10, titulo="Diagrama de fiabilidad",
                      etiqueta_x="Probabilidad predicha (media del tramo)",
                      etiqueta_y="Fracción observada de positivos"):
    """Diagrama de fiabilidad: en cada tramo de probabilidad predicha (por
    cuantiles), la media predicha contra la fracción observada. Debajo, el
    histograma de las probabilidades. `curvas` es `{nombre: (y, proba)}`."""
    fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(6.6, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    maximo = 0.0
    for (nombre, (y, proba)), color in zip(curvas.items(), PALETA):
        proba = np.asarray(proba, dtype=float)
        observada, predicha = calibration_curve(np.asarray(y), proba, n_bins=n_tramos, strategy="quantile")
        ax.plot(predicha, observada, marker="o", color=color, label=nombre)
        ax_h.hist(proba, bins=40, range=(0, 1), color=color, alpha=0.45)
        maximo = max(maximo, float(np.max(predicha)), float(np.max(observada)))
    limite = min(1.0, maximo * 1.1 + 0.02)
    ax.plot([0, limite], [0, limite], linestyle="--", color=COLORES["gris"], linewidth=1,
            label="Calibración perfecta")
    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    ax.set_ylabel(etiqueta_y)
    ax.set_title(titulo)
    ax.legend(loc="upper left")
    _ejes(ax, rejilla="both")
    ax_h.set_xlabel(etiqueta_x)
    ax_h.set_ylabel("Filas")
    _ejes(ax_h)
    fig.tight_layout()
    return fig


def figura_umbrales(tabla, umbral_elegido, umbral_teorico=None, columna_semana="Alertas por semana"):
    """Recorrido de umbrales en tres paneles con el mismo eje: costo,
    precisión y recall, y alertas por semana. Líneas verticales en el umbral
    elegido y, si se da, en el teórico."""
    fig, ejes = plt.subplots(3, 1, figsize=(7.6, 7.4), sharex=True)
    x = tabla["Umbral"].to_numpy(dtype=float)
    ejes[0].plot(x, tabla["Costo"], color=COLORES["rojo"], marker="o", markersize=3)
    ejes[0].set_ylabel("Costo")
    ejes[1].plot(x, tabla["Precisión"], color=COLORES["azul"], marker="o", markersize=3, label="Precisión")
    ejes[1].plot(x, tabla["Recall"], color=COLORES["verde"], marker="o", markersize=3, label="Recall")
    ejes[1].set_ylabel("Proporción")
    ejes[1].set_ylim(0, 1.02)
    ejes[1].legend(loc="upper right")
    ejes[2].plot(x, tabla[columna_semana], color=COLORES["naranja"], marker="o", markersize=3)
    ejes[2].set_ylabel("Alertas por semana")
    ejes[2].set_xlabel("Umbral de probabilidad")
    distinto = umbral_teorico is not None and not np.isclose(umbral_teorico, umbral_elegido, atol=1e-3)
    for i, ax in enumerate(ejes):
        ax.axvline(umbral_elegido, color=COLORES["tinta"], linewidth=1.2,
                   label=(f"Elegido ({umbral_elegido:.3f})" if distinto or umbral_teorico is None else
                          f"Elegido = teórico ({umbral_elegido:.3f})") if i == 0 else None)
        if distinto:
            ax.axvline(umbral_teorico, color=COLORES["gris"], linewidth=1, linestyle="--",
                       label=f"Teórico ({umbral_teorico:.3f})" if i == 0 else None)
        _ejes(ax)
    ejes[0].legend(loc="upper right")
    ejes[0].set_title("Costo, aciertos y carga de alertas según el umbral")
    fig.tight_layout()
    return fig


def figura_confusion(tn, fp, fn, tp, titulo="Matriz de confusión",
                     clases=("conforme", "no conforme"), acciones=("sin alerta", "alerta")):
    """Matriz de confusión 2 × 2 con el conteo y el porcentaje de la fila
    (de cada clase real, qué fracción recibe cada acción)."""
    matriz = np.array([[tn, fp], [fn, tp]], dtype=float)
    por_fila = matriz / matriz.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.imshow(por_fila, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(matriz[i, j])}\n({100 * por_fila[i, j]:.1f} %)", ha="center", va="center",
                    color="white" if por_fila[i, j] > 0.55 else COLORES["tinta"], fontsize=11)
    ax.set_xticks([0, 1], [f"Predicho: {a}" for a in acciones])
    ax.set_yticks([0, 1], [f"Real: {c}" for c in clases])
    ax.grid(False)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def figura_vpp_ensayo(tabla, columna="vpp", inferior="vpp_inf", superior="vpp_sup", columna_real=None,
                      titulo="Valor predictivo positivo por ensayo"):
    """Barras horizontales del VPP de cada ensayo con su intervalo; si se da
    `columna_real`, un rombo con el valor verdadero (solo en la auditoría de
    la simulación). `tabla` trae una fila por ensayo y la columna `ensayo`."""
    tabla = tabla.sort_values(columna)
    posiciones = np.arange(len(tabla))
    valor = tabla[columna].to_numpy(dtype=float)
    errores = np.vstack([valor - tabla[inferior].to_numpy(dtype=float),
                         tabla[superior].to_numpy(dtype=float) - valor])
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(tabla) + 1.6))
    ax.barh(posiciones, valor, color=COLORES["azul"], alpha=0.8, label="Estimado (IC 95 %)")
    ax.errorbar(valor, posiciones, xerr=np.clip(errores, 0, None), fmt="none", ecolor=COLORES["tinta"], capsize=3)
    if columna_real is not None:
        ax.scatter(tabla[columna_real], posiciones, marker="D", color=COLORES["naranja"], zorder=3,
                   label="Verdadero (simulación)")
    ax.set_yticks(posiciones, tabla["ensayo"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(contaminada | resultado no conforme)")
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _ejes(ax, rejilla="x")
    fig.tight_layout()
    return fig


def figura_vpp_muestra(tabla, referencia, columna="vpp", grupo="ensayo",
                       titulo="VPP por muestra entre los positivos de validación"):
    """Distribución del VPP de cada muestra positiva, por ensayo (caja y
    puntos), con un rombo en el VPP del ensayo sin previa por muestra
    (`referencia`: `{ensayo: vpp}`)."""
    orden = sorted(tabla[grupo].unique(), key=lambda g: referencia.get(g, np.nan))
    datos = [tabla.loc[tabla[grupo] == g, columna].to_numpy(dtype=float) for g in orden]
    fig, ax = plt.subplots(figsize=(7.8, 0.6 * len(orden) + 1.8))
    ax.boxplot(datos, vert=False, widths=0.55, showfliers=False,
               medianprops={"color": COLORES["tinta"]}, boxprops={"color": COLORES["azul"]})
    generador = np.random.default_rng(0)
    for i, valores in enumerate(datos, start=1):
        ax.scatter(valores, i + generador.uniform(-0.18, 0.18, len(valores)), s=7, alpha=0.35,
                   color=COLORES["azul"])
    ax.scatter([referencia.get(g, np.nan) for g in orden], np.arange(1, len(orden) + 1), marker="D", s=40,
               color=COLORES["naranja"], zorder=3, label="VPP del ensayo (previa común)")
    ax.set_yticks(np.arange(1, len(orden) + 1), [f"{g} (n = {len(d)})" for g, d in zip(orden, datos)])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("VPP(x) = P(contaminada | positivo, variables de la muestra)")
    ax.set_title(titulo)
    ax.legend(loc="upper left")
    _ejes(ax, rejilla="x")
    fig.tight_layout()
    return fig


def figura_tasa_por_tramo(tabla, etiqueta_x, etiqueta_y, titulo=None, referencia=None):
    """Tasa observada por tramo de una variable, con su intervalo y el número
    de filas encima. `tabla` trae `tramo`, `tasa`, `inf`, `sup` y `n`."""
    x = np.arange(len(tabla))
    tasa = tabla["tasa"].to_numpy(dtype=float)
    errores = np.vstack([tasa - tabla["inf"].to_numpy(dtype=float), tabla["sup"].to_numpy(dtype=float) - tasa])
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(x, tasa, color=COLORES["azul"], alpha=0.8)
    ax.errorbar(x, tasa, yerr=np.clip(errores, 0, None), fmt="none", ecolor=COLORES["tinta"], capsize=3)
    for xi, sup, n in zip(x, tabla["sup"], tabla["n"]):
        ax.text(xi, float(sup) + 0.02, f"n = {int(n)}", ha="center", fontsize=8.5, color=COLORES["gris"])
    if referencia is not None:
        ax.axhline(referencia, color=COLORES["rojo"], linestyle="--", linewidth=1,
                   label=f"Referencia ({referencia:.2f})")
        ax.legend(loc="upper right")
    ax.set_xticks(x, tabla["tramo"], rotation=20, ha="right")
    ax.set_ylim(0, min(1.0, float(np.nanmax(tabla["sup"])) + 0.12))
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    if titulo:
        ax.set_title(titulo)
    _ejes(ax)
    fig.tight_layout()
    return fig


def _condicion(nombre, umbral, a_la_izquierda, binarias):
    if nombre in binarias:
        return f"{nombre} = {0 if a_la_izquierda else 1}"
    return f"{nombre} {'≤' if a_la_izquierda else '>'} {umbral:.4g}"


def reglas_arbol(arbol, nombres, binarias=(), maximo=6):
    """Hojas de un árbol de clasificación ajustado como reglas legibles: la
    conjunción de condiciones del camino, las filas de entrenamiento que caen
    en la hoja y su fracción de la clase 1. Devuelve las `maximo` hojas con
    mayor fracción. `binarias` son las columnas 0/1 (indicadoras), que se
    escriben como «= 0» o «= 1»."""
    estructura = arbol.tree_
    binarias = set(binarias)
    filas = []

    def recorrer(nodo, condiciones):
        izquierda, derecha = estructura.children_left[nodo], estructura.children_right[nodo]
        if izquierda == derecha:
            valores = np.asarray(estructura.value[nodo][0], dtype=float)
            filas.append({"Regla": " y ".join(condiciones) or "(todas las filas)",
                          "Filas": int(estructura.n_node_samples[nodo]),
                          "Fracción de positivos": float(valores[-1] / valores.sum())})
            return
        nombre = nombres[estructura.feature[nodo]]
        umbral = float(estructura.threshold[nodo])
        recorrer(izquierda, condiciones + [_condicion(nombre, umbral, True, binarias)])
        recorrer(derecha, condiciones + [_condicion(nombre, umbral, False, binarias)])

    recorrer(0, [])
    tabla = pd.DataFrame(filas).sort_values("Fracción de positivos", ascending=False, kind="mergesort")
    return tabla.head(maximo).reset_index(drop=True)
