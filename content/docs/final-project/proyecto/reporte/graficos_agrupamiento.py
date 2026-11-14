"""Capa 7 · Figuras del agrupamiento.

Responsabilidad única: dibujar las figuras de la etapa de agrupamiento a
partir de tablas ya calculadas: método del codo con la silueta, curva de
k-distancias de DBSCAN, proyección en dos componentes principales coloreada
por grupo y mapa de calor de los perfiles estandarizados. Cada función
recibe datos en memoria, no lee archivos y devuelve la figura de matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np

from proyecto.base.config import COLORES, PALETA
from proyecto.reporte.graficos_modelado import _ejes


def figura_codo_silueta(tabla_k, k_elegido, titulo=None):
    """Dos paneles con la inercia (método del codo) y la silueta media de
    K-means para cada k; el k elegido queda marcado en ambos. `tabla_k` es la
    salida de `agrupamiento.explorar_kmeans`."""
    fig, (ax_codo, ax_sil) = plt.subplots(1, 2, figsize=(10, 3.8))
    ks = tabla_k["k"].to_numpy()
    for ax, columna, color, etiqueta in ((ax_codo, "inercia", COLORES["azul"], "Inercia (suma de distancias²)"),
                                         (ax_sil, "silueta", COLORES["verde"], "Silueta media")):
        valores = tabla_k[columna].to_numpy(dtype=float)
        ax.plot(ks, valores, marker="o", color=color, linewidth=1.8)
        elegido = ks == k_elegido
        ax.scatter(ks[elegido], valores[elegido], s=140, facecolor="none", edgecolor=COLORES["rojo"],
                   linewidth=2, zorder=3, label=f"k elegido = {k_elegido}")
        ax.set_xticks(ks)
        ax.set_xlabel("Número de grupos (k)")
        ax.set_ylabel(etiqueta)
        _ejes(ax)
    ax_codo.set_title("Método del codo", fontsize=11)
    ax_sil.set_title("Silueta (más alta = grupos más separados)", fontsize=11)
    ax_sil.legend(frameon=False, fontsize=9)
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    return fig


def figura_k_distancia(distancias, k, eps_elegido, eps_rejilla=None, titulo=None):
    """Curva ordenada de la distancia de cada punto a su k-ésimo vecino, con
    la franja de `eps` explorada y el `eps` elegido. Los puntos por encima de
    la línea del `eps` elegido no tienen k vecinos tan cerca: son candidatos
    a ruido."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    distancias = np.asarray(distancias, dtype=float)
    posicion = np.arange(1, len(distancias) + 1)
    ax.plot(posicion, distancias, color=COLORES["azul"], linewidth=1.6)
    if eps_rejilla is not None and len(eps_rejilla):
        ax.axhspan(min(eps_rejilla), max(eps_rejilla), color=COLORES["naranja"], alpha=0.12,
                   label="franja de eps explorada")
    ax.axhline(eps_elegido, color=COLORES["rojo"], linestyle="--", linewidth=1.4,
               label=f"eps elegido = {eps_elegido:.2f}")
    ax.set_xlabel("Puntos ordenados por su distancia")
    ax.set_ylabel(f"Distancia al vecino {k}")
    ax.set_title(titulo or f"Curva de k-distancias (k = {k})", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _ejes(ax)
    fig.tight_layout()
    return fig


def figura_pca_grupos(coordenadas, etiquetas, varianza, nombres=None, titulo=None, ax=None):
    """Dispersión en las dos primeras componentes principales, un color por
    grupo; la etiqueta −1 (ruido de DBSCAN) va en gris. `varianza` es la
    proporción de varianza explicada por cada componente y `nombres` un
    diccionario opcional {etiqueta: rótulo}."""
    propia = ax is None
    if propia:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    coordenadas = np.asarray(coordenadas, dtype=float)
    etiquetas = np.asarray(etiquetas)
    nombres = nombres or {}
    colores = [c for c in PALETA if c != COLORES["gris"]]
    grupos = [g for g in np.unique(etiquetas) if g != -1]
    tamano = 8 if len(etiquetas) > 1000 else 22
    if (etiquetas == -1).any():
        ruido = etiquetas == -1
        ax.scatter(coordenadas[ruido, 0], coordenadas[ruido, 1], s=tamano, color=COLORES["gris"], alpha=0.5,
                   marker="x", linewidth=0.8, label=nombres.get(-1, "ruido"))
    for i, grupo in enumerate(grupos):
        miembros = etiquetas == grupo
        ax.scatter(coordenadas[miembros, 0], coordenadas[miembros, 1], s=tamano, alpha=0.6,
                   color=colores[i % len(colores)], edgecolor="none",
                   label=nombres.get(grupo, f"grupo {grupo}"))
    ax.set_xlabel(f"Componente 1 ({100 * varianza[0]:.1f} % de la varianza)")
    ax.set_ylabel(f"Componente 2 ({100 * varianza[1]:.1f} % de la varianza)")
    ax.set_title(titulo or "Proyección en dos componentes principales", fontsize=11)
    ax.legend(frameon=False, fontsize=8, markerscale=1.8, loc="best")
    _ejes(ax, rejilla="both")
    if propia:
        fig.tight_layout()
    return fig


def figura_pca_comparada(coordenadas, soluciones, varianza, titulo=None):
    """Varios paneles de `figura_pca_grupos` lado a lado sobre las mismas
    coordenadas. `soluciones` es un diccionario {título del panel:
    (etiquetas, nombres)}."""
    fig, ejes = plt.subplots(1, len(soluciones), figsize=(6.2 * len(soluciones), 4.8), squeeze=False)
    for ax, (rotulo, (etiquetas, nombres)) in zip(ejes[0], soluciones.items()):
        figura_pca_grupos(coordenadas, etiquetas, varianza, nombres=nombres, titulo=rotulo, ax=ax)
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    return fig


def figura_perfil_calor(perfil_z, etiquetas_variables=None, titulo=None):
    """Mapa de calor de las medias estandarizadas de cada grupo: filas =
    grupos (índice de `perfil_z`), columnas = variables. Cada celda dice a
    cuántas desviaciones estándar de la media general queda el grupo."""
    etiquetas_variables = etiquetas_variables or {}
    valores = perfil_z.to_numpy(dtype=float)
    limite = max(1.0, float(np.nanmax(np.abs(valores))))
    ancho = max(7.0, 0.85 * perfil_z.shape[1] + 2.5)
    fig, ax = plt.subplots(figsize=(ancho, 0.6 * perfil_z.shape[0] + 2.4))
    imagen = ax.imshow(valores, cmap="RdBu_r", vmin=-limite, vmax=limite, aspect="auto")
    ax.set_xticks(range(perfil_z.shape[1]))
    ax.set_xticklabels([etiquetas_variables.get(c, str(c)) for c in perfil_z.columns], rotation=35, ha="right",
                       fontsize=9)
    ax.set_yticks(range(perfil_z.shape[0]))
    ax.set_yticklabels([str(g) for g in perfil_z.index], fontsize=9)
    for fila in range(valores.shape[0]):
        for columna in range(valores.shape[1]):
            valor = valores[fila, columna]
            if np.isfinite(valor):
                ax.text(columna, fila, f"{valor:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(valor) > 0.5 * limite else COLORES["tinta"])
    barra = fig.colorbar(imagen, ax=ax, fraction=0.03, pad=0.02)
    barra.set_label("Desviaciones estándar respecto a la media general", fontsize=9)
    ax.set_title(titulo or "Perfil de cada grupo (medias estandarizadas)", fontsize=11)
    ax.grid(False)
    ax.tick_params(length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)
    fig.tight_layout()
    return fig
