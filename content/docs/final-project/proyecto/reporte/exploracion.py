"""Capa 7 · Figuras de la exploración de datos.

Responsabilidad única: dibujar las figuras del análisis exploratorio. Cada
función recibe las tablas ya limpias (y ya filtradas al periodo de
desarrollo por quien la llama), no lee archivos y devuelve la figura de
matplotlib. Si a la tabla le falta una columna opcional que la figura
necesita, se devuelve una figura con un aviso en lugar de fallar, de forma
que la exploración siga funcionando con una exportación real incompleta.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.colors import LinearSegmentedColormap

from proyecto.base.config import COLORES, PALETA
from proyecto.datos import esquema

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MINIMO_CELDA = 30   # filas mínimas para mostrar una tasa en una celda o barra

MAPA_SECUENCIAL = LinearSegmentedColormap.from_list("pf_secuencial", ["#FFFFFF", COLORES["azul"]])
MAPA_DIVERGENTE = LinearSegmentedColormap.from_list(
    "pf_divergente", [COLORES["azul"], "#FFFFFF", COLORES["rojo"]])
MAPA_RIESGO = LinearSegmentedColormap.from_list("pf_riesgo", ["#FFFFFF", COLORES["rojo"]])


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def _ejes(ax, rejilla="y"):
    """Estilo común: sin bordes superior ni derecho y rejilla tenue."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if rejilla:
        ax.grid(axis=rejilla, color=COLORES["gris"], alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=COLORES["tinta"], labelsize=9)
    return ax


def _aviso(mensaje):
    """Figura con un aviso (cuando faltan las columnas que la figura necesita)."""
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.axis("off")
    ax.text(0.5, 0.5, mensaje, ha="center", va="center", fontsize=10, color=COLORES["tinta"], wrap=True)
    return fig


def _faltan(tabla, columnas):
    return [c for c in columnas if c not in tabla.columns]


def _wilson(positivos, n, z=1.96):
    """Intervalo de Wilson al 95 % para una proporción (vectorizado)."""
    positivos, n = np.asarray(positivos, dtype=float), np.asarray(n, dtype=float)
    n_seguro = np.where(n > 0, n, np.nan)
    p = positivos / n_seguro
    centro = (p + z ** 2 / (2 * n_seguro)) / (1 + z ** 2 / n_seguro)
    radio = z * np.sqrt(p * (1 - p) / n_seguro + z ** 2 / (4 * n_seguro ** 2)) / (1 + z ** 2 / n_seguro)
    return centro - radio, centro + radio


def _tinta_sobre(valor, maximo):
    """Color de texto legible sobre una celda de un mapa de calor."""
    return "#FFFFFF" if maximo > 0 and valor / maximo > 0.6 else COLORES["tinta"]


# ----------------------------------------------------------------------
# 1. Calidad
# ----------------------------------------------------------------------
def figura_faltantes(resumen, maximo=25):
    """Porcentaje de faltantes por columna, separando los estructurales (el
    dato no puede existir) de los demás. `resumen` es la tabla que devuelve
    `limpieza.faltantes_por_columna`."""
    datos = resumen[resumen["faltantes"] > 0].copy()
    if datos.empty:
        return _aviso("Ninguna columna del contrato tiene valores faltantes.")
    datos["pct_estructural"] = 100 * datos["estructurales"] / datos["filas"].clip(lower=1)
    datos = datos.sort_values("pct_faltante", ascending=False).head(maximo).iloc[::-1]
    etiquetas = datos["tabla"] + "." + datos["columna"]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(datos) + 1.4))
    ax.barh(etiquetas, datos["pct_no_estructural"], color=COLORES["rojo"], label="No estructural (dato perdido)")
    ax.barh(etiquetas, datos["pct_estructural"], left=datos["pct_no_estructural"], color=COLORES["gris"],
            alpha=0.6, label="Estructural (el dato no puede existir)")
    for y, total in enumerate(datos["pct_faltante"]):
        ax.text(total + 1, y, f"{total:.1f}", va="center", fontsize=8, color=COLORES["tinta"])
    ax.set_xlim(0, 108)
    ax.set_xlabel("Filas con el dato faltante (%)")
    ax.set_title("Valores faltantes por columna", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _ejes(ax, "x")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2–3. Volumen y estacionalidad
# ----------------------------------------------------------------------
def _semana_de(fechas):
    fechas = pd.to_datetime(fechas)
    return (fechas - pd.to_timedelta(fechas.dt.dayofweek, unit="D")).dt.normalize()


def figura_volumen_semanal(muestras, calendario=None, columna_grupo="sector"):
    """Muestras recibidas por semana, un panel por sector (o uno solo si no
    hay sector). Las semanas con algún festivo entre lunes y viernes se
    marcan con una línea vertical tenue."""
    datos = muestras.dropna(subset=["fecha_recepcion"]).copy()
    datos["_semana"] = _semana_de(datos["fecha_recepcion"])
    if columna_grupo in datos.columns:
        conteo = datos.groupby(["_semana", columna_grupo]).size().unstack(fill_value=0)
        conteo = conteo[conteo.sum().sort_values(ascending=False).index]
    else:
        conteo = datos.groupby("_semana").size().to_frame("Total")
    semanas = pd.date_range(conteo.index.min(), conteo.index.max(), freq="7D")
    conteo = conteo.reindex(semanas, fill_value=0)
    con_festivo = []
    if calendario is not None and {"fecha", "es_festivo"} <= set(calendario.columns):
        festivos = calendario[calendario["es_festivo"].astype(bool) & (calendario["fecha"].dt.dayofweek < 5)]
        con_festivo = sorted(set(_semana_de(festivos["fecha"])) & set(semanas))
    n = conteo.shape[1]
    fig, ejes = plt.subplots(n, 1, figsize=(10, 1.55 * n + 0.9), sharex=True, squeeze=False)
    for i, (ax, grupo) in enumerate(zip(ejes[:, 0], conteo.columns)):
        for semana in con_festivo:
            ax.axvline(semana, color=COLORES["gris"], alpha=0.35, linewidth=0.8, zorder=1)
        ax.plot(conteo.index, conteo[grupo], color=PALETA[i % len(PALETA)], linewidth=1.1, zorder=2)
        ax.set_ylim(0, None)
        ax.set_title(str(grupo), loc="left", fontsize=9.5, color=COLORES["tinta"], pad=3)
        _ejes(ax)
    ejes[n // 2, 0].set_ylabel("Muestras por semana")
    titulo = "Muestras recibidas por semana"
    if con_festivo:
        titulo += " (líneas grises: semanas con festivo)"
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def figura_estacionalidad(muestras, columna_grupo="sector"):
    """Mapa de calor mes × sector del índice estacional: volumen medio del
    mes dividido por el volumen mensual medio del sector (1 = mes típico)."""
    faltan = _faltan(muestras, [columna_grupo])
    if faltan:
        return _aviso(f"No se puede dibujar la estacionalidad: falta la columna «{faltan[0]}».")
    datos = muestras.dropna(subset=["fecha_recepcion", columna_grupo])
    fechas = datos["fecha_recepcion"]
    mensual = datos.groupby([fechas.dt.year.rename("anio"), fechas.dt.month.rename("mes"),
                             datos[columna_grupo]]).size().unstack(fill_value=0)
    indice = (mensual / mensual.mean()).groupby(level="mes").mean().T
    indice = indice.reindex(columns=range(1, 13))
    desvio = np.nanmax(np.abs(indice.to_numpy() - 1))
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(indice) + 1.6))
    imagen = ax.imshow(indice.to_numpy(), cmap=MAPA_DIVERGENTE, vmin=1 - desvio, vmax=1 + desvio, aspect="auto")
    for i in range(indice.shape[0]):
        for j in range(indice.shape[1]):
            valor = indice.iat[i, j]
            if not np.isnan(valor):
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8,
                        color=_tinta_sobre(abs(valor - 1), desvio))
    ax.set_xticks(range(12))
    ax.set_xticklabels(MESES)
    ax.set_yticks(range(len(indice)))
    ax.set_yticklabels(indice.index)
    ax.tick_params(length=0, labelsize=9)
    for borde in ax.spines.values():
        borde.set_visible(False)
    fig.colorbar(imagen, ax=ax, fraction=0.03, pad=0.02, label="Índice (1 = mes medio)")
    ax.set_title(f"Estacionalidad del volumen por {columna_grupo} y mes", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 4. Tasas por categoría
# ----------------------------------------------------------------------
def figura_tasa_por_categoria(muestras, columnas=("sector", "matriz", "ensayo"), objetivo="no_conforme",
                              etiqueta="Muestras no conformes (%)"):
    """Tasa del objetivo (0/1) por nivel de cada columna categórica, con su
    intervalo de Wilson al 95 % y el número de muestras de cada nivel (en una
    columna a la derecha de cada panel). La línea punteada es la tasa global."""
    columnas = [c for c in columnas if c in muestras.columns]
    if objetivo not in muestras.columns or not columnas:
        return _aviso(f"No se puede dibujar la tasa por categoría: falta «{objetivo}» o las columnas de agrupación.")
    datos = muestras.dropna(subset=[objetivo])
    tasa_global = 100 * datos[objetivo].mean()
    alturas = [max(datos[c].nunique(), 2) for c in columnas]
    fig, ejes = plt.subplots(len(columnas), 1, figsize=(8.5, 0.33 * sum(alturas) + 1.1 * len(columnas) + 0.6),
                             gridspec_kw={"height_ratios": alturas}, squeeze=False)
    for ax, columna in zip(ejes[:, 0], columnas):
        grupo = datos.groupby(columna)[objetivo].agg(["sum", "count"])
        grupo = grupo[grupo["count"] >= MINIMO_CELDA]
        grupo["tasa"] = 100 * grupo["sum"] / grupo["count"]
        grupo = grupo.sort_values("tasa")
        inferior, superior = _wilson(grupo["sum"], grupo["count"])
        errores = np.vstack([grupo["tasa"] - 100 * inferior, 100 * superior - grupo["tasa"]]).clip(min=0)
        ax.barh(grupo.index.astype(str), grupo["tasa"], color=COLORES["azul"], alpha=0.85,
                xerr=errores, error_kw={"ecolor": COLORES["tinta"], "elinewidth": 0.9, "capsize": 2})
        ax.axvline(tasa_global, color=COLORES["rojo"], linestyle="--", linewidth=1)
        tope = max(float((100 * superior).max()), tasa_global)
        # Los tamaños van en una columna fija a la derecha del panel, fuera del
        # área de datos, para que nunca se crucen con la línea de la tasa global.
        columna_n = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for y, n in enumerate(grupo["count"]):
            ax.text(1.02, y, f"n = {int(n)}", transform=columna_n, ha="left", va="center", fontsize=8,
                    color=COLORES["gris"])
        ax.set_xlim(0, tope * 1.05)
        ax.set_title(f"Por {columna}", loc="left", fontsize=9.5)
        _ejes(ax, "x")
    ejes[-1, 0].set_xlabel(f"{etiqueta} · línea punteada: tasa global ({tasa_global:.1f} %)")
    fig.suptitle("Tasa por categoría con intervalo de 95 %", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 5–6. Variables numéricas y resultado
# ----------------------------------------------------------------------
def figura_distribuciones_por_resultado(muestras, columnas=None, objetivo="no_conforme",
                                        nombres=("Conforme", "No conforme")):
    """Histogramas (densidad) de las variables numéricas conocidas al llegar
    la muestra, separados por resultado. Con densidades las dos clases son
    comparables aunque una sea mucho menos frecuente."""
    if columnas is None:
        columnas = esquema.columnas("muestras", tipo="numero", hasta="llegada")
    columnas = [c for c in columnas if c in muestras.columns and muestras[c].notna().sum() > MINIMO_CELDA]
    if objetivo not in muestras.columns or not columnas:
        return _aviso("No hay variables numéricas con datos para comparar por resultado.")
    datos = muestras.dropna(subset=[objetivo])
    n_columnas = min(3, len(columnas))
    n_filas = int(np.ceil(len(columnas) / n_columnas))
    fig, ejes = plt.subplots(n_filas, n_columnas, figsize=(3.6 * n_columnas, 2.6 * n_filas + 0.5), squeeze=False)
    for ax, columna in zip(ejes.ravel(), columnas):
        valores = datos[columna].dropna()
        limites = np.linspace(valores.quantile(0.005), valores.quantile(0.995), 31)
        for clase, color, nombre in [(0, COLORES["azul"], nombres[0]), (1, COLORES["rojo"], nombres[1])]:
            de_clase = datos.loc[datos[objetivo] == clase, columna].dropna()
            if len(de_clase):
                ax.hist(de_clase, bins=limites, density=True, histtype="stepfilled", alpha=0.35, color=color,
                        edgecolor=color, linewidth=1.1, label=f"{nombre} (n = {len(de_clase)})")
        ax.set_title(columna, loc="left", fontsize=9.5)
        ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False)
        _ejes(ax, None)
    for ax in ejes.ravel()[len(columnas):]:
        ax.axis("off")
    fig.suptitle("Distribución de las variables numéricas según el resultado (densidad)", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    return fig


def figura_transporte(muestras, objetivo="no_conforme",
                      cortes_temperatura=(4, 8, 12, 16, 20), cortes_horas=(6, 12, 18, 24, 36)):
    """Tasa del objetivo por temperatura y horas de transporte (mapa de
    calor). Cada celda muestra la tasa y el número de muestras; las celdas
    con pocas muestras quedan en blanco."""
    necesarias = ["temperatura_transporte_c", "horas_transporte", objetivo]
    faltan = _faltan(muestras, necesarias)
    if faltan:
        return _aviso(f"No se puede dibujar el transporte: falta «{', '.join(faltan)}».")
    datos = muestras.dropna(subset=necesarias)

    def tramos(cortes, unidad):
        nombres = ([f"< {cortes[0]} {unidad}"]
                   + [f"{a}–{b} {unidad}" for a, b in zip(cortes[:-1], cortes[1:])] + [f"≥ {cortes[-1]} {unidad}"])
        return [-np.inf, *cortes, np.inf], nombres

    bordes_t, nombres_t = tramos(list(cortes_temperatura), "°C")
    bordes_h, nombres_h = tramos(list(cortes_horas), "h")
    fila = pd.cut(datos["temperatura_transporte_c"], bordes_t, labels=nombres_t, right=False)
    columna = pd.cut(datos["horas_transporte"], bordes_h, labels=nombres_h, right=False)
    cuenta = pd.crosstab(fila, columna, dropna=False).reindex(index=nombres_t, columns=nombres_h, fill_value=0)
    positivos = pd.crosstab(fila, columna, values=datos[objetivo], aggfunc="sum", dropna=False)
    positivos = positivos.reindex(index=nombres_t, columns=nombres_h).fillna(0)
    tasa = (100 * positivos / cuenta.where(cuenta >= MINIMO_CELDA)).iloc[::-1]
    cuenta = cuenta.iloc[::-1]
    maximo = float(np.nanmax(tasa.to_numpy())) if tasa.notna().any().any() else 1.0
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    imagen = ax.imshow(tasa.to_numpy(dtype=float), cmap=MAPA_RIESGO, vmin=0, vmax=maximo, aspect="auto")
    for i in range(tasa.shape[0]):
        for j in range(tasa.shape[1]):
            n, valor = int(cuenta.iat[i, j]), tasa.iat[i, j]
            if np.isnan(valor):
                ax.text(j, i, f"n = {n}", ha="center", va="center", fontsize=7.5, color=COLORES["gris"])
            else:
                ax.text(j, i, f"{valor:.1f} %\nn = {n}", ha="center", va="center", fontsize=8,
                        color=_tinta_sobre(valor, maximo))
    ax.set_xticks(range(len(nombres_h)))
    ax.set_xticklabels(nombres_h)
    ax.set_yticks(range(len(nombres_t)))
    ax.set_yticklabels(nombres_t[::-1])
    ax.set_xlabel("Horas de transporte")
    ax.set_ylabel("Temperatura al llegar")
    ax.tick_params(length=0, labelsize=9)
    for borde in ax.spines.values():
        borde.set_visible(False)
    fig.colorbar(imagen, ax=ax, fraction=0.035, pad=0.02, label="Muestras no conformes (%)")
    ax.set_title(f"Resultado según las condiciones de transporte (celdas con n < {MINIMO_CELDA} en blanco)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 7. Tiempos de entrega
# ----------------------------------------------------------------------
def figura_dias_entrega(muestras, columna_grupo="ensayo"):
    """Días calendario entre recepción y reporte por ensayo (cajas, sin
    atípicos dibujados) con el plazo prometido mediano como referencia y el
    porcentaje de entregas tardías."""
    faltan = _faltan(muestras, ["dias_entrega", columna_grupo])
    if faltan or muestras["dias_entrega"].notna().sum() == 0:
        return _aviso("No se pueden dibujar los tiempos de entrega: no hay fechas de reporte.")
    datos = muestras.dropna(subset=["dias_entrega", columna_grupo])
    orden = datos.groupby(columna_grupo)["dias_entrega"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(orden) + 1.6))
    cajas = ax.boxplot([datos.loc[datos[columna_grupo] == g, "dias_entrega"] for g in orden], vert=False,
                       showfliers=False, patch_artist=True, widths=0.6,
                       medianprops={"color": COLORES["tinta"], "linewidth": 1.4})
    for caja in cajas["boxes"]:
        caja.set(facecolor=COLORES["azul"], alpha=0.55, edgecolor=COLORES["azul"])
    ax.set_yticks(range(1, len(orden) + 1))
    ax.set_yticklabels(orden)
    con_plazo = "dias_prometidos" in datos.columns and datos["dias_prometidos"].notna().any()
    if con_plazo:
        plazo = datos.groupby(columna_grupo)["dias_prometidos"].median().reindex(orden)
        ax.scatter(plazo, range(1, len(orden) + 1), marker="|", s=260, color=COLORES["rojo"], linewidths=2,
                   zorder=3, label="Plazo prometido (mediana)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    if "entrega_tardia" in datos.columns and datos["entrega_tardia"].notna().any():
        tardias = 100 * datos.groupby(columna_grupo)["entrega_tardia"].mean().reindex(orden)
        derecha = ax.get_xlim()[1]
        for y, valor in enumerate(tardias, start=1):
            ax.text(derecha, y, f"  {valor:.1f} % tardías", va="center", fontsize=8, color=COLORES["gris"])
    ax.set_xlabel("Días calendario entre la recepción y el reporte")
    ax.set_title(f"Tiempo de entrega por {columna_grupo}", loc="left", fontsize=11)
    _ejes(ax, "x")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 8. Deriva en el tiempo
# ----------------------------------------------------------------------
def figura_tasa_mensual(muestras, objetivo="no_conforme", por=None, columna_fecha="fecha_recepcion",
                        etiqueta="Muestras no conformes (%)"):
    """Tasa mensual del objetivo a lo largo del tiempo, con banda de Wilson
    al 95 %. Con `por` (una columna categórica) dibuja una línea por nivel
    (los seis más frecuentes), útil para ver deriva localizada."""
    if objetivo not in muestras.columns or (por is not None and por not in muestras.columns):
        return _aviso(f"No se puede dibujar la tasa mensual: falta «{objetivo if por is None else por}».")
    datos = muestras.dropna(subset=[objetivo, columna_fecha]).copy()
    datos["_mes"] = datos[columna_fecha].dt.to_period("M").dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    if por is None:
        mensual = datos.groupby("_mes")[objetivo].agg(["sum", "count"])
        inferior, superior = _wilson(mensual["sum"], mensual["count"])
        ax.fill_between(mensual.index, 100 * inferior, 100 * superior, color=COLORES["azul"], alpha=0.18,
                        linewidth=0, label="Intervalo de 95 %")
        ax.plot(mensual.index, 100 * mensual["sum"] / mensual["count"], color=COLORES["azul"], linewidth=1.6,
                marker="o", markersize=3, label="Tasa mensual")
        media = 100 * datos[objetivo].mean()
        ax.axhline(media, color=COLORES["rojo"], linestyle="--", linewidth=1, label=f"Tasa del periodo ({media:.1f} %)")
    else:
        niveles = datos[por].value_counts().head(len(PALETA)).index.tolist()
        for color, nivel in zip(PALETA, sorted(niveles)):
            mensual = datos[datos[por] == nivel].groupby("_mes")[objetivo].agg(["sum", "count"])
            mensual = mensual[mensual["count"] >= MINIMO_CELDA]
            ax.plot(mensual.index, 100 * mensual["sum"] / mensual["count"], color=color, linewidth=1.4,
                    marker="o", markersize=2.5, label=str(nivel))
    ax.set_ylim(0, None)
    ax.set_ylabel(etiqueta)
    ax.set_title("Evolución mensual" + (f" por {por}" if por else ""), loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    _ejes(ax)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 9. Correlaciones
# ----------------------------------------------------------------------
def figura_correlaciones(muestras, columnas=None, metodo="spearman"):
    """Matriz de correlación (Spearman por defecto) de las columnas numéricas.
    Por defecto usa los números del contrato más los resultados numéricos
    derivados que existan."""
    if columnas is None:
        columnas = (esquema.columnas("muestras", tipo="numero")
                    + ["distancia_km", "antiguedad_dias", "recuento_ufc", "dias_entrega", "no_conforme",
                       "entrega_tardia"])
    columnas = [c for c in dict.fromkeys(columnas)
                if c in muestras.columns and muestras[c].notna().sum() > MINIMO_CELDA and muestras[c].nunique() > 1]
    if len(columnas) < 2:
        return _aviso("No hay suficientes columnas numéricas para calcular correlaciones.")
    correlacion = muestras[columnas].astype(float).corr(method=metodo)
    n = len(columnas)
    fig, ax = plt.subplots(figsize=(0.62 * n + 2.6, 0.55 * n + 1.8))
    visible = np.tril(np.ones((n, n), dtype=bool))
    imagen = ax.imshow(np.where(visible, correlacion.to_numpy(), np.nan), cmap=MAPA_DIVERGENTE, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            valor = correlacion.iat[i, j]
            if i != j and not np.isnan(valor):
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=7,
                        color=_tinta_sobre(abs(valor), 1.0))
    ax.set_xticks(range(n))
    ax.set_xticklabels(columnas, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(columnas)
    ax.tick_params(length=0, labelsize=8)
    for borde in ax.spines.values():
        borde.set_visible(False)
    fig.colorbar(imagen, ax=ax, fraction=0.035, pad=0.02, label=f"Correlación de {metodo.capitalize()}")
    ax.set_title("Correlación entre columnas numéricas", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 10. Clientes
# ----------------------------------------------------------------------
def figura_volumen_clientes(muestras, columna_cliente="id_cliente"):
    """Concentración del volumen por cliente: muestras de cada cliente
    ordenadas de mayor a menor (escala logarítmica) y participación acumulada
    (cola pesada: pocos clientes aportan gran parte del volumen)."""
    volumen = muestras[columna_cliente].dropna().value_counts().sort_values(ascending=False)
    if volumen.empty:
        return _aviso("No hay muestras con cliente identificado.")
    puesto = np.arange(1, len(volumen) + 1)
    acumulado = 100 * volumen.cumsum().to_numpy() / volumen.sum()
    fig, (izquierda, derecha) = plt.subplots(1, 2, figsize=(10, 3.8))
    izquierda.bar(puesto, volumen.to_numpy(), width=1.0, color=COLORES["azul"], alpha=0.85)
    izquierda.set_yscale("log")
    izquierda.axhline(volumen.median(), color=COLORES["rojo"], linestyle="--", linewidth=1,
                      label=f"Mediana: {volumen.median():.0f} muestras")
    izquierda.set_xlabel("Clientes ordenados por volumen")
    izquierda.set_ylabel("Muestras en el periodo (escala log)")
    izquierda.set_title("Volumen por cliente", loc="left", fontsize=11)
    izquierda.legend(fontsize=8, frameon=False)
    _ejes(izquierda)
    pct_clientes = 100 * puesto / len(volumen)
    derecha.plot(pct_clientes, acumulado, color=COLORES["azul"], linewidth=1.8)
    derecha.plot([0, 100], [0, 100], color=COLORES["gris"], linestyle=":", linewidth=1, label="Reparto uniforme")
    quinto = max(int(np.ceil(0.2 * len(volumen))), 1)
    derecha.scatter([pct_clientes[quinto - 1]], [acumulado[quinto - 1]], color=COLORES["rojo"], zorder=3)
    derecha.annotate(f"El 20 % de los clientes\naporta el {acumulado[quinto - 1]:.0f} % de las muestras",
                     (pct_clientes[quinto - 1], acumulado[quinto - 1]), xytext=(12, -28),
                     textcoords="offset points", fontsize=8.5, color=COLORES["tinta"])
    derecha.set_xlim(0, 100)
    derecha.set_ylim(0, 102)
    derecha.set_xlabel("Clientes, de mayor a menor volumen (%)")
    derecha.set_ylabel("Muestras acumuladas (%)")
    derecha.set_title("Concentración del volumen", loc="left", fontsize=11)
    derecha.legend(fontsize=8, frameon=False, loc="lower right")
    _ejes(derecha, "both")
    fig.tight_layout()
    return fig
