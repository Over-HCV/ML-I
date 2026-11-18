"""Capa 3 · Variables de la exportación real por cliente y por muestra.

Responsabilidad única: armar, solo con fechas de desarrollo, las tablas que
usa la etapa de datos reales: el recorte al desarrollo con su partición
(entrenamiento y validación), los conteos semanales por cliente recortados a
las semanas en que el cliente estuvo activo, la tabla de perfil por cliente
(volumen, dispersión semanal, lotes, tipo de muestreo, no conformidad) y las
variables de cada muestra conocidas al ingresarla (sin fugas: la historia del
cliente solo cuenta resultados ya conocidos ese día).

Recibe las tablas de `proyecto.datos.rodam` (muestras con `asignar_lotes` y
lotes); no lee archivos.
"""
import numpy as np
import pandas as pd

from proyecto.datos import rodam
from proyecto.variables.comunes import es_desarrollo, es_entrenamiento, historial_conocido, tasa_suavizada

MINIMO_MUESTRAS = 20        # muestras de desarrollo para que un cliente entre en el perfil
MINIMO_LOTES = 20           # lotes de un tipo para medir la dispersión semanal de ese tipo
NO_APLICA = {"N.A.", "N.A", "n.a", "-", "nan", "None", "", "<NA>"}

VARIABLES_CLIENTE = ["muestras_semana_media", "dispersion_semanal", "proporcion_semanas_activas", "lotes_semana",
                     "tamano_lote_medio", "proporcion_planta", "tasa_no_conforme"]
NUMERICAS_INGRESO = ["n_especificaciones", "tamano_lote", "tasa_cliente", "historia_cliente"]
CATEGORICAS_INGRESO = ["sector", "origen_muestra", "forma_farmaceutica", "cuadro_complejo", "tipo_lote"]


# ----------------------------------------------------------------------
# Periodo de desarrollo
# ----------------------------------------------------------------------
def del_desarrollo(muestras):
    """Muestras con ingreso anterior al corte de prueba, con la columna
    `particion` ("entrenamiento" o "validacion") y la forma farmacéutica con
    sus variantes de "no aplica" unidas en un solo nivel."""
    muestras = muestras[es_desarrollo(muestras["fecha_ingreso"])].copy()
    muestras["particion"] = np.where(es_entrenamiento(muestras["fecha_ingreso"]), "entrenamiento", "validacion")
    forma = muestras["forma_farmaceutica"].astype(str).str.strip()
    muestras["forma_farmaceutica"] = np.where(forma.isin(NO_APLICA), "no aplica", forma.str.lower())
    return muestras


def semanas_de_desarrollo(muestras):
    """Primer y último lunes del periodo (la última semana empieza antes del
    corte de prueba)."""
    semana = muestras["dia"].dt.to_period("W-SUN").dt.start_time
    return semana.min(), semana.max()


def recortar_a_actividad(ancha):
    """Pone NaN en las semanas anteriores a la primera y posteriores a la
    última semana con algún conteo de cada fila: un cliente que llegó tarde o
    se fue no aporta ceros que no son suyos."""
    valores = ancha.to_numpy(dtype=float)
    con_dato = valores > 0
    primera = np.where(con_dato.any(axis=1), con_dato.argmax(axis=1), valores.shape[1])
    ultima = valores.shape[1] - 1 - np.where(con_dato.any(axis=1), con_dato[:, ::-1].argmax(axis=1), valores.shape[1])
    columnas = np.arange(valores.shape[1])[None, :]
    fuera = (columnas < primera[:, None]) | (columnas > ultima[:, None])
    return ancha.mask(fuera)


def conteos_semanales(tabla, desde, hasta, clientes=None):
    """Conteos por cliente y semana de una tabla de muestras o de lotes,
    recortados a la actividad de cada cliente. Con `clientes`, solo esas filas."""
    ancha = recortar_a_actividad(rodam.conteos_por_semana(tabla, desde=desde, hasta=hasta))
    return ancha if clientes is None else ancha.reindex(clientes).dropna(how="all")


# ----------------------------------------------------------------------
# Perfil por cliente
# ----------------------------------------------------------------------
def tabla_clientes(muestras, tabla_lotes, minimo=MINIMO_MUESTRAS):
    """Una fila por cliente con al menos `minimo` muestras de desarrollo:
    `muestras` (total, sirve de peso), `sector`, `muestras_semana_media`,
    `dispersion_semanal` (varianza sobre media de las muestras por semana),
    `proporcion_semanas_activas`, `lotes_semana`, `tamano_lote_medio`,
    `proporcion_planta` y `tasa_no_conforme`. Las semanas se cuentan entre la
    primera y la última semana con ingreso del cliente."""
    desde, hasta = semanas_de_desarrollo(muestras)
    por_semana = conteos_semanales(muestras, desde, hasta)
    lotes_semana = conteos_semanales(tabla_lotes, desde, hasta).reindex(por_semana.index)
    activas = por_semana.notna().sum(axis=1)
    tabla = pd.DataFrame({
        "muestras": muestras.groupby("cliente").size(),
        "sector": muestras.groupby("cliente")["sector"].agg(lambda s: s.mode().iat[0]),
        "muestras_semana_media": por_semana.mean(axis=1),
        "dispersion_semanal": por_semana.var(axis=1, ddof=1) / por_semana.mean(axis=1),
        "proporcion_semanas_activas": (por_semana > 0).sum(axis=1) / activas,
        "lotes_semana": lotes_semana.mean(axis=1),
        "tamano_lote_medio": tabla_lotes.groupby("cliente")["n_muestras"].mean(),
        "proporcion_planta": muestras.assign(planta=muestras["tipo_lote"] == "planta").groupby("cliente")["planta"].mean(),
        "tasa_no_conforme": muestras.groupby("cliente")["no_conforme"].mean().astype(float),
    })
    tabla = tabla[tabla["muestras"] >= minimo].dropna(subset=["dispersion_semanal"])
    tabla["tasa_no_conforme"] = tabla["tasa_no_conforme"].fillna(0.0)
    return tabla.rename_axis("cliente").reset_index()


def dispersion_por_tipo(muestras, tabla_lotes, minimo_lotes=MINIMO_LOTES):
    """Conteos semanales por tipo de lote ("producto" o "planta") para los
    clientes con al menos `minimo_lotes` lotes de ese tipo: un diccionario
    tipo → (muestras por semana, lotes por semana), dos tablas anchas."""
    desde, hasta = semanas_de_desarrollo(muestras)
    salida = {}
    for tipo in rodam.TIPOS_LOTE:
        lotes_tipo = tabla_lotes[tabla_lotes["tipo_lote"] == tipo]
        cuantos = lotes_tipo.groupby("cliente").size()
        clientes = cuantos[cuantos >= minimo_lotes].index
        muestras_tipo = muestras[muestras["tipo_lote"] == tipo]
        salida[tipo] = (conteos_semanales(muestras_tipo, desde, hasta, clientes),
                        conteos_semanales(lotes_tipo, desde, hasta, clientes))
    return salida


def indice_compuesto_por_cliente(tabla_lotes):
    """Índice de dispersión que tendrían las muestras por semana de cada
    cliente y tipo si sus lotes llegaran como Poisson: E[X²]/E[X], con X el
    tamaño de sus lotes (Poisson compuesta). Una fila por cliente y tipo."""
    tamanos = tabla_lotes.assign(cuadrado=tabla_lotes["n_muestras"].astype(float) ** 2)
    medias = tamanos.groupby(["cliente", "tipo_lote"])[["n_muestras", "cuadrado"]].mean()
    return (medias["cuadrado"] / medias["n_muestras"]).rename("indice").reset_index()


# ----------------------------------------------------------------------
# Variables de cada muestra al ingresar
# ----------------------------------------------------------------------
def variables_de_ingreso(muestras, tabla_lotes):
    """Una fila por muestra con dictamen y las variables conocidas al
    ingresarla: las de `CATEGORICAS_INGRESO` y `NUMERICAS_INGRESO`.

    `tamano_lote` es el número de muestras de su lote (llegan juntas el mismo
    día). `historia_cliente` es el número de dictámenes del cliente ya
    conocidos antes del ingreso (su última lectura anterior al ingreso) y
    `tasa_cliente` su no conformidad encogida hacia la tasa de
    entrenamiento. El objetivo es `no_conforme` (0 o 1)."""
    tabla = muestras.dropna(subset=["no_conforme"]).copy()
    tabla["no_conforme"] = tabla["no_conforme"].astype(int)
    tabla = tabla.merge(tabla_lotes[["id_lote", "n_muestras"]].rename(columns={"n_muestras": "tamano_lote"}),
                        on="id_lote", how="left")
    previa = float(tabla.loc[tabla["particion"] == "entrenamiento", "no_conforme"].mean())
    historia = historial_conocido(tabla, "cliente", "fecha_ingreso", "ultima_lectura", "no_conforme")
    tabla["historia_cliente"] = historia["n"].to_numpy()
    tabla["tasa_cliente"] = tasa_suavizada(historia["suma"], historia["n"], previa)
    tabla["cuadro_complejo"] = tabla["cuadro_complejo"].astype(str)
    return tabla.reset_index(drop=True), previa
