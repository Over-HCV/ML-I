"""Capa 7 · Nombres y lectura operativa de los grupos.

Responsabilidad única: convertir los perfiles estandarizados de un
agrupamiento en texto legible: etiquetas en español de las variables, nombre
y descripción de cada grupo a partir de las variables en que más se aparta
de la media, sugerencias operativas tomadas de una tabla de reglas escrita
de antemano, tabla de tamaño y desempeño de los segmentos y vista de los
casos que DBSCAN marca como ruido. No ajusta modelos: recibe resultados de
`proyecto.modelos.agrupamiento`.
"""
import numpy as np
import pandas as pd

from proyecto.modelos.agrupamiento import desviaciones_de_filas

ETIQUETAS = {
    "volumen_ml": "volumen de muestra", "hora_recepcion": "hora de recepción",
    "temperatura_transporte_c": "temperatura de transporte", "horas_transporte": "horas de transporte",
    "dias_prometidos": "días prometidos", "carga_dia": "carga del día", "control_ambiental_ufc": "control ambiental",
    "dias_entrega": "días de entrega", "muestras_mes_log": "volumen mensual", "tamano_lote_medio": "tamaño de lote",
    "productos_distintos": "variedad de productos", "ensayos_distintos": "variedad de ensayos",
    "proporcion_urgente": "urgencias", "proporcion_planta": "muestreo en planta", "tasa_no_conforme": "no conformidad",
    "proporcion_tardia": "entregas tardías", "temperatura_media": "temperatura de llegada",
    "regularidad_cv": "irregularidad de envíos", "concentracion_estacional": "estacionalidad",
    "antiguedad_dias": "antigüedad",
}

# Lectura operativa: qué podría hacer el laboratorio distinto con un grupo que
# queda claramente por encima (+1) o por debajo (−1) de la media en una
# variable. Son sugerencias, no hallazgos.
SUGERENCIAS = {
    ("tasa_no_conforme", 1): "ofrecer acompañamiento técnico (revisión del plan de muestreo y de higiene) y seguir de cerca sus resultados",
    ("proporcion_tardia", 1): "revisar la capacidad comprometida con estos clientes antes de prometer plazos",
    ("muestras_mes_log", 1): "negociar acuerdos de volumen y rutas de recolección dedicadas",
    ("muestras_mes_log", -1): "estudiar campañas comerciales para aumentar la frecuencia de muestreo",
    ("proporcion_urgente", 1): "hacer explícita una franja de servicio urgente y reservarle capacidad",
    ("regularidad_cv", 1): "planear con el cliente un calendario de muestreo que suavice la carga",
    ("concentracion_estacional", 1): "anticipar su temporada alta con personal y medios de cultivo",
    ("antiguedad_dias", -1): "hacer seguimiento de incorporación a los clientes recientes",
    ("tamano_lote_medio", 1): "pedir aviso previo de los lotes grandes para programar la siembra",
    ("proporcion_planta", 1): "coordinar las visitas de muestreo en planta con la logística de transporte",
    ("temperatura_media", 1): "reforzar la cadena de frío en el transporte (neveras y geles refrigerantes)",
    ("temperatura_transporte_c", 1): "reforzar la cadena de frío en el transporte (neveras y geles refrigerantes)",
    ("horas_transporte", 1): "revisar rutas o puntos de entrega para acortar el transporte",
    ("dias_entrega", 1): "revisar en qué paso del proceso se demoran estas muestras",
    ("carga_dia", 1): "desplazar parte de la recepción hacia días de menor carga",
    ("control_ambiental_ufc", 1): "revisar la limpieza del área en los días con control ambiental alto",
    ("volumen_ml", -1): "recordar al cliente el volumen mínimo de muestra",
    ("hora_recepcion", 1): "incentivar la entrega temprana para no aplazar la siembra",
}

# Frases para los nombres de grupo cuando "más/menos <variable>" suena forzado.
FRASES = {
    ("hora_recepcion", 1): "recepción tardía", ("hora_recepcion", -1): "recepción temprana",
    ("temperatura_transporte_c", 1): "transporte más cálido", ("temperatura_transporte_c", -1): "transporte más frío",
    ("horas_transporte", 1): "transporte largo", ("horas_transporte", -1): "transporte corto",
    ("dias_prometidos", 1): "plazo prometido largo", ("dias_prometidos", -1): "plazo prometido corto",
    ("dias_entrega", 1): "entrega lenta", ("dias_entrega", -1): "entrega rápida",
    ("antiguedad_dias", 1): "clientes antiguos", ("antiguedad_dias", -1): "clientes recientes",
}

# Tasas que se resumen por segmento, ponderadas por el número de muestras.
TASAS_SEGMENTO = (("tasa_no_conforme", "No conformidad"), ("proporcion_tardia", "Entregas tardías"))


def etiqueta(columna):
    """Nombre legible de una variable (la propia columna si no está en
    `ETIQUETAS`)."""
    return ETIQUETAS.get(columna, columna.replace("_", " "))


def frase(columna, z):
    """Fragmento de nombre de grupo para una variable con media `z`
    desviaciones estándar por encima (z > 0) o por debajo de la general."""
    signo = int(np.sign(z))
    return FRASES.get((columna, signo), f"{'más' if signo > 0 else 'menos'} {etiqueta(columna)}")


def formato_ari(valor):
    """Índice de Rand ajustado con tres decimales y sin "-0.000"."""
    return f"{round(float(valor), 3) + 0.0:.3f}"


def nombrar_grupos(perfil_z, umbral, variables):
    """Nombre, descripción y sugerencias de cada grupo.

    El nombre une las `variables` primeras variables cuya media en el grupo
    se aleja al menos `umbral` desviaciones estándar de la media general
    (ordenadas por distancia); la descripción lista todas las que superan el
    umbral y las sugerencias salen de `SUGERENCIAS`. Los nombres repetidos se
    distinguen con el número de grupo. Devuelve tres diccionarios grupo →
    (nombre, descripción, lista de sugerencias)."""
    nombres, descripciones, sugerencias = {}, {}, {}
    for grupo, fila in perfil_z.iterrows():
        destacadas = fila[fila.abs() >= umbral].reindex(fila.abs().sort_values(ascending=False).index).dropna()
        if destacadas.empty:
            nombre = "Perfil cercano al promedio"
        else:
            nombre = " y ".join(frase(c, z) for c, z in destacadas.iloc[:variables].items())
            nombre = nombre[0].upper() + nombre[1:]
        nombres[grupo] = nombre
        descripciones[grupo] = (", ".join(f"{etiqueta(c)} {z:+.1f} DE" for c, z in destacadas.items())
                                or f"ninguna variable a más de {umbral} DE de la media")
        sugerencias[grupo] = [SUGERENCIAS[(c, int(np.sign(z)))] for c, z in destacadas.items()
                              if (c, int(np.sign(z))) in SUGERENCIAS]
    repetidos = pd.Series(nombres).duplicated(keep=False)
    for grupo in repetidos[repetidos].index:
        nombres[grupo] = f"{nombres[grupo]} (grupo {grupo})"
    return nombres, descripciones, sugerencias


def tabla_segmentos(tabla, etiquetas, nombres, peso="muestras"):
    """Tamaño y desempeño de cada segmento: clientes, proporción de clientes
    y, si existe la columna `peso`, proporción de muestras y las tasas de
    `TASAS_SEGMENTO` ponderadas por ese peso. Una fila por grupo, en el orden
    de `nombres`."""
    filas = []
    for grupo, nombre in nombres.items():
        miembros = tabla[etiquetas == grupo]
        fila = {"Segmento": nombre, "Clientes": len(miembros), "Proporción de clientes": len(miembros) / len(tabla)}
        if peso in tabla.columns:
            pesos = miembros[peso]
            fila["Proporción de muestras"] = pesos.sum() / tabla[peso].sum()
            for origen, destino in TASAS_SEGMENTO:
                if origen in tabla.columns:
                    fila[destino] = float(np.average(miembros[origen], weights=pesos))
        filas.append(fila)
    return pd.DataFrame(filas)


def vista_ruido(tabla, etiquetas_db, numericas, identificadores, filas):
    """Casos que DBSCAN marca como ruido, los `filas` más apartados de la
    media, con la variable en que más se apartan. Devuelve (vista con
    columnas renombradas, conteo de casos por variable más apartada)."""
    ruido = etiquetas_db == -1
    z = desviaciones_de_filas(tabla, ruido, numericas)
    columna_extrema = z.abs().idxmax(axis=1)
    vista = tabla.loc[ruido].assign(**{"Variable más apartada": [f"{etiqueta(c)} ({z.loc[i, c]:+.1f} DE)"
                                                                 for i, c in columna_extrema.items()]})
    vista = vista.loc[z.abs().max(axis=1).sort_values(ascending=False).index[:filas],
                      [*identificadores, *numericas, "Variable más apartada"]]
    return vista.rename(columns={c: etiqueta(c) for c in numericas}), columna_extrema.map(etiqueta).value_counts()
