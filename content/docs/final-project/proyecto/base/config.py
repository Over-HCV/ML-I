"""Capa 0 · Parámetros globales del proyecto.

Responsabilidad única: declarar las constantes que comparten todas las capas
(semilla, periodo, cortes temporales, orden de las etapas y paleta). No
contiene lógica ni importa ninguna otra capa del proyecto.
"""
import os

SEMILLA = 42

TITULO_PROYECTO = "Analítica predictiva sobre la base de muestras de un laboratorio de microbiología"
ROTULO_PROYECTO = "Aprendizaje Automático de Máquina I, proyecto final"
LISTA_AUTORES = ["Tania Fernanda Mora Cubillos", "Over Haider Castrillón Valencia", "Daniel Arias Mutis"]
AUTORES = ", ".join(LISTA_AUTORES[:-1]) + " y " + LISTA_AUTORES[-1]

# Origen de los datos: "simulacion" genera la base con `datos/simulacion.py`;
# "real" lee la exportación del laboratorio descrita en
# `configuracion/mapeo_columnas.json`. Puede cambiarse sin tocar código con la
# variable de entorno PROYECTO_ORIGEN.
ORIGEN_DATOS = os.environ.get("PROYECTO_ORIGEN", "simulacion")

# Periodo cubierto por la base simulada.
FECHA_INICIO = "2023-01-02"
FECHA_FIN = "2026-10-30"

# Partición temporal (cada bloque cumple un único papel):
#   entrenamiento : fecha <  CORTE_VALIDACION   → ajustar y validar en cruzado
#   validación    : [CORTE_VALIDACION, CORTE_PRUEBA) → decidir y congelar
#   prueba        : fecha >= CORTE_PRUEBA       → evaluar una sola vez
CORTE_VALIDACION = "2025-07-01"
CORTE_PRUEBA = "2026-01-01"

# Número de bloques temporales de la validación cruzada dentro de entrenamiento.
BLOQUES_VALIDACION_CRUZADA = 4

# Orden de las etapas (un notebook por etapa). La partición de prueba solo
# puede leerse desde las etapas de ETAPA_PRUEBA: la 02 abre una sola vez el
# bloque de 2026 de la exportación real (después de congelar su modelo) y la
# 06 abre la prueba de la base simulada. Se nombran por separado para no
# depender del orden de la tupla.
ETAPAS = [
    "01_datos_exploracion",
    "02_base_real",
    "03_cribado_regresion",
    "04_clasificacion_bayes",
    "05_agrupamiento",
    "06_evaluacion_final",
]
ETAPA_PRUEBA_REAL = "02_base_real"
ETAPA_PRUEBA_SIMULACION = "06_evaluacion_final"
ETAPA_PRUEBA = (ETAPA_PRUEBA_REAL, ETAPA_PRUEBA_SIMULACION)

TITULOS_ETAPAS = {
    "01_datos_exploracion": "Datos y exploración",
    "02_base_real": "Base real",
    "03_cribado_regresion": "Cribado y regresión",
    "04_clasificacion_bayes": "Clasificación y Bayes",
    "05_agrupamiento": "Agrupamiento",
    "06_evaluacion_final": "Evaluación final",
}

# Relación de costos de la clasificación: un falso negativo (muestra de riesgo
# que no se prioriza) cuesta COSTO_FN veces lo que un falso positivo (alerta
# innecesaria). Se declara antes de mirar resultados.
COSTO_FP = 1
COSTO_FN = 5

COLORES = {
    "azul": "#4C72B0",
    "rojo": "#C44E52",
    "verde": "#55A868",
    "naranja": "#DD8452",
    "morado": "#8172B3",
    "gris": "#8C8C8C",
    "tinta": "#1F2933",
    "fondo": "#F5F7FA",
}
PALETA = [COLORES[c] for c in ("azul", "naranja", "verde", "rojo", "morado", "gris")]
