"""Capa 1 · Calendario laboral colombiano.

Responsabilidad única: calcular los festivos de Colombia (fijos, trasladados
al lunes por la Ley 51 de 1983 y dependientes de la Pascua) y construir la
tabla de calendario que se integra con las demás. No depende de paquetes
externos, de forma que el resultado es idéntico en cualquier entorno.
"""
from datetime import date, timedelta

import pandas as pd

_FIJOS = {
    (1, 1): "Año Nuevo",
    (5, 1): "Día del Trabajo",
    (7, 20): "Día de la Independencia",
    (8, 7): "Batalla de Boyacá",
    (12, 8): "Inmaculada Concepción",
    (12, 25): "Navidad",
}
# Se trasladan al lunes siguiente cuando no caen en lunes.
_TRASLADABLES = {
    (1, 6): "Reyes Magos",
    (3, 19): "San José",
    (6, 29): "San Pedro y San Pablo",
    (8, 15): "Asunción de la Virgen",
    (10, 12): "Día de la Raza",
    (11, 1): "Todos los Santos",
    (11, 11): "Independencia de Cartagena",
}
# Días respecto al domingo de Pascua (los tres últimos ya caen en lunes).
_PASCUA = {
    -3: "Jueves Santo",
    -2: "Viernes Santo",
    43: "Ascensión del Señor",
    64: "Corpus Christi",
    71: "Sagrado Corazón",
}


def domingo_de_pascua(anio):
    """Algoritmo gregoriano anónimo (Meeus/Jones/Butcher)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = (h + l - 7 * m + 114) % 31 + 1
    return date(anio, mes, dia)


def _lunes_siguiente(fecha):
    return fecha + timedelta(days=(7 - fecha.weekday()) % 7)


def festivos_colombia(anio):
    """Diccionario {fecha: nombre} con los festivos del año (18 reglas; son 17
    fechas cuando dos festivos trasladados coinciden, como el 30 de junio de
    2025, en cuyo caso se conserva el último nombre asignado)."""
    festivos = {date(anio, m, d): nombre for (m, d), nombre in _FIJOS.items()}
    for (m, d), nombre in _TRASLADABLES.items():
        festivos[_lunes_siguiente(date(anio, m, d))] = nombre
    pascua = domingo_de_pascua(anio)
    for desfase, nombre in _PASCUA.items():
        festivos[pascua + timedelta(days=desfase)] = nombre
    return festivos


def tabla_calendario(inicio, fin):
    """Una fila por día entre `inicio` y `fin`, con festivos y días hábiles
    (lunes a viernes no festivos)."""
    dias = pd.date_range(inicio, fin, freq="D")
    festivos = {}
    for anio in range(dias[0].year, dias[-1].year + 1):
        festivos.update(festivos_colombia(anio))
    festivos = {pd.Timestamp(f): n for f, n in festivos.items()}
    cal = pd.DataFrame({"fecha": dias})
    cal["es_festivo"] = cal["fecha"].isin(list(festivos))
    cal["nombre_festivo"] = cal["fecha"].map(festivos).fillna("")
    cal["es_habil"] = (cal["fecha"].dt.dayofweek < 5) & ~cal["es_festivo"]
    pascuas = [pd.Timestamp(domingo_de_pascua(a)) for a in range(dias[0].year, dias[-1].year + 1)]
    cal["es_semana_santa"] = False
    for pascua in pascuas:
        cal.loc[cal["fecha"].between(pascua - pd.Timedelta(days=6), pascua), "es_semana_santa"] = True
    return cal
