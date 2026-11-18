"""Capa 1 · Generador de la base de datos simulada del laboratorio.

Responsabilidad única: producir tablas crudas con la forma del contrato
(`esquema.py`) mientras no se disponga de la exportación real.

Es el **único** módulo que conoce las variables latentes (contaminación
real, higiene y segmento del cliente, error de cada analista). Esas
variables no se escriben en las tablas crudas: salen aparte, en la «verdad
oculta», que solo se consulta en las celdas de auditoría de la simulación.
Ninguna otra capa importa este módulo.

Decisiones de diseño para no favorecer a ningún modelo por construcción:
la señal mezcla efectos lineales, umbrales, interacciones y episodios
locales en el tiempo; hay columnas distractoras y redundantes; y las tablas
salen «sucias» (faltantes, categorías mal escritas, duplicados, valores
imposibles y errores de transcripción).
"""
import numpy as np
import pandas as pd

from proyecto.base.config import FECHA_FIN, FECHA_INICIO, SEMILLA
from proyecto.datos.calendario import tabla_calendario

# ----------------------------------------------------------------------
# Catálogos
# ----------------------------------------------------------------------
SECTORES = ["cosmético", "farmacéutico", "alimentos", "veterinario", "central de mezclas"]
P_SECTOR = [0.38, 0.22, 0.22, 0.10, 0.08]
P_SECTOR_TARDIO = [0.25, 0.15, 0.42, 0.10, 0.08]   # deriva: desde 2025 llegan más clientes de alimentos
ZONAS = ["Bogotá", "Tocancipá", "Chía", "Cota", "Mosquera"]
P_ZONA = [0.52, 0.14, 0.10, 0.12, 0.12]
DISTANCIA_ZONA = [9.0, 47.0, 28.0, 22.0, 25.0]
TAMANOS = ["micro", "pequeña", "mediana", "grande"]
PLANES = ["ninguno", "trimestral", "bimestral", "mensual"]
DIAS_PLAN = {"trimestral": 91, "bimestral": 61, "mensual": 30}

# Segmentos latentes de cliente (no se exportan).
SEGMENTOS = ["industrial regular", "pyme esporádica", "crítico urgente", "estacional"]
P_SEGMENTO = [0.20, 0.45, 0.15, 0.20]
SEG = {
    "mediana_semanal": [3.0, 0.5, 1.2, 1.0],
    "sigma": [0.6, 0.8, 0.7, 0.6],
    "lote_medio": [3.0, 1.8, 2.1, 2.6],
    "higiene_a": [7.0, 5.0, 3.0, 5.5],
    "higiene_b": [2.5, 3.0, 3.5, 3.0],
    "p_urgente": [0.06, 0.12, 0.33, 0.14],
    "p_plan": [0.65, 0.15, 0.30, 0.25],
    "p_marco": [0.75, 0.20, 0.40, 0.50],
    "retiro_anual": [0.03, 0.12, 0.15, 0.08],
    "p_tamano": [[0.02, 0.18, 0.45, 0.35], [0.40, 0.42, 0.15, 0.03],
                 [0.15, 0.40, 0.35, 0.10], [0.12, 0.38, 0.38, 0.12]],
}
# Dispersión entre clientes de un mismo segmento (para que los segmentos se
# solapen y no puedan recuperarse a la perfección con agregados por cliente).
CONCENTRACION_URGENCIA = 7.0      # parámetro a + b de la beta de la propensión a pedir urgencia
SIGMA_LOTE = 0.30                 # desviación (log) del tamaño medio de envío
# Autocorrelación semanal de la actividad de cada sector.
AR_PHI, AR_SIGMA = 0.85, 0.10
# Componente común a todo el laboratorio (coyuntura) y actividad propia de cada cliente.
AR_COMUN_PHI, AR_COMUN_SIGMA = 0.92, 0.08
AR_CLIENTE_PHI, AR_CLIENTE_SIGMA = 0.95, 0.10
TENDENCIA_ANUAL = 0.09

# Familias de producto por sector: (nombre, aw medio, pH medio, prob. de conservante)
PRODUCTOS = {
    0: [("crema", 0.96, 6.0, 0.90), ("champú", 0.97, 5.5, 0.90), ("maquillaje en polvo", 0.55, 7.0, 0.40),
        ("gel", 0.95, 6.5, 0.85), ("loción", 0.97, 5.8, 0.90), ("labial", 0.45, 7.0, 0.30)],
    1: [("jarabe", 0.88, 4.8, 0.80), ("tableta", 0.40, 7.0, 0.0), ("crema tópica", 0.95, 6.0, 0.90),
        ("solución oral", 0.97, 5.5, 0.80), ("cápsula", 0.45, 7.0, 0.0)],
    2: [("lácteo", 0.98, 6.4, 0.0), ("cárnico", 0.97, 6.0, 0.20), ("bebida", 0.98, 3.6, 0.50),
        ("panadería", 0.80, 5.8, 0.30), ("salsa", 0.93, 4.0, 0.60), ("snack", 0.35, 6.0, 0.0)],
    3: [("suplemento", 0.60, 6.5, 0.20), ("suspensión oral", 0.96, 5.5, 0.80),
        ("alimento concentrado", 0.55, 6.0, 0.10), ("ungüento", 0.90, 6.5, 0.70)],
    4: [("nutrición parenteral", 0.99, 6.0, 0.0), ("mezcla oncológica", 0.99, 6.5, 0.0),
        ("solución estéril", 0.99, 6.8, 0.0)],
}
MATERIAS_PRIMAS = [("extracto vegetal", 0.85, 5.5, 0.0), ("excipiente en polvo", 0.35, 7.0, 0.0),
                   ("aceite base", 0.30, 7.0, 0.0), ("harina", 0.60, 6.0, 0.0), ("colorante", 0.50, 7.0, 0.0)]
AGUAS = [("agua de proceso", 1.00, 7.0, 0.0), ("agua purificada", 1.00, 6.8, 0.0)]
MATRICES_ENVIO = ["producto terminado", "materia prima", "agua"]
MATRICES_PLANTA = ["superficie", "ambiente", "manipulador", "equipo"]
P_MATRIZ_PLANTA = [0.50, 0.25, 0.15, 0.10]
EFECTO_MATRIZ = {"producto terminado": 0.0, "materia prima": 0.8, "agua": -1.5,
                 "superficie": 0.4, "ambiente": -0.5, "manipulador": 0.6, "equipo": 0.2}
# Efecto del sector sobre la contaminación (las líneas con BPM farmacéuticas son más limpias).
EFECTO_SECTOR = np.array([0.0, -0.55, 0.50, 0.30, -0.70])
INTERCEPTO_CONTAMINACION = -5.25
# Combinaciones matriz/familia × ensayo con riesgo propio (interacciones: el
# microorganismo «típico» de cada tipo de muestra). Claves: índice del ensayo.
RIESGO_MATRIZ_ENSAYO = {
    ("agua", 3): 2.0, ("agua", 2): 0.9, ("manipulador", 4): 2.2, ("manipulador", 2): 1.0,
    ("ambiente", 1): 1.7, ("materia prima", 1): 1.2, ("superficie", 0): 0.6, ("equipo", 0): 0.5,
}
RIESGO_FAMILIA_ENSAYO = {
    ("lácteo", 2): 1.3, ("cárnico", 2): 1.3, ("cárnico", 5): 1.2, ("lácteo", 5): 0.6,
    ("crema", 3): 1.2, ("champú", 3): 1.2, ("gel", 3): 0.9, ("loción", 3): 0.9,
    ("maquillaje en polvo", 1): 1.0, ("panadería", 1): 1.2, ("extracto vegetal", 1): 1.0,
    ("harina", 1): 0.9, ("jarabe", 1): 0.8, ("crema tópica", 4): 0.9, ("ungüento", 4): 0.9,
}

ENSAYOS = ["mesófilos aerobios", "mohos y levaduras", "coliformes y E. coli",
           "Pseudomonas aeruginosa", "Staphylococcus aureus", "Salmonella spp.", "esterilidad"]
ES_CUANTITATIVO = np.array([True, True, True, False, False, False, False])
DIAS_INCUBACION = np.array([2, 5, 2, 3, 2, 4, 14])
LIMITE_UFC = np.array([1000.0, 100.0, 10.0, np.nan, np.nan, np.nan, np.nan])
FACTOR_INCONTABLE = 30        # por encima de 30 veces el límite la placa se reporta «>N»
CONTROLES_POR_DIA = 2         # controles positivos y negativos por ensayo y día (uno por turno)
SENSIBILIDAD = np.array([0.97, 0.95, 0.94, 0.92, 0.93, 0.90, 0.96])
ESPECIFICIDAD = np.array([0.987, 0.982, 0.987, 0.991, 0.991, 0.996, 0.980])
EFECTO_ENSAYO = np.array([0.6, 0.2, -0.2, -0.8, -0.9, -2.1, -1.0])
# Probabilidad de cada ensayo según el grupo de muestra.
P_ENSAYO = {
    "general":   [0.35, 0.25, 0.10, 0.17, 0.13, 0.00, 0.00],
    "alimentos": [0.30, 0.20, 0.30, 0.00, 0.08, 0.12, 0.00],
    "mezclas":   [0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80],
    "agua":      [0.45, 0.00, 0.35, 0.20, 0.00, 0.00, 0.00],
    "planta":    [0.45, 0.30, 0.15, 0.00, 0.10, 0.00, 0.00],
}
GRUPOS_ENSAYO = list(P_ENSAYO)

# Analistas: fecha de ingreso (los tres últimos llegan dentro del periodo y
# cometen más errores mientras aprenden) y propensión latente al falso positivo.
INGRESO_ANALISTAS = ["2015-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2021-06-01",
                     "2023-09-01", "2024-06-03", "2025-04-01"]
ERROR_ANALISTA = np.array([-0.5, -0.3, -0.1, 0.2, 0.4, 0.3, 0.1, 0.5])
INCUBADORA_DEFECTUOSA = (3, "2024-02-01", "2024-07-15")   # deriva: se reemplaza a mediados de 2024

# Estacionalidad de la demanda por sector (fila) y mes (columna).
ESTACIONALIDAD = np.array([
    [0.80, 0.95, 1.10, 1.20, 1.00, 0.90, 0.95, 1.00, 1.15, 1.25, 1.20, 0.75],   # cosmético
    [0.90, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.05, 0.85],   # farmacéutico
    [0.85, 0.90, 0.95, 1.00, 1.05, 1.15, 1.10, 0.95, 1.00, 1.10, 1.25, 1.20],   # alimentos
    [0.90, 1.00, 1.05, 1.00, 0.95, 1.00, 1.05, 1.00, 1.00, 1.00, 1.00, 0.85],   # veterinario
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.95],   # central de mezclas
])
FACTOR_DIA_SEMANA = np.array([1.15, 1.10, 1.00, 0.95, 0.80])
MESES_LLUVIA = [4, 5, 10, 11]


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def _logistica(z):
    return 1.0 / (1.0 + np.exp(-z))


def _elegir(rng, probabilidades):
    """Índice aleatorio por fila de una matriz de probabilidades [n, k]."""
    acumulada = np.cumsum(np.asarray(probabilidades, dtype=float), axis=1)
    u = rng.random(len(acumulada))[:, None]
    return np.minimum((u > acumulada).sum(axis=1), acumulada.shape[1] - 1)


def _dia(fecha):
    return np.datetime64(fecha, "D")


def _ar1(rng, k, n, phi, sigma):
    """k series AR(1) de longitud n, arrancadas en su distribución estacionaria."""
    x = np.zeros((k, n))
    x[:, 0] = rng.normal(0, sigma / np.sqrt(1 - phi ** 2), k)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + rng.normal(0, sigma, k)
    return x


def _rotulos(plantilla, numeros):
    """Arreglo de textos `plantilla % numero` (tipo objeto, admite None después)."""
    return np.array([plantilla % int(k) for k in numeros], dtype=object)


def _habil(fechas, desplazamiento, festivos):
    """Suma días hábiles (lunes a viernes no festivos)."""
    return np.busday_offset(fechas, desplazamiento, roll="forward", holidays=festivos)


# ----------------------------------------------------------------------
# Clientes y productos
# ----------------------------------------------------------------------
def _clientes(rng, n, inicio, fin):
    seg = rng.choice(4, n, p=P_SEGMENTO)
    antiguo = rng.random(n) < 0.6
    dias_periodo = int((fin - inicio).astype(int))
    vinculacion = np.where(antiguo,
                           _dia("2015-01-01") + rng.integers(0, 2900, n),
                           inicio + rng.integers(0, int(dias_periodo * 0.9), n))
    tardio = vinculacion >= _dia("2025-03-01")
    sector = np.where(tardio, rng.choice(5, n, p=P_SECTOR_TARDIO), rng.choice(5, n, p=P_SECTOR))
    zona = rng.choice(5, n, p=P_ZONA)
    tamano = _elegir(rng, np.array(SEG["p_tamano"])[seg])
    higiene = rng.beta(np.array(SEG["higiene_a"])[seg], np.array(SEG["higiene_b"])[seg])
    bpm = rng.random(n) < _logistica(7.0 * (higiene - 0.62) + 0.4 * (tamano == 3))
    marco = rng.random(n) < np.array(SEG["p_marco"])[seg]
    con_plan = rng.random(n) < np.array(SEG["p_plan"])[seg]
    frecuencia = np.where(seg == 0, 1 + _elegir(rng, np.tile([0.10, 0.30, 0.60], (n, 1))),
                          1 + _elegir(rng, np.tile([0.35, 0.40, 0.25], (n, 1))))
    plan = np.where(con_plan, frecuencia, 0)
    tasa = np.exp(rng.normal(np.log(np.array(SEG["mediana_semanal"])[seg]), np.array(SEG["sigma"])[seg]))
    tasa = tasa * np.array([0.6, 0.85, 1.15, 1.6])[tamano]
    anios_hasta_retiro = 0.5 + rng.exponential(1.0 / np.array(SEG["retiro_anual"])[seg])
    retiro = np.maximum(vinculacion, inicio) + (anios_hasta_retiro * 365).astype(int)
    retiro = np.where(retiro < fin, retiro, np.datetime64("NaT"))
    cadena_fria = rng.random(n) < np.clip(0.2 + 0.7 * higiene, 0, 1)
    mes_campana = rng.choice([3, 4, 10, 11], n)
    media_urgente = np.array(SEG["p_urgente"])[seg]
    p_urgente = rng.beta(CONCENTRACION_URGENCIA * media_urgente, CONCENTRACION_URGENCIA * (1 - media_urgente))
    lote = (np.array(SEG["lote_medio"])[seg] * np.array([0.8, 0.9, 1.1, 1.3])[tamano]
            * rng.lognormal(0, SIGMA_LOTE, n))
    campana_base, campana_pico = rng.uniform(0.30, 0.65, n), rng.uniform(1.0, 2.4, n)

    clientes = pd.DataFrame({
        "id_cliente": [f"C{i + 1:03d}" for i in range(n)],
        "sector": np.array(SECTORES)[sector],
        "zona": np.array(ZONAS)[zona],
        "tamano": np.array(TAMANOS)[tamano],
        "tipo_contrato": np.where(marco, "contrato marco", "por demanda"),
        "certificacion_bpm": np.where(bpm, "sí", "no"),
        "plan_muestreo": np.array(PLANES)[plan],
        "fecha_vinculacion": pd.to_datetime(vinculacion),
        "distancia_km": np.round(np.array(DISTANCIA_ZONA)[zona] * rng.lognormal(0, 0.25, n), 1),
        "canal_captacion": rng.choice(["referido", "página web", "feria", "llamada comercial"], n),
        "ejecutivo_cuenta": rng.choice(["Ejecutivo 1", "Ejecutivo 2", "Ejecutivo 3", "Ejecutivo 4"], n),
    })
    latentes = {
        "segmento": seg, "sector": sector, "tamano": tamano, "higiene": higiene, "tasa": tasa,
        "plan": plan, "vinculacion": vinculacion, "retiro": retiro, "cadena_fria": cadena_fria,
        "mes_campana": mes_campana, "distancia": clientes["distancia_km"].to_numpy(),
        "p_urgente": p_urgente, "lote": lote, "campana_base": campana_base, "campana_pico": campana_pico,
    }
    return clientes, latentes


def _productos(rng, lat):
    """Catálogo de referencias de cada cliente, con su efecto latente."""
    n = len(lat["segmento"])
    cuantos = np.array([rng.integers(a, b + 1) for a, b in
                        np.array([(2, 4), (3, 7), (5, 12), (8, 18)])[lat["tamano"]]])
    cliente = np.repeat(np.arange(n), cuantos)
    total = len(cliente)
    matriz = _elegir(rng, np.tile([0.62, 0.23, 0.15], (total, 1)))
    nombre, aw, ph, cons = [], np.zeros(total), np.zeros(total), np.zeros(total, dtype=bool)
    for i in range(total):
        opciones = (PRODUCTOS[int(lat["sector"][cliente[i]])], MATERIAS_PRIMAS, AGUAS)[matriz[i]]
        familia, aw_m, ph_m, p_cons = opciones[rng.integers(len(opciones))]
        nombre.append(familia)
        aw[i] = np.clip(aw_m + rng.normal(0, 0.02), 0.20, 1.0)
        ph[i] = np.clip(ph_m + rng.normal(0, 0.35), 2.5, 9.5)
        cons[i] = rng.random() < p_cons
    numero = np.concatenate([np.arange(1, c + 1) for c in cuantos])
    return {
        "cliente": cliente, "matriz": matriz, "familia": np.array(nombre), "aw": aw, "ph": ph,
        "conservante": cons, "efecto": rng.normal(0, 0.50, total),
        "codigo": np.array([f"P-C{c + 1:03d}-{k:02d}" for c, k in zip(cliente, numero)]),
        "inicio": np.concatenate([[0], np.cumsum(cuantos)[:-1]]), "cuantos": cuantos,
    }


# ----------------------------------------------------------------------
# Demanda: envíos de clientes y muestreos en planta
# ----------------------------------------------------------------------
def _factor_demanda(rng, lat, habiles, cal):
    """Matriz [cliente, día hábil] con el multiplicador de la tasa de envío."""
    fechas = pd.DatetimeIndex(habiles)
    mes = fechas.month.to_numpy()
    semana = ((habiles - habiles[0]).astype(int) // 7)
    # Fin y comienzo de mes (liberación de lotes de producción).
    periodo = fechas.to_period("M")
    orden = pd.Series(np.arange(len(fechas))).groupby(periodo).rank().to_numpy()
    tamano_mes = pd.Series(np.ones(len(fechas))).groupby(periodo).transform("sum").to_numpy()
    cierre, apertura = (tamano_mes - orden) < 5, orden <= 5
    # Día hábil siguiente a un festivo y receso de fin de año.
    festivos = cal.loc[cal["es_festivo"], "fecha"].to_numpy("datetime64[D]")
    tras_festivo = np.isin(habiles - 1, festivos) | (
        (fechas.dayofweek == 0) & np.isin(habiles - 3, festivos))
    receso = ((mes == 12) & (fechas.day >= 20)) | ((mes == 1) & (fechas.day <= 8))
    santa = cal.set_index("fecha")["es_semana_santa"].reindex(fechas).to_numpy()
    comun = (FACTOR_DIA_SEMANA[fechas.dayofweek] * np.where(tras_festivo, 1.20, 1.0)
             * np.where(receso, 0.45, 1.0) * np.where(santa, 0.60, 1.0)
             * (1 + TENDENCIA_ANUAL * (habiles - habiles[0]).astype(int) / 365.0))
    # Actividad de cada sector: proceso AR(1) semanal (da memoria a la serie).
    n_semanas = int(semana.max()) + 1
    ar = _ar1(rng, 5, n_semanas, AR_PHI, AR_SIGMA)
    ar_comun = _ar1(rng, 1, n_semanas, AR_COMUN_PHI, AR_COMUN_SIGMA)[0]
    ar_cliente = _ar1(rng, len(lat["sector"]), n_semanas, AR_CLIENTE_PHI, AR_CLIENTE_SIGMA)
    por_sector = (ESTACIONALIDAD[:, mes - 1] * np.exp(ar[:, semana] + ar_comun[semana])
                  * np.where(cierre, np.array([1.22, 1.22, 1.08, 1.08, 1.0])[:, None], 1.0)
                  * np.where(apertura, 0.92, 1.0))
    factor = por_sector[lat["sector"]] * comun * np.exp(ar_cliente[:, semana])
    # Clientes estacionales: la demanda se concentra alrededor de su campaña.
    distancia_mes = np.abs(mes[None, :] - lat["mes_campana"][:, None])
    distancia_mes = np.minimum(distancia_mes, 12 - distancia_mes)
    campana = lat["campana_base"][:, None] + lat["campana_pico"][:, None] * np.exp(-(distancia_mes / 1.3) ** 2)
    factor = np.where((lat["segmento"] == 3)[:, None], factor * campana, factor)
    # Vida del cliente: activo entre vinculación y retiro, con caída previa al retiro.
    activo = habiles[None, :] >= lat["vinculacion"][:, None]
    con_retiro = ~np.isnat(lat["retiro"])
    faltan = (lat["retiro"][:, None] - habiles[None, :]).astype("timedelta64[D]").astype(float)
    faltan = np.where(con_retiro[:, None], faltan, np.inf)
    caida = np.clip(0.3 + 0.7 * faltan / 90.0, 0.3, 1.0)
    return factor * activo * (faltan > 0) * caida


def _envios(rng, lat, prod, habiles, cal, objetivo):
    """Una fila por muestra enviada por los clientes."""
    factor = _factor_demanda(rng, lat, habiles, cal)
    tasa = (lat["tasa"] / 5.0)[:, None] * factor
    tasa = tasa * objetivo / tasa.sum()
    lote = lat["lote"]
    n_lotes = rng.poisson(tasa / lote[:, None])
    # Los clientes «industrial regular» despachan según su plan de producción:
    # el número de lotes de la semana es casi fijo (redondeo del valor esperado)
    # y solo varía el día en que llegan.
    semana = (habiles - habiles[0]).astype(int) // 7
    for c in np.nonzero(lat["segmento"] == 0)[0]:
        esperados = tasa[c] / lote[c]
        por_semana = np.bincount(semana, weights=esperados)
        lotes_semana = np.floor(por_semana + rng.random(len(por_semana))).astype(int)
        pesos = esperados / np.maximum(por_semana[semana], 1e-12)
        n_lotes[c] = 0
        for w in np.nonzero(lotes_semana)[0]:
            dias_w = np.nonzero(semana == w)[0]
            n_lotes[c, dias_w] = rng.multinomial(lotes_semana[w], pesos[dias_w] / pesos[dias_w].sum())
    cli, dia = np.nonzero(n_lotes)
    repeticiones = n_lotes[cli, dia]
    cli, dia = np.repeat(cli, repeticiones), np.repeat(dia, repeticiones)
    orden = np.argsort(dia, kind="stable")
    cli, dia = cli[orden], dia[orden]
    tamano_lote = 1 + rng.poisson(np.maximum(lote[cli] - 1, 0.05))
    n_envios = len(cli)
    producto = prod["inicio"][cli] + (rng.random(n_envios) * prod["cuantos"][cli]).astype(int)
    # Transporte, por envío.
    refrigerado = rng.random(n_envios) < np.where(lat["cadena_fria"][cli], 0.85, 0.10)
    temperatura = np.where(refrigerado, rng.normal(6, 2, n_envios), rng.normal(17.5, 3.5, n_envios))
    trasnocha = rng.random(n_envios) < np.clip(0.12 + 0.004 * lat["distancia"][cli], 0, 0.6)
    horas = np.where(trasnocha, rng.uniform(18, 44, n_envios),
                     rng.lognormal(np.log(3 + lat["distancia"][cli] / 12), 0.4, n_envios))
    urgente = rng.random(n_envios) < lat["p_urgente"][cli]
    documentos = rng.random(n_envios) > (0.03 + 0.06 * (lat["tamano"][cli] == 0))
    hora = 7 + 10 * rng.beta(2, 2.5, n_envios)

    filas = np.repeat(np.arange(n_envios), tamano_lote)
    n = len(filas)
    return pd.DataFrame({
        "cliente": cli[filas], "envio": filas, "visita": -1, "producto": producto[filas],
        "fecha_recepcion": habiles[dia[filas]], "origen": "enviada por el cliente",
        "matriz": np.array(MATRICES_ENVIO)[prod["matriz"][producto[filas]]],
        "temperatura": temperatura[filas] + rng.normal(0, 0.3, n), "horas": horas[filas],
        "urgente": urgente[filas], "documentos": documentos[filas], "hora": hora[filas],
    })


def _muestreos(rng, lat, habiles, festivos):
    """Visitas de muestreo en planta y las muestras que genera cada una."""
    inicio, fin = habiles[0], habiles[-1]
    visitas = []
    for c in np.nonzero(lat["plan"] > 0)[0]:
        paso = DIAS_PLAN[PLANES[lat["plan"][c]]]
        desde = max(lat["vinculacion"][c], inicio) + rng.integers(0, paso)
        hasta = fin if np.isnat(lat["retiro"][c]) else min(lat["retiro"][c], fin)
        dia = desde
        while dia < hasta:
            visitas.append((c, dia + rng.integers(-3, 4)))
            dia = dia + paso
    cli = np.array([v[0] for v in visitas])
    programada = _habil(np.array([v[1] for v in visitas], dtype="datetime64[D]"), 0, festivos)
    programada = np.clip(programada, inicio, fin)
    reprogramada = rng.random(len(cli)) < 0.15
    visita = _habil(programada, np.where(reprogramada, rng.integers(1, 6, len(cli)), 0), festivos)
    visita = np.minimum(visita, fin)
    puntos_previstos = 1 + rng.poisson(np.array([2.0, 3.0, 4.0, 6.0])[lat["tamano"][cli]])
    puntos = np.maximum(puntos_previstos + rng.integers(-1, 2, len(cli)), 1)
    orden = np.argsort(visita, kind="stable")
    cli, programada, visita = cli[orden], programada[orden], visita[orden]
    puntos_previstos, puntos = puntos_previstos[orden], puntos[orden]

    tabla = pd.DataFrame({
        "id_muestreo": [f"V{i + 1:04d}" for i in range(len(cli))],
        "id_cliente": [f"C{c + 1:03d}" for c in cli],
        "fecha_programada": pd.to_datetime(programada),
        "puntos_programados": puntos_previstos,
        "tecnico": rng.choice(["Técnico 1", "Técnico 2", "Técnico 3"], len(cli)),
        "fecha_visita": pd.to_datetime(visita),
        "duracion_horas": np.round(0.8 + 0.25 * puntos + rng.normal(0, 0.4, len(cli)), 1).clip(0.5),
    })
    filas = np.repeat(np.arange(len(cli)), puntos)
    n = len(filas)
    muestras = pd.DataFrame({
        "cliente": cli[filas], "envio": -1, "visita": filas, "producto": -1,
        "fecha_recepcion": visita[filas], "origen": "muestreo en planta",
        "matriz": np.array(MATRICES_PLANTA)[_elegir(rng, np.tile(P_MATRIZ_PLANTA, (n, 1)))],
        "temperatura": rng.normal(5.5, 1.5, n), "horas": rng.uniform(1, 6, n),
        "urgente": False, "documentos": True, "hora": np.clip(rng.normal(14.5, 1.5, n), 9, 17.5),
    })
    return tabla, muestras


# ----------------------------------------------------------------------
# Atributos de la muestra, proceso de laboratorio y resultado
# ----------------------------------------------------------------------
def _condiciones_laboratorio(rng, fechas, ensayo, habiles, n):
    """Analista, incubadora y lote de medio para un conjunto de montajes, con
    los indicadores latentes que degradan la especificidad."""
    ingreso = np.array(INGRESO_ANALISTAS, dtype="datetime64[D]")
    disponibles = (fechas[:, None] >= ingreso[None, :]).sum(axis=1)
    analista = (rng.random(n) * disponibles).astype(int)
    meses = (fechas - ingreso[analista]).astype(int) / 30.0
    incubadora = rng.integers(1, 5, n)
    numero, desde, hasta = INCUBADORA_DEFECTUOSA
    defectuosa = (incubadora == numero) & (fechas >= _dia(desde)) & (fechas < _dia(hasta))
    indice_lote = ((fechas - habiles[0]).astype(int) + 5 * ensayo) // 21
    return analista, meses, incubadora, defectuosa, indice_lote


def _prob_falso_positivo(ensayo, error_analista, meses, carga_alta, lote_malo, defectuosa, ambiente_alto):
    """Probabilidad de que una muestra limpia dé positivo, según las
    condiciones del montaje."""
    log_mult = (0.6 * error_analista + 0.8 * np.exp(-meses / 6.0) + 0.5 * carga_alta
                + 1.3 * lote_malo + 1.5 * defectuosa + 0.9 * ambiente_alto)
    return np.minimum((1 - ESPECIFICIDAD[ensayo]) * np.exp(log_mult), 0.5)


def simular(semilla=SEMILLA, n_clientes=220, objetivo_envios=30000):
    """Genera la base completa.

    Devuelve `(tablas, verdad)`: `tablas` es un diccionario con las tablas
    crudas del contrato (más `calendario`) y `verdad` contiene las variables
    latentes, para auditar la simulación.
    """
    rng = np.random.default_rng(semilla)
    cal = tabla_calendario(FECHA_INICIO, FECHA_FIN)
    habiles = cal.loc[cal["es_habil"], "fecha"].to_numpy("datetime64[D]")
    festivos = cal.loc[cal["es_festivo"], "fecha"].to_numpy("datetime64[D]")
    inicio, fin = habiles[0], habiles[-1]

    clientes, lat = _clientes(rng, n_clientes, inicio, fin)
    prod = _productos(rng, lat)
    muestreos, de_planta = _muestreos(rng, lat, habiles, festivos)
    enviadas = _envios(rng, lat, prod, habiles, cal, objetivo_envios)
    m = pd.concat([enviadas, de_planta], ignore_index=True)
    m = m.sort_values(["fecha_recepcion", "hora"], kind="stable").reset_index(drop=True)
    n = len(m)
    cli = m["cliente"].to_numpy()
    pr = m["producto"].to_numpy()
    con_producto = pr >= 0
    recepcion = m["fecha_recepcion"].to_numpy("datetime64[D]")
    matriz = m["matriz"].to_numpy()
    sector = lat["sector"][cli]

    # --- Atributos declarados por el cliente --------------------------------
    familia = np.where(con_producto, prod["familia"][pr], "no aplica")
    aw = np.where(con_producto, np.clip(prod["aw"][pr] + rng.normal(0, 0.012, n), 0.2, 1.0), np.nan)
    ph = np.where(con_producto, np.clip(prod["ph"][pr] + rng.normal(0, 0.2, n), 2.0, 10.0), np.nan)
    conservante = np.where(con_producto, np.where(prod["conservante"][pr], "sí", "no"), "no aplica")
    grupo = np.select(
        [~con_producto, matriz == "agua", sector == 4, sector == 2],
        [GRUPOS_ENSAYO.index("planta"), GRUPOS_ENSAYO.index("agua"),
         GRUPOS_ENSAYO.index("mezclas"), GRUPOS_ENSAYO.index("alimentos")],
        default=GRUPOS_ENSAYO.index("general"))
    ensayo = _elegir(rng, np.array([P_ENSAYO[g] for g in GRUPOS_ENSAYO])[grupo])
    temperatura = m["temperatura"].to_numpy()
    horas = np.maximum(m["horas"].to_numpy(), 0.3)
    envase_integro = rng.random(n) > 0.015
    documentos = m["documentos"].to_numpy()
    urgente = m["urgente"].to_numpy()
    hora = m["hora"].to_numpy()
    # Días completos de viaje, más uno si la muestra salió antes de la medianoche.
    toma = recepcion - (np.floor(horas / 24.0).astype(int) + (horas % 24 > hora).astype(int))

    # --- Recepción -----------------------------------------------------------
    z_rechazo = (-4.2 + 1.5 * (temperatura > 15) + 1.3 * (horas > 36) + 2.8 * ~envase_integro
                 + 1.0 * ~documentos - 2.0 * ~con_producto)
    rechazada = rng.random(n) < _logistica(z_rechazo)
    aceptada = ~rechazada

    # --- Contaminación real (latente) ---------------------------------------
    higiene = lat["higiene"][cli]
    episodio = np.zeros(n, dtype=bool)
    n_episodios = rng.poisson(0.6 + 3.0 * (1 - lat["higiene"]) ** 2)
    for c in np.nonzero(n_episodios)[0]:
        del_cliente = cli == c
        for _ in range(n_episodios[c]):
            comienzo = inicio + rng.integers(0, int((fin - inicio).astype(int)))
            episodio |= del_cliente & (toma >= comienzo) & (toma < comienzo + rng.integers(14, 46))
    mes_toma = pd.DatetimeIndex(toma).month.to_numpy()
    aw0, ph0 = np.nan_to_num(aw, nan=0.85), np.nan_to_num(ph, nan=6.5)
    tipico = np.array([RIESGO_MATRIZ_ENSAYO.get((a, e), 0.0) + RIESGO_FAMILIA_ENSAYO.get((f, e), 0.0)
                       for a, f, e in zip(matriz, familia, ensayo)])
    z = (INTERCEPTO_CONTAMINACION
         + np.array([EFECTO_MATRIZ[x] for x in matriz])
         + EFECTO_ENSAYO[ensayo]
         + EFECTO_SECTOR[sector]
         + tipico                                                              # interacción categórica
         + np.where(con_producto, 3.0, 3.6) * (0.5 - higiene)                 # lineal (latente)
         + np.where(con_producto, prod["efecto"][pr], 0.0)                     # propio de la referencia
         + 1.9 * (aw0 > 0.92) + 0.5 * np.clip((aw0 - 0.8) / 0.1, 0, 1.2)      # umbral + rampa
         + 1.1 * np.exp(-((ph0 - 6.5) / 1.2) ** 2)                             # local (pH neutro)
         - 1.8 * ((conservante == "sí") & (aw0 > 0.85))                        # interacción
         + 2.2 * ((temperatura > 12) & (horas > 18))                           # interacción
         + 0.04 * np.clip(temperatura - 8, 0, 15)                              # lineal
         + 1.8 * ~envase_integro
         + 0.40 * np.isin(mes_toma, MESES_LLUVIA)
         + 1.5 * episodio)                                                     # local en el tiempo
    p_contaminacion = _logistica(z)
    contaminada = rng.random(n) < p_contaminacion

    # --- Siembra -------------------------------------------------------------
    siembra = _habil(recepcion, (hora >= 14).astype(int), festivos)
    analista, meses, incubadora, defectuosa, indice_lote = _condiciones_laboratorio(
        rng, siembra, ensayo, habiles, n)
    lote_malo_tabla = rng.random((len(ENSAYOS), int(indice_lote.max()) + 2)) < 0.07
    lote_malo = lote_malo_tabla[ensayo, indice_lote]
    cabina = rng.integers(1, 4, n)
    dia_siembra = (siembra - inicio).astype(int)
    carga = pd.Series(dia_siembra[aceptada]).value_counts()
    carga_dia = pd.Series(dia_siembra).map(carga).fillna(0).to_numpy()
    carga_alta = carga_dia > np.quantile(carga_dia[aceptada], 0.85)
    ambiente_malo = rng.random((int(dia_siembra.max()) + 1, 4)) < 0.05
    ambiente_tabla = rng.poisson(np.where(ambiente_malo, 9.0, 1.5))
    ambiente = ambiente_tabla[dia_siembra, cabina]
    p_fp = _prob_falso_positivo(ensayo, ERROR_ANALISTA[analista], meses, carga_alta, lote_malo,
                                defectuosa, ambiente >= 6)

    # --- Lectura y resultado -------------------------------------------------
    u = rng.random(n)
    positivo = np.where(contaminada, u < SENSIBILIDAD[ensayo], u < p_fp)
    lectura = _habil(siembra + DIAS_INCUBACION[ensayo], 0, festivos)
    limite = LIMITE_UFC[ensayo]
    razon = np.select(
        [positivo & contaminada, positivo & ~contaminada, ~positivo & contaminada],
        [1.05 + rng.lognormal(1.2, 1.0, n), 1.02 + rng.lognormal(-0.6, 0.8, n), rng.uniform(0.3, 0.99, n)],
        default=np.where(rng.random(n) < 0.6, 0.0, 10 ** rng.uniform(-2, -0.1, n)))
    recuento = np.round(razon * np.nan_to_num(limite, nan=1.0))
    cuantitativo = ES_CUANTITATIVO[ensayo]
    recuento = np.where(cuantitativo & positivo, np.maximum(recuento, limite + 1), recuento)
    recuento = np.where(cuantitativo & ~positivo, np.minimum(recuento, limite), recuento)
    tope = FACTOR_INCONTABLE * np.nan_to_num(limite, nan=1.0)
    texto = np.array([str(int(r)) for r in np.nan_to_num(recuento)], dtype=object)
    texto = np.where(recuento > tope, np.array([f">{int(t)}" for t in tope], dtype=object), texto)
    texto = np.where(recuento < 10, "<10", texto)
    reportado = np.where(cuantitativo, texto, np.where(positivo, "Presencia", "Ausencia"))

    # --- Reensayo ------------------------------------------------------------
    reensayo = rng.random(n) < np.where(positivo, 0.88, 0.02)
    u2 = rng.random(n)
    positivo_2 = np.where(contaminada, u2 < SENSIBILIDAD[ensayo], u2 < 1 - ESPECIFICIDAD[ensayo])

    # --- Reporte -------------------------------------------------------------
    semana = dia_siembra // 7
    carga_semana = pd.Series(semana).map(pd.Series(semana[aceptada]).value_counts()).fillna(0).to_numpy()
    z_carga = (carga_semana - carga_semana[aceptada].mean()) / carga_semana[aceptada].std()
    cola = rng.poisson(0.4 + 0.8 * np.clip(z_carga, 0, None))
    cola = np.where(urgente, cola // 2, 1 + cola)
    reporte = _habil(lectura, cola, festivos)
    reporte = np.where(reensayo, _habil(reporte + DIAS_INCUBACION[ensayo], 1, festivos), reporte)
    prometidos = DIAS_INCUBACION[ensayo] + np.where(urgente, 3, 6)

    # --- Tabla de muestras (aún limpia) --------------------------------------
    nat = np.datetime64("NaT")
    muestras = pd.DataFrame({
        "id_muestra": [f"M{i + 1:06d}" for i in range(n)],
        "id_cliente": clientes["id_cliente"].to_numpy()[cli],
        "id_lote_envio": np.where(m["envio"] >= 0, _rotulos("L%05d", m["envio"].to_numpy() + 1), None),
        "id_muestreo": np.where(m["visita"] >= 0,
                                muestreos["id_muestreo"].to_numpy()[np.maximum(m["visita"].to_numpy(), 0)], None),
        "fecha_toma": pd.to_datetime(toma),
        "origen": m["origen"].to_numpy(),
        "matriz": matriz,
        "tipo_producto": familia,
        "codigo_producto": np.where(con_producto, prod["codigo"][pr], None),
        "ensayo": np.array(ENSAYOS)[ensayo],
        "actividad_agua": np.round(aw, 3),
        "ph": np.round(ph, 2),
        "conservante": conservante,
        "volumen_ml": np.round(rng.lognormal(np.log(120), 0.6, n)),
        "urgencia": np.where(urgente, "prioritaria", "normal"),
        "fecha_recepcion": pd.to_datetime(recepcion),
        "hora_recepcion": np.round(hora, 2),
        "temperatura_transporte_c": np.round(temperatura, 1),
        "horas_transporte": np.round(horas, 1),
        "envase_integro": np.where(envase_integro, "sí", "no"),
        "documentacion_completa": np.where(documentos, "sí", "no"),
        "recepcionista": rng.choice(["Recepción 1", "Recepción 2", "Recepción 3"], n),
        "estado_recepcion": np.where(aceptada, "aceptada", "rechazada"),
        "dias_prometidos": prometidos,
        "fecha_siembra": pd.to_datetime(np.where(aceptada, siembra, nat)),
        "analista": np.where(aceptada, _rotulos("Analista %d", analista + 1), None),
        "turno": np.where(aceptada, np.where(hora < 12, "mañana", "tarde"), None),
        "cabina": np.where(aceptada, _rotulos("Cabina %d", cabina), None),
        "incubadora": np.where(aceptada, _rotulos("Incubadora %d", incubadora), None),
        "lote_medio": np.where(aceptada, [f"LM{e + 1}-{k:03d}" for e, k in zip(ensayo, indice_lote)], None),
        "carga_dia": np.where(aceptada, carga_dia, np.nan),
        "fecha_lectura": pd.to_datetime(np.where(aceptada, lectura, nat)),
        "control_ambiental_ufc": np.where(aceptada, ambiente, np.nan),
        "recuento_reportado": np.where(aceptada, reportado, None),
        "limite_ufc": limite,
        "resultado": np.where(aceptada, np.where(positivo, "no conforme", "conforme"), None),
        "reensayo": np.where(aceptada, np.where(reensayo, "sí", "no"), None),
        "resultado_reensayo": np.where(aceptada & reensayo,
                                       np.where(positivo_2, "no conforme", "conforme"), None),
        "fecha_reporte": pd.to_datetime(np.where(aceptada, reporte, nat)),
    })

    # --- Controles de calidad diarios ---------------------------------------
    k = len(habiles) * len(ENSAYOS) * CONTROLES_POR_DIA
    fecha_c = np.repeat(habiles, len(ENSAYOS) * CONTROLES_POR_DIA)
    ensayo_c = np.tile(np.repeat(np.arange(len(ENSAYOS)), CONTROLES_POR_DIA), len(habiles))
    controles = []
    for tipo in ("positivo", "negativo"):
        analista_c, meses_c, incubadora_c, defectuosa_c, lote_c = _condiciones_laboratorio(
            rng, fecha_c, ensayo_c, habiles, k)
        dia_c = (fecha_c - inicio).astype(int)
        carga_c = pd.Series(dia_c).map(carga).fillna(0).to_numpy() > np.quantile(carga_dia[aceptada], 0.85)
        ambiente_c = ambiente_tabla[dia_c, rng.integers(1, 4, k)]
        p_fp_c = _prob_falso_positivo(ensayo_c, ERROR_ANALISTA[analista_c], meses_c, carga_c,
                                      lote_malo_tabla[ensayo_c, lote_c], defectuosa_c, ambiente_c >= 6)
        crece = rng.random(k) < (SENSIBILIDAD[ensayo_c] if tipo == "positivo" else p_fp_c)
        controles.append(pd.DataFrame({
            "fecha": pd.to_datetime(fecha_c), "ensayo": np.array(ENSAYOS)[ensayo_c], "tipo_control": tipo,
            "analista": _rotulos("Analista %d", analista_c + 1),
            "incubadora": _rotulos("Incubadora %d", incubadora_c),
            "lote_medio": [f"LM{e + 1}-{j:03d}" for e, j in zip(ensayo_c, lote_c)],
            "resultado_control": np.where(crece, "crecimiento", "sin crecimiento"),
        }))
    controles = pd.concat(controles, ignore_index=True)
    controles = controles.sort_values(["fecha", "ensayo", "tipo_control"], kind="stable").reset_index(drop=True)
    controles.insert(0, "id_control", [f"Q{i + 1:06d}" for i in range(len(controles))])

    verdad = {
        "muestras": pd.DataFrame({
            "id_muestra": muestras["id_muestra"], "contaminada_real": contaminada.astype(int),
            "prob_contaminacion": p_contaminacion, "prob_falso_positivo": p_fp,
            "positivo_sin_ruido": (positivo & aceptada).astype(int),
            "en_episodio": episodio.astype(int), "logit_contaminacion": z,
        }),
        "clientes": pd.DataFrame({
            "id_cliente": clientes["id_cliente"], "segmento_latente": np.array(SEGMENTOS)[lat["segmento"]],
            "higiene": lat["higiene"], "tasa_semanal": lat["tasa"],
            "fecha_retiro": pd.to_datetime(lat["retiro"]),
        }),
        "ensayos": pd.DataFrame({
            "ensayo": ENSAYOS, "sensibilidad": SENSIBILIDAD, "especificidad_nominal": ESPECIFICIDAD,
            "especificidad_efectiva": [1 - p_fp[aceptada & ~contaminada & (ensayo == e)].mean()
                                       for e in range(len(ENSAYOS))],
            "prevalencia_real": [contaminada[aceptada & (ensayo == e)].mean() for e in range(len(ENSAYOS))],
        }),
    }
    tablas = {
        "clientes": _ensuciar_clientes(rng, clientes),
        "muestreos": muestreos,
        "muestras": _ensuciar_muestras(rng, muestras, lat["tamano"][cli]),
        "controles": controles,
        "calendario": cal,
    }
    return tablas, verdad


# ----------------------------------------------------------------------
# Suciedad realista
# ----------------------------------------------------------------------
def _variantes(rng, valores, proporcion):
    """Reescribe una fracción de las etiquetas con mayúsculas, espacios o sin
    tildes, como ocurre al digitar a mano."""
    valores = np.array(valores, dtype=object)
    elegidas = np.nonzero((rng.random(len(valores)) < proporcion) & pd.notna(valores))[0]
    sin_tildes = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
    formas = [str.upper, str.title, lambda s: s + " ", lambda s: s.translate(sin_tildes),
              lambda s: " " + s.replace(" ", "  ")]
    for i in elegidas:
        valores[i] = formas[rng.integers(len(formas))](valores[i])
    return valores


def _ensuciar_clientes(rng, clientes):
    sucia = clientes.copy()
    sucia["zona"] = _variantes(rng, sucia["zona"], 0.06)
    sucia["sector"] = _variantes(rng, sucia["sector"], 0.05)
    sucia.loc[rng.random(len(sucia)) < 0.04, "distancia_km"] = np.nan
    return sucia


def _ensuciar_muestras(rng, muestras, tamano_cliente):
    sucia = muestras.copy()
    n = len(sucia)
    aceptada = (sucia["estado_recepcion"] == "aceptada").to_numpy()
    # Errores de transcripción del resultado (solo en la columna de texto).
    cambia = aceptada & (rng.random(n) < 0.006)
    sucia.loc[cambia, "resultado"] = np.where(sucia.loc[cambia, "resultado"] == "conforme",
                                              "no conforme", "conforme")
    # Faltantes: al azar y dependientes del tamaño del cliente.
    sucia.loc[rng.random(n) < 0.10 + 0.10 * (tamano_cliente <= 1), "actividad_agua"] = np.nan
    sucia.loc[rng.random(n) < 0.09, "ph"] = np.nan
    sucia.loc[rng.random(n) < 0.07, "temperatura_transporte_c"] = np.nan
    sucia.loc[rng.random(n) < 0.02, "horas_transporte"] = np.nan
    sucia.loc[rng.random(n) < 0.03, "volumen_ml"] = np.nan
    sucia.loc[(rng.random(n) < 0.05) & (sucia["conservante"] != "no aplica"), "conservante"] = None
    # Valores imposibles.
    sucia.loc[rng.random(n) < 0.003, "temperatura_transporte_c"] = 999.0
    negativas = rng.random(n) < 0.002
    sucia.loc[negativas, "horas_transporte"] = -sucia.loc[negativas, "horas_transporte"]
    sin_punto = (rng.random(n) < 0.004) & sucia["ph"].notna()
    sucia.loc[sin_punto, "ph"] = sucia.loc[sin_punto, "ph"] * 10
    sucia.loc[(rng.random(n) < 0.002) & sucia["actividad_agua"].notna(), "actividad_agua"] = 1.5
    # Etiquetas mal digitadas.
    for columna, proporcion in [("tipo_producto", 0.04), ("matriz", 0.03), ("ensayo", 0.02),
                                ("urgencia", 0.02), ("envase_integro", 0.02), ("resultado", 0.015),
                                ("estado_recepcion", 0.01), ("recuento_reportado", 0.01)]:
        sucia[columna] = _variantes(rng, sucia[columna], proporcion)
    # Recuentos digitados con la unidad.
    con_unidad = (rng.random(n) < 0.01) & sucia["recuento_reportado"].notna() & sucia["limite_ufc"].notna()
    sucia.loc[con_unidad, "recuento_reportado"] = sucia.loc[con_unidad, "recuento_reportado"] + " UFC"
    # Fechas de reporte digitadas con el año anterior (quedan antes de la recepción).
    anio_errado = (rng.random(n) < 0.001) & sucia["fecha_reporte"].notna()
    sucia.loc[anio_errado, "fecha_reporte"] = sucia.loc[anio_errado, "fecha_reporte"] - pd.Timedelta(days=365)
    # Filas duplicadas.
    repetidas = sucia.iloc[np.nonzero(rng.random(n) < 0.004)[0]]
    sucia = pd.concat([sucia, repetidas]).sort_values("id_muestra", kind="stable").reset_index(drop=True)
    return sucia
