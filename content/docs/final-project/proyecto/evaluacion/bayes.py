"""Capa 5 · Teorema de Bayes para resultados de ensayo.

Responsabilidad única: pasar de «el ensayo dio positivo» a «qué tan probable
es que la condición sea real». Estima la sensibilidad y la especificidad de
cada ensayo con sus controles de calidad, corrige la proporción observada de
positivos para obtener la probabilidad previa y aplica el teorema de Bayes
(valor predictivo positivo, probabilidad de falso positivo y actualización
tras varios resultados).

Todas las funciones de probabilidad están vectorizadas: aceptan escalares,
arreglos de numpy o Series de pandas (en ese caso conservan el índice).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def _resultado(valores, *entradas):
    """Devuelve `valores` con la forma de las entradas: Serie si alguna lo
    era, escalar si todas eran escalares, arreglo en otro caso."""
    for entrada in entradas:
        if isinstance(entrada, pd.Series):
            return pd.Series(np.broadcast_to(valores, entrada.shape), index=entrada.index)
    valores = np.asarray(valores, dtype=float)
    return float(valores) if valores.ndim == 0 else valores


def _numeros(*entradas):
    return [np.asarray(entrada, dtype=float) for entrada in entradas]


def intervalo_wilson(exitos, n, nivel=0.95):
    """Intervalo de Wilson para una proporción (éxitos / n). Devuelve
    (inferior, superior); NaN cuando n = 0. A diferencia del intervalo
    normal clásico, no se sale de [0, 1] ni colapsa cuando la proporción
    observada es 0 o 1, que es lo habitual en controles de calidad."""
    exitos, n = _numeros(exitos, n)
    z = norm.ppf(1 - (1 - nivel) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = exitos / n
        denominador = 1 + z**2 / n
        centro = (p + z**2 / (2 * n)) / denominador
        mitad = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominador
    # Con 0 o n éxitos el extremo correspondiente es exactamente 0 o 1 (se
    # fija para que el redondeo de máquina no lo deje del otro lado).
    inferior = np.where(exitos == 0, 0.0, np.clip(centro - mitad, 0, 1))
    superior = np.where(exitos == n, 1.0, np.clip(centro + mitad, 0, 1))
    sin_datos = n == 0
    return np.where(sin_datos, np.nan, inferior), np.where(sin_datos, np.nan, superior)


def estimar_sens_espec(controles, columna_ensayo="ensayo", columna_tipo="tipo_control",
                       columna_resultado="resultado_control", nivel=0.95):
    """Sensibilidad y especificidad de cada ensayo a partir de sus controles.

    - Sensibilidad: fracción de controles **positivos** (inoculados) en los
      que hubo crecimiento.
    - Especificidad: fracción de controles **negativos** (estériles) en los
      que no hubo crecimiento.

    Espera las etiquetas limpias `positivo`/`negativo` y
    `crecimiento`/`sin crecimiento`; los controles con otro valor o sin
    resultado se ignoran. Los intervalos son de Wilson al nivel indicado
    (95 % por defecto).

    Devuelve una fila por ensayo con `n_positivos`, `sensibilidad`,
    `sens_inf`, `sens_sup`, `n_negativos`, `especificidad`, `espec_inf`,
    `espec_sup`. Limitación: un control positivo lleva una carga conocida y
    alta, por lo que la sensibilidad frente a muestras reales con carga baja
    puede ser menor que la estimada aquí.
    """
    tipo = controles[columna_tipo].astype(str).str.strip().str.lower()
    resultado = controles[columna_resultado].astype(str).str.strip().str.lower()
    crece = resultado == "crecimiento"
    valido = resultado.isin(["crecimiento", "sin crecimiento"])
    conteos = pd.DataFrame({
        "ensayo": controles[columna_ensayo].to_numpy(),
        "n_positivos": ((tipo == "positivo") & valido).to_numpy().astype(int),
        "detectados": ((tipo == "positivo") & valido & crece).to_numpy().astype(int),
        "n_negativos": ((tipo == "negativo") & valido).to_numpy().astype(int),
        "limpios": ((tipo == "negativo") & valido & ~crece).to_numpy().astype(int),
    }).groupby("ensayo", sort=True).sum().reset_index()

    with np.errstate(divide="ignore", invalid="ignore"):
        sensibilidad = conteos["detectados"].to_numpy() / conteos["n_positivos"].to_numpy()
        especificidad = conteos["limpios"].to_numpy() / conteos["n_negativos"].to_numpy()
    sens_inf, sens_sup = intervalo_wilson(conteos["detectados"], conteos["n_positivos"], nivel)
    espec_inf, espec_sup = intervalo_wilson(conteos["limpios"], conteos["n_negativos"], nivel)
    return pd.DataFrame({
        "ensayo": conteos["ensayo"],
        "n_positivos": conteos["n_positivos"],
        "sensibilidad": sensibilidad, "sens_inf": sens_inf, "sens_sup": sens_sup,
        "n_negativos": conteos["n_negativos"],
        "especificidad": especificidad, "espec_inf": espec_inf, "espec_sup": espec_sup,
    })


def previa_corregida(p_positivo, sens, espec):
    """Probabilidad previa (prevalencia real) a partir de la proporción
    observada de positivos, con la corrección de Rogan–Gladen.

    En palabras: a la proporción de positivos observada se le resta la tasa
    de falsos positivos (1 − especificidad) y el resultado se divide por
    (sensibilidad + especificidad − 1):

        previa = (p_positivo − (1 − espec)) / (sens + espec − 1)

    El resultado se recorta a [0, 1]. Si sens + espec ≤ 1 el ensayo no
    discrimina y la corrección no está definida: se devuelve NaN.
    """
    p, s, e = _numeros(p_positivo, sens, espec)
    youden = s + e - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        previa = np.where(youden > 0, (p - (1 - e)) / np.where(youden > 0, youden, np.nan), np.nan)
    return _resultado(np.clip(previa, 0, 1), p_positivo, sens, espec)


def _actualizar(previa, verosimilitud_si, verosimilitud_no):
    """Bayes: P(condición | dato) con las verosimilitudes del dato cuando la
    condición está presente y cuando no."""
    numerador = verosimilitud_si * previa
    denominador = numerador + verosimilitud_no * (1 - previa)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominador > 0, numerador / np.where(denominador > 0, denominador, np.nan), np.nan)


def valor_predictivo_positivo(previa, sens, espec):
    """Probabilidad de que la condición sea real dado un resultado positivo.

    En palabras: de todos los positivos que se esperan, qué fracción son
    verdaderos. Verdaderos positivos = sensibilidad × previa; falsos
    positivos = (1 − especificidad) × (1 − previa):

        VPP = sens·previa / (sens·previa + (1 − espec)·(1 − previa))

    NaN cuando no se espera ningún positivo.
    """
    pi, s, e = _numeros(previa, sens, espec)
    return _resultado(_actualizar(pi, s, 1 - e), previa, sens, espec)


def prob_falso_positivo(previa, sens, espec):
    """Probabilidad de que un resultado positivo sea falso: 1 − VPP."""
    pi, s, e = _numeros(previa, sens, espec)
    return _resultado(1 - _actualizar(pi, s, 1 - e), previa, sens, espec)


def _es_positivo(resultado):
    if isinstance(resultado, str):
        texto = resultado.strip().lower()
        if texto in ("positivo", "no conforme", "crecimiento", "+"):
            return True
        if texto in ("negativo", "conforme", "sin crecimiento", "-"):
            return False
        raise ValueError(f"Resultado de ensayo no reconocido: {resultado!r}")
    return bool(resultado)


def posterior_secuencial(previa, sens, espec, resultados):
    """Probabilidad de la condición después de una lista de resultados del
    mismo ensayo (por ejemplo, ensayo y reensayo).

    En palabras: se parte de la previa y, por cada resultado, la posterior
    del paso anterior hace de previa del siguiente. Tras un positivo la
    actualización es la del valor predictivo positivo; tras un negativo,

        posterior = (1 − sens)·previa / ((1 − sens)·previa + espec·(1 − previa))

    `resultados` es una lista con un elemento por repetición: verdadero / 1 /
    "positivo" o falso / 0 / "negativo". Supone que las repeticiones son
    independientes dada la condición real (un error sistemático que se
    repite en el reensayo rompe ese supuesto). Con un único positivo
    coincide con `valor_predictivo_positivo`.
    """
    pi, s, e = _numeros(previa, sens, espec)
    posterior = pi
    for resultado in resultados:
        if _es_positivo(resultado):
            posterior = _actualizar(posterior, s, 1 - e)
        else:
            posterior = _actualizar(posterior, 1 - s, e)
    return _resultado(posterior, previa, sens, espec)
