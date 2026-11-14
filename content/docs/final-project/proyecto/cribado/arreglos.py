"""Capa 6 · Arreglos ortogonales.

Responsabilidad única: construir y verificar los diseños de experimentos del
cribado: el arreglo ortogonal L16 de Taguchi en su versión 4¹ × 2¹² (un
factor de cuatro niveles y hasta doce de dos niveles en 16 corridas) y el
factorial completo con el que se puede contrastar.

El L16 se genera a partir de cuatro columnas binarias base (a, b, c, d) cuyas
16 filas son todas sus combinaciones. Cada columna de dos niveles es la suma
módulo 2 (XOR) de un subconjunto de las base y se nombra por ese
subconjunto. La columna de cuatro niveles `A` codifica el par (a, b) y por
eso consume tres columnas: a, b y ab.
"""
import itertools

import pandas as pd

COLUMNA_CUATRO_NIVELES = "A"
COLUMNAS_DOS_NIVELES = ["c", "ac", "bc", "abc", "d", "ad", "bd", "abd", "cd", "acd", "bcd", "abcd"]
_GENERADORES_DE_A = ["a", "b", "ab"]


def arreglo_l16():
    """Arreglo L16 (4¹ × 2¹²): 16 corridas, la columna `A` con niveles 0–3 y
    doce columnas de niveles 0–1 nombradas por su generador."""
    base = pd.DataFrame(list(itertools.product([0, 1], repeat=4)), columns=list("abcd"))
    arreglo = pd.DataFrame({COLUMNA_CUATRO_NIVELES: 2 * base["a"] + base["b"]})
    for nombre in COLUMNAS_DOS_NIVELES:
        arreglo[nombre] = base[list(nombre)].sum(axis=1) % 2
    arreglo.index = pd.RangeIndex(1, len(arreglo) + 1, name="corrida")
    return arreglo


def verificar_ortogonalidad(arreglo):
    """Comprueba que el arreglo es ortogonal: para cada par de columnas,
    todas las combinaciones de niveles aparecen y lo hacen el mismo número de
    veces. Devuelve True o False."""
    columnas = list(arreglo.columns)
    if len(columnas) < 2:
        return False
    for col_a, col_b in itertools.combinations(columnas, 2):
        conteos = pd.crosstab(arreglo[col_a], arreglo[col_b]).to_numpy()
        if conteos.min() == 0 or conteos.min() != conteos.max():
            return False
    return True


def _generadores(columna):
    if columna == COLUMNA_CUATRO_NIVELES:
        return _GENERADORES_DE_A
    if columna in COLUMNAS_DOS_NIVELES:
        return [columna]
    raise ValueError(f"Columna desconocida del L16: {columna!r}")


def columnas_de_interaccion(col_a, col_b):
    """Columnas del L16 en las que cae la interacción entre dos columnas.

    La interacción de dos columnas de dos niveles está en la columna cuyo
    generador es el XOR de los suyos (por ejemplo, c × d → cd). La de `A` con
    una columna de dos niveles ocupa tres columnas (A × c → ac, bc, abc). Si
    el XOR resulta ser a, b o ab, la interacción está confundida con la
    propia columna `A` y se devuelve `["A"]`.
    """
    if col_a == col_b:
        raise ValueError("Una columna no interactúa consigo misma.")
    resultado = []
    for gen_a in _generadores(col_a):
        for gen_b in _generadores(col_b):
            producto = "".join(sorted(set(gen_a) ^ set(gen_b)))
            nombre = COLUMNA_CUATRO_NIVELES if producto in _GENERADORES_DE_A else producto
            if nombre not in resultado:
                resultado.append(nombre)
    orden = [COLUMNA_CUATRO_NIVELES] + COLUMNAS_DOS_NIVELES
    return sorted(resultado, key=orden.index)


def asignar_factores(arreglo, factores, columnas):
    """Convierte el arreglo en un diseño con las etiquetas reales.

    `factores` es {nombre del factor: lista de niveles} y `columnas` es
    {nombre del factor: columna del arreglo}. El código i de la columna se
    traduce al nivel i del factor. Se rechaza un factor cuyo número de
    niveles no coincide con el de su columna, una columna inexistente y una
    columna asignada a dos factores. Devuelve una tabla con una fila por
    corrida y una columna por factor.
    """
    faltan = [f for f in factores if f not in columnas]
    if faltan:
        raise ValueError(f"Factores sin columna asignada: {faltan}")
    usadas = [columnas[f] for f in factores]
    repetidas = sorted({c for c in usadas if usadas.count(c) > 1})
    if repetidas:
        raise ValueError(f"Columnas asignadas a más de un factor: {repetidas}")

    diseno = pd.DataFrame(index=arreglo.index)
    for factor, niveles in factores.items():
        columna = columnas[factor]
        if columna not in arreglo.columns:
            raise ValueError(f"El arreglo no tiene la columna {columna!r} (factor {factor!r}).")
        codigos = sorted(arreglo[columna].unique())
        if len(codigos) != len(niveles):
            raise ValueError(
                f"El factor {factor!r} tiene {len(niveles)} niveles y la columna "
                f"{columna!r} tiene {len(codigos)}."
            )
        traduccion = dict(zip(codigos, niveles))
        diseno[factor] = pd.Series(
            [traduccion[codigo] for codigo in arreglo[columna]], index=arreglo.index, dtype=object
        )
    return diseno


def confusiones(columnas):
    """Para cada par de factores asignados al L16, dice en qué columnas cae
    su interacción y qué factores ocupan esas columnas (lista vacía si están
    libres). Un efecto principal que comparte columna con una interacción no
    puede distinguirse de ella. `columnas` es {factor: columna del arreglo}."""
    ocupante = {columna: factor for factor, columna in columnas.items()}
    filas = []
    for factor_a, factor_b in itertools.combinations(columnas, 2):
        interaccion = columnas_de_interaccion(columnas[factor_a], columnas[factor_b])
        filas.append({
            "factor_1": factor_a, "factor_2": factor_b,
            "columnas_interaccion": interaccion,
            "confundida_con": [ocupante[c] for c in interaccion if c in ocupante],
        })
    return pd.DataFrame(filas, columns=["factor_1", "factor_2", "columnas_interaccion", "confundida_con"])


def factorial_completo(factores):
    """Diseño con todas las combinaciones de niveles de `factores`
    ({nombre: lista de niveles}), una fila por corrida."""
    nombres = list(factores)
    combinaciones = list(itertools.product(*(factores[n] for n in nombres)))
    diseno = pd.DataFrame(index=pd.RangeIndex(1, len(combinaciones) + 1, name="corrida"))
    for posicion, nombre in enumerate(nombres):
        diseno[nombre] = pd.Series(
            [combinacion[posicion] for combinacion in combinaciones], index=diseno.index, dtype=object
        )
    return diseno
