"""Capa 7 · Cifras y tablas para el informe en LaTeX.

Responsabilidad única: llevar al informe las cifras y tablas que calculan las
etapas, sin copiarlas a mano. Cada etapa registra sus cifras con `registrar`
(una macro `\\Nombre` por cifra) y sus tablas con `exportar_tabla` (un
`tabular` de booktabs por tabla). Al cerrar la etapa, `guardar_etapa` deja el
registro en `artefactos/macros_<etapa>.json` y `escribir_macros` arma, con
los registros de todas las etapas, `informe/generado/resultados.tex` (las
macros) e `informe/generado/indice_macros.tex` (la lista comentada de macros
y tablas, para que quien redacta el informe las encuentre).

Convenciones: nombres de macro solo con letras, punto decimal, sin separador
de miles y leyendas con «Tabla». Las figuras no pasan por aquí: viven en
`figuras/` (PNG a 200 ppp) y el informe las toma con
`\\graphicspath{{../figuras/}}`.
"""
import json
import math
import re

import numpy as np
import pandas as pd

from proyecto.base import artefactos, rutas

_MACROS = {}   # nombre → {"valor": texto LaTeX, "descripcion": str, "etapa": str}
_TABLAS = {}   # nombre → {"descripcion": str, "etapa": str, "filas": int}

_REEMPLAZOS = [
    ("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("$", r"\$"),
    ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ("²", r"\textsuperscript{2}"), ("±", r"$\pm$"), ("≤", r"$\le$"), ("≥", r"$\ge$"), ("Δ", r"$\Delta$"),
    ("−", r"$-$"), ("×", r"$\times$"), ("π", r"$\pi$"), ("→", r"$\to$"), ("·", r"$\cdot$"),
    ("≈", r"$\approx$"), ("…", r"\ldots{}"), ("✓", r"$\checkmark$"), ("ŷ", r"$\hat{y}$"),
]


def _etapa():
    return artefactos.etapa_actual() or "sin_etapa"


def escapar(texto):
    """Texto plano → texto seguro para LaTeX (caracteres especiales y símbolos)."""
    texto = str(texto)
    # La barra invertida va primero y con marcador, para no escapar lo que agregan los demás reemplazos.
    texto = texto.replace("\\", "\0")
    for original, nuevo in _REEMPLAZOS[1:]:
        texto = texto.replace(original, nuevo)
    return texto.replace("\0", _REEMPLAZOS[0][1])


def formatear(valor, decimales=4):
    """Una cifra como texto LaTeX: punto decimal, sin separador de miles y signo menos tipográfico."""
    if valor is None:
        return "---"
    if isinstance(valor, (bool, np.bool_)):
        return "sí" if valor else "no"
    if isinstance(valor, (int, np.integer)):
        texto = str(int(valor))
    elif isinstance(valor, (float, np.floating)):
        if math.isnan(float(valor)):
            return "---"
        texto = f"{float(valor):.{decimales}f}"
        if float(texto) == 0:
            texto = texto.lstrip("-")   # sin «−0.000»
    elif isinstance(valor, pd.Timestamp):
        return f"{valor:%Y-%m-%d}"
    else:
        return escapar(valor)
    return "$-$" + texto[1:] if texto.startswith("-") else texto


def iniciar_registro(etapa):
    """Olvida lo registrado antes por la etapa (al volver a ejecutarla)."""
    for registro in (_MACROS, _TABLAS):
        for nombre in [n for n, d in registro.items() if d["etapa"] == etapa]:
            del registro[nombre]


def registrar(nombre, valor, descripcion, decimales=4):
    """Registra una cifra como macro `\\<nombre>` y devuelve el valor sin tocarlo.
    Uso: `registrar("RegRMSEValidacion", rmse, "RMSE del modelo final en validación", 2)`."""
    if not re.fullmatch(r"[A-Za-z]+", nombre):
        raise ValueError(f"Nombre de macro inválido para LaTeX (solo letras): {nombre!r}")
    etapa = _etapa()
    previo = _MACROS.get(nombre)
    if previo is not None and previo["etapa"] != etapa:
        raise ValueError(f"La macro {nombre} ya la registró la etapa {previo['etapa']}")
    _MACROS[nombre] = {"valor": formatear(valor, decimales), "descripcion": descripcion, "etapa": etapa}
    return valor


def registrar_varias(cifras, decimales=4):
    """`cifras`: lista de (nombre, valor, descripción[, decimales])."""
    for nombre, valor, descripcion, *resto in cifras:
        registrar(nombre, valor, descripcion, resto[0] if resto else decimales)


def _celda(valor, decimales):
    if isinstance(valor, str):
        return escapar(valor)
    return formatear(valor, decimales)


def exportar_tabla(tabla, nombre, descripcion, decimales=4, indice=False, alineacion=None):
    """Escribe `informe/generado/tabla_<nombre>.tex` con un `tabular` de booktabs.
    `decimales` es un entero o un diccionario `{columna: decimales}` (por defecto 4).
    Devuelve la ruta del archivo."""
    if not re.fullmatch(r"[a-z0-9_]+", nombre):
        raise ValueError(f"Nombre de tabla inválido: {nombre!r} (minúsculas, cifras y «_»)")
    df = tabla.reset_index() if indice else tabla.copy()
    por_columna = decimales if isinstance(decimales, dict) else {}
    defecto = 4 if isinstance(decimales, dict) else decimales

    def es_numerica(columna):
        return pd.api.types.is_numeric_dtype(df[columna]) and not pd.api.types.is_bool_dtype(df[columna])

    alineacion = alineacion or "".join("r" if es_numerica(c) else "l" for c in df.columns)
    encabezado = " & ".join(escapar(c) for c in df.columns) + r" \\"
    filas = [" & ".join(_celda(v, por_columna.get(c, defecto)) for c, v in zip(df.columns, fila)) + r" \\"
             for fila in df.itertuples(index=False)]
    cuerpo = "\n".join([f"\\begin{{tabular}}{{{alineacion}}}", r"\toprule", encabezado, r"\midrule",
                        *filas, r"\bottomrule", r"\end{tabular}"])
    rutas.DIR_INFORME_GENERADO.mkdir(parents=True, exist_ok=True)
    ruta = rutas.DIR_INFORME_GENERADO / f"tabla_{nombre}.tex"
    ruta.write_text(f"% Generado por la etapa {_etapa()} (no editar a mano). {descripcion}\n{cuerpo}\n",
                    encoding="utf-8")
    _TABLAS[nombre] = {"descripcion": descripcion, "etapa": _etapa(), "filas": int(len(df))}
    return ruta


def guardar_etapa(etapa):
    """Deja en `artefactos/macros_<etapa>.json` lo que registró la etapa."""
    registro = {
        "macros": {n: d for n, d in sorted(_MACROS.items()) if d["etapa"] == etapa},
        "tablas": {n: d for n, d in sorted(_TABLAS.items()) if d["etapa"] == etapa},
    }
    return artefactos.guardar_json(registro, rutas.DIR_ARTEFACTOS / f"macros_{etapa}.json")


def escribir_macros():
    """Arma `resultados.tex` e `indice_macros.tex` con los registros guardados de todas las
    etapas (sirve también tras reconstruir una sola). Devuelve (macros, tablas)."""
    macros, tablas = {}, {}
    for ruta in sorted(rutas.DIR_ARTEFACTOS.glob("macros_*.json")):
        registro = artefactos.leer_json(ruta)
        for nombre, datos in registro["macros"].items():
            if nombre in macros and macros[nombre]["etapa"] != datos["etapa"]:
                raise ValueError(f"Macro {nombre} repetida en {macros[nombre]['etapa']} y {datos['etapa']}")
            macros[nombre] = datos
        tablas.update(registro["tablas"])
    etapas = sorted({d["etapa"] for d in [*macros.values(), *tablas.values()]})
    aviso = "% Generado por la construcción de los notebooks (no editar a mano)."
    lineas = [aviso, "% Cifras citadas en el informe; el índice comentado está en indice_macros.tex.", ""]
    indice = [aviso, "% Índice de macros y tablas por etapa. \\input{generado/resultados} define las macros;",
              "% cada tabla se incluye con \\input{generado/tabla_<nombre>}.", "%"]
    for etapa in etapas:
        propias = sorted((n, d) for n, d in macros.items() if d["etapa"] == etapa)
        lineas.append(f"% --- {etapa} ---")
        lineas += [f"\\newcommand{{\\{n}}}{{{d['valor']}}}" for n, d in propias]
        lineas.append("")
        indice.append(f"% ===== {etapa}: {len(propias)} macros =====")
        ancho = max((len(n) for n, _ in propias), default=0) + 1
        indice += [f"% \\{n.ljust(ancho)} = {d['valor']:<12} {d['descripcion']}" for n, d in propias]
        propias_t = sorted((n, d) for n, d in tablas.items() if d["etapa"] == etapa)
        if propias_t:
            indice.append(f"% --- tablas de {etapa} ---")
            indice += [f"% tabla_{n}.tex ({d['filas']} filas): {d['descripcion']}" for n, d in propias_t]
        indice.append("%")
    rutas.DIR_INFORME_GENERADO.mkdir(parents=True, exist_ok=True)
    (rutas.DIR_INFORME_GENERADO / "resultados.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    (rutas.DIR_INFORME_GENERADO / "indice_macros.tex").write_text("\n".join(indice) + "\n", encoding="utf-8")
    return len(macros), len(tablas)


def cifras_registradas():
    """Copia del registro en memoria (para revisar desde un notebook)."""
    return json.loads(json.dumps(_MACROS))
