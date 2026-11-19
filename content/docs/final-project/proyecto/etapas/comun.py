"""Capa 8 · Piezas comunes de las etapas.

Responsabilidad única: dar a los módulos de `proyecto/etapas/` una forma
única de devolver lo que calcula cada sección (un `Resultado` con las piezas
que se muestran y las cifras que la narración necesita), de abrir y cerrar
una etapa (estilo de figuras, registro de cifras para el informe, marca de
etapa terminada) y unos pocos formatos de texto que comparten.

Una etapa puede tener dos partes, cada una en su módulo (por ejemplo la
etapa 01 con `datos` y `exploracion`). El módulo de la primera parte abre la
etapa con `abrir` y la deja abierta con `cerrar_parte`. El de la segunda toma
un contexto nuevo con `continuar` (mismo registro de cifras, sin encabezado)
y la cierra con `cerrar`. Así la etapa se abre y se cierra una sola vez.

Cada sección de una etapa es una función `seccion(c) -> Resultado`, donde
`c` es el `Contexto` de la etapa (lo que las secciones anteriores dejaron).
El guion del notebook solo llama `seccion(c).mostrar()`; una presentación
puede llamar la misma función y mostrar solo la pieza que necesite
(`r.salidas[i]`) o leer sus cifras (`r["clave"]`).
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from proyecto.base import config
from proyecto.base.artefactos import cerrar_etapa, etapa_actual
from proyecto.base.salida import mostrar
from proyecto.reporte import latex
from proyecto.reporte.figuras import configurar_estilo, guardar_figura
from proyecto.reporte.html import encabezado_etapa, nota

ES_SIMULACION = config.ORIGEN_DATOS == "simulacion"


@dataclass
class Resultado:
    """Lo que produce una sección: piezas para mostrar (HTML, tablas, figuras ya guardadas)
    y un diccionario pequeño de cifras."""
    salidas: list = field(default_factory=list)
    valores: dict = field(default_factory=dict)

    def agregar(self, *piezas):
        self.salidas.extend(p for p in piezas if p is not None)
        return self

    def figura(self, fig, nombre):
        """Guarda la figura en `figuras/<nombre>.png`, la cierra (para que el notebook no la
        dibuje dos veces) y la deja lista para mostrarse."""
        guardar_figura(fig, nombre)
        plt.close(fig)
        self.salidas.append(fig)
        return fig

    def mostrar(self):
        mostrar(*self.salidas)

    def __getitem__(self, clave):
        return self.valores[clave]


class Contexto(SimpleNamespace):
    """Estado de una etapa entre secciones (tablas, modelos y cifras ya calculadas)."""


_PARTES = {}   # etapa → resumen de su primera parte, que se suma a la marca al cerrar la etapa


def abrir(etapa, subtitulo):
    """Prepara una etapa ya iniciada con `iniciar_etapa`: estilo de figuras, registro de
    cifras vacío y encabezado. Devuelve el contexto, con el encabezado en `c.encabezado`."""
    if etapa_actual() != etapa:
        raise RuntimeError(f"Llame primero a iniciar_etapa({etapa!r}).")
    configurar_estilo()
    latex.iniciar_registro(etapa)
    _PARTES.pop(etapa, None)
    return Contexto(etapa=etapa, encabezado=encabezado_etapa(etapa, subtitulo=subtitulo))


def continuar(etapa):
    """Contexto de la segunda parte de una etapa que otro módulo abrió con `abrir` y dejó abierta
    con `cerrar_parte`: mismo registro de cifras y sin encabezado nuevo."""
    if etapa_actual() != etapa or etapa not in _PARTES:
        raise RuntimeError(f"La primera parte de {etapa!r} debe abrirse con abrir() y dejarse con cerrar_parte().")
    configurar_estilo()
    return Contexto(etapa=etapa)


def cerrar_parte(resumen=None):
    """Termina la primera parte de una etapa sin cerrarla: guarda su resumen para la marca de la etapa."""
    _PARTES[etapa_actual()] = dict(resumen or {})


def cerrar(resumen=None):
    """Guarda las cifras registradas, rehace `informe/generado/resultados.tex` y deja la marca
    de etapa terminada (con el resumen de la primera parte, si la etapa tiene dos)."""
    etapa = etapa_actual()
    previo = _PARTES.pop(etapa, {})
    repetidas = sorted(set(previo) & set(resumen or {}))
    if repetidas:
        raise ValueError(f"Claves repetidas en el resumen de las dos partes de {etapa}: {repetidas}")
    latex.guardar_etapa(etapa)
    latex.escribir_macros()
    cerrar_etapa({**previo, **(resumen or {})})


def aviso_origen(texto_simulacion, texto_real="Los datos provienen de la exportación real del laboratorio."):
    """Aviso de datos simulados (o de datos reales) al pie de una sección."""
    return nota(texto_simulacion, "simulacion") if ES_SIMULACION else nota(texto_real, "info")


def pct(valor, decimales=1):
    """Proporción escrita como porcentaje, con punto decimal."""
    return f"{100 * valor:.{decimales}f} %"


def legible(valor):
    """Etiqueta de un nivel de factor o hiperparámetro para tablas y texto."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "ninguna"
    if isinstance(valor, (bool, np.bool_)):
        return "sí" if valor else "no"
    if isinstance(valor, (float, np.floating)) and float(valor).is_integer():
        return str(int(valor))
    return str(valor)


def enumerar(partes):
    """Partes de texto separadas por comas y "y" antes de la última."""
    partes = [str(p) for p in partes]
    if len(partes) <= 1:
        return "".join(partes)
    return ", ".join(partes[:-1]) + " y " + partes[-1]


def lista_nombres(nombres):
    """Nombres entre comillas normales, separados por comas y "y"."""
    return enumerar(f'"{n}"' for n in nombres)
