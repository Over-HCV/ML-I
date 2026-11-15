"""Capa 8 · Piezas comunes de las etapas.

Responsabilidad única: dar a los módulos de `proyecto/etapas/` una forma
única de devolver lo que calcula cada sección (un `Resultado` con las piezas
que se muestran y las cifras que la narración necesita), de abrir y cerrar
una etapa (estilo de figuras, registro de cifras para el informe, marca de
etapa terminada) y unos pocos formatos de texto que comparten.

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


def abrir(etapa, subtitulo):
    """Prepara una etapa ya iniciada con `iniciar_etapa`: estilo de figuras, registro de
    cifras vacío y encabezado. Devuelve el contexto, con el encabezado en `c.encabezado`."""
    if etapa_actual() != etapa:
        raise RuntimeError(f"Llame primero a iniciar_etapa({etapa!r}).")
    configurar_estilo()
    latex.iniciar_registro(etapa)
    return Contexto(etapa=etapa, encabezado=encabezado_etapa(etapa, subtitulo=subtitulo))


def cerrar(resumen=None):
    """Guarda las cifras registradas, rehace `informe/generado/resultados.tex` y deja la marca
    de etapa terminada."""
    etapa = etapa_actual()
    latex.guardar_etapa(etapa)
    latex.escribir_macros()
    cerrar_etapa(resumen)


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
