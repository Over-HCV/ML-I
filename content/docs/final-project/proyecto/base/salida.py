"""Capa 0 · Salida en pantalla.

Responsabilidad única: mostrar objetos (tablas, HTML, figuras) en el
notebook. Toda la salida del proyecto pasa por `mostrar`, de forma que una
etapa pueda ejecutarse en silencio cuando otra la necesita como requisito.
"""
_SILENCIO = [False]


def en_silencio():
    return _SILENCIO[0]


def fijar_silencio(valor):
    _SILENCIO[0] = bool(valor)


def mostrar(*objetos):
    """Muestra cada objeto con `IPython.display.display` (o `print` si no hay
    IPython). No hace nada cuando la etapa se está ejecutando en silencio."""
    if en_silencio():
        return
    try:
        from IPython.display import display
    except ImportError:  # ejecución fuera de un notebook
        for objeto in objetos:
            print(objeto)
        return
    for objeto in objetos:
        display(objeto)


def decir(*partes):
    """`print` que respeta el modo silencioso."""
    if not en_silencio():
        print(*partes)
