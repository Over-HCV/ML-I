# Notebooks anteriores (respaldo del 2026-11-21)

Aquí quedan los seis notebooks de etapa tal como estaban antes de reordenar el 03 y el 04 para la exposición
(commit `8dff55f` de `main`). Se conservan con sus salidas y con su archivo de código en `codigo/`, así que se
pueden abrir y leer sin ejecutar nada.

- `01` a `06`: Los notebooks construidos, con todas sus secciones numeradas como antes.
- `codigo/`: El paquete con que se construyó cada uno (el arranque de cada notebook lo busca ahí).
- `src/`: Los guiones anteriores del 03 y el 04. Los del 01, 02, 05 y 06 no cambiaron y siguen en `notebooks/src/`.

Para volver a la versión anterior basta con copiar `src/03_cribado_regresion.py` y `src/04_clasificacion_bayes.py`
a `notebooks/src/` y ejecutar `make etapa N=03` y `make etapa N=04`.
