# Taller 2 — Modelos de aprendizaje supervisado

Aprendizaje Automático de Máquina I · Maestría en Matemáticas y Ciencias
de la Computación · Universidad del Rosario
Estudiantes: Tania Fernanda Mora Cubillos · Over Haider Castrillón Valencia

Enunciado: [`taller-02.md`](taller-02.md).

## Estructura

```
workshop-02/
├── taller-02.md            Enunciado del taller
├── latex/                  Entrega formal en LaTeX (Parte I y II)
│   ├── main.tex             Ensambla el documento (sin contenido propio)
│   ├── preambulo.tex        Paquetes, colores, footer institucional
│   ├── portada.tex          Página de título
│   ├── secciones/           Un .tex por ejercicio (1 a 10)
│   ├── figuras/             Logos + PNGs que genera el notebook (ex6_* … ex10_*)
│   └── generado/            Lo escribe el notebook: resultados.tex (macros con
│                            las cifras citadas), tabla_*.tex y regla_ex9.tex
│
├── notebook/               Componente práctico-computacional (ejercicios 6-10)
│   ├── Taller_2_Notebook.ipynb   Notebook final, ya ejecutado
│   └── src/                      Fuente en formato "percent" (jupytext)
│       ├── 00_portada.py          imports, versiones, helpers, carga y partición
│       ├── 01_ejercicio6.py … 05_ejercicio10.py   un .py por ejercicio
│       └── build.py               concatena, ejecuta y escribe el .ipynb
│
└── verificacion/           Recalculo independiente de la aritmética de la Parte I
```

Los datos (`load_diabetes`, `load_breast_cancer`) vienen incluidos en
scikit-learn, así que no hay descargas externas.

## Todo en un comando

```bash
cd content/docs/workshop-02
make            # verificación de teoría → notebook (figuras, tablas, macros) → PDF
make teoria     # solo verificacion/teoria.py
make notebook   # solo build.py
make pdf        # solo latexmk
make limpiar    # auxiliares de LaTeX y el .py combinado
```

## Cómo reconstruir el notebook

Desde la raíz del repositorio (las dependencias, incluidas `jupytext` y
`nbclient`, están en `pyproject.toml`):

```bash
uv sync
uv run python content/docs/workshop-02/notebook/src/build.py
```

`build.py` puede lanzarse desde cualquier carpeta: ejecuta el notebook con
`notebook/` como directorio de trabajo. Además del `.ipynb`, la ejecución
regenera `latex/figuras/ex*.png` y todo `latex/generado/`, de modo que el
PDF cita siempre las cifras de la última corrida (no hay números copiados
a mano en las secciones 6-10). Con `--no-execute` solo arma el `.ipynb`
sin salidas.

Los `.py` de `src/` no son independientes: cada uno usa variables
definidas en los anteriores (p. ej. `pipe_multi`, `logit_sel`, `t_logit`),
por lo que solo funcionan concatenados en orden por `build.py`. La prueba
(`X*_test`, `y*_test`) solo se usa en `05_ejercicio10.py`.

## Cómo compilar el LaTeX

Requiere una distribución TeX (probado con TeX Live 2025) con `babel`
en español y `pgfplots`:

```bash
cd content/docs/workshop-02/latex
latexmk -pdf main.tex
```

Si `generado/resultados.tex` no existe (notebook sin ejecutar), el
documento compila igual pero las macros de las secciones 6-10 quedan
indefinidas: primero hay que correr `build.py`.
