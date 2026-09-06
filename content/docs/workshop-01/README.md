# Taller 1 — KNN y Métricas de Clasificación

Aprendizaje Automático de Máquina I · Maestría en Matemáticas y Ciencias
de la Computación · Universidad del Rosario
Estudiante: Over Haider Castrillón Valencia

## Estructura

```
taller-1-knn/
├── latex/                 Entrega formal en LaTeX (Parte I y II completas)
│   ├── main.tex            Ensambla el documento (sin contenido propio)
│   ├── preambulo.tex        Paquetes, colores, footer institucional
│   ├── portada.tex          Página de título
│   ├── secciones/           Un .tex por ejercicio (1 a 10), autocontenido
│   ├── figuras/              PNGs generados por el notebook + logos
│   └── main.pdf              Compilado, para revisar sin instalar nada
│
├── notebook/               Componente práctico-computacional (ejercicios 6-10)
│   ├── Taller_1_KNN_Notebook.ipynb   Notebook final, ya ejecutado
│   ├── cs-training.csv                Dataset (Give Me Some Credit)
│   └── src/                           Fuente en formato "percent" (jupytext),
│       ├── 00_portada.py … 05_ejercicio10.py   un .py por ejercicio
│       └── build.py                             arma + ejecuta el .ipynb
│
├── borradores/             Versiones de trabajo en Markdown (previas al LaTeX)
│   ├── Taller_1_KNN_Plantilla.md      Enunciados en blanco
│   └── Taller_1_KNN_Respuestas.md     Respuestas completas en Markdown
│
└── revision-critica/        Auditoría propia del contenido antes de entregar
    └── Revision_Critica_Taller1.md
```

## Cómo compilar el LaTeX

Requiere una distribución TeX (probado con TeX Live) y el paquete de
idioma español de `babel`:

```bash
cd latex
sudo apt install texlive-lang-spanish   # si babel no reconoce "spanish"
latexmk -pdf main.tex
```

Cada archivo en `secciones/` es independiente: se puede editar el
ejercicio que se necesite sin tocar los demás, y comentar sus `\input`
en `main.tex` para compilar solo una parte mientras se trabaja.

## Cómo reconstruir el notebook

El notebook ya viene ejecutado y listo para abrir. Si se edita algún
`.py` en `src/` y se necesita regenerar el `.ipynb`:

```bash
cd notebook
pip install jupytext nbclient nbformat ipykernel --break-system-packages
python3 -m ipykernel install --user --name python3
python3 src/build.py
```

Importante: `build.py` debe lanzarse desde la carpeta `notebook/` (no
desde `src/`), para que el directorio de trabajo coincida con el de
`cs-training.csv`. Si el CSV no está presente, el propio notebook lo
descarga automáticamente de un mirror público en GitHub.

## Pendiente

- Aplicar (si se desea) los hallazgos de menor prioridad de
  `revision-critica/` que aún no se llevaron al LaTeX.
- Revisar `borradores/` solo como referencia histórica: el contenido
  vigente y con todas las correcciones ya está en `latex/`.
