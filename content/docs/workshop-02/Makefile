# Taller 2 — construcción completa en un comando.
# Uso (desde esta carpeta):  make            → teoría + notebook + PDF
#                            make teoria | notebook | pdf | limpiar

RAIZ := $(shell git rev-parse --show-toplevel)
UV   := uv run --project $(RAIZ)

.PHONY: all teoria notebook pdf limpiar

all: teoria notebook pdf

teoria:
	$(UV) python verificacion/teoria.py

notebook:
	$(UV) python notebook/src/build.py

pdf:
	cd latex && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

limpiar:
	cd latex && latexmk -c main.tex
	rm -f notebook/src/_combined_notebook.py
