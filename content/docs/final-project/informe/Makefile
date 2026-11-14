# Informe final — compilación.
# Uso (desde esta carpeta):
#   make pdf      compila main.pdf con latexmk (las cifras vienen de generado/)
#   make limpiar  borra los auxiliares de LaTeX (conserva main.pdf)

.PHONY: pdf limpiar

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

limpiar:
	latexmk -c main.tex
	rm -f main.bbl main.run.xml
