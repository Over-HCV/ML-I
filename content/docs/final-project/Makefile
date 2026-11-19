# Proyecto final — construcción y verificación.
# Uso (desde esta carpeta):
#   make notebooks     ejecuta todas las etapas y escribe notebooks/*.ipynb
#   make etapa N=03    vuelve a ejecutar solo la etapa 03 (y rehace el notebook completo)
#   make presentacion  rehace solo notebooks/Presentacion.ipynb con los artefactos actuales
#   make colab         prueba que los notebooks construidos funcionan solos en una carpeta vacía
#   make verificar     ejecuta verificacion/verificar.py (si existe)
#   make manim         renderiza los videos de Manim de la presentación (entorno aparte, MANIM_VENV=...)
#   make entrega       arma entrega/TallerFinal_GrupoXX.zip (GRUPO=07 make entrega), sin la base real
#   make limpiar       borra datos generados, artefactos y cachés

RAIZ := $(shell git rev-parse --show-toplevel)
UV   := PYTHONPATH=. uv run --project $(RAIZ)
N    ?=
# Manim no es dependencia del proyecto: vive en su propio entorno (ver notebooks/manim/escenas.py).
MANIM_VENV   ?= $(HOME)/.cache/proyecto_manim_venv
MEDIOS_MANIM ?= $(or $(TMPDIR),/tmp)/proyecto_manim

.PHONY: notebooks etapa presentacion colab verificar limpiar manim entrega

notebooks:
	$(UV) python notebooks/src/build.py

etapa:
	@test -n "$(N)" || { echo "Uso: make etapa N=03"; exit 1; }
	$(UV) python notebooks/src/build.py --solo $(N)

presentacion:
	$(UV) python notebooks/src/build.py --presentacion

# La carpeta temporal se conserva si la prueba falla, para poder revisarla.
colab:
	@carpeta="$$(mktemp -d)" && \
	$(UV) python notebooks/src/build.py --sin-construir --prueba-colab "$$carpeta/colab" && \
	rm -rf "$$carpeta"

verificar:
	@if [ -f verificacion/verificar.py ]; then \
		$(UV) python verificacion/verificar.py; \
	else \
		echo "Todavía no existe verificacion/verificar.py."; \
	fi

# Zip para el aula virtual: informe, notebooks, código, cinco filas anonimizadas y el diccionario.
entrega:
	$(UV) python notebooks/src/empaquetar.py

# No toca datos/reales ni datos/externos (archivos traídos a mano).
limpiar:
	rm -rf artefactos datos/crudos datos/auditoria datos/limpios datos/modelado datos/prueba
	find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +

# Videos de la presentación (figuras/manim/*.mp4 y su portada .png). No lo exige ningún otro objetivo.
manim:
	@test -x "$(MANIM_VENV)/bin/manim" || { echo "No hay Manim en $(MANIM_VENV): cree el entorno (notebooks/manim/escenas.py) o use make manim MANIM_VENV=/ruta"; exit 1; }
	"$(MANIM_VENV)/bin/manim" -qm --media_dir "$(MEDIOS_MANIM)" notebooks/manim/escenas.py EscenaBayes -o bayes
	"$(MANIM_VENV)/bin/manim" -qm --media_dir "$(MEDIOS_MANIM)" notebooks/manim/escenas.py EscenaRetropropagacion -o red
	"$(MANIM_VENV)/bin/python" notebooks/manim/escenas.py "$(MEDIOS_MANIM)/videos/escenas/720p30" figuras/manim
