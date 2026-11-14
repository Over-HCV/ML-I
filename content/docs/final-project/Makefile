# Proyecto final — construcción y verificación.
# Uso (desde esta carpeta):
#   make notebooks     ejecuta todas las etapas y escribe notebooks/*.ipynb
#   make etapa N=03    vuelve a ejecutar solo la etapa 03 (y rehace el notebook completo)
#   make colab         prueba que los notebooks construidos funcionan solos en una carpeta vacía
#   make verificar     ejecuta verificacion/verificar.py (si existe)
#   make limpiar       borra datos generados, artefactos y cachés

RAIZ := $(shell git rev-parse --show-toplevel)
UV   := PYTHONPATH=. uv run --project $(RAIZ)
N    ?=

.PHONY: notebooks etapa colab verificar limpiar

notebooks:
	$(UV) python notebooks/src/build.py

etapa:
	@test -n "$(N)" || { echo "Uso: make etapa N=03"; exit 1; }
	$(UV) python notebooks/src/build.py --solo $(N)

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

# No toca datos/reales ni datos/externos (archivos traídos a mano).
limpiar:
	rm -rf artefactos datos/crudos datos/auditoria datos/limpios datos/modelado datos/prueba
	find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
