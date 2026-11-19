# Analítica predictiva sobre la base de muestras de un laboratorio de microbiología

**Aprendizaje Automático de Máquina I, proyecto final** (Universidad del Rosario)

**Autores:** Tania Fernanda Mora Cubillos, Over Haider Castrillón Valencia y Daniel Arias Mutis

**Exposición.** La charla recorre los seis notebooks de etapa, de `01_datos_exploracion.ipynb` a `06_evaluacion_final.ipynb`, en orden. Los notebooks 03 a 05 traen paneles interactivos (umbral de alerta, KNN, poda del árbol, red neuronal, Bayes, K-means y Taguchi) que solo se mueven con el núcleo en marcha: Antes de exponer hay que ejecutar el notebook (sin núcleo cada panel muestra la figura de su estado inicial). `Presentacion.ipynb` queda como un resumen corto de once diapositivas.

Se aplica aprendizaje automático a la base de muestras de un laboratorio de microbiología de alimentos con el objetivo de responder cuatro preguntas de la operación: cuántas muestras llegarán (regresión), cuáles están en riesgo de resultar no conformes (clasificación), qué tan probable es que un positivo sea falso (Bayes y regla de reensayo) y qué grupos de muestras y clientes existen (agrupamiento). La base del proyecto es la exportación real del laboratorio (notebook 02). Como no trae los controles de calidad ni los datos de recepción, la complementamos con una base **simulada** de la misma estructura (notebooks 01 y 03 a 06) para aplicar más técnicas del curso; la exportación real también puede conectarse a esas etapas con un mapeo de columnas, sin cambiar el código.

> **Base real y complemento simulado.** El notebook 02 usa la exportación real anonimizada (confidencial, nunca se versiona ni se entrega; ver "Notebook 02 y entrega"). Las cifras de los notebooks 01 y 03 a 06 salen de la base complementaria que genera `proyecto/datos/simulacion.py` (semilla fija).

## Estructura

```
final-project/
├── Makefile                    construcción, verificación y limpieza
├── configuracion/
│   └── mapeo_columnas.json     mapeo exportación real → nombres canónicos
├── proyecto/                   el paquete: toda la lógica, por capas
│   ├── base/                   capa 0
│   ├── datos/                  capa 1
│   ├── preparacion/            capa 2
│   ├── variables/              capa 3
│   ├── modelos/                capa 4
│   ├── evaluacion/             capa 5
│   ├── cribado/                capa 6
│   ├── reporte/                capa 7
│   └── etapas/                 capa 8: una función por sección de cada notebook
├── notebooks/
│   ├── src/                    guiones de etapa (formato percent) y build.py
│   ├── 01_datos_exploracion.ipynb … 06_evaluacion_final.ipynb
│   ├── Proyecto_Final.ipynb    las seis etapas en un solo notebook (entregable funcional)
│   └── Presentacion.ipynb      resumen corto de once diapositivas (la exposición recorre los notebooks 01 a 06)
├── figuras/                    PNG de cada etapa (se versionan; el informe las toma de aquí)
├── informe/                    informe final en LaTeX (main.tex, secciones/, make pdf)
│   └── generado/               macros (resultados.tex), índice comentado y tablas tabla_*.tex
├── verificacion/
│   └── verificar.py            comprobaciones independientes
├── datos/                      generado, no se versiona
│   ├── crudos/  limpios/  modelado/  prueba/  auditoria/
│   ├── reales/                 aquí va la exportación real
│   └── externos/               fuentes públicas descargadas a mano (opcional)
└── artefactos/                 generado, no se versiona: JSON y modelos entre etapas
```

## Arquitectura por capas

Cada subcarpeta de `proyecto/` es una capa, y **un módulo solo puede importar de su propia capa o de una de número menor**. Además, nada fuera de `proyecto/datos/` importa el generador (`simulacion`), de forma que el resto del código no distingue entre datos simulados y reales. Cada módulo empieza con `Capa N ·` (el encabezado que exige la verificación) y un párrafo `Responsabilidad única:`.

| Capa | Carpeta | Responsabilidad |
|---|---|---|
| 0 | `base` | Configuración, rutas, salida en pantalla, artefactos entre etapas y custodia de la prueba |
| 1 | `datos` | Contrato de columnas (`esquema.py`), calendario y festivos, carga, mapeo de la exportación real, simulación |
| 2 | `preparacion` | Limpieza determinista, integración de tablas y partición temporal |
| 3 | `variables` | Tablas de modelado con el momento de disponibilidad de cada variable (sin fugas) |
| 4 | `modelos` | Tuberías de scikit-learn, registro de modelos y rejillas, agrupamiento |
| 5 | `evaluacion` | Validación por bloques temporales, métricas, umbral por costo, bootstrap, Bayes |
| 6 | `cribado` | Catálogo de análisis, cribado rápido, arreglo ortogonal L16 y corridas de Taguchi |
| 7 | `reporte` | Figuras, componentes HTML y exportación de cifras y tablas a LaTeX (`latex.py`) |
| 8 | `etapas` | Una función por sección de cada notebook: calcula, guarda artefactos, registra cifras y devuelve lo que se muestra |

Los guiones de `notebooks/src/` solo narran y llaman funciones de `proyecto/etapas/` (sin `def`, celdas de pocas líneas): cada sección es `modulo.seccion(c).mostrar()`, donde `c` es el contexto de la etapa. La función devuelve un `Resultado` con las piezas que se muestran (`r.salidas`) y unas pocas cifras (`r["clave"]`), de forma que una presentación puede llamar la misma función y mostrar solo la pieza que necesite.

## Flujo de datos entre etapas

Las etapas se comunican **solo por archivos**; ninguna depende de variables en memoria de otra.

| Etapa | Lee | Escribe |
|---|---|---|
| `01_datos_exploracion` | configuración o exportación real (parte A), crudos (parte B) | `datos/crudos/*.csv`, `datos/auditoria/` (solo simulación), `datos/limpios/*.parquet`, `artefactos/01_calidad.json` |
| `02_base_real` | exportación real en `datos/reales/` (el bloque de 2026 una sola vez, con `leer_con_prueba`) | `artefactos/02_base_real.json`, `artefactos/02_congelado.json`, `artefactos/modelos/riesgo_real.joblib` |
| `03_cribado_regresion` | limpios (parte A), modelado del análisis elegido (parte B) | `datos/modelado/<análisis>.{parquet,json}`, `datos/prueba/<análisis>.parquet`, `artefactos/03_cribado.json`, `artefactos/03_regresion.json`, `artefactos/modelos/regresion.joblib` |
| `04_clasificacion_bayes` | modelado elegido, controles limpios | `artefactos/04_clasificacion.json`, `artefactos/modelos/clasificacion.joblib` |
| `05_agrupamiento` | modelado elegido | `artefactos/05_agrupamiento.json` |
| `06_evaluacion_final` | modelos congelados y la prueba (`leer_prueba`) | `artefactos/06_evaluacion.json` |

Las etapas 01 y 03 tienen dos partes y cada parte tiene su módulo en `proyecto/etapas/` (`datos` y `exploracion`, `cribado` y `regresion`). El módulo de la parte A abre la etapa con `abrir` y la deja abierta con `cerrar_parte`. El de la parte B sigue con `continuar` (mismo registro de cifras, sin encabezado nuevo) y la cierra con `cerrar`, de modo que cada etapa se abre y se cierra una sola vez y deja un solo registro de cifras.

El protocolo es **ajustar con entrenamiento, decidir con validación y evaluar una sola vez con prueba**. Los cortes están en `proyecto/base/config.py` (`CORTE_VALIDACION`, `CORTE_PRUEBA`), y la partición de prueba solo puede leerse en la etapa 06 (la de la base simulada) y en la 02 (el bloque de 2026 de la exportación real), que dejan cada lectura en `artefactos/_etapas/lecturas_prueba.log`.

## Cómo ejecutar

Desde esta carpeta, con [uv](https://docs.astral.sh/uv/) y el entorno del repositorio:

| Comando | Qué hace |
|---|---|
| `uv sync` | Instala las dependencias (en la raíz del repositorio) |
| `make notebooks` | Ejecuta las seis etapas y escribe `notebooks/*.ipynb` (unos tres minutos) |
| `make etapa N=04` | Vuelve a ejecutar solo una etapa y rehace el notebook completo |
| `make presentacion` | Rehace solo `Presentacion.ipynb` a partir de los artefactos ya calculados (segundos) |
| `cd informe && make pdf` | Compila el informe en LaTeX (requiere haber ejecutado los notebooks) |
| `make colab` | Copia los notebooks a una carpeta vacía y los ejecuta allí, sin el paquete, para probar que son autosuficientes |
| `make verificar` | Ejecuta `verificacion/verificar.py` (menos de un minuto) |
| `make manim` | Renderiza los dos videos de Manim de la presentación en `figuras/manim/` (Manim en un entorno aparte, `MANIM_VENV=...`) |
| `make limpiar` | Borra datos generados, artefactos y cachés (no toca `datos/reales` ni `datos/externos`) |

Si una etapa se abre sola y faltan los artefactos de las anteriores, `iniciar_etapa` las ejecuta primero en silencio.

**Presentación y estilo.** `make presentacion` primero escribe `artefactos/presentacion_datos.npz` (unos 260 kB: probabilidades de validación del clasificador, curvas del perceptrón entrenado época por época, una muestra de los perfiles de agrupamiento, unos días de la carga diaria y una submuestra fija de entrenamiento y otra de validación ya transformadas para los paneles interactivos, todo del desarrollo y nunca de la prueba) y comprueba que reproduce el ROC-AUC y el costo de validación de la etapa 04; luego ejecuta el guion, que intercala las diapositivas con fragmentos de código, ecuaciones en LaTeX con términos coloreados y tres animaciones de `matplotlib.animation`. Seis paneles con controles de `ipywidgets` (`proyecto/reporte/interactivo.py`: umbral de alerta, KNN, poda del árbol, red neuronal, Bayes y K-means) permiten, con el núcleo en marcha, mover un parámetro y ver cómo cambia la métrica optimizada frente al valor que eligió el proyecto; sin núcleo, cada celda conserva la figura de su estado inicial. Diapositivas, componentes HTML y figuras comparten un sistema editorial (Fraunces, IBM Plex Sans y JetBrains Mono): el HTML las toma de Google Fonts y matplotlib las busca en el sistema o las descarga una vez de `google/fonts` a `~/.cache/proyecto_fuentes/`; sin conexión, todo cae en Georgia/DejaVu sin errores.

Dos videos de Manim (`notebooks/manim/escenas.py`: Bayes término a término con el ensayo de esterilidad y la retropropagación en el MLP) leen sus cifras de los artefactos y se versionan ya renderizados en `figuras/manim/`; la presentación los incrusta en su salida y, si faltan, muestra un aviso en su lugar.
Manim no es dependencia del proyecto: `make manim MANIM_VENV=/ruta/al/entorno` los rehace con un entorno aparte (instrucciones en el encabezado de `escenas.py`), y `make notebooks` no lo necesita.

`make verificar` comprueba, entre otras cosas, la regla de capas (incluida la capa 8), que el generador sea determinista, los festivos de Colombia, la ortogonalidad del L16, las fórmulas de Bayes contra un cálculo a mano, la ausencia de fugas por momento, que solo las etapas 02 y 06 hayan leído la prueba, que el análisis elegido en el cribado sea el que reportan las etapas siguientes, que el texto de las etapas no tenga cifras escritas a mano, que los guiones no definan funciones y que las macros del informe estén completas.

## Cifras y tablas para el informe

Cada función de `proyecto/etapas/` registra las cifras que un informe citaría (`latex.registrar`) y exporta sus tablas principales (`latex.exportar_tabla`). Al cerrar cada etapa se guarda su registro en `artefactos/macros_<etapa>.json` y se rehacen, con los registros de todas las etapas (sirve también tras `make etapa N=04`):

- `informe/generado/resultados.tex`: una macro `\NombreDeCifra` por cifra (punto decimal, sin separador de miles).
- `informe/generado/indice_macros.tex`: lista comentada de todas las macros (valor, etapa y descripción) y de las tablas.
- `informe/generado/tabla_<nombre>.tex`: un `tabular` de booktabs por tabla.

El informe incluye `\input{generado/resultados}` y toma las figuras con `\graphicspath{{../figuras/}}`. Las figuras se quedan en PNG a 200 ppp: a ancho de texto dan unos 1250 píxeles, suficiente para imprimir, y varias (dispersión de miles de muestras, mapas de calor) serían archivos vectoriales pesados y lentos de compilar.

## Conectar la exportación real

1. Copiar los archivos exportados en `datos/reales/`.
2. Completar `configuracion/mapeo_columnas.json`: en `archivo`, el nombre de cada archivo; en `columnas`, el nombre real de cada columna canónica (`null` si no existe); en `valores`, la traducción de las etiquetas reales a las canónicas. El propio archivo trae las instrucciones.
3. Reconstruir con `PROYECTO_ORIGEN=real make notebooks`.

Solo las columnas **obligatorias** son imprescindibles; las demás son opcionales y los análisis que las necesitan se omiten si faltan.

| Tabla | Columnas obligatorias |
|---|---|
| `clientes` | `id_cliente`, `sector` |
| `muestreos` | `id_muestreo`, `id_cliente`, `fecha_programada` |
| `muestras` | `id_muestra`, `id_cliente`, `ensayo`, `fecha_recepcion`, `resultado` |
| `controles` | `id_control`, `fecha`, `ensayo`, `tipo_control`, `resultado_control` |

Qué revisar primero:

- **Tabla de cobertura (etapa 01):** dice, por tabla, cuántas columnas opcionales trae la exportación y cuáles faltan, de forma que se sabe de antemano qué análisis quedarán fuera del cribado.
- **Fechas de corte:** `CORTE_VALIDACION` y `CORTE_PRUEBA` en `proyecto/base/config.py` deben caer dentro del periodo exportado.
- **Controles de calidad:** sin ellos no hay sensibilidad ni especificidad, y la parte de Bayes no aplica.
- **Auditoría de limpieza (etapa 01, parte B):** lista los problemas encontrados y las filas afectadas.

Las celdas de auditoría de la simulación (verdad oculta) se saltan solas cuando los datos son reales.

## Abrir en Google Colab

Se sube a Colab `notebooks/Proyecto_Final.ipynb` (las seis etapas) o un notebook de etapa suelto, por ejemplo `notebooks/04_clasificacion_bayes.ipynb`, y se ejecuta todo. La primera sección, "Código del proyecto", es una sola celda que recrea el paquete desde un archivo .tar.xz embebido en base64 (y, en los notebooks de etapa, los guiones de las etapas previas), comprobando su SHA-256 y sin reemplazar archivos que ya existan, de forma que no hace falta clonar el repositorio; un notebook de etapa suelto reconstruye en silencio las etapas anteriores antes de empezar.

El notebook completo se probó de punta a punta, en una carpeta vacía y sin el paquete instalado, con una pila parecida a la de Colab (Python 3.12, pandas 2.2.2, NumPy 2.0.2, scikit-learn 1.6.1, imbalanced-learn 0.13.0, matplotlib 3.10.0, pyarrow 18.1.0, SciPy 1.14.1, statsmodels 0.14.4), así como con la pila local del repositorio (pandas 3, scikit-learn 1.9), y en ambas termina sin errores, con las mismas selecciones y el mismo umbral (las cifras de la etapa 04 difieren solo desde el cuarto o quinto decimal, por diferencias numéricas del optimizador entre versiones de scikit-learn).

## Notebook 02 y entrega

El notebook 02 (`02_base_real.ipynb`) trabaja con la exportación real del laboratorio, que se copia a mano en `datos/reales/muestras_*.parquet` (una fila por muestra, especificación y lectura; clientes, puntos y analistas con códigos aleatorios). Con ella revisamos si las llegadas son de Poisson (los lotes sí, las muestras no: Poisson compuesta), segmentamos clientes y clasificamos la no conformidad al ingreso con referencia, regla de la tasa histórica, regresión logística y árbol (tres configuraciones cada uno), umbral por costo, cálculo ilustrativo y una sola evaluación en 2026 tras congelar el modelo en `artefactos/02_congelado.json`. Solo las etapas 02 y 06 pueden leer la prueba y cada lectura queda en `artefactos/_etapas/lecturas_prueba.log`. Sin el archivo cada sección del notebook 02 muestra una nota y el resto del proyecto corre igual.

`make entrega` arma `entrega/TallerFinal_GrupoXX.zip` (`GRUPO=07 make entrega` cambia el número) con el informe, `Modelo_GrupoXX.ipynb` (copia de `Proyecto_Final.ipynb`), los notebooks de etapa, el código, `README.md`, `requirements.txt` y en `datos/` solo cinco filas anonimizadas y el diccionario de las 36 columnas (`proyecto/datos/muestra_publica.py`). El script falla si encuentra datos reales o algún código de cliente en el zip. Semilla 42, Python 3.12 y las versiones de `requirements.txt`.

## Limitaciones conocidas

- **Base complementaria:** las cifras de los notebooks 01 y 03 a 06 salen de la base simulada; el generador fija relaciones que los modelos luego encuentran, y con la base real (notebook 02) los resultados son más modestos.
- **Brechas entre validación y prueba:** la etapa 06 las mide y las reporta; no se corrige nada después de abrir la prueba.
- **Ensayos con pocos casos:** *Salmonella* y esterilidad tienen pocos positivos, así que su VPP y su tasa de reensayos conformes son poco precisos.
- **Agrupamiento:** las siluetas son bajas, así que los grupos son descriptivos; la silueta del cribado no es comparable con la de la etapa 05, porque cambian las filas (entrenamiento frente a todo el desarrollo) y el preprocesamiento (log1p antes de escalar), y la propia etapa 05 separa los dos efectos y dice si, con su preprocesamiento, el análisis complementario habría puntuado más alto (con los datos simulados sí, y la selección se mantiene porque la regla se fijó de antemano).
- **Intervalos bootstrap:** remuestrean filas como si fueran independientes, lo que en la serie diaria ignora la autocorrelación.
- **Reensayo como referencia:** un reensayo conforme no equivale a un falso positivo; con datos reales no existe otra verdad.
