# %% [markdown]
# # Etapa 08: Datos reales
#
# Las etapas 01 a 07 las hicimos sobre una base simulada porque la exportación del laboratorio no estaba lista. Al
# final del semestre Daniel Arias Mutis, que trabaja en el laboratorio, consiguió la exportación real (con los clientes
# anonimizados) y nos propuso dos cosas: Clasificar a los clientes por su volumen semanal o mensual y a los lotes por
# su tamaño. Además nos dijo que las llegadas de muestras le sonaban a un proceso de Poisson. Esta etapa cuenta qué
# pasó cuando pusimos a prueba esa idea.
#
# Dos reglas de entrada. La primera es que solo usamos fechas de desarrollo: Todo lo que ingresó desde el corte de
# prueba (es decir 2026) se queda por fuera de cada cuenta. Así no gastamos un periodo que nadie ha mirado y además es
# donde la base empieza a cambiar de comportamiento. La segunda es que el archivo es confidencial y no viaja con el
# proyecto así que si no está (por ejemplo en Colab) cada sección muestra una nota en vez de cifras.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import datos_reales

iniciar_etapa("08_datos_reales")
c = datos_reales.iniciar()
mostrar(c.encabezado)

# %% [markdown]
# ## 1. La base real
#
# La exportación no viene con una fila por muestra. Trae una fila por cada muestra, especificación y lectura así que
# una muestra con seis especificaciones y un reanálisis ocupa siete u ocho filas. Lo primero que hicimos fue volver a
# una fila por muestra y después agrupar las muestras en **lotes**. Un lote son las muestras que un mismo cliente
# ingresa el mismo día y del mismo tipo: **Producto** (producto terminado, materia prima, agua, granel o envase) o
# **planta** (superficies, ambientes y personal que el laboratorio recoge en un muestreo en la planta del cliente).

# %%
datos_reales.la_base(c).mostrar()

# %% [markdown]
# ## 2. Llegadas: Muestras o lotes
#
# ### 2.1 La idea de Poisson
#
# Un conteo de Poisson describe cuántos eventos caen en un intervalo cuando llegan sueltos, sin ponerse de acuerdo y
# con una tasa ${\color{#3B64A8}{\lambda}}$ estable. La probabilidad de ver $k$ llegadas en una semana es
#
# $$P(N = k) = \frac{e^{-{\color{#3B64A8}{\lambda}}}\,{\color{#3B64A8}{\lambda}}^{k}}{k!}$$
#
# y tiene una propiedad que se puede revisar con los datos: Su varianza es igual a su media. Por eso usamos el
# **índice de dispersión**, la varianza sobre la media de los conteos semanales de cada cliente. Si vale cerca de 1
# los conteos se parecen a una Poisson, si es mucho mayor hay **sobredispersión** (semanas vacías mezcladas con semanas
# llenas) y si es menor los envíos son más regulares de lo que daría el azar:
#
# $$I = \frac{{\color{#B8423F}{s^2}}}{{\color{#2F8A4E}{\bar x}}}, \qquad
# D = \sum_{t=1}^{n} \frac{(x_t - {\color{#2F8A4E}{\bar x}})^2}{{\color{#2F8A4E}{\bar x}}} \sim \chi^2_{n-1}$$
#
# El estadístico $D$ es la prueba de dispersión: Si los conteos son Poisson se distribuye como una ji cuadrado con
# $n - 1$ grados de libertad (una por semana menos una) y de ahí sale un valor p para cada cola.
#
# Esperábamos que las muestras por semana de cada cliente pasaran la prueba. No pasó así. Antes de descartar la idea
# de Daniel nos dimos cuenta de que el laboratorio no recibe muestras sueltas sino lotes (una caja con varias muestras
# o una visita de muestreo) así que repetimos la cuenta con los lotes. Para cada tipo usamos los clientes con
# suficientes lotes y las semanas entre su primer y su último envío, para que un cliente que llegó tarde no sume ceros
# que no son suyos. La línea de cada figura es lo que se esperaría si cada cliente fuera una Poisson con su propia tasa.

# %%
datos_reales.llegadas(c).mostrar()

# %% [markdown]
# ### 2.2 Por qué las muestras no son Poisson aunque los lotes sí
#
# Si llegan $N$ lotes en la semana y el lote $i$ trae $X_i$ muestras, el total de muestras es una **Poisson
# compuesta**:
#
# $$S = \sum_{i=1}^{{\color{#3B64A8}{N}}} {\color{#C46A2B}{X_i}}, \qquad {\color{#3B64A8}{N}} \sim
# \mathrm{Poisson}({\color{#3B64A8}{\lambda}})$$
#
# $$E[S] = {\color{#3B64A8}{\lambda}}\, E[{\color{#C46A2B}{X}}], \qquad
# \mathrm{Var}[S] = {\color{#3B64A8}{\lambda}}\, E[{\color{#C46A2B}{X}}^2], \qquad
# \frac{\mathrm{Var}[S]}{E[S]} = \frac{E[{\color{#C46A2B}{X}}^2]}{E[{\color{#C46A2B}{X}}]}$$
#
# O sea que aunque los lotes sean Poisson el índice de las muestras ya no es 1 sino el cociente entre el segundo
# momento y la media del tamaño de lote. Si todos los lotes trajeran una muestra volveríamos a la Poisson y entre más
# varíe el tamaño más grande es el índice. Esa es la cuenta que aparece en la segunda nota de arriba y explica por qué
# los clientes de planta (con lotes de tamaño muy variable) tienen índices de muestras tan altos.

# %% [markdown]
# ## 3. Tamaño de los lotes
#
# Si los lotes llegan como Poisson lo que queda por describir es cuántas muestras trae cada uno. Como un lote tiene al
# menos una muestra modelamos el tamaño como uno más un conteo y probamos tres opciones: Uno más una Poisson (lo que
# hace nuestra simulación), uno más una geométrica (cada muestra adicional llega con la misma probabilidad que la
# anterior) y uno más una binomial negativa, que agrega un parámetro $r$ para colas más largas (con $r = 1$ es la
# geométrica). Las comparamos con el AIC, que premia el ajuste y castiga cada parámetro adicional (gana el menor).

# %%
datos_reales.lotes(c).mostrar()

# %% [markdown]
# ## 4. Segmentos de clientes
#
# La otra propuesta de Daniel era clasificar a los clientes por su volumen. Usamos el mismo procedimiento de la etapa 06
# (K-means con la regla de la silueta y nombres que salen de las variables en que cada grupo más se aparta) pero ahora
# con variables que salen de las secciones anteriores: Muestras y lotes por semana, dispersión semanal, proporción de
# semanas con envío, tamaño de lote, proporción de muestreo en planta y no conformidad.

# %%
datos_reales.segmentos(c).mostrar()

# %% [markdown]
# ## 5. Riesgo de no conformidad al ingreso
#
# Para no quedarnos solo en la descripción probamos rápido si la base real deja anticipar qué muestras van a salir no
# conformes con lo que se sabe al recibirlas: Sector, origen de la muestra, forma farmacéutica, si el cuadro analítico
# es complejo, número de especificaciones, tipo y tamaño del lote y la historia del cliente. La historia solo cuenta los
# dictámenes que ya se conocían el día del ingreso para que el modelo no vea el futuro. Ajustamos en entrenamiento y
# medimos en validación con el ROC-AUC (qué tan bien ordena el riesgo) y la precisión media (qué tan limpia sale la
# lista de alertas cuando la clase positiva es rara).

# %%
datos_reales.riesgo(c).mostrar()

# %% [markdown]
# ## 6. Qué tan lejos estaba la simulación
#
# Le dimos varias vueltas a si valía la pena rehacer las etapas anteriores con este hallazgo. Antes de decidir
# comparamos pieza por pieza el mecanismo de llegada de la simulación con lo que encontramos en la base real.

# %%
datos_reales.comparacion_simulacion(c).mostrar()

# %% [markdown]
# ## 7. Cierre

# %%
datos_reales.cierre(c).mostrar()
