# %% [markdown]
# # Etapa 02: La base real del laboratorio
#
# Esta etapa trabaja la base real del proyecto: La exportación del laboratorio que consiguió Daniel Arias Mutis (con
# los clientes anonimizados). Los notebooks 01 y 03 a 06 la complementan con una base simulada de la misma estructura
# para aplicar más técnicas del curso y aquí trabajamos con los datos reales. Daniel trabaja en el laboratorio y nos
# propuso dos cosas: Clasificar a los clientes por su volumen semanal o mensual y a los lotes por su tamaño. Además
# nos dijo que las llegadas de muestras le sonaban a un proceso de Poisson. Esta etapa cuenta qué pasó cuando pusimos
# a prueba esa idea.
#
# Dos reglas de entrada. La primera es que usamos solo fechas de desarrollo: Todo lo que ingresó desde el corte de
# prueba (es decir 2026) se queda por fuera de cada cuenta hasta la sección 7, donde lo abrimos una sola vez para
# evaluar el clasificador ya congelado. Así no gastamos antes de tiempo un periodo que nadie ha mirado. La segunda es
# que el archivo es confidencial y no viaja con el proyecto así que si no está (por ejemplo en Colab) cada sección
# muestra una nota en vez de cifras.

# %%
from proyecto.base.artefactos import iniciar_etapa
from proyecto.base.salida import mostrar
from proyecto.etapas import datos_reales

iniciar_etapa("02_base_real")
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
# La otra propuesta de Daniel era clasificar a los clientes por su volumen. Usamos el mismo procedimiento de la etapa 05
# (K-means con la regla de la silueta y nombres que salen de las variables en que cada grupo más se aparta) pero ahora
# con variables que salen de las secciones anteriores: Muestras y lotes por semana, dispersión semanal, proporción de
# semanas con envío, tamaño de lote, proporción de muestreo en planta y no conformidad.

# %%
datos_reales.segmentos(c).mostrar()

# %% [markdown]
# ## 5. Riesgo de no conformidad al ingreso
#
# Para no quedarnos solo en la descripción nos preguntamos si la base real deja anticipar qué muestras van a salir no
# conformes con lo que se sabe en el momento de recibirlas. Si se pudiera, recepción podría priorizar esas muestras
# (revisarlas primero o avisarle al cliente). La **clase 1** es la muestra con dictamen "no cumple" y la clase 0 la
# que cumple.
#
# ### 5.1 Protocolo y particiones
#
# Antes de comparar modelos fijamos las reglas. Los predictores son los que existen al registrar la muestra: Sector,
# origen de la muestra, forma farmacéutica, si el cuadro analítico es complejo, número de especificaciones (lo fija el
# cuadro que se asigna al recibirla), tipo de lote, cuántas muestras del mismo lote ya se habían registrado antes y la
# historia del cliente. Le dimos varias vueltas a dos fugas. La primera era el tamaño del lote: Antes usábamos el total
# de muestras del lote en el día, que solo se sabe cuando el día se cierra. Ahora contamos solo las muestras
# del mismo cliente, día y tipo registradas antes que esta. La segunda era la historia del cliente: Solo cuenta los
# dictámenes cuya última lectura es anterior al ingreso (y nunca antes del ingreso de esa muestra porque hay lecturas
# con fecha mal digitada).
#
# La guía del curso propone como referencia 60 % de entrenamiento, 20 % de validación y 20 % de prueba. Nosotros
# mantuvimos los mismos cortes por fecha de todo el proyecto (entrenamiento antes de julio de 2025, validación hasta
# diciembre de 2025 y prueba en 2026) porque así el modelo siempre se mide en un periodo posterior al que vio, que es
# como se usaría en recepción. La guía permite proporciones distintas si se justifican. Toda la preparación
# (imputación, escalado y codificación) vive dentro de una tubería que se ajusta solo con entrenamiento.

# %%
datos_reales.riesgo_protocolo(c).mostrar()

# %% [markdown]
# ### 5.2 Modelos y configuraciones
#
# Comparamos cuatro cosas. La **referencia** le da a todas las muestras la probabilidad de entrenamiento y no ordena
# nada. La **regla simple** usa como probabilidad la tasa histórica de no conformidad del cliente (encogida hacia la
# tasa general cuando tiene poca historia). La **regresión logística** la probamos con tres valores de C, el inverso de
# la regularización (con C chico los coeficientes se encogen más). El **árbol de decisión** lo probamos con tres
# profundidades.
#
# Como la clase 1 es rara la métrica principal es el **PR-AUC** (el área bajo la curva de precisión y recall), que
# dice qué tan limpia sale la lista de alertas cuando se ordena por riesgo. Si el modelo ordenara al azar el PR-AUC
# sería la prevalencia así que esa es la vara. Al lado reportamos el ROC-AUC (con la probabilidad de la clase 1) y la
# precisión, el recall y el F1 con el umbral de menor costo en validación de cada configuración. La elegida es la de
# mayor PR-AUC en validación.

# %%
datos_reales.riesgo_modelos(c).mostrar()

# %% [markdown]
# ### 5.3 Umbral por costo
#
# El modelo entrega una probabilidad y para decidir hay que cortarla en un umbral. No tenemos un costo real de cada
# error así que lo declaramos como un **supuesto académico**, el mismo de la etapa 04: Un falso negativo (una muestra no
# conforme que no se prioriza) cuesta lo mismo que cinco falsas alarmas. Recorrimos los umbrales en validación, nos
# quedamos con el de menor costo y miramos la matriz de confusión con las etiquetas 0 y 1.

# %%
datos_reales.riesgo_umbral(c).mostrar()

# %% [markdown]
# ### 5.4 Cálculo ilustrativo
#
# Para ver qué hace cada modelo por dentro tomamos la muestra no conforme de validación a la que la regresión
# logística le dio más riesgo y rehicimos su probabilidad a mano. El modelo suma un intercepto y cada coeficiente por el
# valor de la variable y después pasa ese logit por la sigmoide:
#
# $$z = {\color{#3B64A8}{\beta_0}} + \sum_j {\color{#C46A2B}{\beta_j}}\, x_j, \qquad p = \frac{1}{1 + e^{-z}}$$
#
# En el árbol miramos el primer corte. El índice de Gini de un nodo con una fracción $p$ de no conformes es
# $G = 1 - p^2 - (1 - p)^2 = 2p(1 - p)$ y el árbol elige el corte que más baja el promedio de los Gini de los dos hijos,
# pesado por cuántas muestras cae en cada uno.

# %%
datos_reales.riesgo_calculo(c).mostrar()

# %% [markdown]
# ### 5.5 Qué dicen los coeficientes y las reglas
#
# La razón de odds es $e^{\beta}$: Cuánto se multiplican las odds de no conformidad cuando la variable sube en una
# unidad (en las numéricas, una desviación estándar) y lo demás se queda igual. En el árbol cada hoja es una regla
# que se puede leer en voz alta.

# %%
datos_reales.riesgo_interpretacion(c).mostrar()

# %% [markdown]
# ## 6. Qué tan lejos estaba la simulación
#
# Le dimos varias vueltas a si valía la pena rehacer la base simulada con este hallazgo. Antes de decidir
# comparamos pieza por pieza el mecanismo de llegada de la simulación con lo que encontramos en la base real.

# %%
datos_reales.comparacion_simulacion(c).mostrar()

# %% [markdown]
# ## 7. La prueba de 2026
#
# Con el modelo, el umbral y los predictores ya decididos en validación guardamos la decisión completa (con una huella
# del modelo para comprobar después que no cambió) y solo entonces abrimos el bloque de 2026. Lo leemos una sola vez y
# la lectura queda en el mismo registro de custodia que usa la etapa 06. Ya sabíamos que en 2026 la base cambia
# (llegaron clientes nuevos y la no conformidad bajó) así que lo que nos interesa es qué tanto aguanta el modelo ese
# cambio y si le gana a la prevalencia y a la regla simple.

# %%
datos_reales.prueba_real(c).mostrar()

# %% [markdown]
# ## 8. Cierre

# %%
datos_reales.cierre(c).mostrar()
