Aprendizaje Automático 1 2026-2

# Taller final de Machine Learning I

Solución de un problema del entorno laboral

Aprendizaje Automático 1 · Semestre 2026-2 Profesor: Daniel Rambaut Modalidad: equipos de 1 a 3 estudiantes Fecha límite y espacio de entrega: 22 de Noviembre

## 1 Propósito del taller

Cada equipo seleccionará un problema real del trabajo de uno de sus integrantes y desarrollará una solución basada en aprendizaje supervisado. El proyecto debe conectar una necesidad del negocio con datos, modelos, evaluación y una recomendación sustentada. El resultado esperado es un prototipo reproducible y una propuesta de uso; no se exige un sistema en producción.

## 2 Resultados de aprendizaje

Formular un problema laboral como una tarea de regresión o clasificación y definir qué decisión apoyará la predicción.

Preparar datos, evitar fuga de información y comparar modelos mediante un protocolo de validación coherente.

Explicar los fundamentos de los modelos utilizados e interpretar sus resultados en el contexto del problema.

Reconocer limitaciones y proponer una solución viable con evidencia, incluso si los modelos no superan la referencia.

## 3 Organización y elección del caso

Cada equipo tendrá máximo tres integrantes y presentará una sola entrega con sus nombres y aportes. El caso puede provenir del trabajo de cualquiera de ellos. Todos deben comprender el proceso completo y poder explicar el código, las decisiones y las conclusiones. Elijan un solo tipo de tarea. En regresión, la variable objetivo es numérica continua; en clasificación, es una categoría. Para este taller se recomienda clasificación binaria. Si el caso exige varias clases, acuerden con el docente cómo evaluarlo.

1

Aprendizaje Automático 1 2026-2

Problema laboral Tarea y variable objetivo

Estimar tiempos de atención o entrega Regresión: duración en horas o días

Anticipar consumo o demanda Regresión: cantidad en un período definido

Priorizar solicitudes o incidentes Clasificación: incumple o cumple el plazo

Detectar operaciones que requieren revisión Clasificación: requiere revisión o no

Estos ejemplos orientan la selección; cada equipo debe definir su propio contexto y justificar por qué dispone de datos adecuados.

2

Aprendizaje Automático 1 2026-2

## 4 Formulación del problema y datos

Describan el proceso actual, quién toma la decisión, qué dificultad existe y cómo se atiende hoy. Delimiten la población, el período y la unidad de análisis: cada fila debe representar una observación claramente definida.

Pregunta predictiva: ¿qué se quiere estimar o clasificar, para quién y con cuánta anticipación?

Objetivo general y entre dos y cuatro objetivos específicos verificables.

Variable objetivo: definición, unidad o clases, origen de la etiqueta y momento en que se conoce. En clasificación binaria, indiquen qué evento representa la clase 1.

Predictores: qué información estará disponible en el momento real de la predicción. Excluyan campos creados después del resultado y variables que revelen directamente la respuesta.

Criterio de éxito: métrica principal, resultado de referencia y mejora mínima deseada. Si no existe una meta de negocio sustentada, declárenla como supuesto académico antes de evaluar.

## Disponibilidad y tratamiento de los datos

Usen datos del entorno laboral cuyo uso esté autorizado y que puedan compartir de forma anonimizada. Si no pueden acceder a ellos, podrán emplear datos abiertos de un problema equivalente o datos sintéticos, con validación previa del docente. Identifiquen esa condición y limiten las conclusiones al conjunto realmente analizado. No incluyan nombres, documentos de identidad, credenciales ni información confidencial de clientes o empresas. La entrega debe permitir reproducir el análisis sin acceder a sistemas privados. Pueden entregar una versión anonimizada o una muestra autorizada; si usan datos sintéticos, incluyan el generador y sus supuestos.

Documenten fuente, permiso o condiciones de uso, período, número de filas y variables, criterios de inclusión y exclusión y limitaciones de representatividad.

Construyan un diccionario con nombre de variable, descripción, tipo, unidad o categorías, rol de objetivo o predictor y disponibilidad al predecir.

Revisen faltantes, duplicados, valores imposibles, atípicos y, si aplica, desbalance de clases. Justifiquen cada tratamiento; no eliminen observaciones solo para mejorar métricas.

## 5 Fundamento conceptual aplicado

Expliquen para cada modelo su forma de predicción, función de pérdida o criterio de división, supuestos y parámetros que controlan la complejidad. Relacionen estas ideas con las variables del proyecto, sin limitarse a copiar definiciones.

3

Aprendizaje Automático 1 2026-2

Incluyan un cálculo ilustrativo con una observación o un subconjunto: una predicción lineal, el cálculo de una probabilidad logística o una división de un árbol. Usen valores del modelo entrenado o identifiquen explícitamente un ejemplo didáctico. Distingan asociación de causalidad.

4

Aprendizaje Automático 1 2026-2

## 6 Protocolo experimental obligatorio

Definan el protocolo antes de comparar modelos. Conserven la misma separación de observaciones para todos los candidatos y reporten el tamaño y la composición de cada partición.

Separación de datos: como referencia, 60 % entrenamiento, 20 % validación y 20 % prueba. Para observaciones independientes, usen random_state=42 y estratifiquen en clasificación cuando sea viable.

Si hay fechas, respeten el orden temporal. Si hay varias filas por cliente, equipo u otra entidad, eviten que la misma entidad aparezca en particiones incompatibles con el uso previsto. Justifiquen cualquier proporción distinta o validación cruzada.

Ajusten imputación, codificación, escalamiento y selección de variables únicamente con entrenamiento. Utilicen pipelines cuando corresponda. La exploración que oriente ajustes debe realizarse con entrenamiento.

Usen validación para elegir modelos, hiperparámetros y umbrales. Con pocos datos pueden aplicar validación cruzada dentro del conjunto de desarrollo y conservar una prueba independiente.

Antes de consultar prueba, registren el candidato final, predictores, transformaciones, hiperparámetros y umbral, si aplica. Congelen esas decisiones y evalúen una sola vez en prueba; no la reutilicen para ajustar.

## 7 Modelos mínimos según el problema

Ruta Comparación obligatoria Regresión Referencia constante con la media de entrenamiento; regresión lineal múltiple; árbol de regresión. Si solo existe un predictor pertinente, justifiquen una regresión simple. Clasificación Referencia de clase mayoritaria o probabilidades de entrenamiento; regresión logística; árbol de clasificación.

No se exige resolver ambas rutas. La regresión lineal simple puede añadirse como comparación. Otros algoritmos son opcionales y no reemplazan los modelos mínimos del curso. Exploren al menos tres configuraciones justificadas para cada familia con hiperparámetros; en la regresión lineal sin regularización, comparen y justifiquen conjuntos de predictores cuando proceda.

## 8 Métricas e interpretación

En regresión reporten MAE, RMSE y R 2 ; seleccionen una métrica principal según el costo del error. Analicen residuos, errores extremos y la brecha entre entrenamiento y validación.

5

Aprendizaje Automático 1 2026-2

En clasificación binaria reporten precisión, recall, F1, ROC-AUC y matriz de confusión con etiquetas [0, 1]. Calculen ROC-AUC con probabilidades de la clase 1. Si el desbalance es relevante, añadan PR-AUC. Cuando una métrica no pueda calcularse, expliquen la causa. Justifiquen la métrica principal y el umbral según las consecuencias de falsos positivos y negativos. Si usan costos, aclaren su origen o carácter supuesto. Interpreten coeficientes o reglas del árbol y expliquen sus límites. Identifiquen siempre la partición de cada tabla y figura.

6

Aprendizaje Automático 1 2026-2

## 9 Estructura del informe que debe diligenciar el equipo

Copien los siguientes apartados en su informe y reemplacen las indicaciones entre corchetes por el desarrollo del proyecto. Las tablas, figuras y resultados deben estar integrados en las secciones correspondientes y explicados con sus propias palabras. Identificación del proyecto [Título del proyecto, integrantes, contexto laboral o sector anonimizado, fecha y enlace al notebook o nombre del archivo. Indiquen la ruta elegida: regresión o clasificación.] Resumen ejecutivo [En 150 a 250 palabras: problema, datos utilizados, modelos comparados, resultado principal, recomendación y limitación más relevante.] 1 Problema y objetivos [Describan el proceso actual, la pregunta predictiva, la decisión que se apoyará, la unidad de análisis, el horizonte de predicción y los objetivos. Definan el criterio de éxito antes de presentar resultados.] 2 Datos y análisis exploratorio [Documenten fuente, permisos, período, tamaño y diccionario de variables. Incluyan calidad de los datos, distribución del objetivo y entre tres y cinco visualizaciones pertinentes con interpretación.] 3 Preparación y diseño experimental [Expliquen limpieza, transformaciones, prevención de fuga, particiones o validación cruzada. Reporten cuántas observaciones se usaron y por qué el diseño representa el uso previsto.] 4 Fundamento y construcción de los modelos [Presenten referencia y modelos obligatorios. Expliquen sus fundamentos, incluyan el cálculo ilustrativo y documenten predictores, configuraciones exploradas y criterio de selección.] 5 Resultados y evaluación final [Incluyan tablas comparativas de entrenamiento y validación, selección previa a prueba y métricas finales. Analicen sobreajuste, errores, coeficientes o reglas. Expliquen si se cumplió el criterio de éxito.] 6 Propuesta de solución para el trabajo [Describan quién utilizaría el resultado, cuándo se produciría la predicción, qué acción seguiría, qué revisión humana se requeriría y cómo se monitorearía. Distingan el desempeño medido del impacto laboral aún no validado.] 7 Conclusiones y limitaciones [Respondan la pregunta inicial y formulen recomendaciones respaldadas por resultados. Incluyan al menos tres limitaciones y dos mejoras futuras viables. Si ningún modelo mejora la referencia, expliquen las causas y el siguiente paso.]

7

Aprendizaje Automático 1 2026-2

8 Referencias y contribuciones [Citen datos, bibliografía, código y recursos externos utilizados. Declaren herramientas de IA y cómo verificaron sus aportes. Describan la contribución concreta de cada integrante.]

8

Aprendizaje Automático 1 2026-2

## 10 Entrega en el aula virtual

Un integrante cargará un solo archivo TallerFinal_GrupoXX.zip por equipo. El informe debe identificar a todos los integrantes. La fecha, la hora y el enlace del aula serán informados por el docente.

Informe_GrupoXX.pdf: desarrollo siguiendo la estructura anterior. Extensión orientativa de 8 a 12 páginas, sin contar portada, referencias y anexos.

Modelo_GrupoXX.ipynb: notebook ejecutado, con explicación, código, salidas, tablas y gráficos. Debe poder ejecutarse de principio a fin y usar rutas relativas.

datos/: datos autorizados y anonimizados utilizados o instrucciones precisas de acceso a una fuente pública estable. Si los datos son sintéticos, incluyan el script para generarlos.

README.md y requirements.txt: instrucciones de ejecución, estructura de archivos, versiones, semilla y dependencias. Incluyan los scripts auxiliares necesarios.

El PDF comunica el análisis; el notebook demuestra cómo se obtuvo. Una captura de pantalla o una salida de Python sin explicación no sustituye el desarrollo. Los resultados deben coincidir entre los archivos.

## 11 Rúbrica de evaluación sobre 100 puntos

Criterio Pts. Evidencia esperada para puntaje completo

Formulación del problema 15 Necesidad laboral clara, pregunta viable, objetivo y criterio de éxito verificables. Datos y exploración 15 Fuente documentada, diccionario, calidad y visualizaciones pertinentes. Preparación y validación 20 Transformaciones correctas, particiones justificadas y control de fuga de información. Fundamento y modelos 20 Referencia y modelos mínimos, explicación conceptual, cálculo aplicado y selección documentada. Evaluación e interpretación 15 Métricas adecuadas, prueba reservada y análisis de errores e interpretabilidad. Solución laboral y límites 10 Propuesta viable, conclusiones sustentadas, límites y mejoras futuras. Reproducibilidad y comunicación 5 Entrega completa, código ejecutable, informe claro y contribuciones identificadas.

Cada criterio se calificará según el cumplimiento de su evidencia: completo y sustentado, 100 % de sus puntos; adecuado con omisiones menores, 75 %; parcial, 50 %; incipiente, 25 %; ausente o sin evidencia verificable, 0 %. El puntaje no depende de obtener métricas altas; se valora la calidad del proceso y la honestidad del análisis.

9

Aprendizaje Automático 1 2026-2

## 12 Verificación antes de enviar

Confirmen que el equipo tiene máximo tres integrantes, el problema proviene de un contexto laboral, la entrega permite reproducir los resultados, no contiene información confidencial y presenta una recomendación coherente con la evidencia. Verifiquen que prueba no haya intervenido en los ajustes y que todos los integrantes puedan explicar la solución. Referencia conceptual sugerida: Müller, A. C. y Guido, S. Introduction to Machine Learning with Python. O’Reilly Media, 2017. Consulten además la documentación oficial de la versión de scikit-learn utilizada y citen los recursos efectivamente consultados.

10