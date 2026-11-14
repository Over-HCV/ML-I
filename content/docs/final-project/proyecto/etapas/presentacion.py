"""Capa 8 · Presentación oral del proyecto.

Responsabilidad única: armar las once diapositivas de la presentación de
diez minutos (`notebooks/Presentacion.ipynb`) a partir de lo que las etapas
ya calcularon y guardaron (`artefactos/*.json` y el registro de cifras
`artefactos/macros_*.json`), y preparar, en tiempo de construcción, el
paquete pequeño de arreglos que necesitan los fragmentos de código y las
animaciones (`artefactos/presentacion_datos.npz`).

- `preparar_datos()` se ejecuta al construir (con el repositorio y las etapas
  terminadas): lee las tablas de modelado de desarrollo (nunca la prueba),
  reproduce las probabilidades de validación del clasificador congelado y
  comprueba que su ROC-AUC coincida con el del artefacto, ajusta una vez el
  perceptrón de la pasada didáctica (curvas `loss_curve_` y
  `validation_scores_`) y lo vuelve a entrenar época por época, y guarda una
  muestra de los perfiles de agrupamiento con su proyección en dos
  componentes, más una submuestra fija de entrenamiento y otra de
  validación ya transformadas (en float16) con que los paneles interactivos
  de `proyecto/reporte/interactivo.py` reentrenan en vivo la red, KNN y el
  árbol. Cada bloque es una función `_datos_*` que devuelve
  `(arreglos, meta)`; agregar un bloque basta para llevar más arreglos.
- `iniciar()` fija el estilo y devuelve esos datos (sin recalcular nada).
- Cada `diapositiva_*()` devuelve un `Resultado` con una sola pieza de HTML y
  las cifras que muestra.
"""
import json
import time
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd

from proyecto.base import config, rutas
from proyecto.base.artefactos import leer_json, leer_modelado
from proyecto.datos.esquema import MOMENTOS
from proyecto.etapas.comun import ES_SIMULACION, Resultado
from proyecto.reporte.figuras import configurar_estilo
from proyecto.reporte.html import (barras_animadas, cifra_grande, columnas, comparacion, diapositiva, embudo,
                                   fichas, flujo, linea_svg, linea_tiempo, matriz_confusion, nota, recuadros,
                                   tabla_html, tarjetas)

TOTAL = 11
FECHA_PRESENTACION = "21 de noviembre de 2026"
INSTITUCION = "Universidad del Rosario"
CORRIDAS_L16 = 16   # definición del arreglo ortogonal L16, no un resultado
ARCHIVOS = ["02_calidad", "03_cribado", "04_regresion", "05_clasificacion", "06_agrupamiento", "07_evaluacion"]
ARCHIVO_DATOS = "presentacion_datos.npz"
MUESTRA_AGRUPAMIENTO = 600      # filas de la animación de K-means
DIAS_REGRESION = 10             # días de validación que recibe el modelo congelado en la demostración
EPOCAS_MAXIMAS = 40             # tope de cuadros de la animación de la red
TRAMOS_HISTOGRAMA = 30
SUBMUESTRA_ENTRENAMIENTO = 3000  # filas con que los paneles interactivos reentrenan (red, KNN y árbol)
SUBMUESTRA_VALIDACION = 2000     # filas de validación con que los paneles miden el ROC-AUC

# Qué ocurre en cada momento del proceso (rótulos para la línea de tiempo del contrato).
QUE_PASA = {
    "maestro": "datos del cliente", "programacion": "agenda de muestreos", "toma": "toma en planta",
    "llegada": "condiciones al llegar", "recepcion": "aceptar o rechazar", "siembra": "montaje del ensayo",
    "lectura": "resultado", "reensayo": "repetición", "reporte": "informe",
}
CON_TILDE = {"programacion": "programación", "recepcion": "recepción"}
NOMBRES_REFERENCIA = {
    "base_dia_anterior": "Referencia: día anterior",
    "base_media_5": "Referencia: media de 5 días",
    "Dummy (media)": "Dummy (media)",
    "Dummy (proporción)": "Dummy (proporción)",
}
TEMAS_CURSO = ["Preprocesamiento", "KNN", "Regresión lineal", "Regresión logística · ROC · umbral",
               "Árboles con poda", "Validación cruzada y rejilla", "Perceptrón multicapa", "K-means", "DBSCAN"]


# ----------------------------------------------------------------------
# Lectura de lo ya calculado
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _artefacto(nombre):
    return leer_json(rutas.DIR_ARTEFACTOS / f"{nombre}.json")


def artefacto(nombre):
    """Artefacto JSON de una etapa (por ejemplo `"05_clasificacion"`), ya leído."""
    return _artefacto(nombre)


@lru_cache(maxsize=None)
def _macros():
    """Todas las cifras registradas por las etapas: nombre → texto (sin el escape de LaTeX)."""
    cifras = {}
    for ruta in sorted(rutas.DIR_ARTEFACTOS.glob("macros_*.json")):
        for nombre, entrada in json.loads(ruta.read_text(encoding="utf-8"))["macros"].items():
            cifras[nombre] = (str(entrada["valor"]).replace("\\_", "_").replace("$-$", "-")
                              .replace("$\\cdot$", "·").replace("\\%", "%"))
    return cifras


def pct(valor, decimales=1):
    """Proporción como porcentaje, con espacio duro para que «%» no quede solo en otra línea."""
    return f"{100 * valor:.{decimales}f} %".replace("-", "−")


def _m(nombre):
    return _macros()[nombre]


def _n(nombre):
    return float(_m(nombre))


def _pie():
    origen = "Datos simulados con la estructura del laboratorio" if ES_SIMULACION else "Exportación real del laboratorio"
    return [config.ROTULO_PROYECTO, origen]


def _diapo(numero, titulo, cuerpo, rotulo, subtitulo=None, valores=None, **opciones):
    pieza = diapositiva(titulo, cuerpo, numero=numero, total=TOTAL, rotulo=rotulo, subtitulo=subtitulo,
                        pie=opciones.pop("pie", _pie()), **opciones)
    return Resultado(salidas=[pieza], valores=valores or {})


def comprobar_artefactos():
    """Lista de artefactos que faltan (vacía si la presentación puede armarse)."""
    faltan = [f"{n}.json" for n in ARCHIVOS if not (rutas.DIR_ARTEFACTOS / f"{n}.json").exists()]
    if not list(rutas.DIR_ARTEFACTOS.glob("macros_*.json")):
        faltan.append("macros_*.json")
    if not (rutas.DIR_ARTEFACTOS / ARCHIVO_DATOS).exists():
        faltan.append(ARCHIVO_DATOS)
    return faltan


# ----------------------------------------------------------------------
# Datos pequeños de la presentación (se preparan al construir)
# ----------------------------------------------------------------------
def _clasificacion_desarrollo():
    from proyecto.evaluacion import validacion
    from proyecto.variables.comunes import columnas_de_grupo

    nombre = _artefacto("03_cribado")["seleccion"]["clasificacion"]
    papeles = leer_json(rutas.DIR_MODELADO / f"{nombre}.json")
    desarrollo = leer_modelado(nombre).reset_index(drop=True)
    ent, val = validacion.separar_desarrollo(desarrollo, papeles["fecha"])
    numericas, categoricas = columnas_de_grupo(papeles, "ampliado")
    columnas = numericas + categoricas
    return SimpleNamespace(
        papeles=papeles, numericas=numericas, categoricas=categoricas, fecha=papeles["fecha"],
        ent=ent.reset_index(drop=True), X_ent=ent[columnas].reset_index(drop=True),
        y_ent=ent[papeles["objetivo"]].to_numpy(dtype=int), X_val=val[columnas].reset_index(drop=True),
        y_val=val[papeles["objetivo"]].to_numpy(dtype=int))


def _datos_umbral(c):
    """Probabilidades de validación del modelo elegido (ajustado con entrenamiento y calibrado con los
    bloques), las mismas con las que la etapa 05 eligió el umbral."""
    from sklearn.calibration import CalibratedClassifierCV

    from proyecto.etapas import taguchi
    from proyecto.evaluacion import metricas, validacion

    clf = _artefacto("05_clasificacion")
    fila = pd.Series({**clf["configuracion"], "hiper": clf["hiperparametros"]}, dtype=object)
    base = taguchi.constructor("clasificacion", c.papeles, clf["discretas"])(fila)
    if clf["calibracion"]["usada"]:
        base = CalibratedClassifierCV(base, method=clf["calibracion"]["metodo"],
                                      cv=validacion.bloques_temporales(c.ent[c.fecha]))
    proba = metricas.prediccion_de(base.fit(c.X_ent, c.y_ent), c.X_val)
    alerta = proba >= clf["umbral"]
    costo = config.COSTO_FP * int((alerta & (c.y_val == 0)).sum()) + config.COSTO_FN * int((~alerta & (c.y_val == 1)).sum())
    arreglos = {"proba_validacion": proba.astype(np.float32), "y_validacion": c.y_val.astype(np.int8),
                "fechas_entrenamiento": pd.to_datetime(c.ent[c.fecha]).to_numpy().astype("datetime64[D]")}
    meta = {"auc_reproducido": float(metricas.roc_auc(c.y_val, proba)), "costo_reproducido": costo,
            "numericas": c.numericas, "categoricas": c.categoricas}
    return arreglos, meta


def _datos_red(c):
    """Perceptrón de la pasada didáctica: un ajuste con parada temprana (sus curvas) y otro época por época
    con `partial_fit` (pérdida, ROC-AUC de validación y distribución de probabilidades en cada época)."""
    from sklearn.base import clone
    from sklearn.neural_network import MLPClassifier

    from proyecto.evaluacion import metricas
    from proyecto.modelos import tuberias

    tuberia = tuberias.construir_tuberia("clasificacion", "mlp", c.numericas, c.categoricas).fit(c.X_ent, c.y_ent)
    red = tuberia.named_steps["modelo"]
    preparar = tuberia.named_steps["preparar"]
    X_ent, X_val = preparar.transform(c.X_ent), preparar.transform(c.X_val)
    params = {k: v for k, v in red.get_params().items() if k not in ("early_stopping", "max_iter", "warm_start")}
    paso = clone(MLPClassifier(**params))
    epocas = min(red.n_iter_, EPOCAS_MAXIMAS)
    perdida, auc, histos, probas = [], [], [], []
    for _ in range(epocas):
        paso.partial_fit(X_ent, c.y_ent, classes=np.array([0, 1]))
        p = paso.predict_proba(X_val)[:, 1]
        perdida.append(paso.loss_)
        auc.append(metricas.roc_auc(c.y_val, p))
        probas.append(p)
    tope = float(np.quantile(np.concatenate(probas), 0.995))
    bordes = np.linspace(0, tope, TRAMOS_HISTOGRAMA + 1)
    for p in probas:
        histos.append([np.histogram(np.clip(p[c.y_val == clase], 0, tope), bordes, density=True)[0] for clase in (0, 1)])
    arreglos = {"mlp_perdida": np.array(red.loss_curve_), "mlp_validacion": np.array(red.validation_scores_),
                "red_perdida": np.array(perdida), "red_auc": np.array(auc),
                "red_histogramas": np.array(histos, dtype=np.float32), "red_bordes": bordes}
    meta = {"mlp": {"ocultas": list(red.hidden_layer_sizes), "activacion": red.activation,
                    "n_entradas": int(red.n_features_in_), "epocas": int(red.n_iter_),
                    "mejor_epoca": int(np.argmax(red.validation_scores_)) + 1, "paciencia": int(red.n_iter_no_change),
                    "fraccion_validacion": float(red.validation_fraction), "alpha": float(red.alpha),
                    "parametros": int(sum(w.size for w in red.coefs_) + sum(b.size for b in red.intercepts_)),
                    "auc_validacion_final": float(auc[-1]),
                    "tasa_mayoritaria": float(max(c.y_ent.mean(), 1 - c.y_ent.mean()))}}
    return arreglos, meta


def _datos_interactivo(c):
    """Submuestras fijas (con semilla) de entrenamiento y validación, ya transformadas por un
    preprocesamiento ajustado solo con entrenamiento, para los paneles que reentrenan en vivo (red, KNN y
    árbol), y los valores congelados que cada panel marca como «elegido». Se guardan en float16: bastan para
    una demostración y mantienen el paquete pequeño."""
    from proyecto.modelos import registro, tuberias

    clf, agr = _artefacto("05_clasificacion"), _artefacto("06_agrupamiento")
    preparar = tuberias.preprocesador(c.numericas, c.categoricas).fit(c.X_ent)
    azar = np.random.default_rng(config.SEMILLA)
    filas_ent = np.sort(azar.choice(len(c.X_ent), min(SUBMUESTRA_ENTRENAMIENTO, len(c.X_ent)), replace=False))
    filas_val = np.sort(azar.choice(len(c.X_val), min(SUBMUESTRA_VALIDACION, len(c.X_val)), replace=False))
    arreglos = {"sub_X_ent": np.asarray(preparar.transform(c.X_ent.iloc[filas_ent]), dtype=np.float16),
                "sub_y_ent": c.y_ent[filas_ent].astype(np.int8),
                "sub_X_val": np.asarray(preparar.transform(c.X_val.iloc[filas_val]), dtype=np.float16),
                "sub_y_val": c.y_val[filas_val].astype(np.int8)}
    knn = registro.crear_modelo("clasificacion", "knn")
    arbol = registro.crear_modelo("clasificacion", "arbol").get_params()
    arbol.update(clf.get("rejilla", {}).get("arbol", {}).get("hiper", {}))
    ensayos = [{k: f[k] for k in ("ensayo", "sensibilidad", "especificidad", "prevalencia_corregida", "vpp")}
               for f in clf["bayes"]["vpp_por_ensayo"]]
    meta = {"interactivo": {
        "variables": [n.split("__", 1)[-1] for n in preparar.get_feature_names_out()],
        "umbral": clf["umbral"], "umbrales": [u["Umbral"] for u in clf["tabla_umbrales"]], "costos": clf["costos"],
        "semanas_validacion": clf["semanas_validacion"],
        "ensayos": ensayos, "knn": {"k": int(knn.n_neighbors), "pesos": knn.weights},
        "arbol": {"ccp_alpha": float(arbol["ccp_alpha"]), "max_depth": arbol["max_depth"],
                  "min_samples_leaf": int(arbol["min_samples_leaf"])},
        "kmeans_exploracion": agr["seleccionado"]["kmeans"]["exploracion"]}}
    return arreglos, meta


def _datos_kmeans():
    """Muestra de la matriz de agrupamiento elegida y la proyección en dos componentes de toda la matriz."""
    from sklearn.decomposition import PCA

    from proyecto.etapas.agrupamiento import ASIMETRIA_LOG
    from proyecto.modelos.agrupamiento import preparar_matrices
    from proyecto.variables.comunes import columnas_de_grupo

    nombre = _artefacto("06_agrupamiento")["seleccionado"]["analisis"]
    papeles = leer_json(rutas.DIR_MODELADO / f"{nombre}.json")
    numericas, categoricas = columnas_de_grupo(papeles, "ampliado")
    matriz, _, _ = preparar_matrices(leer_modelado(nombre), numericas, categoricas, ASIMETRIA_LOG)
    pca = PCA(n_components=2, random_state=config.SEMILLA).fit(matriz)
    filas = np.random.default_rng(config.SEMILLA).choice(len(matriz), min(MUESTRA_AGRUPAMIENTO, len(matriz)),
                                                         replace=False)
    arreglos = {"kmeans_muestra": matriz[np.sort(filas)].astype(np.float32),
                "kmeans_componentes": pca.components_.astype(np.float32), "kmeans_media": pca.mean_.astype(np.float32)}
    return arreglos, {"kmeans": {"analisis": nombre, "k": int(_artefacto("06_agrupamiento")["seleccionado"]["kmeans"]["k"])}}


def _datos_regresion():
    """Los últimos días de validación de la carga diaria, con las columnas que recibe el modelo congelado."""
    from proyecto.evaluacion import validacion

    reg = _artefacto("04_regresion")
    nombre = _artefacto("03_cribado")["seleccion"]["regresion"]
    papeles = leer_json(rutas.DIR_MODELADO / f"{nombre}.json")
    _, val = validacion.separar_desarrollo(leer_modelado(nombre), papeles["fecha"])
    dias = val.sort_values(papeles["fecha"]).tail(DIAS_REGRESION)[[papeles["fecha"], papeles["objetivo"],
                                                                    *reg["variables_entrada"]]]
    dias = dias.assign(**{papeles["fecha"]: pd.to_datetime(dias[papeles["fecha"]]).dt.strftime("%Y-%m-%d")})
    filas = json.loads(dias.to_json(orient="records"))
    return {}, {"regresion": {"fecha": papeles["fecha"], "objetivo": papeles["objetivo"],
                              "variables": reg["variables_entrada"], "filas": filas}}


def preparar_datos():
    """Escribe `artefactos/presentacion_datos.npz` desde las salidas reales de las etapas (sin tocar la
    prueba). Devuelve un resumen con el tamaño, el tiempo y las comprobaciones de reproducción."""
    inicio = time.perf_counter()
    c = _clasificacion_desarrollo()
    arreglos, meta = {}, {}
    for bloque in (lambda: _datos_umbral(c), lambda: _datos_red(c), lambda: _datos_interactivo(c), _datos_kmeans,
                   _datos_regresion):
        a, m = bloque()
        arreglos.update(a)
        meta.update(m)
    clf = _artefacto("05_clasificacion")
    meta["comprobacion"] = {"auc_artefacto": clf["calibracion"]["auc_calibrado" if clf["calibracion"]["usada"] else "auc_sin"],
                            "costo_artefacto": clf["umbral_elegido"]["Costo"]}
    ruta = rutas.DIR_ARTEFACTOS / ARCHIVO_DATOS
    np.savez_compressed(ruta, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arreglos)
    return {"ruta": ruta, "kB": ruta.stat().st_size / 1024, "segundos": time.perf_counter() - inicio,
            "auc": (meta["auc_reproducido"], meta["comprobacion"]["auc_artefacto"]),
            "costo": (meta["costo_reproducido"], meta["comprobacion"]["costo_artefacto"])}


@lru_cache(maxsize=None)
def _cargar():
    with np.load(rutas.DIR_ARTEFACTOS / ARCHIVO_DATOS) as archivo:
        arreglos = {k: archivo[k] for k in archivo.files}
    meta = json.loads(str(arreglos.pop("meta")))
    reg = meta["regresion"]
    dias = pd.DataFrame(reg["filas"])
    dias[reg["fecha"]] = pd.to_datetime(dias[reg["fecha"]])
    return SimpleNamespace(**arreglos, **{k: v for k, v in meta.items() if k != "regresion"},
                           regresion=SimpleNamespace(**{**reg, "dias": dias}))


def iniciar():
    """Estilo editorial de las figuras y datos pequeños de la presentación (ya calculados al construir)."""
    configurar_estilo()
    return _cargar()


# ----------------------------------------------------------------------
# Diapositivas
# ----------------------------------------------------------------------
def diapositiva_portada():
    """1 · Título, curso, autores, fecha y contexto del laboratorio."""
    autores = [a.strip() for a in config.AUTORES.split("·")]
    contexto = ("Rodam Analisis S.A.S: Laboratorio de microbiología de alimentos en Bogotá.")
    return _diapo(1, config.TITULO_PROYECTO, [fichas(autores), contexto],
                  rotulo=f"{config.ROTULO_PROYECTO} · {INSTITUCION}",
                  subtitulo="Tres preguntas operativas, modelos de clase puestos a prueba",
                  pie=[FECHA_PRESENTACION, INSTITUCION], portada=True, valores={"autores": autores})


def diapositiva_problema():
    """2 · Las tres preguntas del laboratorio y el requisito del curso."""
    preguntas = recuadros([
        {"marca": "1", "titulo": "¿Cuántas muestras llegarán?", "color": "azul",
         "texto": "**Regresión** de la carga diaria, para planear personal y medios de cultivo."},
        {"marca": "2", "titulo": "¿Qué muestras son de riesgo?", "color": "naranja",
         "texto": "**Clasificación** de la no conformidad en el momento de recibir la muestra."},
        {"marca": "3", "titulo": "¿Qué tan probable es que un positivo sea falso?", "color": "rojo",
         "texto": "**Bayes** con la sensibilidad y la especificidad de los controles de cada ensayo."},
    ], por_fila=3)
    curso = fichas(TEMAS_CURSO, "Requisito del curso: resolverlo con los modelos vistos en clase")
    extra = nota("El **agrupamiento** (K-means y DBSCAN) describe perfiles de muestras y segmentos de clientes "
                 "como apoyo a las tres preguntas.", "info", titulo="Complemento")
    return _diapo(2, "El problema", [preguntas, curso, extra], rotulo="Contexto",
                  subtitulo="Pasar de una base de registros a decisiones de la operación diaria")


def diapositiva_datos():
    """3 · El contrato de datos (momentos del proceso), tamaños de las tablas y origen."""
    calidad = _artefacto("02_calidad")
    momento_reg, momento_clf = _m("RegMomento"), _m("ClfMomento")
    hitos = []
    for i, momento in enumerate(MOMENTOS, start=1):
        hito = {"fecha": f"{i:02d}", "titulo": CON_TILDE.get(momento, momento), "detalle": QUE_PASA.get(momento, ""),
                "color": "#B7B1A5"}
        if momento == momento_reg:
            hito.update(color="azul", detalle=f"{hito['detalle']} · **predice la regresión**")
        elif momento == momento_clf:
            hito.update(color="naranja", detalle=f"{hito['detalle']} · **predice la clasificación**")
        hitos.append(hito)
    linea = linea_tiempo(hitos, titulo="Cada columna declara en qué momento se conoce: un modelo solo usa lo "
                                       "anterior a su momento de predicción")
    filas = calidad["filas_limpias"]
    cifras = tarjetas([
        {"titulo": "Muestras", "valor": filas["muestras"], "color": "azul", "detalle": "una fila por muestra"},
        {"titulo": "Clientes", "valor": filas["clientes"], "color": "verde"},
        {"titulo": "Visitas de muestreo", "valor": filas["muestreos"], "color": "morado"},
        {"titulo": "Controles de calidad", "valor": filas["controles"], "color": "naranja",
         "detalle": "positivos y negativos"},
        {"titulo": "Columnas del contrato", "valor": int(_n("DatColumnasContrato")), "color": "gris",
         "detalle": f"{_m('DatColumnasObligatorias')} obligatorias"},
    ])
    if ES_SIMULACION:
        origen = nota("La base es una **réplica simulada** con la misma estructura (semilla fija). La exportación real "
                      "se conecta con `configuracion/mapeo_columnas.json` y el resto del código no cambia.", "simulacion")
    else:
        origen = nota("Los datos provienen de la exportación real del laboratorio, conectada con el mapeo de columnas.",
                      "info")
    periodo = f"Del {_m('DatDesde')} al {_m('DatHasta')}"
    return _diapo(3, "Los datos", [cifras, linea, origen], rotulo="Contrato de datos", subtitulo=periodo,
                  valores={"muestras": filas["muestras"]})


def diapositiva_exploracion():
    """4 · Tres hallazgos: desbalance, condiciones del transporte y estacionalidad."""
    tasa, conformes = _n("ExpTasaNoConformePct"), int(_n("ExpConformesPorNoConforme"))
    desbalance = cifra_grande({
        "valor": tasa, "decimales": 1, "sufijo": " %", "color": "rojo", "etiqueta": "de las muestras no son conformes",
        "detalle": f"{conformes} conformes por cada no conforme: se mide con ROC-AUC y costo, no con exactitud"})
    indice_alto, indice_bajo = _n("ExpIndiceMesAlto"), _n("ExpIndiceMesBajo")
    volumen = cifra_grande({
        "valor": int(_n("ExpMuestrasSemana")), "color": "azul", "etiqueta": "muestras por semana en promedio",
        "detalle": (f"entre {_m('ExpMuestrasSemanaMin')} y {_m('ExpMuestrasSemanaMax')}; {_m('ExpMesAlto')} "
                    f"{100 * (indice_alto - 1):+.0f} % sobre lo normal y {_m('ExpMesBajo')} "
                    f"{100 * (indice_bajo - 1):+.0f} %".replace("-", "−"))})
    transporte = {
        "Temperatura: cuarto más frío": _n("ExpTemperaturaCuartoBajoPct"),
        "Temperatura: cuarto más cálido": _n("ExpTemperaturaCuartoAltoPct"),
        "Transporte: cuarto más corto": _n("ExpHorasCuartoBajoPct"),
        "Transporte: cuarto más largo": _n("ExpHorasCuartoAltoPct"),
    }
    sube_temp = transporte["Temperatura: cuarto más cálido"] > transporte["Temperatura: cuarto más frío"]
    sube_horas = transporte["Transporte: cuarto más largo"] > transporte["Transporte: cuarto más corto"]
    barras = barras_animadas(transporte, "No conformes (%) según el transporte", formato="{:.1f} %",
                             resaltar=[k for k, v in transporte.items() if v == max(transporte.values())],
                             color="naranja")
    if sube_temp and sube_horas:
        lectura = ("Un transporte **más cálido o más largo** acompaña más no conformidades: son variables que se "
                   "conocen al llegar la muestra y le sirven al clasificador.")
    else:
        lectura = ("Las condiciones del transporte no ordenan la no conformidad en un mismo sentido: el clasificador "
                   "las usa, pero sin una lectura simple.")
    cuerpo = [columnas([desbalance, volumen], [barras, nota(lectura, "hallazgo")], proporciones="1fr 1.15fr")]
    return _diapo(4, "Lo que se ve antes de modelar", cuerpo, rotulo="Exploración",
                  valores={"tasa_no_conforme_pct": tasa})


def diapositiva_metodo():
    """5 · Embudo de cribado, protocolo entrenamiento → validación → prueba y Taguchi L16."""
    regla = _artefacto("03_cribado")["constantes_regla"]
    filtro = embudo([
        ("análisis candidatos", int(_n("CriCandidatos")), "los bayesianos son analíticos y no se criban"),
        ("con tabla de modelado y prueba rápida", int(_n("CriCribados")),
         f"le ganan a su línea base en los {regla['BLOQUES_VALIDACION_CRUZADA']} bloques y no dependen de una sola "
         "variable"),
        ("elegibles", int(_n("CriElegibles")), "puntaje = habilidad × valor para el laboratorio"),
        ("elegidos: uno por tipo", int(_n("CriElegidos"))),
    ], titulo="Qué se modela a fondo")
    partes = {p["Bloque"]: p for p in _artefacto("02_calidad")["particion"]}
    protocolo = []
    for bloque, accion, color in (("entrenamiento", "ajustar y comparar (CV por bloques)", "azul"),
                                  ("validacion", "decidir modelo y umbral", "morado"),
                                  ("prueba", "evaluar una sola vez", "rojo")):
        p = partes.get(bloque, {})
        protocolo.append({"marca": f"{p.get('Desde', '')[:7]} → {p.get('Hasta', '')[:7]}",
                          "titulo": "validación" if bloque == "validacion" else bloque,
                          "detalle": f"{accion} · {p.get('Filas', '—')} muestras", "color": color,
                          "lleno": bloque == "prueba"})
    rho, iguales = _n("RegRhoCompleto"), int(_n("RegFactoresMismoOptimo"))
    factores = int(_n("RegFactores"))
    acuerdo = "el mismo orden de importancia" if rho >= 0.8 else "un orden parecido"
    taguchi = cifra_grande({
        "valor": CORRIDAS_L16, "color": "verde", "etiqueta": f"corridas del L16 de Taguchi en vez de "
                                                             f"{_m('RegCorridasCompleto')}",
        "detalle": (f"Regresión: {acuerdo} de factores que el factorial completo (Spearman {rho:.2f}) y "
                    f"{iguales} de {factores} factores con el mismo mejor nivel; en clasificación "
                    f"{_n('ClfSegundosLSixteen'):.0f} s en vez de unos {_n('ClfSegundosCompletoEstimados'):.0f} s.")})
    derecha = [flujo(protocolo, titulo="Protocolo temporal: nunca se entrena con datos del futuro"), taguchi]
    return _diapo(5, "Cribar, decidir y probar una vez", [columnas(filtro, derecha, proporciones="1fr 1.3fr")],
                  rotulo="Método", valores={"rho_l16": rho})


def diapositiva_regresion():
    """6 · Carga diaria: modelos del curso comparados, ganador y error en prueba frente a la referencia."""
    reg, final = _artefacto("04_regresion"), _artefacto("07_evaluacion")["regresion"]
    rmse = {NOMBRES_REFERENCIA.get(f["Modelo"], f["Modelo"].split(" (")[0]): f["RMSE"]
            for f in reg["didactica"] if f["Modelo"] != "base_dia_anterior"}
    mejor = _m("RegMejorDidactico")
    barras = barras_animadas(dict(sorted(rmse.items(), key=lambda par: par[1])),
                             "Validación cruzada: RMSE medio por bloque (muestras/día, menos es mejor)",
                             formato="{:.1f}", resaltar=mejor)
    filas = {f["Predictor"]: f for f in final["tabla"]}
    modelo, referencia = filas[_m("FinRegModelo")], filas[final["referencia"]]
    op = final["operacion"]
    tolerancia = op["tolerancia"]
    duelo = comparacion([
        {"titulo": NOMBRES_REFERENCIA.get(final["referencia"], final["referencia"]), "valor": referencia["MAE"],
         "decimales": 1, "etiqueta": "error absoluto medio en prueba (muestras/día)",
         "puntos": [f"RMSE {referencia['RMSE']:.1f}", f"{pct(op['dias_fuera_referencia'], 0)} de días fuera de "
                                                       f"±{tolerancia:.0f}"]},
        {"titulo": _m("FinRegModelo"), "valor": modelo["MAE"], "decimales": 1,
         "etiqueta": "error absoluto medio en prueba (muestras/día)",
         "puntos": [f"RMSE {modelo['RMSE']:.1f} · R² {modelo['R2']:.2f}",
                    f"{pct(op['dias_fuera'], 0)} de días fuera de ±{tolerancia:.0f}"]},
    ], elegida=1, sello="Congelado")
    cambio = _n("FinCambioRMSEPct")
    sentido = "menor" if cambio < 0 else "mayor"
    gana = modelo["MAE"] < referencia["MAE"]
    lectura = (f"Con unas {op['media_prueba']:.0f} muestras por día hábil, el modelo se equivoca en promedio en "
               f"{modelo['MAE']:.0f} muestras ({'menos' if gana else 'más'} que la referencia, "
               f"{referencia['MAE']:.0f}); la tolerancia práctica es ±{tolerancia:.0f} "
               f"({_m('RegToleranciaOperativaPct')} % de la media). En prueba el RMSE fue "
               f"{abs(cambio):.0f} % {sentido} que en validación.")
    cuerpo = [columnas(barras, duelo, proporciones="1fr 1fr"), nota(lectura, "hallazgo", titulo="Lectura operativa")]
    return _diapo(6, "¿Cuántas muestras llegarán mañana?", cuerpo, rotulo="Pregunta 1 · regresión de la carga diaria",
                  subtitulo=f"Se predice en el momento de {CON_TILDE.get(_m('RegMomento'), _m('RegMomento'))}, con la agenda ya conocida",
                  valores={"mae_prueba": modelo["MAE"], "mae_referencia": referencia["MAE"]})


def diapositiva_clasificacion():
    """7 · Riesgo de no conformidad: modelos, umbral por costo, alertas por semana, validación frente a prueba."""
    clf, final = _artefacto("05_clasificacion"), _artefacto("07_evaluacion")["clasificacion"]
    auc = {NOMBRES_REFERENCIA.get(f["Modelo"], f["Modelo"]): f["ROC-AUC"] for f in clf["didactica"]}
    barras = barras_animadas(dict(sorted(auc.items(), key=lambda par: -par[1])),
                             "Validación cruzada: ROC-AUC medio (más es mejor)", formato="{:.3f}",
                             resaltar=_m("ClfMejorDidactico"), color="naranja")
    umbrales = clf["tabla_umbrales"]
    umbral = final["umbral"]
    costos = clf["costos"]
    curva = linea_svg([u["Umbral"] for u in umbrales], {"Costo en validación": ([u["Costo"] for u in umbrales],
                                                                                 "naranja")},
                      titulo=f"Costo = {costos['falso_positivo']}·FP + {costos['falso_negativo']}·FN según el umbral",
                      marca=umbral, rotulo_marca=f"umbral {umbral:.3f}", etiqueta_x="umbral de alerta",
                      formato_y="{:.0f}", alto=165)
    m = final["metricas"]
    matriz = matriz_confusion(m["TN"], m["FP"], m["FN"], m["TP"],
                              titulo=f"Prueba: {final['semanas']} semanas con el umbral congelado")
    val = final["validacion"]
    alertas_val = clf["umbral_elegido"]["Alertas por semana"]
    cifras = tarjetas([
        {"titulo": "Alertas por semana", "valor": final["alertas_por_semana"], "decimales": 1, "color": "naranja",
         "detalle": f"validación: {alertas_val:.1f}"},
        {"titulo": "Precisión", "valor": 100 * m["Precisión"], "decimales": 0, "sufijo": " %", "color": "azul",
         "detalle": f"validación: {pct(val['Precisión'], 0)}"},
        {"titulo": "Recall", "valor": 100 * m["Recall"], "decimales": 0, "sufijo": " %", "color": "verde",
         "detalle": f"validación: {pct(val['Recall'], 0)}"},
    ])
    cada = 1 / m["Precisión"] if m["Precisión"] else float("nan")
    ahorro = _n("FinAhorroPct")
    lectura = (f"Cada semana se revisan unas {final['alertas_por_semana']:.0f} muestras y aproximadamente 1 de cada "
               f"{cada:.0f} resulta no conforme; el costo baja {ahorro:.0f} % frente a la mejor política sin modelo. "
               f"ROC-AUC: {val['ROC-AUC']:.3f} en validación, {m['ROC-AUC']:.3f} en prueba.")
    if m["Recall"] < val["Recall"]:
        lectura += " En prueba se detectan menos no conformes que en validación."
    cuerpo = [columnas([barras, curva], [matriz, cifras], proporciones="1fr 1fr"), nota(lectura, "hallazgo")]
    return _diapo(7, "¿Qué muestras son de riesgo?", cuerpo, rotulo="Pregunta 2 · clasificación de la no conformidad",
                  subtitulo=(f"{_m('ClfModeloFinal')} al recibir la muestra · perder una no conforme cuesta "
                             f"{costos['falso_negativo'] // costos['falso_positivo']} veces una falsa alarma"),
                  valores={"umbral": umbral, "alertas_semana": final["alertas_por_semana"]})


def diapositiva_red():
    """8 · Perceptrón multicapa: arquitectura usada, parada temprana y comparación con la regresión logística."""
    didactica = {f["Modelo"]: f for f in _artefacto("05_clasificacion")["didactica"]}
    red, log = didactica["Perceptrón multicapa"], didactica["Regresión logística"]
    mlp = _cargar().mlp
    duelo = comparacion([
        {"titulo": "Regresión logística", "valor": log["ROC-AUC"], "decimales": 3, "etiqueta": "ROC-AUC medio por bloque",
         "puntos": [f"± {log['± ROC-AUC']:.3f} entre bloques", f"{mlp['n_entradas'] + 1} parámetros",
                    f"Brier {log['Brier']:.4f}"]},
        {"titulo": "Perceptrón multicapa", "valor": red["ROC-AUC"], "decimales": 3, "etiqueta": "ROC-AUC medio por bloque",
         "puntos": [f"± {red['± ROC-AUC']:.3f} entre bloques", f"{mlp['parametros']} parámetros",
                    f"Brier {red['Brier']:.4f}"]},
    ], elegida=0 if log["ROC-AUC"] >= red["ROC-AUC"] else 1, sello="Ordena mejor",
        titulo="Pasada didáctica: mismos bloques, mismas variables")
    capas = " → ".join(str(x) for x in [mlp["n_entradas"], *mlp["ocultas"], 1])
    cifras = tarjetas([
        {"titulo": "Neuronas ocultas", "valor": sum(mlp["ocultas"]), "color": "morado",
         "detalle": f"{capas} · activación {mlp['activacion']}"},
        {"titulo": "Épocas", "valor": mlp["epocas"], "color": "rojo",
         "detalle": f"la mejor fue la {mlp['mejor_epoca']}; parada temprana"},
        {"titulo": "Regularización (alpha)", "valor": mlp["alpha"], "decimales": 3, "color": "gris",
         "detalle": "penaliza los pesos grandes"},
    ])
    if red["ROC-AUC"] <= log["ROC-AUC"]:
        lectura = (f"Con {mlp['parametros']} parámetros frente a {mlp['n_entradas'] + 1}, la red no ordena mejor que la "
                   f"logística ({red['ROC-AUC']:.3f} frente a {log['ROC-AUC']:.3f}): el riesgo se explica sobre todo por "
                   "efectos que suman en la escala logit, que la logística ya captura, y la capa oculta no encuentra "
                   "interacciones que compensen su costo. Por eso el modelo congelado es la logística.")
    else:
        lectura = (f"La red ({red['ROC-AUC']:.3f}) ordena mejor que la logística ({log['ROC-AUC']:.3f}): hay algo de no "
                   f"linealidad o de interacción que la capa oculta aprovecha, a cambio de {mlp['parametros']} parámetros.")
    cuerpo = [columnas(duelo, cifras, proporciones="1.1fr 1fr"), nota(lectura, "hallazgo", titulo="Por qué")]
    return _diapo(8, "Redes neuronales: del perceptrón al MLP", cuerpo, rotulo="Tema del curso · perceptrón multicapa",
                  subtitulo="La misma pregunta de riesgo, con una capa oculta entre las variables y la probabilidad",
                  valores={"auc_red": red["ROC-AUC"], "auc_logistica": log["ROC-AUC"]})


def diapositiva_bayes():
    """9 · ¿Falso positivo?: VPP por ensayo, el mismo positivo con riesgo bajo y alto, regla de reensayo."""
    bayes = _artefacto("05_clasificacion")["bayes"]
    vpp = {f["ensayo"]: f["vpp"] for f in sorted(bayes["vpp_por_ensayo"], key=lambda f: f["vpp"])}
    peor = _m("ClfVPPPeorEnsayo")
    barras = barras_animadas(vpp, "VPP por ensayo (desarrollo)", formato="{:.2f}", resaltar=peor, maximo=1,
                             color="#B8423F")
    ejemplos = bayes["vpp_por_muestra"]["ejemplos"]
    bajo, alto = ejemplos["bajo"], ejemplos["alto"]
    mismo = comparacion([
        {"titulo": "Muestra de riesgo bajo", "valor": 100 * (1 - bajo["vpp_x"]), "decimales": 0, "sufijo": " %",
         "etiqueta": "probabilidad de que el positivo sea falso",
         "puntos": [f"previa del clasificador: {pct(bajo['p_x'], 0)}"]},
        {"titulo": "Muestra de riesgo alto", "valor": 100 * (1 - alto["vpp_x"]), "decimales": 0, "sufijo": " %",
         "etiqueta": "probabilidad de que el positivo sea falso",
         "puntos": [f"previa del clasificador: {pct(alto['p_x'], 0)}"]},
    ], titulo=f"Mismo ensayo ({bajo['ensayo']}), mismo positivo", elegida=0, sello="Reensayar")
    veces, tasa_bajo, tasa_alto = _n("ClfReglaVecesLimite"), _n("FinReglaTasaBajoPct"), _n("FinReglaTasaAltoPct")
    regla = (f"**Regla de reensayo:** si el recuento queda por debajo de {veces:.2f} veces el límite, se reensaya; "
             f"en prueba el {tasa_bajo:.0f} % de esos reensayos resultó conforme, frente al {tasa_alto:.0f} % por "
             f"encima del borde.")
    cuerpo = [columnas(barras, mismo, proporciones="1fr 1.1fr"), nota(regla, "hallazgo")]
    return _diapo(9, "¿Qué tan probable es que un positivo sea falso?", cuerpo,
                  rotulo="Pregunta 3 · Bayes y falsos positivos",
                  subtitulo=(f"En el laboratorio, cerca de 1 de cada {_m('FinUnoDeCadaFalso')} positivos sería falso; "
                             f"{peor} es el ensayo menos fiable"),
                  valores={"vpp_peor": vpp[peor] if peor in vpp else None})


def diapositiva_agrupamiento():
    """10 · Segmentos de clientes y perfiles de muestra, con la advertencia sobre siluetas bajas."""
    agr = _artefacto("06_agrupamiento")
    sel, comp = agr["seleccionado"], agr["complementario"]
    segmentos = pd.DataFrame([{"Segmento de clientes": s["Segmento"], "Clientes": int(s["Clientes"]),
                               "% de muestras": 100 * s["Proporción de muestras"],
                               "% no conformes": 100 * s["No conformidad"]} for s in comp["segmentos_operacion"]])
    mayor = segmentos["% no conformes"].idxmax()
    tabla = tabla_html(segmentos, "Tabla. Segmentos de clientes (K-means)", resaltar=mayor,
                       decimales={"% de muestras": 1, "% no conformes": 1}, barras="% de muestras")
    perfiles = barras_animadas({s["nombre"]: 100 * s["proporcion"] for s in sel["segmentos"]},
                               "Perfiles de muestra (K-means): % de muestras", formato="{:.0f} %", color="morado")
    silueta_sel, silueta_comp = sel["kmeans"]["silueta"], comp["kmeans"]["silueta"]
    if max(silueta_sel, silueta_comp) < 0.25:
        honesto = (f"Las siluetas son bajas ({silueta_sel:.2f} y {silueta_comp:.2f}): los grupos sirven para "
                   f"**describir** la operación, no son fronteras nítidas. Hopkins ≈ {sel['hopkins']:.2f} indica "
                   "que sí hay algo de estructura.")
        tipo = "limite"
    else:
        honesto = f"Siluetas de {silueta_sel:.2f} y {silueta_comp:.2f}: grupos razonablemente separados."
        tipo = "hallazgo"
    cuerpo = [columnas(tabla, perfiles, proporciones="1.25fr 1fr"), nota(honesto, tipo)]
    return _diapo(10, "Perfiles y segmentos", cuerpo, rotulo="Complemento · agrupamiento",
                  subtitulo=f"{sel['kmeans']['k']} perfiles de muestra y {comp['kmeans']['k']} segmentos de clientes",
                  valores={"silueta_perfiles": silueta_sel, "silueta_segmentos": silueta_comp})


def diapositiva_conclusiones():
    """11 · Respuesta a cada pregunta, limitaciones y pasos siguientes."""
    final = _artefacto("07_evaluacion")
    reg, clf = final["regresion"], final["clasificacion"]
    modelo = {f["Predictor"]: f for f in reg["tabla"]}[_m("FinRegModelo")]
    m = clf["metricas"]
    respuestas = recuadros([
        {"marca": "1", "color": "azul", "titulo": "Carga diaria",
         "texto": (f"{_m('FinRegModelo')}: error medio de {modelo['MAE']:.1f} muestras/día en prueba y RMSE "
                   f"{pct(modelo['Habilidad'], 0)} menor que la mejor referencia.")},
        {"marca": "2", "color": "naranja", "titulo": "Muestras de riesgo",
         "texto": (f"{_m('FinClfModelo')} (ROC-AUC {m['ROC-AUC']:.2f}): {clf['alertas_por_semana']:.0f} alertas por "
                   f"semana, precisión {pct(m['Precisión'], 0)}, recall {pct(m['Recall'], 0)}, costo "
                   f"{_n('FinAhorroPct'):.0f} % menor.")},
        {"marca": "3", "color": "rojo", "titulo": "Falsos positivos",
         "texto": (f"Cerca de 1 de cada {_m('FinUnoDeCadaFalso')} positivos sería falso; {_m('ClfVPPPeorEnsayo')} es "
                   f"el ensayo menos fiable y la regla de reensayo concentra el {_n('FinReglaCapturaPct'):.0f} % de "
                   "los reensayos conformes.")},
    ], titulo="Respuestas (prueba, abierta una sola vez)", por_fila=3)
    if ES_SIMULACION:
        limites = ("**Datos simulados:** las cifras prueban que el procedimiento funciona de punta a punta, no "
                   "describen todavía al laboratorio. La señal es moderada y los grupos, poco nítidos.")
    else:
        limites = "La señal es moderada y los grupos, poco nítidos: las cifras deben leerse con sus intervalos."
    pasos_siguientes = ("Conectar la exportación real con el mapeo de columnas y reconstruir con "
                        "`PROYECTO_ORIGEN=real make notebooks`; revisar con el laboratorio los costos de falso "
                        "positivo y falso negativo y la tolerancia diaria; reestimar S y E con los controles reales.")
    cuerpo = [respuestas, nota(limites, "limite", titulo="Limitaciones"),
              nota(pasos_siguientes, "info", titulo="Siguientes pasos")]
    return _diapo(11, "Conclusiones y siguientes pasos", cuerpo, rotulo="Cierre",
                  valores={"mae_prueba": modelo["MAE"], "auc_prueba": m["ROC-AUC"]})


DIAPOSITIVAS = [diapositiva_portada, diapositiva_problema, diapositiva_datos, diapositiva_exploracion,
                diapositiva_metodo, diapositiva_regresion, diapositiva_clasificacion, diapositiva_red,
                diapositiva_bayes, diapositiva_agrupamiento, diapositiva_conclusiones]
