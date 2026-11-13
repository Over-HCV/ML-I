# %% [markdown]
# ## Ejercicio 9 — Caso B: árbol de clasificación e interpretación

# %% [markdown]
# ### a. Cuadrícula de `DecisionTreeClassifier`
#
# Se usan los mismos predictores originales y particiones, sin estandarizar
# (los cortes de un árbol son invariantes a transformaciones monótonas de
# cada variable). La cuadrícula se recorre en el orden del enunciado,
# `max_depth` por fuera y `min_samples_leaf` por dentro, porque ese orden es
# el último criterio de desempate.

# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree

filas, arboles_clf = [], []
for max_depth in [2, 3, 5, None]:
    for min_leaf in [1, 5, 15]:
        arbol = DecisionTreeClassifier(
            criterion="gini", max_depth=max_depth,
            min_samples_leaf=min_leaf, random_state=RANDOM_STATE,
        ).fit(Xc_train, yc_train)
        arboles_clf.append(arbol)
        p_tr = arbol.predict_proba(Xc_train)[:, 1]
        p_val = arbol.predict_proba(Xc_val)[:, 1]
        filas.append({
            "max_depth": "None" if max_depth is None else max_depth,
            "min_samples_leaf": min_leaf,
            "ROC-AUC ent.": roc_auc_score(yc_train, p_tr),
            "ROC-AUC val.": roc_auc_score(yc_val, p_val),
            "F1 ent. (t=0.5)": f1_score(yc_train, predecir_umbral(p_tr, 0.5), zero_division=0),
            "F1 val. (t=0.5)": f1_score(yc_val, predecir_umbral(p_val, 0.5), zero_division=0),
            "Profundidad real": arbol.get_depth(),
            "Hojas": arbol.get_n_leaves(),
        })
tabla_9a = pd.DataFrame(filas)
print("Tabla 9.1 — Cuadrícula de árboles de clasificación (entrenamiento y validación)")
tabla_9a

# %% [markdown]
# Los doce árboles se ajustaron sobre los predictores originales sin escalar.
# Se observa que `min_samples_leaf` limita la profundidad efectiva, de forma
# que con `min_samples_leaf` igual a 5 o 15 los árboles con `max_depth = 5` y
# `max_depth = None` resultan idénticos (profundidad real 4), y los de `(2,
# 1)` y `(2, 5)` también coinciden; en el otro extremo, el árbol sin
# restricciones `(None, 1)` alcanza ROC-AUC y F1 de entrenamiento iguales a 1,
# señal de que memoriza el entrenamiento.

# %% [markdown]
# ### b. Selección del árbol
#
# Regla: mayor ROC-AUC de validación; en empate, menor profundidad real,
# menos hojas y, finalmente, el primero en el orden de la cuadrícula (el
# ordenamiento estable de `seleccionar` garantiza esto último).

# %%
exportar_tabla(
    tabla_9a.rename(columns={"F1 ent. (t=0.5)": "F1 ent.", "F1 val. (t=0.5)": "F1 val.", "Profundidad real": "Prof. real"}),
    "ex9_arboles", indice=False,
)
libre = tabla_9a.iloc[9]  # max_depth=None, min_samples_leaf=1
registrar("ArbolLibreAUCTr", libre["ROC-AUC ent."])
registrar("ArbolLibreAUCVal", libre["ROC-AUC val."])
registrar("ArbolLibreFUnoTr", libre["F1 ent. (t=0.5)"])
registrar("ArbolLibreFUnoVal", libre["F1 val. (t=0.5)"])

sel_arbol_clf = seleccionar(
    tabla_9a, [("ROC-AUC val.", False), ("Profundidad real", True), ("Hojas", True)]
)
arbol_clf = arboles_clf[sel_arbol_clf.name]
empate_arb = np.isclose(tabla_9a["ROC-AUC val."], sel_arbol_clf["ROC-AUC val."], rtol=0, atol=1e-10)
print(f"Árbol seleccionado: fila {sel_arbol_clf.name} — max_depth={sel_arbol_clf['max_depth']}, "
      f"min_samples_leaf={sel_arbol_clf['min_samples_leaf']} "
      f"(ROC-AUC validación = {sel_arbol_clf['ROC-AUC val.']:.4f}, "
      f"profundidad real = {sel_arbol_clf['Profundidad real']}, hojas = {sel_arbol_clf['Hojas']})")
print(f"Filas empatadas en el ROC-AUC máximo: {tabla_9a.index[empate_arb].tolist()}")

registrar("ArbolClfMaxDepth", str(sel_arbol_clf["max_depth"]))
registrar("ArbolClfMinLeaf", int(sel_arbol_clf["min_samples_leaf"]))
registrar("ArbolClfProf", int(sel_arbol_clf["Profundidad real"]))
registrar("ArbolClfHojas", int(sel_arbol_clf["Hojas"]))
registrar("ArbolClfAUCTr", sel_arbol_clf["ROC-AUC ent."])
registrar("ArbolClfAUCVal", sel_arbol_clf["ROC-AUC val."])
registrar("ArbolClfFUnoTr", sel_arbol_clf["F1 ent. (t=0.5)"])
registrar("ArbolClfFUnoVal", sel_arbol_clf["F1 val. (t=0.5)"])
_ = registrar("ArbolClfEmpatados", int(empate_arb.sum()))

# %% [markdown]
# El árbol seleccionado es el de **`max_depth = 3` y `min_samples_leaf =
# 15`**, con el mayor ROC-AUC de validación (0.9745), profundidad real 3 y 5
# hojas; su máximo es único, por lo que no se aplicaron desempates. Le siguen
# de cerca los árboles con `min_samples_leaf = 15` y profundidad real 4
# (0.9735), mientras que los árboles con hojas de una sola muestra y
# profundidad real de 3 o más quedan entre los peores en validación (0.8898 a
# 0.9270) pese a tener los mejores valores de entrenamiento.

# %% [markdown]
# ### c. Umbral del árbol y candidato final de clasificación
#
# Mismos siete umbrales y mismos desempates del ejercicio 8 (menor $L$,
# mayor recall, menor umbral). Después se elige el candidato final de
# clasificación por menor costo; en empate, mayor ROC-AUC y después
# regresión logística.

# %%
p_val_arbol = arbol_clf.predict_proba(Xc_val)[:, 1]
tabla_9c = tabla_umbrales(yc_val, p_val_arbol)
print("Tabla 9.2 — Umbrales para el árbol seleccionado (validación)")
display(tabla_9c)
print("Probabilidades distintas que asigna el árbol en validación:", np.unique(p_val_arbol).round(4).tolist())

exportar_tabla(tabla_9c.assign(Umbral=tabla_9c["Umbral"].map(lambda t: f"{t:.1f}")), "ex9_umbrales", indice=False)

sel_t_arbol = seleccionar(tabla_9c, [("Costo L", True), ("Recall", False), ("Umbral", True)])
t_arbol = float(sel_t_arbol["Umbral"])
print(f"Umbral seleccionado para el árbol (congelado): t = {t_arbol} con L = {int(sel_t_arbol['Costo L'])}")

# %%
comparacion_clf = pd.DataFrame([
    {"Modelo": f"Regresión logística (C = {C_sel:g})", "Umbral": t_logit, "Costo L": int(sel_t_logit["Costo L"]),
     "ROC-AUC val.": sel_C["ROC-AUC val."], "Orden": 0},
    {"Modelo": f"Árbol (max_depth={sel_arbol_clf['max_depth']}, min_samples_leaf={sel_arbol_clf['min_samples_leaf']})",
     "Umbral": t_arbol, "Costo L": int(sel_t_arbol["Costo L"]),
     "ROC-AUC val.": sel_arbol_clf["ROC-AUC val."], "Orden": 1},
])
print("Tabla 9.3 — Comparación de candidatos de clasificación (validación)")
display(comparacion_clf.set_index("Modelo").drop(columns="Orden"))

candidato_clf = seleccionar(comparacion_clf, [("Costo L", True), ("ROC-AUC val.", False), ("Orden", True)])
print(f"Candidato final de clasificación (congelado): {candidato_clf['Modelo']} "
      f"con t = {candidato_clf['Umbral']}")

registrar("ArbolT", f"{t_arbol:.1f}")
registrar("ArbolCostoSel", int(sel_t_arbol["Costo L"]))
registrar("ArbolFPSel", int(sel_t_arbol["FP"]))
_ = registrar("ArbolFNSel", int(sel_t_arbol["FN"]))

# %% [markdown]
# El árbol solo asigna cinco probabilidades distintas (las proporciones de
# malignos de sus hojas: 0.0054, 0.2, 0.5714, 0.6364 y 1), de modo que todos
# los umbrales comprendidos entre dos de esos valores producen las mismas
# predicciones (por eso coinciden las filas de 0.3 a 0.5 y las de 0.7 y 0.8).
# El menor costo se obtiene con **$t = 0.2$** ($L = 18$, con 13 falsos
# positivos y 1 falso negativo), donde la regla $p \ge t$ incluye la hoja con
# probabilidad exactamente 0.2 y gana un maligno a cambio de cuatro falsos
# positivos adicionales frente a $t = 0.3$. Este costo es claramente mayor que
# el de la regresión logística ($L = 5$), por lo que el candidato final de
# clasificación es la **regresión logística con $C = 1$ y $t = 0.7$**, sin
# necesidad de desempates; además su ROC-AUC de validación también es mayor
# (0.9957 frente a 0.9745).

# %% [markdown]
# ### Puente con el ejercicio 3d: umbral teórico y calibración (validación)
#
# En el ejercicio 3d se demostró que, con probabilidades bien calibradas, el
# umbral que minimiza el costo esperado es $t^* = c_{FP}/(c_{FP} + c_{FN})$,
# que para $L = FP + 5FN$ vale $t^* = 1/(1+5) \approx 0.1667$. Aquí se
# contrasta esa predicción teórica con los umbrales elegidos empíricamente y
# se pone a prueba el supuesto de calibración. Este análisis es solo
# descriptivo, pues el enunciado fija la rejilla de siete umbrales, de modo
# que $t^*$ no se usa para seleccionar ni cambia ningún candidato congelado.

# %%
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

T_TEORICO = 1 / (1 + 5)
puente = []
curvas_cal = {}
for nombre, proba, t_emp in [("Regresión logística", p_val_logit, t_logit), ("Árbol", p_val_arbol, t_arbol)]:
    m_emp, m_teo = metricas_clf(yc_val, proba, t_emp), metricas_clf(yc_val, proba, T_TEORICO)
    puente.append({
        "Modelo": nombre, "Brier": brier_score_loss(yc_val, proba),
        "t elegido": t_emp, "L (t elegido)": m_emp["Costo L"],
        "L (t* = 1/6)": m_teo["Costo L"], "FP (t*)": m_teo["FP"], "FN (t*)": m_teo["FN"],
    })
    curvas_cal[nombre] = calibration_curve(yc_val, proba, n_bins=5, strategy="quantile")
tabla_puente = pd.DataFrame(puente).set_index("Modelo")
print("Tabla 9.4 — Umbral teórico t* = 1/6 frente al umbral elegido, y puntaje de Brier (validación)")
display(tabla_puente)
exportar_tabla(tabla_puente.assign(**{"t elegido": tabla_puente["t elegido"].map(lambda t: f"{t:.1f}")}), "ex9_puente")

registrar("TTeorico", T_TEORICO)
for nombre, clave in [("Regresión logística", "Logit"), ("Árbol", "Arbol")]:
    registrar(f"{clave}Brier", tabla_puente.loc[nombre, "Brier"])
    registrar(f"{clave}CostoTeo", int(tabla_puente.loc[nombre, "L (t* = 1/6)"]))
    registrar(f"{clave}FPTeo", int(tabla_puente.loc[nombre, "FP (t*)"]))
    _ = registrar(f"{clave}FNTeo", int(tabla_puente.loc[nombre, "FN (t*)"]))

# Grupo de la curva de calibración más cercano a t* y menor probabilidad asignada a un maligno
frac_lg, media_lg = curvas_cal["Regresión logística"]
i_cerca = int(np.argmin(np.abs(media_lg - T_TEORICO)))
registrar("LogitCalMedia", media_lg[i_cerca])
registrar("LogitCalFrac", frac_lg[i_cerca])
_ = registrar("LogitPMalMin", p_val_logit[yc_val.to_numpy() == 1].min())

# %%
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0, 1], [0, 1], color=COLORES["gris"], linestyle=":", label="Calibración perfecta")
for (nombre, (frac, media)), color, marca in zip(curvas_cal.items(), [COLORES["azul"], COLORES["verde"]], ["o", "s"]):
    ax.plot(media, frac, marker=marca, color=color, label=nombre)
ax.axvline(T_TEORICO, color=COLORES["rojo"], linestyle="--", label="t* = 1/6 (ejercicio 3d)")
ax.set_title("Figura 9.1 — Curva de calibración (validación,\n5 grupos por cuantiles)")
ax.set_xlabel("Probabilidad predicha media del grupo")
ax.set_ylabel("Fracción observada de malignos")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
guardar_fig(fig, "ex9_calibracion")
plt.show()

# %% [markdown]
# En el árbol la teoría y la selección empírica coinciden, pues con
# $t^* = 1/6$ se obtiene exactamente el mismo resultado que con $t = 0.2$
# ($L = 18$), ya que ninguna hoja tiene una probabilidad comprendida entre
# 0.1667 y 0.2 y la regla $p \ge t$ produce las mismas predicciones (el umbral
# elegido es, de hecho, el valor de la rejilla más cercano a $t^*$). En la
# regresión logística, en cambio, aplicar $t^*$ daría $L = 15$ (10 falsos
# positivos y 1 falso negativo), el triple del costo del umbral elegido
# ($t = 0.7$, $L = 5$), y esta discrepancia se explica por las dos condiciones
# que exige el resultado del ejercicio 3d. Primeramente, la calibración no es
# perfecta, pues en el grupo de probabilidades bajas (media predicha 0.1079)
# la fracción observada de malignos es solo 0.0455, es decir, el modelo
# sobreestima el riesgo justo en la zona donde cae $t^*$, y por eso bajar el
# umbral hasta allí agrega falsos positivos que en realidad son benignos.
# Progresivamente se suma un hecho puntual de esta validación, el único
# maligno no detectado tiene $p = 0.1094$, por debajo incluso de $t^*$, de
# modo que ningún umbral de la rejilla (ni el teórico) lo recupera y bajar el
# umbral solo cuesta falsos positivos. Finalmente, el puntaje de Brier
# (0.0201 frente a 0.0528 del árbol) confirma que la logística entrega
# probabilidades más precisas, aunque con 114 muestras y cinco grupos de unas
# 23 observaciones cada punto de la curva tiene un error de muestreo
# considerable. La lectura de conjunto es que $t^*$ es una referencia teórica
# valiosa cuyo supuesto clave (probabilidades calibradas) debe verificarse
# antes de aplicarla, y que el umbral alto de la logística descansa en muy
# pocos casos, advertencia que se retoma al evaluar en prueba.

# %% [markdown]
# ### d. Visualización y regla de una hoja

# %%
fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(
    arbol_clf, max_depth=3, feature_names=list(Xc_train.columns),
    class_names=["benigno (0)", "maligno (1)"], filled=True, rounded=True,
    impurity=True, proportion=False, fontsize=8, ax=ax,
)
truncado = arbol_clf.get_depth() > 3
ax.set_title(
    "Figura 9.2 — Árbol de clasificación seleccionado (entrenamiento)"
    + (f"\nVisualización truncada a 3 niveles; profundidad real = {arbol_clf.get_depth()}" if truncado else "")
)
guardar_fig(fig, "ex9_arbol")
plt.show()

# %%
def regla_hasta_hoja(arbol, hoja, nombres):
    """Reconstruye la ruta raíz → hoja como lista de condiciones."""
    t = arbol.tree_
    padre = {}
    for nodo in range(t.node_count):
        for hijo, lado in [(t.children_left[nodo], "<="), (t.children_right[nodo], ">")]:
            if hijo != -1:
                padre[hijo] = (nodo, lado)
    condiciones, nodo = [], hoja
    while nodo in padre:
        p, lado = padre[nodo]
        condiciones.append(f"{nombres[t.feature[p]]} {lado} {t.threshold[p]:.4f}")
        nodo = p
    return condiciones[::-1]


t_ = arbol_clf.tree_
hojas = np.where(t_.children_left == -1)[0]
# Proporción de malignos por hoja contada directamente en entrenamiento
# (independiente de cómo guarde `tree_.value` cada versión de scikit-learn).
hoja_train = arbol_clf.apply(Xc_train)
info_hojas = pd.DataFrame({
    "Muestras": [int((hoja_train == h).sum()) for h in hojas],
    "Proporción de malignos": [yc_train[hoja_train == h].mean() for h in hojas],
}, index=hojas)
info_hojas.index.name = "Hoja"
print("Tabla 9.4 — Hojas del árbol seleccionado (entrenamiento)")
display(info_hojas.sort_values("Muestras", ascending=False))

# Se ilustra la hoja con más muestras de entrenamiento clasificadas como malignas.
hoja_ej = info_hojas[info_hojas["Proporción de malignos"] >= 0.5]["Muestras"].idxmax()
condiciones = regla_hasta_hoja(arbol_clf, hoja_ej, list(Xc_train.columns))
regla = "\n".join(f"{'    ' * i}if {c}:" for i, c in enumerate(condiciones))
regla += f"\n{'    ' * len(condiciones)}predecir maligno  # p = {info_hojas.loc[hoja_ej, 'Proporción de malignos']:.4f}"
print(f"Ruta hasta la hoja {hoja_ej} "
      f"({info_hojas.loc[hoja_ej, 'Muestras']} muestras de entrenamiento, "
      f"proporción de malignos = {info_hojas.loc[hoja_ej, 'Proporción de malignos']:.4f}):\n")
print(regla)

registrar("HojaEjMuestras", int(info_hojas.loc[hoja_ej, "Muestras"]))
_ = registrar("HojaEjProp", info_hojas.loc[hoja_ej, "Proporción de malignos"])
(MACROS_PATH.parent / "regla_ex9.tex").write_text(
    "% Archivo generado automáticamente por el notebook (no editar a mano).\n"
    "\\begin{verbatim}\n" + regla + "\n\\end{verbatim}\n", encoding="utf-8")

# %% [markdown]
# El árbol seleccionado tiene profundidad real 3, por lo que la Figura 9.2 lo
# muestra completo y no está truncado. La raíz divide por `worst perimeter` y
# el segundo nivel por `worst concave points` en ambas ramas, y solo la rama
# izquierda usa `area error` en el tercer nivel. La ruta impresa arriba lleva
# a la hoja con más muestras malignas, en forma de regla: si `worst perimeter`
# > 112.8 y además `worst concave points` > 0.1396, se predice maligno; esta
# hoja contiene 97 muestras de entrenamiento, todas malignas (proporción
# 1.0000).

# %% [markdown]
# ### e. Importancias, coeficientes y sobreajuste

# %%
importancias = pd.Series(arbol_clf.feature_importances_, index=Xc_train.columns).sort_values(ascending=False)
top5 = importancias.head(5)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.barh(top5.index[::-1], top5.values[::-1], color=COLORES["verde"], label="Importancia (reducción de Gini)")
ax.set_title("Figura 9.3 — Cinco mayores importancias del árbol seleccionado\n(entrenamiento)")
ax.set_xlabel("feature_importances_ (suman 1)")
ax.set_ylabel("Predictor")
ax.legend(loc="lower right")
plt.tight_layout()
guardar_fig(fig, "ex9_importancias")
plt.show()

registrar("ImpPerimetro", importancias["worst perimeter"])
registrar("ImpPuntos", importancias["worst concave points"])
registrar("ImpAreaError", importancias["area error"])
registrar("ImpNoNulas", int((importancias > 0).sum()))
comparacion_imp = pd.DataFrame({
    "Importancia árbol": top5,
    "Coef. logística": coefs_logit.loc[top5.index, "Coeficiente"],
    "Rango |coef.| logística": coefs_logit["|Coeficiente|"].rank(ascending=False).loc[top5.index].astype(int),
})
print(f"Predictores con importancia > 0: {(importancias > 0).sum()} de {len(importancias)}")
registrar("RangoPerimetro", int(comparacion_imp.loc["worst perimeter", "Rango |coef.| logística"]))
registrar("RangoPuntos", int(comparacion_imp.loc["worst concave points", "Rango |coef.| logística"]))
exportar_tabla(comparacion_imp.rename(columns={"Rango |coef.| logística": "Rango $|\\beta|$ logística"})
               .rename_axis("Predictor"), "ex9_importancias")
print("Tabla 9.5 — Importancias del árbol frente a coeficientes de la logística")
comparacion_imp

# %%
sobreajuste = tabla_9a.assign(
    **{"Brecha AUC (ent − val)": tabla_9a["ROC-AUC ent."] - tabla_9a["ROC-AUC val."],
       "Brecha F1 (ent − val)": tabla_9a["F1 ent. (t=0.5)"] - tabla_9a["F1 val. (t=0.5)"]}
)[["max_depth", "min_samples_leaf", "Profundidad real", "Hojas",
   "ROC-AUC ent.", "ROC-AUC val.", "Brecha AUC (ent − val)", "Brecha F1 (ent − val)"]]
print("Tabla 9.6 — Brechas entre entrenamiento y validación en la cuadrícula de árboles")
sobreajuste

# %% [markdown]
# El árbol concentra casi toda la importancia en `worst perimeter` (0.8686) y
# `worst concave points` (0.1234), con un aporte marginal de `area error`
# (0.0080), y las 27 variables restantes tienen importancia cero (las dos
# últimas barras de la figura son nulas). La regresión logística, en cambio,
# reparte el peso entre muchas variables, de forma que `worst perimeter`
# aparece apenas en el puesto 11 por magnitud de coeficiente y `worst concave
# points` en el 15. Las escalas no son comparables porque la importancia es la
# reducción total de Gini atribuida a cada variable, normalizada para sumar 1,
# siempre no negativa y sin signo, que además depende de qué variable eligió
# primero el árbol entre varias casi intercambiables (perímetro, radio y
# área), mientras que el coeficiente logístico es un cambio condicional en
# log-odds por desviación estándar, con signo; ninguna de las dos implica
# efectos causales.
#
# Sobre el sobreajuste, la Tabla 9.6 muestra evidencia clara en los árboles
# con `min_samples_leaf = 1`, como `(None, 1)` y `(3, 1)`, con brechas de
# ROC-AUC de 0.0958 y 0.0995 y la mayor brecha de F1 (0.1236 en el árbol sin
# restricciones), mientras que los árboles con `min_samples_leaf = 15`,
# incluido el seleccionado (brecha de 0.0110), son los que mejor trasladan su
# desempeño a validación.
