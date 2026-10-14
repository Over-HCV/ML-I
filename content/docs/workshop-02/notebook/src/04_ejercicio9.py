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
# TODO: análisis 9a

# %% [markdown]
# ### b. Selección del árbol
#
# Regla: mayor ROC-AUC de validación; en empate, menor profundidad real,
# menos hojas y, finalmente, el primero en el orden de la cuadrícula (el
# ordenamiento estable de `seleccionar` garantiza esto último).

# %%
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
# TODO: análisis 9b

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
# TODO: análisis 9c

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
    "Figura 9.1 — Árbol de clasificación seleccionado (entrenamiento)"
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

# %% [markdown]
# TODO: análisis 9d

# %% [markdown]
# ### e. Importancias, coeficientes y sobreajuste

# %%
importancias = pd.Series(arbol_clf.feature_importances_, index=Xc_train.columns).sort_values(ascending=False)
top5 = importancias.head(5)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.barh(top5.index[::-1], top5.values[::-1], color=COLORES["verde"], label="Importancia (reducción de Gini)")
ax.set_title("Figura 9.2 — Cinco mayores importancias del árbol seleccionado\n(entrenamiento)")
ax.set_xlabel("feature_importances_ (suman 1)")
ax.set_ylabel("Predictor")
ax.legend(loc="lower right")
plt.tight_layout()
guardar_fig(fig, "ex9_importancias")
plt.show()

comparacion_imp = pd.DataFrame({
    "Importancia árbol": top5,
    "Coef. logística": coefs_logit.loc[top5.index, "Coeficiente"],
    "Rango |coef.| logística": coefs_logit["|Coeficiente|"].rank(ascending=False).loc[top5.index].astype(int),
})
print(f"Predictores con importancia > 0: {(importancias > 0).sum()} de {len(importancias)}")
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
# TODO: análisis 9e
