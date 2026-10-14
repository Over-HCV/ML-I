# %% [markdown]
# ## Ejercicio 8 — Caso B: regresión logística y elección del umbral

# %% [markdown]
# ### a. Particiones, recodificación y referencia

# %%
import warnings

from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

print("Tabla 8.1 — Recodificación de la clase positiva (conjunto completo)")
display(pd.crosstab(
    pd.Series(cancer.target_names[cancer.target], name="Etiqueta original"),
    pd.Series(y_maligno.to_numpy(), name="y recodificada (1 = maligno)"),
))

prop = pd.DataFrame({
    "n": [len(yc_train), len(yc_val)],
    "Malignos": [int(yc_train.sum()), int(yc_val.sum())],
    "Proporción de malignos": [yc_train.mean(), yc_val.mean()],
}, index=["Entrenamiento", "Validación"])
print("Tabla 8.2 — Proporción de la clase positiva por partición")
display(prop)
registrar("PropMalTr", yc_train.mean())
_ = registrar("PropMalVal", yc_val.mean())

# %%
dummy_clf = DummyClassifier(strategy="prior").fit(Xc_train, yc_train)
filas = []
for particion, X_, y_ in [("Entrenamiento", Xc_train, yc_train), ("Validación", Xc_val, yc_val)]:
    filas.append({"Partición": particion, **metricas_clf(y_, dummy_clf.predict_proba(X_)[:, 1], 0.5)})
tabla_8a = pd.DataFrame(filas).set_index("Partición")
print(f"Probabilidad constante de maligno que asigna la referencia: {dummy_clf.predict_proba(Xc_val[:1])[0, 1]:.4f}")
print("Tabla 8.3 — Clasificador de referencia (prior), umbral 0.5")
tabla_8a

# %% [markdown]
# TODO: análisis 8a

# %% [markdown]
# ### b. Selección de $C$
#
# Regla: mayor ROC-AUC de validación; en empate, menor $C$. La convergencia
# se verifica de dos formas, capturando cualquier `ConvergenceWarning` y
# comprobando que el número de iteraciones usadas sea menor que `max_iter`.

# %%
filas, logits = [], {}
for C in [0.01, 0.1, 1, 10]:
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        logit = Pipeline([
            ("escalador", StandardScaler()),
            ("logistica", LogisticRegression(C=C, solver="lbfgs", max_iter=5000)),
        ]).fit(Xc_train, yc_train)
    logits[C] = logit
    p_tr = logit.predict_proba(Xc_train)[:, 1]
    p_val = logit.predict_proba(Xc_val)[:, 1]
    n_iter = int(logit.named_steps["logistica"].n_iter_[0])
    filas.append({
        "C": C,
        "ROC-AUC ent.": roc_auc_score(yc_train, p_tr),
        "ROC-AUC val.": roc_auc_score(yc_val, p_val),
        "F1 ent. (t=0.5)": f1_score(yc_train, predecir_umbral(p_tr, 0.5), zero_division=0),
        "F1 val. (t=0.5)": f1_score(yc_val, predecir_umbral(p_val, 0.5), zero_division=0),
        "Iteraciones": n_iter,
        "Convergió": n_iter < 5000 and not any(issubclass(a.category, ConvergenceWarning) for a in avisos),
    })
tabla_8b = pd.DataFrame(filas)
print("Tabla 8.4 — Regresión logística según C (entrenamiento y validación)")
tabla_8b

# %%
sel_C = seleccionar(tabla_8b, [("ROC-AUC val.", False), ("C", True)])
C_sel = float(sel_C["C"])
logit_sel = logits[C_sel]
assert tabla_8b["Convergió"].all(), "Algún ajuste no convergió"
empate_C = np.isclose(tabla_8b["ROC-AUC val."], sel_C["ROC-AUC val."], rtol=0, atol=1e-10)
print("ROC-AUC de validación a precisión completa:",
      {f"C={c:g}": f"{v:.10f}" for c, v in zip(tabla_8b["C"], tabla_8b["ROC-AUC val."])})
print(f"Valores de C empatados en el máximo: {tabla_8b.loc[empate_C, 'C'].tolist()}")
print(f"C seleccionado: {C_sel:g} (ROC-AUC validación = {sel_C['ROC-AUC val.']:.4f})")
registrar("LogitEmpatesC", int(empate_C.sum()))
registrar("LogitC", f"{C_sel:g}")
registrar("LogitAUCVal", sel_C["ROC-AUC val."])
registrar("LogitAUCTr", sel_C["ROC-AUC ent."])
registrar("LogitFUnoVal", sel_C["F1 val. (t=0.5)"])
_ = registrar("LogitIterMax", int(tabla_8b["Iteraciones"].max()))

# %% [markdown]
# TODO: análisis 8b

# %% [markdown]
# ### c. Umbral por costo $L = FP + 5FN$
#
# El modelo queda fijo; solo se cambia la regla $\hat y = \mathbb{1}[p \ge t]$
# sobre validación. Regla: menor $L$; en empate, mayor recall y después menor
# umbral.

# %%
UMBRALES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def tabla_umbrales(y, proba):
    cols = ["Umbral", "Precisión", "Recall", "F1", "FP", "FN", "Costo L"]
    return pd.DataFrame([metricas_clf(y, proba, t) for t in UMBRALES])[cols]


p_val_logit = logit_sel.predict_proba(Xc_val)[:, 1]
tabla_8c = tabla_umbrales(yc_val, p_val_logit)
print(f"Tabla 8.5 — Umbrales para la regresión logística con C = {C_sel:g} (validación)")
tabla_8c

# %%
sel_t_logit = seleccionar(tabla_8c, [("Costo L", True), ("Recall", False), ("Umbral", True)])
t_logit = float(sel_t_logit["Umbral"])
fila_05 = tabla_8c.set_index("Umbral").loc[0.5]
print(f"Umbral seleccionado (congelado): t = {t_logit} con L = {int(sel_t_logit['Costo L'])} "
      f"(FP = {int(sel_t_logit['FP'])}, FN = {int(sel_t_logit['FN'])})")
print(f"Con t = 0.5: L = {int(fila_05['Costo L'])} (FP = {int(fila_05['FP'])}, FN = {int(fila_05['FN'])})")
print("Umbrales con el costo mínimo:", tabla_8c.loc[tabla_8c["Costo L"] == sel_t_logit["Costo L"], "Umbral"].tolist())

registrar("LogitT", f"{t_logit:.1f}")
for nombre, fila in [("Sel", sel_t_logit), ("Medio", fila_05)]:
    registrar(f"LogitCosto{nombre}", int(fila["Costo L"]))
    registrar(f"LogitFP{nombre}", int(fila["FP"]))
    registrar(f"LogitFN{nombre}", int(fila["FN"]))
    registrar(f"LogitRecall{nombre}", fila["Recall"])
    _ = registrar(f"LogitPrec{nombre}", fila["Precisión"])

# %% [markdown]
# TODO: análisis 8c

# %% [markdown]
# ### d. Precisión y recall frente al umbral

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tabla_8c["Umbral"], tabla_8c["Precisión"], marker="o", color=COLORES["azul"], label="Precisión")
ax.plot(tabla_8c["Umbral"], tabla_8c["Recall"], marker="s", linestyle="--", color=COLORES["rojo"], label="Recall")
ax.axvline(t_logit, color=COLORES["gris"], linestyle=":", label=f"Umbral elegido (t = {t_logit})")
ax.set_title(f"Figura 8.1 — Precisión y recall según el umbral\n(regresión logística, C = {C_sel:g}, validación)")
ax.set_xlabel("Umbral t")
ax.set_ylabel("Valor de la métrica")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
plt.tight_layout()
guardar_fig(fig, "ex8_precision_recall")
plt.show()

print(f"ROC-AUC de validación (no depende de t): {roc_auc_score(yc_val, p_val_logit):.4f}")

# %% [markdown]
# TODO: análisis 8d

# %% [markdown]
# ### e. Coeficientes estandarizados y odds

# %%
coefs_logit = pd.DataFrame(
    {"Coeficiente": logit_sel.named_steps["logistica"].coef_[0]}, index=Xc_train.columns
)
coefs_logit["exp(β)"] = np.exp(coefs_logit["Coeficiente"])
coefs_logit["|Coeficiente|"] = coefs_logit["Coeficiente"].abs()
coefs_logit = coefs_logit.sort_values("|Coeficiente|", ascending=False)
print(f"Tabla 8.6 — Coeficientes estandarizados de la regresión logística (C = {C_sel:g}), por magnitud absoluta")
coefs_logit

# %% [markdown]
# TODO: análisis 8e
