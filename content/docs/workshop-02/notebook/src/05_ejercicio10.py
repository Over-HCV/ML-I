# %% [markdown]
# ## Ejercicio 10 — Evaluación final y recomendación sustentada

# %% [markdown]
# ### Candidatos congelados antes de abrir prueba
#
# Hasta este punto ninguna celda ha tocado `Xr_test`, `yr_test`, `Xc_test` ni
# `yc_test`. Se registran primero los candidatos con sus predictores,
# hiperparámetros y umbrales, y solo después se calcula cualquier métrica de
# prueba; los modelos se usan tal como quedaron ajustados con entrenamiento.

# %%
candidatos = pd.DataFrame([
    {"Caso": "A (regresión)", "Candidato final": candidato_reg["Modelo"],
     "Predictores": f"{len(cols_reg_final)} ({', '.join(cols_reg_final) if len(cols_reg_final) < 4 else 'todos'})",
     "Hiperparámetros": "StandardScaler + LinearRegression" if modelo_reg_final is pipe_multi
        else ("LinearRegression" if modelo_reg_final is lr_simple
              else f"max_depth={arbol_reg.max_depth}, min_samples_leaf={arbol_reg.min_samples_leaf}"),
     "Umbral": "—", "Criterio de validación": f"RMSE = {candidato_reg['RMSE val.']:.4f}"},
    {"Caso": "B (clasificación)", "Candidato final": candidato_clf["Modelo"],
     "Predictores": f"{Xc_train.shape[1]} (todos)",
     "Hiperparámetros": (f"StandardScaler + LogisticRegression(C={C_sel:g}, lbfgs, L2)"
                         if candidato_clf["Orden"] == 0 else
                         f"max_depth={sel_arbol_clf['max_depth']}, min_samples_leaf={sel_arbol_clf['min_samples_leaf']}"),
     "Umbral": f"{candidato_clf['Umbral']:.1f}", "Criterio de validación": f"L = {int(candidato_clf['Costo L'])}"},
]).set_index("Caso")
print("Tabla 10.1 — Candidatos finales congelados (decididos con validación)")
candidatos

# %% [markdown]
# ### a. Regresión en prueba

# %%
modelos_reg_prueba = [
    ("Constante (media)", dummy_reg, list(Xr_train.columns)),
    ("Regresión simple (bmi)", lr_simple, ["bmi"]),
    ("Regresión múltiple (10 predictores)", pipe_multi, list(Xr_train.columns)),
    (tabla_7d.loc[2, "Modelo"], arbol_reg, list(Xr_train.columns)),
]
tabla_10a = pd.DataFrame(
    [{"Modelo": n, **metricas_reg(yr_test, m.predict(Xr_test[c]))} for n, m, c in modelos_reg_prueba]
).set_index("Modelo")
tabla_10a["RMSE validación"] = [rmse(yr_val, m.predict(Xr_val[c])) for _, m, c in modelos_reg_prueba]
tabla_10a["Candidato elegido"] = ["✓" if n == candidato_reg["Modelo"] else "" for n in tabla_10a.index]
print("Tabla 10.2 — Modelos de regresión en prueba (RMSE de validación como referencia)")
tabla_10a

# %%
claves_reg = {"Constante (media)": "Dummy", "Regresión simple (bmi)": "Simple",
              "Regresión múltiple (10 predictores)": "Multi", tabla_7d.loc[2, "Modelo"]: "ArbolReg"}
for nombre, clave in claves_reg.items():
    for met in ["MAE", "RMSE", "R2"]:
        registrar(f"Test{clave}{met.replace('2', 'dos')}", tabla_10a.loc[nombre, met])
rmse_val_cand, rmse_test_cand = candidato_reg["RMSE val."], tabla_10a.loc[candidato_reg["Modelo"], "RMSE"]
print(f"Candidato elegido: RMSE validación = {rmse_val_cand:.4f} · RMSE prueba = {rmse_test_cand:.4f} "
      f"(diferencia = {rmse_test_cand - rmse_val_cand:+.4f})")
_ = registrar("CandidatoRegRMSETest", rmse_test_cand)

# %% [markdown]
# TODO: análisis 10a

# %% [markdown]
# ### b. Clasificación en prueba

# %%
modelos_clf_prueba = [
    ("Referencia (prior)", dummy_clf, 0.5, "Dummy"),
    (f"Regresión logística (C = {C_sel:g})", logit_sel, t_logit, "Logit"),
    (comparacion_clf.loc[1, "Modelo"], arbol_clf, t_arbol, "ArbolClf"),
]
tabla_10b = pd.DataFrame(
    [{"Modelo": n, **metricas_clf(yc_test, m.predict_proba(Xc_test)[:, 1], t)} for n, m, t, _ in modelos_clf_prueba]
).set_index("Modelo")
print("Tabla 10.3 — Clasificación en prueba: métricas con los umbrales congelados")
display(tabla_10b[["Umbral", "Precisión", "Recall", "F1", "ROC-AUC"]])
print("Tabla 10.4 — Clasificación en prueba: errores y costo L = FP + 5FN")
display(tabla_10b[["Umbral", "TN", "FP", "FN", "TP", "Costo L"]])

for nombre, _, _, clave in modelos_clf_prueba:
    fila = tabla_10b.loc[nombre]
    for met, etiqueta in [("Precisión", "Prec"), ("Recall", "Recall"), ("F1", "FUno"), ("ROC-AUC", "AUC")]:
        registrar(f"Test{clave}{etiqueta}", fila[met])
    for met in ["FP", "FN", "TP", "TN"]:
        registrar(f"Test{clave}{met}", int(fila[met]))
    registrar(f"Test{clave}Costo", int(fila["Costo L"]))

# %% [markdown]
# TODO: análisis 10b

# %% [markdown]
# ### c. Matrices de confusión y costo

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (nombre, modelo, t, _) in zip(axes, modelos_clf_prueba[1:]):
    cm = confusion_matrix(yc_test, predecir_umbral(modelo.predict_proba(Xc_test)[:, 1], t), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["0 (benigno)", "1 (maligno)"])
    ax.set_yticks([0, 1], ["0 (benigno)", "1 (maligno)"])
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"{nombre}\nt = {t}")
fig.suptitle("Figura 10.1 — Matrices de confusión en prueba (orden de etiquetas [0, 1])")
plt.tight_layout()
guardar_fig(fig, "ex10_matrices_confusion")
plt.show()

# %%
fila_cand = tabla_10b.loc[candidato_clf["Modelo"]]
fp, fn = int(fila_cand["FP"]), int(fila_cand["FN"])
print(f"Candidato elegido: {candidato_clf['Modelo']} (t = {candidato_clf['Umbral']})")
print(f"Verificación del costo: L = FP + 5·FN = {fp} + 5·{fn} = {fp + 5 * fn}")
assert fp + 5 * fn == int(fila_cand["Costo L"])

# %% [markdown]
# TODO: análisis 10c

# %% [markdown]
# ### d. Recomendación
#
# El texto se guarda en una variable para contar sus palabras en la misma
# celda y comprobar que está entre 250 y 350.

# %%
from IPython.display import Markdown

RECOMENDACION = """TODO"""

n_palabras = len(RECOMENDACION.split())
assert 250 <= n_palabras <= 350, f"La recomendación tiene {n_palabras} palabras"
display(Markdown(RECOMENDACION))
print(f"Número de palabras: {n_palabras}")

# %% [markdown]
# ### Cifras para el documento LaTeX

# %%
escribir_macros()
