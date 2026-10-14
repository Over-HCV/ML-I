# %% [markdown]
# ## Ejercicio 7 — Caso A: regresión múltiple y árbol de regresión

# %% [markdown]
# ### a. Regresión múltiple con `Pipeline`
#
# El `StandardScaler` vive dentro del `Pipeline`, de modo que sus medias y
# desviaciones se aprenden solo con entrenamiento y se reutilizan tal cual
# al predecir validación.

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

pipe_multi = Pipeline(
    [("escalador", StandardScaler()), ("regresion", LinearRegression())]
).fit(Xr_train, yr_train)

filas = []
for nombre, modelo, cols in [
    ("Constante (media)", dummy_reg, list(Xr_train.columns)),
    ("Regresión simple (bmi)", lr_simple, ["bmi"]),
    ("Regresión múltiple (10 predictores)", pipe_multi, list(Xr_train.columns)),
]:
    for particion, X_, y_ in [("Entrenamiento", Xr_train, yr_train), ("Validación", Xr_val, yr_val)]:
        filas.append({"Modelo": nombre, "Partición": particion, **metricas_reg(y_, modelo.predict(X_[cols]))})
tabla_7a = pd.DataFrame(filas).set_index(["Modelo", "Partición"])
print("Tabla 7.1 — Constante, regresión simple y múltiple (entrenamiento y validación)")
tabla_7a

# %%
for sufijo, particion in [("Tr", "Entrenamiento"), ("Val", "Validación")]:
    for met in ["MAE", "RMSE", "R2"]:
        registrar(f"RegMulti{met.replace('2', 'dos')}{sufijo}",
                  tabla_7a.loc[("Regresión múltiple (10 predictores)", particion), met])

# %% [markdown]
# TODO: análisis 7a

# %% [markdown]
# ### b. Coeficientes estandarizados y correlaciones
#
# Como los predictores entran estandarizados, cada coeficiente es el cambio
# esperado en la variable objetivo ante un aumento de una desviación
# estándar de entrenamiento en ese predictor, manteniendo fijos los demás.

# %%
escalador = pipe_multi.named_steps["escalador"]
coefs_multi = pd.DataFrame(
    {
        "Coeficiente": pipe_multi.named_steps["regresion"].coef_,
        "Desv. est. entrenamiento": escalador.scale_,
        "Corr. con y (entrenamiento)": Xr_train.corrwith(yr_train).to_numpy(),
    },
    index=Xr_train.columns,
)
coefs_multi["|Coeficiente|"] = coefs_multi["Coeficiente"].abs()
coefs_multi = coefs_multi.sort_values("|Coeficiente|", ascending=False)
print(f"Intercepto (media de y en entrenamiento): {pipe_multi.named_steps['regresion'].intercept_:.4f}")
print("Tabla 7.2 — Coeficientes estandarizados de la regresión múltiple, por magnitud absoluta")
coefs_multi

# %%
for var in ["bmi", "s5", "s1", "s2"]:
    nombre = var.capitalize().replace("1", "uno").replace("2", "dos").replace("5", "cinco")
    registrar(f"CoefMulti{nombre}", coefs_multi.loc[var, "Coeficiente"], 2)
    registrar(f"DesvMulti{nombre}", coefs_multi.loc[var, "Desv. est. entrenamiento"], 2)

# %%
corr_tr = Xr_train.corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr_tr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_tr)), corr_tr.columns, rotation=45)
ax.set_yticks(range(len(corr_tr)), corr_tr.columns)
for i in range(len(corr_tr)):
    for j in range(len(corr_tr)):
        v = corr_tr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 else "black")
fig.colorbar(im, ax=ax, label="Correlación de Pearson")
ax.set_title("Figura 7.1 — Matriz de correlaciones de los predictores\n(entrenamiento)")
plt.tight_layout()
guardar_fig(fig, "ex7_correlaciones")
plt.show()

pares = (
    corr_tr.where(np.triu(np.ones(corr_tr.shape, dtype=bool), k=1))
    .stack().rename("Correlación").to_frame()
)
pares["|Correlación|"] = pares["Correlación"].abs()
print("Tabla 7.3 — Pares de predictores más correlacionados (entrenamiento)")
pares.sort_values("|Correlación|", ascending=False).head(5)

# %%
registrar("CorrSunoSdos", corr_tr.loc["s1", "s2"])
_ = registrar("CorrStresScuatro", corr_tr.loc["s3", "s4"])

# %% [markdown]
# TODO: análisis 7b

# %% [markdown]
# ### c. Cuadrícula de `DecisionTreeRegressor`
#
# Se ajustan los nueve árboles sobre los predictores originales (sin
# escalar). Regla de selección: menor RMSE de validación; en empate, menor
# `max_depth` y después mayor `min_samples_leaf`.

# %%
filas, arboles_reg = [], {}
for max_depth in [2, 4, 6]:
    for min_leaf in [5, 15, 30]:
        arbol = DecisionTreeRegressor(
            criterion="squared_error", max_depth=max_depth,
            min_samples_leaf=min_leaf, random_state=RANDOM_STATE,
        ).fit(Xr_train, yr_train)
        arboles_reg[(max_depth, min_leaf)] = arbol
        filas.append({
            "max_depth": max_depth, "min_samples_leaf": min_leaf,
            "RMSE entrenamiento": rmse(yr_train, arbol.predict(Xr_train)),
            "RMSE validación": rmse(yr_val, arbol.predict(Xr_val)),
            "Profundidad real": arbol.get_depth(), "Hojas": arbol.get_n_leaves(),
        })
tabla_7c = pd.DataFrame(filas)
tabla_7c["Brecha (val − ent)"] = tabla_7c["RMSE validación"] - tabla_7c["RMSE entrenamiento"]
print("Tabla 7.4 — Cuadrícula de árboles de regresión (RMSE en entrenamiento y validación)")
tabla_7c

# %%
sel_arbol_reg = seleccionar(
    tabla_7c, [("RMSE validación", True), ("max_depth", True), ("min_samples_leaf", False)]
)
arbol_reg = arboles_reg[(int(sel_arbol_reg["max_depth"]), int(sel_arbol_reg["min_samples_leaf"]))]
n_empatados = int(np.isclose(tabla_7c["RMSE validación"], sel_arbol_reg["RMSE validación"], rtol=0, atol=1e-10).sum())
print(f"Árbol seleccionado: max_depth={int(sel_arbol_reg['max_depth'])}, "
      f"min_samples_leaf={int(sel_arbol_reg['min_samples_leaf'])} "
      f"(RMSE validación = {sel_arbol_reg['RMSE validación']:.4f}; "
      f"árboles con ese mismo RMSE: {n_empatados})")

registrar("ArbolRegMaxDepth", int(sel_arbol_reg["max_depth"]))
registrar("ArbolRegMinLeaf", int(sel_arbol_reg["min_samples_leaf"]))
registrar("ArbolRegProf", int(sel_arbol_reg["Profundidad real"]))
registrar("ArbolRegHojas", int(sel_arbol_reg["Hojas"]))
registrar("ArbolRegRMSETr", sel_arbol_reg["RMSE entrenamiento"])
registrar("ArbolRegRMSEVal", sel_arbol_reg["RMSE validación"])
_ = registrar("ArbolRegEmpatados", n_empatados)

# %% [markdown]
# TODO: análisis 7c

# %% [markdown]
# ### d. Candidato final de regresión
#
# Se elige por menor RMSE de validación; en empate numérico se prefiere el
# modelo simple, luego el múltiple y después el árbol (columna `Orden`).

# %%
filas = []
for orden, (nombre, modelo, cols) in enumerate([
    ("Regresión simple (bmi)", lr_simple, ["bmi"]),
    ("Regresión múltiple (10 predictores)", pipe_multi, list(Xr_train.columns)),
    (f"Árbol (max_depth={arbol_reg.max_depth}, min_samples_leaf={arbol_reg.min_samples_leaf})",
     arbol_reg, list(Xr_train.columns)),
]):
    filas.append({
        "Modelo": nombre, "Orden": orden,
        **{f"{k} ent.": v for k, v in metricas_reg(yr_train, modelo.predict(Xr_train[cols])).items()},
        **{f"{k} val.": v for k, v in metricas_reg(yr_val, modelo.predict(Xr_val[cols])).items()},
    })
tabla_7d = pd.DataFrame(filas)
tabla_7d.loc[len(tabla_7d)] = {
    "Modelo": "Constante (media) — referencia", "Orden": np.nan,
    **{f"{k} ent.": v for k, v in metricas_reg(yr_train, dummy_reg.predict(Xr_train)).items()},
    **{f"{k} val.": v for k, v in metricas_reg(yr_val, dummy_reg.predict(Xr_val)).items()},
}
print("Tabla 7.5 — Comparación de candidatos de regresión (entrenamiento y validación)")
display(tabla_7d.set_index("Modelo").drop(columns="Orden"))

candidato_reg = seleccionar(tabla_7d.dropna(subset=["Orden"]), [("RMSE val.", True), ("Orden", True)])
modelos_reg = {
    "Regresión simple (bmi)": (lr_simple, ["bmi"]),
    "Regresión múltiple (10 predictores)": (pipe_multi, list(Xr_train.columns)),
    tabla_7d.loc[2, "Modelo"]: (arbol_reg, list(Xr_train.columns)),
}
modelo_reg_final, cols_reg_final = modelos_reg[candidato_reg["Modelo"]]
print(f"Candidato final de regresión (congelado): {candidato_reg['Modelo']} "
      f"(RMSE validación = {candidato_reg['RMSE val.']:.4f})")
_ = registrar("CandidatoRegRMSEVal", candidato_reg["RMSE val."])

# %% [markdown]
# TODO: análisis 7d

# %% [markdown]
# ### e. No linealidad y brecha entrenamiento–validación

# %%
brechas = pd.DataFrame({
    "RMSE ent.": tabla_7d.set_index("Modelo")["RMSE ent."],
    "RMSE val.": tabla_7d.set_index("Modelo")["RMSE val."],
})
brechas["Brecha (val − ent)"] = brechas["RMSE val."] - brechas["RMSE ent."]
print("Tabla 7.6 — Brecha de RMSE entre validación y entrenamiento")
brechas

# %% [markdown]
# TODO: análisis 7e
