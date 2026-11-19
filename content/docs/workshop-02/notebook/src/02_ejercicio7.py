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
# La regresión múltiple mejora frente al modelo simple y a la constante en las
# dos particiones, con un RMSE de validación de 51.1937 (frente a 58.5671 del
# simple y 75.6392 de la constante) y un $R^2$ de validación de 0.5216 (frente
# a 0.3739). La mejora en entrenamiento (RMSE de 63.3458 a 54.6104) por sí
# sola no prueba nada, pues como se justificó en el ejercicio 2e añadir
# predictores a mínimos cuadrados nunca puede aumentar el error mínimo de
# entrenamiento; lo relevante es que la mejora se conserva en validación,
# datos que no participaron en el ajuste, y que la brecha entre particiones no
# se abre.

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
    registrar(f"CorrY{nombre}", coefs_multi.loc[var, "Corr. con y (entrenamiento)"])
registrar("CorrYStres", coefs_multi.loc["s3", "Corr. con y (entrenamiento)"])
registrar("CoefMultiStres", coefs_multi.loc["s3", "Coeficiente"], 2)
_ = registrar("BmiPorUnidad", coefs_multi.loc["bmi", "Coeficiente"] / coefs_multi.loc["bmi", "Desv. est. entrenamiento"], 2)

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
exportar_tabla(
    coefs_multi[["Coeficiente", "Desv. est. entrenamiento", "Corr. con y (entrenamiento)"]]
    .rename(columns={"Desv. est. entrenamiento": "Desv. est. (ent.)", "Corr. con y (entrenamiento)": "Corr. con $y$ (ent.)"})
    .rename_axis("Predictor"),
    "ex7_coeficientes",
)

# %% [markdown]
# Como los predictores están estandarizados, cada coeficiente es el cambio
# esperado en la predicción ante un aumento de una desviación estándar de
# entrenamiento (la poblacional que usa `StandardScaler`) en ese predictor,
# manteniendo fijos los demás. Para `s5` un aumento de una desviación (0.50 en
# el logaritmo de los triglicéridos) se asocia con 37.27 unidades más de
# progresión, y para `bmi` un aumento de una desviación (4.37 kg/m²) se asocia
# con 25.08 unidades más, equivalentes a unas 5.73 unidades por kg/m², cerca
# de la mitad de la pendiente del modelo simple (10.7331), ya que allí `bmi`
# absorbía parte de lo que ahora explican predictores correlacionados con él.
#
# La matriz de correlaciones muestra pares fuertemente relacionados, en
# especial `s1` y `s2` (0.9023) y `s3` y `s4` (−0.7143). El caso de `s1` y
# `s2` ilustra el problema: ambos tienen correlación positiva con la variable
# objetivo (0.1623 y 0.1073), pero el modelo les asigna coeficientes de signo
# opuesto y gran magnitud (−51.10 y +27.48), porque al moverse casi juntos el
# ajuste aprovecha su diferencia y no el nivel de cada uno; algo similar
# ocurre con `s3`, cuya correlación con el target es negativa (−0.3798) y su
# coeficiente positivo. Por esto los coeficientes deben leerse como efectos
# parciales condicionados al resto de variables, inestables ante colinealidad
# (pequeños cambios en los datos pueden redistribuirlos entre los predictores
# correlacionados), y no como importancias independientes.

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
exportar_tabla(
    tabla_7c.rename(columns={"RMSE entrenamiento": "RMSE ent.", "RMSE validación": "RMSE val.",
                             "Profundidad real": "Prof. real", "Brecha (val − ent)": "Brecha"}),
    "ex7_arboles", indice=False,
)
profundo = tabla_7c.query("max_depth == 6 and min_samples_leaf == 5").iloc[0]
registrar("ArbolProfundoRMSETr", profundo["RMSE entrenamiento"])
registrar("ArbolProfundoRMSEVal", profundo["RMSE validación"])
registrar("ArbolProfundoHojas", int(profundo["Hojas"]))

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
# Con `min_samples_leaf = 5` se observa el patrón clásico de sobreajuste, pues
# al aumentar la profundidad el RMSE de entrenamiento baja (57.63, 49.01,
# 38.54) mientras que el de validación sube (58.78, 61.05, 63.21), hasta una
# brecha de 24.67 con 31 hojas. Con `min_samples_leaf = 30` la restricción de
# tamaño de hoja domina, de forma que con `max_depth = 6` el árbol no supera
# la profundidad real 4 y resulta idéntico al de `max_depth = 4` (6 hojas,
# RMSE de validación 56.7789 en ambos); ante ese empate exacto la regla elige
# la menor profundidad máxima, es decir, `max_depth = 4` con `min_samples_leaf
# = 30`. Lo mismo ocurre entre `(2, 5)` y `(2, 15)`, árboles idénticos, aunque
# ese empate no afecta la selección.

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
exportar_tabla(tabla_7d.drop(columns="Orden").set_index("Modelo"), "ex7_candidatos")
print(f"Candidato final de regresión (congelado): {candidato_reg['Modelo']} "
      f"(RMSE validación = {candidato_reg['RMSE val.']:.4f})")
_ = registrar("CandidatoRegRMSEVal", candidato_reg["RMSE val."])

# %% [markdown]
# El candidato final de regresión es la **regresión múltiple**, con el menor
# RMSE de validación (51.1937), por delante del árbol seleccionado (56.7789) y
# del modelo simple (58.5671), sin que se requiera aplicar la regla de
# desempate. El predictor constante (75.6392) queda como referencia, y los
# tres modelos lo superan con claridad. Esta decisión queda congelada antes de
# consultar la prueba.

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
# El árbol divide el espacio con umbrales sobre los valores originales de cada
# variable, de modo que solo importa el orden de los valores y no su escala
# (por eso no necesita estandarización), y al asignar una constante por región
# puede representar relaciones no lineales e interacciones sin especificarlas.
# Sin embargo, esa flexibilidad no garantiza superar al modelo múltiple, pues
# con 264 observaciones de entrenamiento y un mínimo de 30 por hoja el árbol
# seleccionado solo produce 6 valores distintos de predicción, una
# aproximación escalonada y gruesa de una relación que en este conjunto parece
# mayormente aditiva y cercana a lineal, donde el modelo múltiple usa los
# datos de forma más eficiente. La Tabla 7.6 lo sustenta, el árbol
# seleccionado tiene una brecha pequeña (58.07 en entrenamiento y 56.78 en
# validación), pero su error es mayor que el del modelo múltiple en ambas
# particiones, y los árboles más flexibles de la Tabla 7.4 (por ejemplo `(6,
# 5)`, con 38.54 en entrenamiento y 63.21 en validación) muestran que la
# flexibilidad adicional se traduce en memorizar el entrenamiento y no en
# generalizar mejor.
