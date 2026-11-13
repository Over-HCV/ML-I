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
exportar_tabla(candidatos.drop(columns="Predictores"), "ex10_candidatos")
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
exportar_tabla(tabla_10a.rename(columns={"RMSE validación": "RMSE val.", "Candidato elegido": "Elegido"}), "ex10_regresion")

# %% [markdown]
# En prueba la regresión múltiple sigue siendo la mejor (RMSE de 54.1564,
# $R^2$ de 0.4464), seguida del árbol (58.2134), el modelo simple (63.8336) y
# la constante (72.9002), el mismo orden que en validación. El candidato
# elegido pasa de un RMSE de validación de 51.1937 a uno de prueba de 54.1564
# (2.96 unidades más), y su $R^2$ de 0.5216 a 0.4464; esta diferencia es
# esperable, pues la validación resultó una partición algo más favorable para
# todos los modelos (como se vio desde el ejercicio 6), y en cualquier caso no
# se cambia de candidato a la luz de estos resultados.

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
exportar_tabla(tabla_10b[["Umbral", "Precisión", "Recall", "F1", "ROC-AUC"]]
               .assign(Umbral=tabla_10b["Umbral"].map(lambda t: f"{t:.1f}")), "ex10_clf_metricas")
exportar_tabla(tabla_10b[["Umbral", "TN", "FP", "FN", "TP", "Costo L"]]
               .assign(Umbral=tabla_10b["Umbral"].map(lambda t: f"{t:.1f}")), "ex10_clf_costos")

# %% [markdown]
# La regresión logística con $t = 0.7$ obtiene en prueba precisión 1, recall
# 0.9286, F1 0.9630 y ROC-AUC 0.9954, con 0 falsos positivos y 3 falsos
# negativos ($L = 15$), mientras que el árbol con $t = 0.2$ obtiene precisión
# 0.8542, recall 0.9762, F1 0.9111 y ROC-AUC 0.9730, con 7 falsos positivos y
# 1 falso negativo ($L = 12$). La referencia predice todo benigno y acumula $L
# = 210$.
#
# Se observa que en prueba el árbol tiene un costo algo menor que el candidato
# elegido, diferencia que se explica por dos falsos negativos adicionales de
# la logística (10 unidades de costo) compensados parcialmente por siete
# falsos positivos menos, y que coincide con la advertencia del ejercicio 8c
# sobre un umbral alto elegido con pocos malignos de validación. Aun así el
# candidato se mantiene, pues la decisión se tomó con validación y cambiarla
# ahora sería usar la prueba para seleccionar, y con 114 muestras (42
# malignas) una diferencia de 3 unidades de costo es compatible con la
# variabilidad muestral; la logística conserva además un ROC-AUC claramente
# superior.

# %% [markdown]
# ### Incertidumbre de la evaluación final: intervalos bootstrap (prueba)
#
# Con 89 y 114 muestras de prueba, una sola cifra puntual oculta cuánto
# podría cambiar con otra muestra del mismo tamaño. Se remuestrea con
# reemplazo el conjunto de prueba ($B = 2000$, semilla 42) y se recalculan
# las métricas con las **mismas predicciones congeladas** (sin reajustar
# nada), reportando intervalos percentiles al 95 %. Para comparar modelos se
# usa la diferencia **pareada** (ambos modelos evaluados sobre el mismo
# remuestreo en cada réplica), que descuenta la variabilidad común. Esto no
# selecciona ni cambia ningún candidato, solo cuantifica la incertidumbre.

# %%
B_BOOT = 2000
rng = np.random.default_rng(RANDOM_STATE)

pred_reg_test = {clave: m.predict(Xr_test[c]) for (n, m, c), clave
                 in zip(modelos_reg_prueba, ["Dummy", "Simple", "Multi", "ArbolReg"])}
proba_clf_test = {clave: m.predict_proba(Xc_test)[:, 1] for _, m, _, clave in modelos_clf_prueba}
umbral_clf = {clave: t for _, _, t, clave in modelos_clf_prueba}
yr_t, yc_t = yr_test.to_numpy(), yc_test.to_numpy()


def costo_L(y, proba, t):
    pred = predecir_umbral(proba, t)
    return int(((pred == 1) & (y == 0)).sum() + 5 * ((pred == 0) & (y == 1)).sum())


boot = {k: [] for k in ["RMSE Multi", "RMSE ArbolReg", "RMSE Simple", "Dif RMSE Multi-Arbol",
                        "Dif RMSE Multi-Simple", "AUC Logit", "AUC ArbolClf", "L Logit", "L ArbolClf",
                        "Dif L Logit-Arbol"]}
for _ in range(B_BOOT):
    ir = rng.integers(0, len(yr_t), len(yr_t))
    r = {k: rmse(yr_t[ir], v[ir]) for k, v in pred_reg_test.items()}
    boot["RMSE Multi"].append(r["Multi"]); boot["RMSE ArbolReg"].append(r["ArbolReg"])
    boot["RMSE Simple"].append(r["Simple"])
    boot["Dif RMSE Multi-Arbol"].append(r["Multi"] - r["ArbolReg"])
    boot["Dif RMSE Multi-Simple"].append(r["Multi"] - r["Simple"])

    ic = rng.integers(0, len(yc_t), len(yc_t))
    while yc_t[ic].min() == yc_t[ic].max():  # el AUC exige ambas clases en la réplica
        ic = rng.integers(0, len(yc_t), len(yc_t))
    L = {k: costo_L(yc_t[ic], proba_clf_test[k][ic], umbral_clf[k]) for k in ["Logit", "ArbolClf"]}
    boot["AUC Logit"].append(roc_auc_score(yc_t[ic], proba_clf_test["Logit"][ic]))
    boot["AUC ArbolClf"].append(roc_auc_score(yc_t[ic], proba_clf_test["ArbolClf"][ic]))
    boot["L Logit"].append(L["Logit"]); boot["L ArbolClf"].append(L["ArbolClf"])
    boot["Dif L Logit-Arbol"].append(L["Logit"] - L["ArbolClf"])

boot = {k: np.asarray(v) for k, v in boot.items()}
puntual = {
    "RMSE Multi": rmse(yr_t, pred_reg_test["Multi"]), "RMSE ArbolReg": rmse(yr_t, pred_reg_test["ArbolReg"]),
    "RMSE Simple": rmse(yr_t, pred_reg_test["Simple"]),
    "Dif RMSE Multi-Arbol": rmse(yr_t, pred_reg_test["Multi"]) - rmse(yr_t, pred_reg_test["ArbolReg"]),
    "Dif RMSE Multi-Simple": rmse(yr_t, pred_reg_test["Multi"]) - rmse(yr_t, pred_reg_test["Simple"]),
    "AUC Logit": roc_auc_score(yc_t, proba_clf_test["Logit"]),
    "AUC ArbolClf": roc_auc_score(yc_t, proba_clf_test["ArbolClf"]),
    "L Logit": costo_L(yc_t, proba_clf_test["Logit"], umbral_clf["Logit"]),
    "L ArbolClf": costo_L(yc_t, proba_clf_test["ArbolClf"], umbral_clf["ArbolClf"]),
}
puntual["Dif L Logit-Arbol"] = puntual["L Logit"] - puntual["L ArbolClf"]

etiquetas = {
    "RMSE Multi": "RMSE regresión múltiple", "RMSE ArbolReg": "RMSE árbol de regresión",
    "RMSE Simple": "RMSE regresión simple",
    "Dif RMSE Multi-Arbol": "Δ RMSE múltiple − árbol", "Dif RMSE Multi-Simple": "Δ RMSE múltiple − simple",
    "AUC Logit": "ROC-AUC logística", "AUC ArbolClf": "ROC-AUC árbol",
    "L Logit": "Costo L logística", "L ArbolClf": "Costo L árbol", "Dif L Logit-Arbol": "Δ L logística − árbol",
}
tabla_boot = pd.DataFrame([
    {"Cantidad": etiquetas[k], "Puntual": puntual[k],
     "IC 95 % inf.": np.percentile(v, 2.5), "IC 95 % sup.": np.percentile(v, 97.5)}
    for k, v in boot.items()
]).set_index("Cantidad")
print(f"Tabla 10.5 — Intervalos bootstrap percentiles al 95 % en prueba (B = {B_BOOT}, predicciones congeladas)")
display(tabla_boot)
_fmt_boot = tabla_boot.apply(lambda fila: fila.map(
    (lambda v: f"{v:.0f}") if fila.name.startswith(("Costo L", "Δ L")) else (lambda v: f"{v:.4f}")), axis=1)
exportar_tabla(_fmt_boot.apply(lambda col: col.str.replace("-", "−", regex=False)), "ex10_bootstrap", alineacion="lrrr")

claves_boot = {"Dif RMSE Multi-Arbol": "DifRMSEMultiArbol", "Dif RMSE Multi-Simple": "DifRMSEMultiSimple",
               "Dif L Logit-Arbol": "DifLLogitArbol", "RMSE Multi": "RMSEMulti", "L Logit": "LLogit",
               "L ArbolClf": "LArbol", "AUC Logit": "AUCLogit", "AUC ArbolClf": "AUCArbol"}
for k, clave in claves_boot.items():
    decs = 0 if k.startswith(("L ", "Dif L")) else 4
    registrar(f"Boot{clave}Inf", float(np.percentile(boot[k], 2.5)), decs)
    registrar(f"Boot{clave}Sup", float(np.percentile(boot[k], 97.5)), decs)
registrar("BootB", B_BOOT)
_ = registrar("BootPropLogitPeor", float((boot["Dif L Logit-Arbol"] > 0).mean()))

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, claves, titulo, xlab in [
    (axes[0], ["Dif RMSE Multi-Simple", "Dif RMSE Multi-Arbol"], "Caso A: Δ RMSE (múltiple − otro)", "Diferencia de RMSE"),
    (axes[1], ["Dif L Logit-Arbol"], "Caso B: Δ costo L (logística − árbol)", "Diferencia de costo L"),
]:
    for i, k in enumerate(claves):
        inf, sup = np.percentile(boot[k], [2.5, 97.5])
        ax.plot([inf, sup], [i, i], color=COLORES["azul"], linewidth=3)
        ax.plot(puntual[k], i, "o", color=COLORES["rojo"], markersize=8)
    ax.axvline(0, color=COLORES["gris"], linestyle="--")
    ax.set_yticks(range(len(claves)), [etiquetas[k] for k in claves])
    ax.set_ylim(-0.6, len(claves) - 0.4)
    ax.set_title(titulo)
    ax.set_xlabel(xlab)
    ax.grid(alpha=0.3, axis="x")
fig.suptitle(f"Figura 10.1 — Diferencias pareadas en prueba con IC bootstrap al 95 % (B = {B_BOOT}); "
             "punto rojo = valor observado", fontsize=10)
plt.tight_layout()
guardar_fig(fig, "ex10_bootstrap")
plt.show()

# %% [markdown]
# Los intervalos muestran cuánto de lo observado en prueba es señal y cuánto
# es ruido. En el caso A la regresión múltiple supera a la simple con un
# intervalo para la diferencia de RMSE de $[-17.58, -2.03]$, que excluye el
# cero, de forma que esa mejora es robusta, mientras que frente al árbol la
# diferencia observada ($-4.06$) tiene un intervalo de $[-12.23, 3.54]$ que sí
# incluye el cero, por lo cual la prueba por sí sola no permite afirmar que la
# múltiple sea mejor que el árbol (la elección se sostiene en la validación y
# en su menor complejidad). En el caso B el costo de la logística tiene un
# intervalo de $[0, 35]$ y el del árbol de $[3, 24]$, y la diferencia pareada
# ($+3$) va de $-16$ a $27$, con la logística más costosa en el 58 % de las
# réplicas, es decir, la ventaja del árbol en prueba es compatible con el azar
# del muestreo, así como la superioridad de la logística en ordenamiento es más
# clara (ROC-AUC con intervalo $[0.9839, 1]$ frente a $[0.9410, 0.9960]$ del
# árbol). Cabe mencionar que el costo $L$ depende de unos pocos falsos
# negativos que valen 5 cada uno, y por eso sus intervalos son tan anchos,
# lo que refuerza que con 114 muestras de prueba la decisión no debe
# revisarse a partir de una diferencia de 3 unidades.

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
fig.suptitle("Figura 10.2 — Matrices de confusión en prueba (orden de etiquetas [0, 1])")
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
# En este conjunto, un **falso positivo** es una muestra de tejido benigno que
# el modelo clasifica como maligna, lo que en un uso hipotético implicaría
# exámenes adicionales innecesarios y angustia para la paciente, mientras que
# un **falso negativo** es una muestra maligna clasificada como benigna, lo
# que retrasaría un diagnóstico y un tratamiento, consecuencia mucho más grave
# y razón por la cual el costo didáctico la pondera cinco veces. Para el
# candidato elegido (regresión logística, $t = 0.7$), la matriz de prueba
# tiene 72 verdaderos negativos, 0 falsos positivos, 3 falsos negativos y 39
# verdaderos positivos, de modo que $L = FP + 5FN = 0 + 5 \cdot 3 = 15$, valor
# que coincide con la Tabla 10.4.

# %% [markdown]
# ### d. Recomendación
#
# El texto se guarda en una variable para contar sus palabras en la misma
# celda y comprobar que está entre 250 y 350.

# %%
from IPython.display import Markdown

_lg = tabla_10b.loc[candidato_clf["Modelo"]]
_ar = tabla_10b.loc[comparacion_clf.loc[1, "Modelo"]]
RECOMENDACION = f"""Para el caso A (progresión de la diabetes) se recomienda la regresión \
lineal múltiple con los diez predictores estandarizados, pues fue el candidato con menor \
RMSE de validación ({rmse_val_cand:.2f}) y conservó el primer lugar en prueba (RMSE de \
{rmse_test_cand:.2f} y $R^2$ de {tabla_10a.loc[candidato_reg["Modelo"], "R2"]:.2f}), por delante \
del árbol y de la regresión simple con `bmi`. Además es interpretable, ya que cada coeficiente \
se lee como el cambio esperado ante una desviación estándar del predictor, con el pequeño \
detalle que la fuerte correlación entre `s1` y `s2` impide leer esos coeficientes como \
importancias independientes o como efectos causales, y un $R^2$ cercano a \
{tabla_10a.loc[candidato_reg["Modelo"], "R2"]:.2f} indica que más de la mitad de la \
variabilidad queda sin explicar.

Para el caso B (clasificación de tejido) se recomienda la regresión logística con $C = {C_sel:g}$ \
y umbral {t_logit}, elegida por menor costo en validación ($L = {int(candidato_clf["Costo L"])}$), \
que en prueba obtuvo ROC-AUC de {_lg["ROC-AUC"]:.4f}, precisión de {_lg["Precisión"]:.2f} y recall de \
{_lg["Recall"]:.2f}, aunque con $L = {int(_lg["Costo L"])}$ frente a $L = {int(_ar["Costo L"])}$ del \
árbol, diferencia que con {len(yc_test)} muestras de prueba no justifica cambiar de candidato \
después de ver los resultados. El árbol, más simple de explicar con dos o tres reglas, \
sacrifica capacidad de ordenamiento y depende de pocas variables.

Si un candidato rinde peor en prueba que en validación, como ocurrió en ambos casos, se \
reporta tal cual como estimación honesta y, en lugar de reajustar con prueba, se revisa el \
análisis sobre el conjunto de desarrollo y, de ser necesario, se reserva un nuevo conjunto \
de prueba.

Cabe mencionar dos limitaciones, primeramente los tamaños son pequeños ({len(yr_val)} y \
{len(yc_val)} observaciones de validación), por lo que umbrales y empates dependen de unos \
pocos casos, y en segundo lugar los datos son observacionales y de una sola fuente, sin \
evaluación externa ni costos clínicos reales detrás de $L = FP + 5FN$. Como mejora futura se \
propone elegir hiperparámetros y umbral con validación cruzada estratificada dentro del \
conjunto de desarrollo, menos sensible a una sola partición, y luego evaluar en una cohorte \
externa."""

n_palabras = len(RECOMENDACION.split())
assert 250 <= n_palabras <= 350, f"La recomendación tiene {n_palabras} palabras"
display(Markdown(RECOMENDACION))
print(f"Número de palabras: {registrar('RecomPalabras', n_palabras)}")

