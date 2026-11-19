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
# La tabla cruzada confirma la recodificación, los 212 tumores `malignant`
# quedan con $y = 1$ y los 357 `benign` con $y = 0$, de modo que las métricas
# de la clase positiva corresponden a muestras malignas. La proporción de
# malignos es 0.3724 en entrenamiento y 0.3772 en validación, prácticamente
# iguales gracias a la estratificación.
#
# El `DummyClassifier(strategy="prior")` asigna a todas las muestras la misma
# probabilidad de maligno (0.3724, la proporción de entrenamiento), por lo que
# con $t = 0.5$ predice siempre benigno, sin verdaderos ni falsos positivos;
# la precisión queda indefinida ($0/0$) y `zero_division=0` la reporta como 0,
# al igual que recall y F1. Su ROC-AUC es exactamente 0.5, pues un puntaje
# constante no ordena a ninguna muestra maligna por encima de una benigna, y
# su costo de validación es $L = 5 \cdot 43 = 215$, el piso que cualquier
# modelo útil debería mejorar.

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
exportar_tabla(tabla_8b.assign(C=tabla_8b["C"].map(lambda c: f"{c:g}")), "ex8_C", indice=False)
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
# Los cuatro ajustes convergieron (entre 10 y 33 iteraciones, lejos del límite
# de 5000, y sin `ConvergenceWarning`). El ROC-AUC de validación crece de
# 0.9905 ($C = 0.01$) a 0.9951 ($C = 0.1$) y luego se estabiliza, pues $C = 1$
# y $C = 10$ obtienen exactamente el mismo valor a precisión completa
# (0.9957418932, ya que con 114 muestras el AUC es un cociente de conteos de
# pares); ante ese empate la regla elige el menor $C$, es decir **$C = 1$**,
# el de mayor regularización entre los empatados. En entrenamiento el ROC-AUC
# y el F1 aumentan monótonamente con $C$ (F1 de 0.9283 a 0.9960), mientras que
# el F1 de validación alcanza su máximo en $C = 0.1$ (0.9762) y cae en $C =
# 10$ (0.9545), señal de que con menos regularización el modelo se ajusta más
# al entrenamiento sin ganancia en datos nuevos; cabe aclarar que el criterio
# de selección es el ROC-AUC, no el F1, por lo que $C = 0.1$ no se elige
# aunque tenga el mejor F1 con umbral 0.5.

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
exportar_tabla(tabla_8c.assign(Umbral=tabla_8c["Umbral"].map(lambda t: f"{t:.1f}")), "ex8_umbrales", indice=False)
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
# El costo baja al subir el umbral, de $L = 13$ con $t = 0.2$ a $L = 7$ entre
# $t = 0.4$ y $t = 0.6$, y a $L = 5$ con $t = 0.7$ y $t = 0.8$ (ningún falso
# positivo y un falso negativo). Entre estos dos últimos el recall es idéntico
# (0.9767), de modo que el segundo desempate elige el menor umbral, **$t =
# 0.7$**. Frente a $t = 0.5$ ($L = 7$, con 2 falsos positivos y 1 falso
# negativo), el umbral elegido elimina los dos falsos positivos sin perder
# ningún maligno.
#
# Cabe una advertencia, el recall es el mismo en los siete umbrales porque el
# único maligno no detectado tiene una probabilidad menor que 0.2, de modo que
# en validación subir el umbral solo elimina falsos positivos y nunca agrega
# falsos negativos. Con probabilidades calibradas, el razonamiento del
# ejercicio 3d daría un umbral teórico de $1/(1+5) \approx 0.167$ para esta
# razón de costos, muy por debajo de 0.7, y la elección de un umbral alto
# descansa en unos pocos casos de una validación de 114 muestras (43
# malignas), por lo que es posible que en datos nuevos aparezcan malignos con
# probabilidades intermedias que este umbral deje pasar.

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
# Al subir el umbral la precisión aumenta (de 0.84 a 1.00) porque se exigen
# probabilidades más altas para declarar maligno y se descartan falsos
# positivos, mientras que el recall se mantiene en 0.9767 por la razón
# explicada en el inciso anterior. El ROC-AUC (0.9957) no cambia con $t$
# porque se calcula con las probabilidades y resume el ordenamiento del modelo
# sobre todos los umbrales posibles (el área bajo la curva completa de tasa de
# verdaderos positivos frente a falsos positivos), mientras que $t$ solo elige
# un punto sobre esa curva. Por eso los dos criterios cumplen papeles
# distintos, el ROC-AUC sirve para comparar modelos (valores de $C$) con
# independencia del punto de operación, y el costo $L$ sirve para fijar, una
# vez congelado el modelo, el umbral que mejor equilibra falsos positivos y
# falsos negativos según sus consecuencias.

# %% [markdown]
# ### e. Coeficientes estandarizados y odds

# %%
coefs_logit = pd.DataFrame(
    {"Coeficiente": logit_sel.named_steps["logistica"].coef_[0]}, index=Xc_train.columns
)
coefs_logit["exp(β)"] = np.exp(coefs_logit["Coeficiente"])
coefs_logit["|Coeficiente|"] = coefs_logit["Coeficiente"].abs()
coefs_logit = coefs_logit.sort_values("|Coeficiente|", ascending=False)
exportar_tabla(coefs_logit.head(10)[["Coeficiente", "exp(β)"]].rename(columns={"exp(β)": r"$e^{\beta}$"})
               .rename_axis("Predictor"), "ex8_coeficientes")
for var, clave in [("worst texture", "TexturaPeor"), ("mean concave points", "PuntosConcavos"),
                   ("compactness error", "CompactError")]:
    registrar(f"Coef{clave}", coefs_logit.loc[var, "Coeficiente"])
    registrar(f"Exp{clave}", coefs_logit.loc[var, "exp(β)"])
print(f"Tabla 8.6 — Coeficientes estandarizados de la regresión logística (C = {C_sel:g}), por magnitud absoluta")
coefs_logit

# %% [markdown]
# Con los predictores estandarizados, un coeficiente positivo indica que
# valores más altos de la variable se asocian con mayores log-odds de
# malignidad manteniendo fijas las demás, y uno negativo lo contrario. `worst
# texture` (+1.3553) y `mean concave points` (+1.1043) tienen signo positivo,
# coherente con que núcleos de textura más irregular y con más puntos cóncavos
# se asocian con malignidad, mientras que `compactness error` (−0.7591) tiene
# signo negativo; este último no significa que valores altos de esa variable
# se asocien marginalmente con benignidad, sino que es un efecto condicional
# al resto de las 29 variables, muchas de ellas casi colineales (radio,
# perímetro y área), lo que vuelve frágil la lectura individual de los signos.
#
# Para `worst texture`, $e^{1.3553} \approx 3.8780$, es decir, un aumento de
# una desviación estándar de entrenamiento en esa variable multiplica los odds
# estimados de malignidad por cerca de 3.88, manteniendo fijas las demás
# variables. Esto describe una asociación dentro del modelo, no un efecto
# causal, y con predictores tan correlacionados "mantener fijas las demás" es
# un escenario en buena parte hipotético. El pipeline con $C = 1$ y el umbral
# $t = 0.7$ quedan congelados.
