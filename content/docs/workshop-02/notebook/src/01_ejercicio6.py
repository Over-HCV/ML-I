# %% [markdown]
# ## Ejercicio 6 — Caso A: regresión lineal simple y modelo de referencia

# %% [markdown]
# ### a. Exploración en entrenamiento
#
# Toda la exploración se hace sobre `Xr_train`/`yr_train`; validación y
# prueba no se miran en esta etapa.

# %%
print("Tabla 6.1 — Tipos, faltantes y rangos (entrenamiento)")
tabla_6a = pd.DataFrame(
    {
        "Tipo": Xr_train.dtypes.astype(str),
        "Faltantes": Xr_train.isnull().sum(),
        "Mínimo": Xr_train.min(),
        "Máximo": Xr_train.max(),
        "Media": Xr_train.mean(),
        "Desv. est.": Xr_train.std(),
    }
)
tabla_6a.loc["target (y)"] = ["float64", int(yr_train.isnull().sum()), yr_train.min(), yr_train.max(),
                              yr_train.mean(), yr_train.std()]
tabla_6a["Faltantes"] = tabla_6a["Faltantes"].astype(int)
for col in ["Mínimo", "Máximo", "Media", "Desv. est."]:
    tabla_6a[col] = tabla_6a[col].astype(float)
exportar_tabla(tabla_6a.drop(columns="Tipo").rename_axis("Variable"), "ex6_exploracion", decimales=2)
tabla_6a

# %%
print(f"Faltantes en la variable objetivo (entrenamiento): {yr_train.isnull().sum()}")
print("Tabla 6.2 — Resumen de la variable objetivo (entrenamiento)")
yr_train.describe().to_frame("target")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(Xr_train["bmi"], yr_train, s=18, alpha=0.6, color=COLORES["azul"], label="Observaciones")
ax.set_title("Figura 6.1 — bmi frente a la progresión de la diabetes\n(entrenamiento)")
ax.set_xlabel("bmi (índice de masa corporal, kg/m²)")
ax.set_ylabel("Progresión a un año (target)")
ax.legend()
plt.tight_layout()
guardar_fig(fig, "ex6_bmi_target")
plt.show()

print(f"Correlación de Pearson bmi–target (entrenamiento): "
      f"{registrar('CorrBmiTarget', Xr_train['bmi'].corr(yr_train)):.4f}")

# %% [markdown]
# Los diez predictores son de tipo `float64` y no presentan valores faltantes
# en entrenamiento (tampoco la variable objetivo), por lo que no se requiere
# imputación. Los rangos están en unidades originales y son muy distintos
# entre sí, por ejemplo `s1` va de 110 a 301 mientras que `s5` va de 3.58 a
# 6.11 (corresponde al logaritmo de los triglicéridos), y `sex` es una
# variable binaria codificada como 1 y 2; esta diferencia de escalas no afecta
# a la regresión simple ni a los árboles, pero sí a la lectura de coeficientes
# del modelo múltiple, razón por la cual allí se estandariza. La variable
# objetivo es una medida cuantitativa de progresión (entre 25 y 346 en
# entrenamiento) cuya unidad no está documentada en scikit-learn, de modo que
# se habla de "unidades de la escala del target".
#
# En la Figura 6.1 se aprecia una tendencia creciente con una dispersión
# vertical amplia (correlación de Pearson de 0.5954), de forma que una recta
# resulta una aproximación razonable para la tendencia promedio aunque dejará
# una buena parte de la variabilidad sin explicar, y la nube parece abrirse
# levemente hacia valores altos de `bmi`. No se elimina ninguna observación,
# pues los puntos alejados de la tendencia son plausibles y nada indica que
# sean errores de registro.

# %% [markdown]
# ### b. Regresión simple con NumPy vs. `LinearRegression`
#
# Se implementan las expresiones del ejercicio 1,
# $\hat\beta_1 = \sum_i (x_i-\bar x)(y_i-\bar y) / \sum_i (x_i-\bar x)^2$ y
# $\hat\beta_0 = \bar y - \hat\beta_1 \bar x$, usando solo `bmi`.

# %%
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

x_tr = Xr_train["bmi"].to_numpy()
y_tr = yr_train.to_numpy()
x_bar, y_bar = x_tr.mean(), y_tr.mean()
beta1_np = np.sum((x_tr - x_bar) * (y_tr - y_bar)) / np.sum((x_tr - x_bar) ** 2)
beta0_np = y_bar - beta1_np * x_bar

lr_simple = LinearRegression().fit(Xr_train[["bmi"]], yr_train)

pred_val_np = beta0_np + beta1_np * Xr_val["bmi"].to_numpy()
pred_val_sk = lr_simple.predict(Xr_val[["bmi"]])
dif_max = np.max(np.abs(pred_val_np - pred_val_sk))

print("Tabla 6.3 — Coeficientes de la regresión simple (entrenamiento)")
display(
    pd.DataFrame(
        {"NumPy": [beta0_np, beta1_np], "LinearRegression": [lr_simple.intercept_, lr_simple.coef_[0]]},
        index=["β0 (intercepto)", "β1 (pendiente bmi)"],
    )
)
print(f"Diferencia absoluta máxima entre predicciones de validación: {dif_max:.3e}")
print(f"np.allclose(NumPy, sklearn): {np.allclose(pred_val_np, pred_val_sk)}")
if dif_max == 0:
    registrar("SimpleDifMax", "0")
else:
    exp = int(np.floor(np.log10(dif_max)))
    registrar("SimpleDifMax", f"{dif_max / 10**exp:.1f}\\times 10^{{{exp}}}")

# %% [markdown]
# Las dos implementaciones producen los mismos coeficientes ($\hat\beta_0 =
# -134.5023$, $\hat\beta_1 = 10.7331$) y la diferencia absoluta máxima entre
# predicciones de validación es del orden de $10^{-14}$, es decir, error de
# redondeo de punto flotante, por lo que `np.allclose` devuelve `True`. Esto
# confirma que `LinearRegression` con un solo predictor resuelve exactamente
# el mismo problema de mínimos cuadrados que las fórmulas cerradas del
# ejercicio 1, con la diferencia de que internamente lo hace mediante una
# factorización numérica en lugar de las sumas explícitas.

# %% [markdown]
# ### c. Comparación con `DummyRegressor`

# %%
dummy_reg = DummyRegressor(strategy="mean").fit(Xr_train, yr_train)

filas = []
for nombre, modelo, cols in [
    ("Constante (media)", dummy_reg, list(Xr_train.columns)),
    ("Regresión simple (bmi)", lr_simple, ["bmi"]),
]:
    for particion, X_, y_ in [("Entrenamiento", Xr_train, yr_train), ("Validación", Xr_val, yr_val)]:
        filas.append({"Modelo": nombre, "Partición": particion, **metricas_reg(y_, modelo.predict(X_[cols]))})
tabla_6c = pd.DataFrame(filas).set_index(["Modelo", "Partición"])
print("Tabla 6.4 — Constante vs. regresión simple (entrenamiento y validación)")
tabla_6c

# %%
for clave, nombre in [("Dummy", "Constante (media)"), ("Simple", "Regresión simple (bmi)")]:
    for sufijo, particion in [("Tr", "Entrenamiento"), ("Val", "Validación")]:
        for met in ["MAE", "RMSE", "R2"]:
            registrar(f"Reg{clave}{met.replace('2', 'dos')}{sufijo}", tabla_6c.loc[(nombre, particion), met])
exportar_tabla(tabla_6c, "ex6_comparacion")

# %% [markdown]
# La regresión simple mejora las tres métricas frente al predictor constante
# en ambas particiones, en validación el RMSE baja de 75.6392 a 58.5671 (cerca
# de un 22.6 % menos) y el MAE de 64.1288 a 50.3475, mientras que $R^2$ pasa
# de −0.0443 a 0.3739, es decir, `bmi` por sí solo explica alrededor del 37 %
# de la variabilidad de la progresión en validación. El $R^2$ negativo de la
# constante no es un error, la media se calcula con entrenamiento (149.8106) y
# en validación el $R^2$ se mide contra la media de validación, de forma que
# cualquier desfase entre ambas medias lo vuelve ligeramente negativo; en este
# caso ninguna métrica empeora con la regresión simple. Cabe mencionar que los
# errores de validación son algo menores que los de entrenamiento en los dos
# modelos (incluida la constante), lo que apunta a una partición de validación
# algo más "fácil" y no a un fenómeno del modelo, y que el error típico sigue
# siendo alto (RMSE cercano a 59 frente a una desviación estándar de 79 del
# target en entrenamiento).

# %% [markdown]
# ### d. Residuos de validación

# %%
res_val = yr_val.to_numpy() - pred_val_sk
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(pred_val_sk, res_val, s=20, alpha=0.7, color=COLORES["azul"], label="Residuo de validación")
ax.axhline(0, color=COLORES["rojo"], linestyle="--", linewidth=1.2, label="Residuo = 0")
ax.set_title("Figura 6.2 — Residuos frente a valores predichos\n(regresión simple con bmi, validación)")
ax.set_xlabel("Valor predicho")
ax.set_ylabel("Residuo (y − ŷ)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
plt.tight_layout()
guardar_fig(fig, "ex6_residuos")
plt.show()

orden_pred = np.argsort(pred_val_sk)
mitad = len(orden_pred) // 2
print(f"Desv. est. de residuos, mitad de predicciones bajas: {res_val[orden_pred[:mitad]].std():.4f}")
print(f"Desv. est. de residuos, mitad de predicciones altas: {res_val[orden_pred[mitad:]].std():.4f}")
print(f"Residuo mínimo: {res_val.min():.4f} · residuo máximo: {res_val.max():.4f}")
registrar("ResStdBaja", res_val[orden_pred[:mitad]].std(), 2)
registrar("ResStdAlta", res_val[orden_pred[mitad:]].std(), 2)
registrar("ResMin", res_val.min(), 2)
_ = registrar("ResMax", res_val.max(), 2)

# %% [markdown]
# Los residuos se distribuyen alrededor de cero sin una curvatura marcada, por
# lo que la forma lineal no parece estar sesgando sistemáticamente las
# predicciones, pero la dispersión crece con el valor predicho (desviación
# estándar de 50.76 en la mitad de predicciones bajas frente a 62.36 en la
# mitad alta), lo que sugiere heterocedasticidad, además de errores grandes en
# ambos sentidos (entre −113.16 y 156.78). Esto implica que el RMSE resume un
# error que en realidad es menor para pacientes con `bmi` bajo y mayor para
# `bmi` alto, que los errores estándar clásicos de mínimos cuadrados (que
# suponen varianza constante) no serían confiables para inferencia, y que la
# gran dispersión residual indica que faltan predictores, lo cual motiva el
# modelo múltiple del ejercicio 7.

# %% [markdown]
# ### e. Interpretación de coeficientes

# %%
print(f"Intercepto: {registrar('SimpleBetaCero', lr_simple.intercept_):.4f}")
print(f"Pendiente (bmi): {registrar('SimpleBetaUno', lr_simple.coef_[0]):.4f}")

# %% [markdown]
# El intercepto ($-134.5023$) es el valor predicho cuando `bmi` vale cero,
# situación fuera del rango observado (18.9 a 41.3 kg/m²), por lo que solo
# cumple el papel de ubicar la recta y no tiene interpretación práctica. La
# pendiente ($10.7331$) indica que, en promedio, cada kg/m² adicional de `bmi`
# se asocia con 10.7331 unidades más de progresión predicha.
#
# Esta asociación no identifica el efecto de intervenir sobre `bmi`, pues los
# datos son observacionales y `bmi` está correlacionado con otros factores
# (presión arterial, lípidos, edad) que también se relacionan con la
# progresión, de modo que la pendiente mezcla el efecto propio de `bmi` con el
# de esas variables; de hecho, en el modelo múltiple del ejercicio 7 la
# pendiente equivalente de `bmi` se reduce aproximadamente a la mitad.
# Identificar un efecto causal exigiría un diseño experimental o supuestos
# causales explícitos sobre las variables de confusión.
