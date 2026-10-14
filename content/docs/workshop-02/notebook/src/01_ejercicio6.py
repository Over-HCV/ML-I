# %% [markdown]
# ## Ejercicio 6 — Caso A: regresión lineal simple y modelo de referencia

# %% [markdown]
# ### a. Exploración en entrenamiento
#
# Toda la exploración se hace sobre `Xr_train`/`yr_train`; validación y
# prueba no se miran en esta etapa.

# %%
print("Tabla 6.1 — Tipos, faltantes y rangos (entrenamiento)")
pd.DataFrame(
    {
        "Tipo": Xr_train.dtypes.astype(str),
        "Faltantes": Xr_train.isnull().sum(),
        "Mínimo": Xr_train.min(),
        "Máximo": Xr_train.max(),
        "Media": Xr_train.mean(),
        "Desv. est.": Xr_train.std(),
    }
)

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
# TODO: análisis 6a

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
# TODO: análisis 6b

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

# %% [markdown]
# TODO: análisis 6c

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
ax.legend()
plt.tight_layout()
guardar_fig(fig, "ex6_residuos")
plt.show()

orden_pred = np.argsort(pred_val_sk)
mitad = len(orden_pred) // 2
print(f"Desv. est. de residuos, mitad de predicciones bajas: {res_val[orden_pred[:mitad]].std():.4f}")
print(f"Desv. est. de residuos, mitad de predicciones altas: {res_val[orden_pred[mitad:]].std():.4f}")
print(f"Residuo mínimo: {res_val.min():.4f} · residuo máximo: {res_val.max():.4f}")

# %% [markdown]
# TODO: análisis 6d

# %% [markdown]
# ### e. Interpretación de coeficientes

# %%
print(f"Intercepto: {registrar('SimpleBetaCero', lr_simple.intercept_):.4f}")
print(f"Pendiente (bmi): {registrar('SimpleBetaUno', lr_simple.coef_[0]):.4f}")

# %% [markdown]
# TODO: análisis 6e
