# %% [markdown]
# ## Ejercicio 6 — Exploración inicial y preparación de datos

# %% [markdown]
# ### a–c. Carga del dataset y diagnóstico general

# %%
DATA_PATH = "cs-training.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/DrIanGregory/Kaggle-GiveMeSomeCredit/"
    "master/data/GiveMeSomeCredit-training.csv"
)

if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

df = pd.read_csv(DATA_PATH, index_col=0)

print(f"Número de observaciones: {df.shape[0]:,}")
print(f"Número de variables: {df.shape[1]}")
df.dtypes.to_frame("Tipo de dato")

# %%
faltantes = df.isnull().sum().to_frame("Faltantes (n)")
faltantes["Faltantes (%)"] = (df.isnull().mean() * 100).round(2)
print("Tabla 6.1 — Valores faltantes por variable")
faltantes[faltantes["Faltantes (n)"] > 0]

# %% [markdown]
# ### d. Balance de clases de la variable objetivo

# %%
balance = df["SeriousDlqin2yrs"].value_counts().sort_index().to_frame("n")
balance["%"] = (
    df["SeriousDlqin2yrs"].value_counts(normalize=True).sort_index() * 100
).round(2)
balance.index = ["0 (no incumple)", "1 (incumple)"]
balance.index.name = "SeriousDlqin2yrs"
print("Tabla 6.2 — Distribución de la variable objetivo")
balance

# %% [markdown]
# ### e. Gráfica de la distribución de la variable objetivo

# %%
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(
    ["No incumple (0)", "Incumple (1)"], balance["n"], color=["#4C72B0", "#C44E52"]
)
ax.set_title("Distribución de la variable objetivo\nSeriousDlqin2yrs")
ax.set_xlabel("Clase")
ax.set_ylabel("Número de observaciones")
for b, v, p in zip(bars, balance["n"], balance["%"]):
    ax.text(
        b.get_x() + b.get_width() / 2, v + 1500, f"{v:,}\n({p}%)",
        ha="center", fontsize=9,
    )
plt.tight_layout()
plt.show()

# %% [markdown]
# ### f. Tratamiento de valores faltantes
#
# Se imputa con la **mediana** de cada variable en lugar de la media,
# porque `MonthlyIncome` y `NumberOfDependents` tienen distribuciones con
# cola larga hacia la derecha (unos pocos valores extremos que arrastrarían
# la media hacia arriba); la mediana es robusta a esos atípicos.

# %%
df_imp = df.copy()
medianas = df_imp.median(numeric_only=True)
for col in ["MonthlyIncome", "NumberOfDependents"]:
    df_imp[col] = df_imp[col].fillna(medianas[col])

assert df_imp.isnull().sum().sum() == 0, "Quedaron valores faltantes sin tratar"
print("Tabla 6.3 — Medianas usadas para imputar")
medianas.to_frame("Mediana").round(4)

# %% [markdown]
# ### g. Partición 80/20 estratificada

# %%
X = df_imp.drop(columns=["SeriousDlqin2yrs"])
y = df_imp["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

tabla_particion = pd.DataFrame(
    {
        "Train (n)": y_train.value_counts().sort_index(),
        "Train (%)": (y_train.value_counts(normalize=True).sort_index() * 100).round(2),
        "Test (n)": y_test.value_counts().sort_index(),
        "Test (%)": (y_test.value_counts(normalize=True).sort_index() * 100).round(2),
    }
)
tabla_particion.index = ["0 (no incumple)", "1 (incumple)"]
tabla_particion.index.name = "SeriousDlqin2yrs"

print(f"Entrenamiento: {X_train.shape} · Prueba: {X_test.shape}")
print("Tabla 6.4 — Distribución de clases en train y test")
tabla_particion

# %% [markdown]
# La estratificación (`stratify=y`) mantiene la proporción de clases
# prácticamente idéntica entre entrenamiento y prueba, evitando que el
# azar de la partición produzca un subconjunto con una tasa de
# incumplimiento notablemente distinta a la del conjunto completo.
