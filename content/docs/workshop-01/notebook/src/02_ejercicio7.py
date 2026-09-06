# %% [markdown]
# ## Ejercicio 7 — KNN con k=5: sin escalar vs. estandarizado

# %%
# --- Sin escalamiento ---
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
pred_raw = knn_raw.predict(X_test)
metricas_raw = calcular_metricas(y_test, pred_raw)
cm_raw = confusion_matrix(y_test, pred_raw)

# --- Estandarizado ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

knn_std = KNeighborsClassifier(n_neighbors=5)
knn_std.fit(X_train_s, y_train)
pred_std = knn_std.predict(X_test_s)
metricas_std = calcular_metricas(y_test, pred_std)
cm_std = confusion_matrix(y_test, pred_std)

tabla_ej7 = pd.DataFrame(
    [metricas_raw, metricas_std], index=["Sin escalamiento", "Estandarizado"]
).round(4)
print("Tabla 7.1 — Comparación k=5, sin escalar vs. estandarizado")
tabla_ej7

# %% [markdown]
# ### Matrices de confusión

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, cm, title in zip(
    axes, [cm_raw, cm_std], ["Sin escalamiento (k=5)", "Estandarizado (k=5)"]
):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0", "1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["0", "1"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11,
            )
fig.suptitle("Figura 7.1 — Matrices de confusión, k=5")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### a. ¿Cambió el desempeño tras estandarizar?
#
# Sí, y de forma sustancial en la capacidad de detectar la clase positiva:
# el Accuracy apenas se mueve, pero el Recall y el F1 aumentan de forma
# considerable (los valores exactos quedan en la Tabla 7.1 de arriba).
#
# ### b. ¿Qué métrica presentó el cambio más importante?
#
# El **Recall**. Esto es consistente con lo demostrado en el ejercicio 3e:
# sin estandarizar, variables como `MonthlyIncome` o `DebtRatio` (rangos
# de miles) dominan el cálculo de distancias frente a `age` o
# `NumberOfDependents` (rangos de decenas o unidades), de modo que el
# modelo termina casi ignorando estas últimas; al estandarizar, todas las
# variables aportan en una escala comparable y el modelo detecta
# muchos más verdaderos incumplimientos.
#
# ### c. Modelo elegido
#
# Se recomienda el modelo **estandarizado**: el Accuracy es prácticamente
# igual, pero el Recall y el F1 son considerablemente superiores. En un
# problema de riesgo crediticio, donde no detectar un incumplimiento suele
# costar más que una alerta equivocada (ver ejercicio 10), esa mejora es
# preferible en casi cualquier escenario de negocio razonable.
