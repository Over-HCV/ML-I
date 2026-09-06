# %% [markdown]
# ## Ejercicio 8 — Efecto del número de vecinos k
#
# Se usa siempre el conjunto **estandarizado** (`X_train_s`, `X_test_s`)
# definido en el ejercicio 7.

# %%
valores_k = [1, 3, 5, 7, 9, 11, 15, 21]
resultados_k = []

for k in valores_k:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    pred = knn.predict(X_test_s)
    resultados_k.append({"k": k, **calcular_metricas(y_test, pred)})

tabla_ej8 = pd.DataFrame(resultados_k).round(4)
print("Tabla 8.1 — Métricas por valor de k (datos estandarizados)")
tabla_ej8

# %% [markdown]
# ### Gráfica: k vs. métricas

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tabla_ej8["k"], tabla_ej8["F1"], marker="o", color="#55A868", label="F1")
ax.plot(
    tabla_ej8["k"], tabla_ej8["Recall"], marker="s", linestyle="--",
    color="#C44E52", label="Recall",
)
ax.plot(
    tabla_ej8["k"], tabla_ej8["Accuracy"], marker="^", linestyle=":",
    color="#4C72B0", label="Accuracy",
)
ax.set_title("Figura 8.1 — Desempeño de KNN según el número de vecinos k\n(datos estandarizados)")
ax.set_xlabel("k (número de vecinos)")
ax.set_ylabel("Valor de la métrica")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
mejor_f1 = tabla_ej8.loc[tabla_ej8["F1"].idxmax()]
mejor_recall = tabla_ej8.loc[tabla_ej8["Recall"].idxmax()]
mejor_acc = tabla_ej8.loc[tabla_ej8["Accuracy"].idxmax()]
top3_k = tabla_ej8.sort_values("F1", ascending=False)["k"].head(3).astype(int).tolist()

print(f"a. Mejor F1:       k={int(mejor_f1['k'])}  (F1={mejor_f1['F1']:.4f})")
print(f"b. Mejor Recall:   k={int(mejor_recall['k'])}  (Recall={mejor_recall['Recall']:.4f})")
print(f"c. Mejor Accuracy: k={int(mejor_acc['k'])}  (Accuracy={mejor_acc['Accuracy']:.4f})")
print(f"   Top 3 valores de k según F1 (se usan en el ejercicio 9): {top3_k}")

# %% [markdown]
# ### d. ¿Coinciden los tres criterios en el mismo modelo?
#
# No: F1 y Recall se maximizan en el mismo k pequeño, mientras que
# Accuracy se maximiza en un k mayor (ver valores impresos arriba). Al
# aumentar k, el modelo promedia sobre vecindarios cada vez más grandes
# que, dado el fuerte desbalance de clases (93 %/7 %), terminan dominados
# por vecinos de la clase mayoritaria, empujando las predicciones hacia la
# clase 0 y sacrificando Recall a cambio de un Accuracy apenas mayor; un k
# pequeño, al fijarse en muy pocos vecinos, es más sensible a los (pocos)
# vecinos positivos cercanos, favoreciendo el Recall aunque a costa de
# mayor varianza (más falsos positivos, como se ve en el ejercicio 10).
