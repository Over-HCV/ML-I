# %% [markdown]
# ## Ejercicio 9 — Distancia Euclidiana vs. Manhattan
#
# Se comparan los tres mejores valores de k según F1 obtenidos en el
# ejercicio 8 (`top3_k`).

# %%
resultados_dist = []
for k in top3_k:
    for dist in ["euclidean", "manhattan"]:
        knn = KNeighborsClassifier(n_neighbors=k, metric=dist)
        knn.fit(X_train_s, y_train)
        pred = knn.predict(X_test_s)
        resultados_dist.append(
            {"k": k, "Distancia": dist, **calcular_metricas(y_test, pred)}
        )

tabla_ej9 = pd.DataFrame(resultados_dist).round(4)
print(f"Tabla 9.1 — Euclidiana vs. Manhattan para k={top3_k}")
tabla_ej9

# %%
mejor_f1_dist = tabla_ej9.loc[tabla_ej9["F1"].idxmax()]
mejor_recall_dist = tabla_ej9.loc[tabla_ej9["Recall"].idxmax()]

print(
    f"b. Mejor F1:     k={int(mejor_f1_dist['k'])}, "
    f"distancia={mejor_f1_dist['Distancia']} (F1={mejor_f1_dist['F1']:.4f})"
)
print(
    f"c. Mejor Recall: k={int(mejor_recall_dist['k'])}, "
    f"distancia={mejor_recall_dist['Distancia']} (Recall={mejor_recall_dist['Recall']:.4f})"
)

# %% [markdown]
# ### a. ¿La métrica de distancia modifica el desempeño?
#
# Sí, pero de forma mucho más discreta que la estandarización o que el
# valor de k: para cada k, Manhattan suele dar Accuracy/Precision
# levemente superiores pero Recall/F1 levemente inferiores a Euclidiana
# (ver Tabla 9.1), con diferencias pequeñas frente a las de los
# ejercicios 7 y 8.
#
# ### d. ¿Coincide la mejor configuración según F1 y según Recall?
#
# Se compara directamente con los valores impresos arriba: cuando ambas
# apuntan al mismo par (k, distancia), la coincidencia tiene sentido
# porque, a diferencia del trade-off marcado entre valores de k muy
# distintos (ejercicio 8), aquí cambiar solo la métrica de distancia no
# altera lo suficiente cuál vecino es "más cercano" como para invertir el
# orden de los modelos — con los datos ya estandarizados, penalizar
# cuadráticamente (Euclidiana) o linealmente (Manhattan) las diferencias
# por variable pesa menos que la elección de k.
