# %% [markdown]
# ## Ejercicio 10 — Análisis final y costo de negocio
#
# Se comparan tres modelos representativos de los ejercicios anteriores:
# el de mejor F1/Recall y el de mejor Accuracy (ambos del ejercicio 8,
# distancia euclidiana sobre datos estandarizados), y el modelo base sin
# estandarizar del ejercicio 7. El costo de negocio se define como
# $L = FP + 5FN$.

# %%
def evaluar_modelo(k, metric, X_tr, X_te):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_tr, y_train)
    pred = knn.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    m = calcular_metricas(y_test, pred)
    return {"k": k, "TP": tp, "TN": tn, "FP": fp, "FN": fn, **m, "Costo L": fp + 5 * fn}


modelo_1 = evaluar_modelo(int(mejor_f1["k"]), "euclidean", X_train_s, X_test_s)  # mejor F1/Recall
modelo_2 = evaluar_modelo(int(mejor_acc["k"]), "euclidean", X_train_s, X_test_s)  # mejor Accuracy

tn3, fp3, fn3, tp3 = cm_raw.ravel()
modelo_3 = {
    "k": 5, "TP": tp3, "TN": tn3, "FP": fp3, "FN": fn3,
    **metricas_raw, "Costo L": fp3 + 5 * fn3,
}

tabla_ej10 = pd.DataFrame(
    [modelo_1, modelo_2, modelo_3],
    index=[
        "Modelo 1 (mejor F1/Recall)",
        "Modelo 2 (mejor Accuracy)",
        "Modelo 3 (base, sin estandarizar)",
    ],
)[["k", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1", "Costo L"]].round(4)

print("Tabla 10.1 — Comparación final de modelos")
tabla_ej10

# %%
menor_costo = tabla_ej10["Costo L"].idxmin()
mayor_acc_final = tabla_ej10["Accuracy"].idxmax()
menor_fn = tabla_ej10["FN"].idxmin()

print(f"Modelo con mayor Accuracy: {mayor_acc_final}")
print(f"Modelo con menor costo L:  {menor_costo}")
print(f"Modelo con menos falsos negativos: {menor_fn}")

# %% [markdown]
# ### a. ¿El modelo de mayor Accuracy es también el de menor costo?
#
# Se responde comparando directamente `mayor_acc_final` y `menor_costo`
# impresos arriba. Si coinciden, vale la pena anotarlo con honestidad:
# esto **no contradice** lo demostrado teóricamente en el ejercicio 5f
# (allí se construyó un ejemplo específico donde ambos criterios sí
# divergían); simplemente muestra que la relación entre Accuracy y costo
# depende de la forma concreta de los errores de cada modelo, y no es una
# ley universal en ningún sentido.
#
# ### b. ¿Qué modelo tiene menos falsos negativos?
#
# Ver `menor_fn` impreso arriba — coincide con el modelo de mayor Recall,
# lo cual es consistente por definición (Recall = TP/(TP+FN)).
#
# ### c. ¿Qué modelo recomendaría para identificar clientes de alto riesgo?
#
# El modelo con mayor Recall (menor FN), aunque no tenga ni el mayor
# Accuracy ni el menor costo L. Si el objetivo explícito del negocio es
# maximizar la detección de incumplimientos reales, ese modelo detecta
# proporcionalmente muchos más de los verdaderos incumplimientos del
# conjunto de prueba que las demás opciones; el costo L tal como está
# definido pondera los falsos positivos con peso 1, lo cual puede
# subestimar el valor real de detectar más incumplimientos si, en la
# práctica, el costo de no detectar a un mal pagador es incluso mayor a 5
# veces el de una alerta equivocada.
#
# ### d. Justificación final
#
# La comparación deja en evidencia una tensión real entre tres criterios
# de selección —Accuracy, un costo explícito ponderado, y Recall puro—
# que no siempre apuntan al mismo modelo. La elección final no es un
# ejercicio puramente estadístico, sino una decisión de negocio que
# depende de cuánto le cuesta realmente a la institución financiera cada
# tipo de error: si ese costo relativo real fuera distinto al 5:1 supuesto
# en el taller, el balance entre los modelos podría cambiar.
