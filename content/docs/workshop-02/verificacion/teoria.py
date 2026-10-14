"""
Verificación numérica independiente de la Parte I (ejercicios 1 a 5).

Responsabilidad única: recalcular toda la aritmética que aparece en
latex/secciones/ejercicio01..05.tex, con fracciones exactas cuando es
posible, para contrastar cada cifra del documento contra este script.

    uv run python content/docs/workshop-02/verificacion/teoria.py
"""
from fractions import Fraction as F
from math import exp, log

import numpy as np


def titulo(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def f4(v):
    return f"{float(v):.4f}"


# ---------------------------------------------------------------------
titulo("Ejercicio 1 — Regresión lineal simple")


def ajuste_simple(x, y):
    n = len(x)
    xb, yb = F(sum(x), n), F(sum(y), n)
    sxy = sum((xi - xb) * (yi - yb) for xi, yi in zip(x, y))
    sxx = sum((xi - xb) ** 2 for xi in x)
    b1 = sxy / sxx
    b0 = yb - b1 * xb
    return xb, yb, sxy, sxx, b0, b1


x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
xb, yb, sxy, sxx, b0, b1 = ajuste_simple(x, y)
print(f"x̄={xb}, ȳ={yb}, Sxy={sxy}, Sxx={sxx}, β1={b1}, β0={b0}")
print(f"Σx={sum(x)}, Σy={sum(y)}, Σx²={sum(v*v for v in x)}, Σxy={sum(a*b for a, b in zip(x, y))}")
yhat = [b0 + b1 * xi for xi in x]
e = [yi - yh for yi, yh in zip(y, yhat)]
print("ŷ =", [f4(v) for v in yhat])
print("e =", [f4(v) for v in e], " Σe =", sum(e))
sse = sum(v * v for v in e)
sst = sum((yi - yb) ** 2 for yi in y)
print(f"SSE={sse}={f4(sse)}, MSE={f4(sse / 5)}, SST={sst}, MSE constante={f4(sst / 5)}, R²={f4(1 - sse / sst)}")
print(f"Reducción relativa del MSE frente a ȳ: {f4(1 - sse / sst)}")
pred6 = b0 + b1 * 6

y2 = [2, 4, 5, 4, 12]
xb2, yb2, sxy2, sxx2, b02, b12 = ajuste_simple(x, y2)
print(f"\n[d] y5=12: ȳ={yb2}, Sxy={sxy2}, β1={b12}, β0={b02}")
pred6b = b02 + b12 * 6
print(f"ŷ(6): original={f4(pred6)}, modificado={f4(pred6b)}")
yhat2 = [b02 + b12 * xi for xi in x]
e2 = [yi - yh for yi, yh in zip(y2, yhat2)]
print("ŷ' =", [f4(v) for v in yhat2], " e' =", [f4(v) for v in e2])
sse2 = sum(v * v for v in e2)
sst2 = sum((yi - yb2) ** 2 for yi in y2)
print(f"SSE'={f4(sse2)}, R²'={f4(1 - sse2 / sst2)}")
print(f"ŷ(20): original={f4(b0 + b1 * 20)}, modificado={f4(b02 + b12 * 20)}")

# ---------------------------------------------------------------------
titulo("Ejercicio 2 — Regresión lineal múltiple")
X = np.array([[1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 0, 1], [1, 1, 1], [1, 2, 1]], dtype=float)
yv = np.array([2, 4, 5, 5, 6, 8], dtype=float)
XtX = X.T @ X
Xty = X.T @ yv
print("X^T X =\n", XtX)
print("X^T y =", Xty)
beta = np.linalg.solve(XtX, Xty)
print("β̂ =", [f4(v) for v in beta])
# Comprobación exacta con fracciones (regla de Cramer sobre el sistema 3x3)
A = [[F(int(v)) for v in fila] for fila in XtX]
bv = [F(int(v)) for v in Xty]


def det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


D = det3(A)
sol = []
for j in range(3):
    Aj = [fila[:] for fila in A]
    for i in range(3):
        Aj[i][j] = bv[i]
    sol.append(det3(Aj) / D)
print("β̂ exacto =", sol, " det(X^T X) =", D)
yh = X @ beta
res = yv - yh
print("ŷ =", [f4(v) for v in yh], " e =", [f4(v) for v in res])
print(f"MSE = {f4(np.mean(res ** 2))}  (SSE={f4(np.sum(res ** 2))}),  Σe={np.sum(res):.2e}")
print(f"ŷ(1.5, 1) = {f4(beta @ [1, 1.5, 1])} centenas = {float(beta @ [1, 1.5, 1]) * 100:.2f} unidades")
X3 = np.column_stack([X, 2 * X[:, 1]])
print("rank([1,x1,x2,2x1]) =", np.linalg.matrix_rank(X3), " det(X3^T X3) =", f4(np.linalg.det(X3.T @ X3)))
bpinv = np.linalg.pinv(X3) @ yv
print("β̂ (pseudoinversa, norma mínima) =", [f4(v) for v in bpinv],
      " ŷ iguales:", np.allclose(X3 @ bpinv, yh))
print(f"β1 + 2β3 (pinv) = {f4(bpinv[1] + 2 * bpinv[3])}")

# ---------------------------------------------------------------------
titulo("Ejercicio 3 — Regresión logística")
sig = lambda z: 1 / (1 + exp(-z))  # noqa: E731
print(f"e^0.6 = {exp(0.6):.4f}  (cambio de odds = +{(exp(0.6) - 1) * 100:.2f} %);  e^1.2 = {exp(1.2):.4f}")
for (a, b) in [((2, 1), (3, 1)), ((4, 2), (5, 2))]:
    pa = sig(-3 + 0.6 * a[0] + 1.2 * a[1]); pb = sig(-3 + 0.6 * b[0] + 1.2 * b[1])
    print(f"x={a}→{b}: p {pa:.4f}→{pb:.4f} (+{(pb - pa) * 100:.2f} pp), "
          f"odds {pa / (1 - pa):.4f}→{pb / (1 - pb):.4f} (×{(pb / (1 - pb)) / (pa / (1 - pa)):.4f})")
maquinas = {"A": (2, 1), "B": (5, 0), "C": (4, 2)}
for nom, (x1, x2) in maquinas.items():
    z = -3 + 0.6 * x1 + 1.2 * x2
    p = sig(z)
    print(f"{nom}: z={z:.4f}, p={p:.4f}, odds={p / (1 - p):.4f}, "
          f"ŷ(t=.5)={int(p >= .5)}, ŷ(t=.8)={int(p >= .8)}, ŷ(t=.2)={int(p >= .2)}")
for t in (0.5, 0.8, 0.2):
    lo = log(t / (1 - t))
    print(f"t={t}: logit(t)={lo:.4f};  frontera 0.6x1 + 1.2x2 = {3 + lo:.4f}"
          f"  →  x2 = {(3 + lo) / 1.2:.4f} - 0.5 x1")
# Umbral de costo: predecir 1 si 2(1-p) <= 8p  ⇔  p >= 2/10
print("t* =", F(2, 2 + 8))
# Ejemplo numérico para (e): MCO sobre un caso que cae fuera de [0, 1]
xe = np.array([0, 1, 2, 3, 10.0]); ye = np.array([0, 0, 1, 1, 1.0])
Xe = np.column_stack([np.ones_like(xe), xe])
be = np.linalg.lstsq(Xe, ye, rcond=None)[0]
print(f"(e) MCO sobre x={xe.tolist()}, y={ye.tolist()}: β={[f4(v) for v in be]}, "
      f"ŷ(0)={f4(be[0])}, ŷ(15)={f4(be[0] + 15 * be[1])}")

# ---------------------------------------------------------------------
titulo("Ejercicio 4 — Pérdida, gradiente y regularización")
xs = np.array([-1.0, 0.0, 1.0]); ys = np.array([0.0, 0.0, 1.0])


def perdida(b0, b1):
    p = 1 / (1 + np.exp(-(b0 + b1 * xs)))
    J = -np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p))
    g0 = np.mean(p - ys); g1 = np.mean((p - ys) * xs)
    return p, J, g0, g1


p, J, g0, g1 = perdida(0, 0)
print(f"p={p}, J={J:.4f} (= ln 2 = {log(2):.4f}), ∂J/∂β0={g0:.4f} (=1/6), ∂J/∂β1={g1:.4f} (=-1/3)")
eta = 0.5
nb0, nb1 = 0 - eta * g0, 0 - eta * g1
p2, J2, g02, g12 = perdida(nb0, nb1)
print(f"β nuevo = ({nb0:.4f}, {nb1:.4f}) = (-1/12, 1/6)")
print("z nuevos =", [f4(v) for v in nb0 + nb1 * xs])
print("p nuevas =", [f4(v) for v in p2], f" J nueva = {J2:.4f}, ΔJ = {J2 - J:.4f}")
print(f"gradiente en β nuevo: ∂J/∂β0={g02:.4f}, ∂J/∂β1={g12:.4f}")
lam = 1
print(f"[d] λ=1: ∂Jλ/∂β0 = {g02:.4f}, ∂Jλ/∂β1 = {g12:.4f} + {lam}·{nb1:.4f} = {g12 + lam * nb1:.4f}")
print(f"     Jλ en β nuevo = {J2 + lam * nb1 ** 2 / 2:.4f}")

# ---------------------------------------------------------------------
titulo("Ejercicio 5 — Árboles de decisión")


def gini(n0, n1):
    n = n0 + n1
    return 1 - F(n0, n) ** 2 - F(n1, n) ** 2


gp = gini(6, 6)
print(f"G padre = {gp} = {f4(gp)}")
for nom, (L, R) in {"A": ((5, 1), (1, 5)), "B": ((4, 0), (2, 6))}.items():
    gl, gr = gini(*L), gini(*R)
    nl, nr = sum(L), sum(R)
    pond = F(nl, 12) * gl + F(nr, 12) * gr
    sinpeso = (gl + gr) / 2
    print(f"{nom}: G_L={gl}={f4(gl)}, G_R={gr}={f4(gr)}, ponderado={pond}={f4(pond)}, "
          f"ΔG={gp - pond}={f4(gp - pond)}, promedio sin pesos={f4(sinpeso)} (Δ={f4(gp - sinpeso)})")
    print(f"   P(1|L)={f4(F(L[1], nl))}, P(1|R)={f4(F(R[1], nr))}")

xr = np.array([1, 2, 3, 4, 5, 6.0]); yr = np.array([2, 3, 7, 8, 9, 12.0])
print(f"SSE sin dividir = {f4(np.sum((yr - yr.mean()) ** 2))} (media {f4(yr.mean())})")
for corte in (2.5, 4.5):
    izq, der = yr[xr <= corte], yr[xr > corte]
    sl = np.sum((izq - izq.mean()) ** 2); sr = np.sum((der - der.mean()) ** 2)
    print(f"x<={corte}: medias=({f4(izq.mean())}, {f4(der.mean())}), "
          f"SSE_L={f4(sl)}, SSE_R={f4(sr)}, total={f4(sl + sr)}")
for corte in (1.5, 3.5, 5.5):
    izq, der = yr[xr <= corte], yr[xr > corte]
    tot = np.sum((izq - izq.mean()) ** 2) + np.sum((der - der.mean()) ** 2)
    print(f"   (referencia) x<={corte}: total={f4(tot)}")
b1r, b0r = np.polyfit(xr, yr, 1)
print(f"Tendencia MCO de referencia (figura 5d): ŷ = {b0r:.4f} + {b1r:.4f} x")
