from __future__ import annotations
import numpy as np

def grad(A: np.ndarray, b: np.ndarray, x: np.ndarray, y: float) -> np.ndarray:
    result = A.T @ (A @ x - b) + y * x
    return result

def des(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    a: float = 0.01,
    y: float = 2.0,
    acc: float = 1e-3,
):
    x = x0.copy().astype(float)
    xs = [x.copy()]
    g  = grad(A, b, x, y)
    k  = 0

    while np.linalg.norm(g, 2) >= acc:
        x = x - a * g
        xs.append(x.copy())
        g = grad(A, b, x, y)
        k += 1

    return np.array(xs), k, np.linalg.norm(g, 2)

def printResult(xs: np.ndarray, first_k_inclusive: int = 5, last_rows: int = 5):
    n_total = xs.shape[0]
    first_end = min(first_k_inclusive, n_total - 1)
    
    print("First 5:")
    for k in range(0, first_end + 1):
        print(f"k = {k},  x^(k) = {[round(float(x), 4) for x in xs[k].tolist()]}")

    print("Last 5:")
    start_last = max(0, n_total - last_rows)
    for idx in range(start_last, n_total):
        print(f"k = {idx},  x^(k) = {[round(float(x), 4) for x in xs[idx].tolist()]}")

A = np.array(
    [
    [ 3,  2,  0, -1],
    [-1,  3,  0,  2],
    [ 0, -4, -2,  7],
    ], dtype=float)

b = np.array([3, 1, -4], dtype=float)

y = 2.0
a = 0.01
x0 = np.array([1, 1, 1, 1], dtype=float)

xs, iters, grad_norm = des(
    A=A, b=b, x0=x0, a=a, y=y, acc=1e-3
)

printResult(xs, first_k_inclusive=5, last_rows=5)

print("Iters: ", iters)
print("Result:", round(float(grad_norm), 6))