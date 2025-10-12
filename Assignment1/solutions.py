####################################################################
##################          PART A            ######################
####################################################################
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

####################################################################
##################          PART B            ######################
####################################################################


import numpy as np
import matplotlib.pyplot as plt
t_var = np.load("t_var.npy")
y_var = np.load("y_var.npy")
plt.plot(t_var, y_var)
plt.show()

def create_W(p):
   ## generate W which is a p-2 x p matrix as defined in the question
    W = np.zeros((p-2, p))
    b = np.array([1, -2, 1])
    for i in range(p-2):
        W[i, i:i+3] = b 
    return W 

def loss(beta, y, W, L):
    ## compute loss for a given vector beta for data y, matrix W, regularization parameter L (lambda)
    # your code here 
    p = y.size
    r1 = y - beta
    r2 = W @ beta
    loss_val = 0.5 / p * float(r1 @ r1) + L * float(r2 @ r2)
    return loss_val

L = 0.9

p = y_var.size
W = create_W(p)
I = np.eye(p)
A = I + (2.0 * L * p) * (W.T @ W)
beta_hat = np.linalg.solve(A, y_var)
L_val = loss(beta_hat, y_var, W, L)

# I dont understand why the source code use y not y_var here. Nothing in source code has defined y, but it should be y_var
y = y_var

plt.plot(t_var, y_var, zorder=1, color='red', label='truth')
plt.plot(t_var, beta_hat, zorder=3, color='blue', 
            linewidth=2, linestyle='--', label='fit')
plt.legend(loc='best')
plt.show()


####################################################################
##################          PART E            ######################
####################################################################
import numpy as np
import matplotlib.pyplot as plt

t_var = np.load("t_var.npy")
y_data = np.load("y_var.npy")

lam = 0.001
p = y_data.size
W = np.zeros((p - 2, p))
tri = np.array([1.0, -2.0, 1.0])
for i in range(p - 2):
    W[i, i:i+3] = tri

A = np.eye(p) + (2.0 * lam * p) * (W.T @ W)
beta_hat = np.linalg.solve(A, y_data)
r1 = y_data - beta_hat
r2 = W @ beta_hat
L_star = 0.5 / p * float(r1 @ r1) + lam * float(r2 @ r2)

alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.2, 2.0]

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharey=True)

betas = [np.ones(p) for _ in alphas]
all_losses = [[] for _ in alphas]

# run 1000 epoch for different learning rates at once
for epoch in range(1000):
    
    for idx, a in enumerate(alphas):
        g = (betas[idx] - y_data) / p + 2.0 * lam * (W.T @ (W @ betas[idx]))
        betas[idx] = betas[idx] - a * g
        r1 = y_data - betas[idx]
        r2 = W @ betas[idx]
        all_losses[idx].append(0.5 / p * float(r1 @ r1) + lam * float(r2 @ r2))

for idx, a in enumerate(alphas):
    deltas = np.array(all_losses[idx]) - L_star
    
    ax = axes[idx // 3, idx % 3]
    ax.plot(range(1, 1001), deltas)
    ax.set_title(f"alpha = {a}")
    ax.grid(True)

plt.show()

####################################################################
##################          PART F            ######################
####################################################################

import numpy as np
import matplotlib.pyplot as plt

t_var = np.load("t_var.npy")
y_data = np.load("y_var.npy")

lam = 0.001
p = y_data.size
W = np.zeros((p - 2, p))
tri = np.array([1.0, -2.0, 1.0])
for i in range(p - 2):
    W[i, i:i+3] = tri

A = np.eye(p) + (2.0 * lam * p) * (W.T @ W)
beta_hat = np.linalg.solve(A, y_data)
r1 = y_data - beta_hat
r2 = W @ beta_hat
L_star = 0.5 / p * float(r1 @ r1) + lam * float(r2 @ r2)

alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.2, 2.0]

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharey=True)

betas = [np.ones(p) for _ in alphas]
all_losses = [[] for _ in alphas]

for epoch in range(4):
    for j in range(p):
        for idx, a in enumerate(alphas):
            g = 2.0 * lam * ((W.T @ W) @ betas[idx])
            g[j] += -(y_data[j] - betas[idx][j])
            g /= p
            
            betas[idx] = betas[idx] - a * g
            
            r1 = y_data - betas[idx]
            r2 = W @ betas[idx]
            all_losses[idx].append(0.5 / p * float(r1 @ r1) + lam * float(r2 @ r2))

for idx, a in enumerate(alphas):
    deltas = np.array(all_losses[idx]) - L_star
    
    ax = axes[idx // 3, idx % 3]
    ax.plot(range(1, 4 * p + 1), deltas)
    ax.set_title(f"alpha = {a}")
    ax.grid(True)

plt.show()


####################################################################
##################          PART H            ######################
####################################################################

import numpy as np
import matplotlib.pyplot as plt

y_data = np.load("y_var.npy")
p = y_data.size
W = np.zeros((p - 2, p))
tri = np.array([1.0, -2.0, 1.0])
for i in range(p - 2):
    W[i, i:i+3] = tri
M = W.T @ W

def loss_val(beta):
    r1 = y_data - beta
    r2 = W @ beta
    return 0.5 / p * float(r1 @ r1) + 0.001 * float(r2 @ r2)

A = np.eye(p) + (2.0 * 0.001 * p) * M
beta_hat = np.linalg.solve(A, y_data)
L_star = loss_val(beta_hat)
beta_cd = np.ones(p)
deltas_cd = [loss_val(beta_cd) - L_star]

diagM = np.diag(M)
for k in range(1000):
    j = k % p
    row_dot = M[j, :] @ beta_cd
    s_excl = row_dot - diagM[j] * beta_cd[j]
    denom = (1.0 / p) + 2.0 * 0.001 * diagM[j]
    numer = (1.0 / p) * y_data[j] - 2.0 * 0.001 * s_excl
    beta_cd[j] = numer / denom
    deltas_cd.append(loss_val(beta_cd) - L_star)

beta_gd = np.ones(p)
deltas_gd = [loss_val(beta_gd) - L_star]
for x in range(1000):
    g = (beta_gd - y_data) / p + 2.0 * 0.001 * (M @ beta_gd)
    beta_gd = beta_gd - 1.0 * g
    deltas_gd.append(loss_val(beta_gd) - L_star)

x_cd = np.arange(0, 1001)
x_gd = np.arange(0, 1001)
plt.plot(x_cd, deltas_cd, label="coordinate scheme", color="blue")
plt.plot(x_gd, deltas_gd, label="gradient descent", color="green")
plt.legend()
plt.grid(True)
plt.show()


