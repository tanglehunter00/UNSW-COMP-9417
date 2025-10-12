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
