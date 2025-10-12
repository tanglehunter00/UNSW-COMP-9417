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
