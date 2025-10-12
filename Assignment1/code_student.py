
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
