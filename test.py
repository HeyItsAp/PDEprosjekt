import numpy as np
import matplotlib.pyplot as plt

# Parametere og diskretisering
N = 50                   # Antall intervaller
x = np.linspace(-1, 1, N + 1)
h = x[1] - x[0]         # Steglengde

# Kildeledd og analytisk løsning
f = np.cos(np.pi * x)
u_eksakt = x + 1 - (1 / np.pi**2) * (1 + np.cos(np.pi * x))

# Sett opp systemet Au = b for de indre punktene
n = N - 1
A = (np.diag(np.ones(n - 1), -1)
     - 2 * np.diag(np.ones(n))
     + np.diag(np.ones(n - 1), 1)) / h**2

b = f[1:-1].copy()

# Korriger b for randbetingelsene u(-1) = 0 og u(1) = 2
b[0]  -= 0 / h**2
b[-1] -= 2 / h**2

# Løs ligningssystemet
u_indre = np.linalg.solve(A, b)
u_num = np.concatenate(([0], u_indre, [2]))

# Visualisering
plt.plot(x, u_eksakt, 'r-',  label='Analytisk løsning', linewidth=2)
plt.plot(x, u_num,   'b--', label='Numerisk løsning (FDM)', linewidth=1.5)
plt.xlabel('x'); plt.ylabel('u(x)')
plt.title('1D Poisson-ligning – analytisk vs. numerisk løsning')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(f"Maksimal absoluttfeil: {np.max(np.abs(u_eksakt - u_num)):.2e}")