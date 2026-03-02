import numpy as np
import matplotlib.pyplot as plt

# 1. Parametere og diskretisering
N = 50                  # Antall intervaller
L = 1                   # Intervallet er [-L, L]
x = np.linspace(-L, L, N+1)
h = x[1] - x[0]         # Steglengde

# 2. Definer kildeleddet f(x) og eksakt løsning
f = np.cos(np.pi * x)
u_eksakt = x + 1 - (1/np.pi**2) * (1 + np.cos(np.pi * x)) # Analytisk løsning

# 3. Sett opp det numeriske systemet Au = b
# Vi løser for de indre punktene (1 til N-1)
n_inner = N - 1
A = (np.diag(np.ones(n_inner-1), -1) - 2*np.diag(np.ones(n_inner)) 
     + np.diag(np.ones(n_inner-1), 1)) / h**2

# Høyre side (b-vektoren)
b = f[1:-1].copy()

# Korriger b for randbetingelser: u(-1)=0 og u(1)=2
u_start = 0
u_slutt = 2
b[0] -= u_start / h**2
b[-1] -= u_slutt / h**2

# 4. Løs det lineære systemet
u_indre = np.linalg.solve(A, b)

# Sett sammen fullstendig løsningsvektor inkludert randpunktene
u_num = np.concatenate(([u_start], u_indre, [u_slutt]))

# 5. Visualisering
plt.figure(figsize=(10, 6))
plt.plot(x, u_eksakt, 'r-', label='Analytisk løsning', linewidth=2)
plt.plot(x, u_num, 'bo--', label='Numerisk løsning (FDM)', markersize=4, alpha=0.7)
plt.title('Løsning av 1D Poisson-ligning')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Skriv ut feilen (maksimal differanse)
print(f"Maksimal feil mellom analytisk og numerisk: {np.max(np.abs(u_eksakt - u_num)):.2e}")