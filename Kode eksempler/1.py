import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

# ------------------------------------------------------------
# Siste kapittel: oppsett av 2D Poisson/Laplace-problem
#
# Vi ser på domenet Ω = (0,1)×(0,1) og bruker Dirichlet-randbetingelser.
# Vi diskretiserer med 5-punkts stencil og bygger operatoren med Kroneckerprodukt.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1) Gitter i x- og y-retning (inkl. randpunkter)
# ------------------------------------------------------------
m = 20                      # antall INDRE punkter i x-retning
n = 20                      # antall INDRE punkter i y-retning

x = np.linspace(0, 1, m + 2)
y = np.linspace(0, 1, n + 2)

h = x[1] - x[0]             # dx
k = y[1] - y[0]             # dy

x_in = x[1:-1]              # indre punkter (m stk)
y_in = y[1:-1]              # indre punkter (n stk)

print(f"dx = {h:.4f}, dy = {k:.4f}")

# ------------------------------------------------------------
# 2) 2D-Laplace-operator L (størrelse (m*n)×(m*n))
#    L = kron(Lx, I) + kron(I, Ly)
# ------------------------------------------------------------

Lx = (1 / h**2) * (
    np.diag((m - 1) * [1], -1) +
    np.diag(m * [-2], 0) +
    np.diag((m - 1) * [1], 1)
)
Ix = np.eye(m)

Ly = (1 / k**2) * (
    np.diag((n - 1) * [1], -1) +
    np.diag(n * [-2], 0) +
    np.diag((n - 1) * [1], 1)
)
Iy = np.eye(n)

L = np.kron(Lx, Iy) + np.kron(Ix, Ly)

print("Størrelse på L:", L.shape)

# ------------------------------------------------------------
# 3) Randbetingelser (Dirichlet)
#
#   y = 0: u(x,0) = f1(x)
#   y = 1: u(x,1) = f2(x)
#   x = 0: u(0,y) = f3(y)
#   x = 1: u(1,y) = f4(y)
# ------------------------------------------------------------

def f1(x):
    """u(x,0) (bunnrand)."""
    return 0 * x

def f2(x):
    """u(x,1) (topprand)."""
    return np.sin(np.pi * x)

def f3(y):
    """u(0,y) (venstre rand)."""
    return 0 * y

def f4(y):
    """u(1,y) (høyre rand)."""
    return 0 * y

# ------------------------------------------------------------
# 4) Bygg høyresiden F fra randbetingelsene
#
# Randbidraget kommer fra at naboverdier utenfor det indre området
# erstattes av kjente randverdier og flyttes over til høyresiden.
# ------------------------------------------------------------

# "Plukketøy" for første/siste indre indeks i hver retning
Zm_l = np.zeros(m); Zm_l[0]  = -1 / h**2   # venstre side (x=0)
Zm_r = np.zeros(m); Zm_r[-1] = -1 / h**2   # høyre side (x=1)
Zn_l = np.zeros(n); Zn_l[0]  = -1 / k**2   # bunn (y=0)
Zn_r = np.zeros(n); Zn_r[-1] = -1 / k**2   # topp (y=1)

# Randbidrag samlet i én vektor (lengde m*n)
F = (
    np.kron(f1(x_in), Zn_l) +   # y=0
    np.kron(f2(x_in), Zn_r) +   # y=1
    np.kron(Zm_l, f3(y_in)) +   # x=0
    np.kron(Zm_r, f4(y_in))     # x=1
)

print("Størrelse på F:", F.shape)

# ------------------------------------------------------------
# 5) Liten sjekk: plott randfunksjonene i x-retning
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(x_in, f1(x_in), "--", label="u(x,0)")
plt.plot(x_in, f2(x_in), "-",  label="u(x,1)")
plt.xlabel("x")
plt.ylabel("randverdi")
plt.title("Randbetingelser på y=0 og y=1")
plt.legend()
plt.show()