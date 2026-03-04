import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Parametere (samme som før) ---
Lx, Ly = 0.11, 0.10
m, n = 30, 30
dx, dy = Lx / (m + 1), Ly / (n + 1)
alpha = 4.1e-7
T_ovn = 200.0
T_start = 15.0

# --- 2. Matriseoppsett (A og F fra din 6b/c) ---
dim = m * n
A = np.zeros((dim, dim))
F = np.zeros(dim)
for j in range(n):
    for i in range(m):
        k = i + j * m
        A[k, k] = -2/dx**2 - 2/dy**2
        if i > 0: A[k, k-1] = 1/dx**2
        else: F[k] -= T_ovn/dx**2
        if i < m-1: A[k, k+1] = 1/dx**2
        else: F[k] -= T_ovn/dx**2
        if j > 0: A[k, k-m] = 1/dy**2
        else: F[k] -= T_ovn/dy**2
        if j < n-1: A[k, k+m] = 1/dy**2
        else: F[k] -= T_ovn/dy**2

L = alpha * A
F_vec = alpha * F

# --- 3. Oppsett for animasjon ---
dt = 2.0  # Vi hopper litt i tid per bilde for at animasjonen skal gå fortere
u = np.full(dim, T_start)

fig, ax = plt.subplots(figsize=(7, 6))
full_grid = np.full((m + 2, n + 2), T_ovn)
full_grid[1:-1, 1:-1] = u.reshape((m, n))

# Initial-plot
im = ax.imshow(full_grid.T, origin="lower", extent=[0, Lx, 0, Ly], 
               cmap="coolwarm", vmin=15, vmax=200)
plt.colorbar(im, label="Temperatur (°C)")
title = ax.set_title("Tid: 0 minutter")

def update(frame):
    global u
    # Vi kjører f.eks. 10 tidssteg per bilde for flyt
    for _ in range(10):
        u = u + dt * (L @ u - F_vec)
    
    # Oppdater bildet
    full_grid[1:-1, 1:-1] = u.reshape((m, n))
    im.set_array(full_grid.T)
    title.set_text(f"Tid: {(frame * dt * 10) / 60:.1f} minutter")
    return im, title

# Lag animasjonen (100 bilder)
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()