import numpy as np
import matplotlib.pyplot as plt

# --- Parametere fra 6a ---
Lx, Ly = 0.11, 0.10
m, n = 30, 30  # Antall indre gitterpunkter
dx, dy = Lx / (m + 1), Ly / (n + 1)
alpha = 4.1e-7
T_ovn = 200.0

# --- Konstruksjon av matrisen A og vektoren F ---
dim = m * n
A = np.zeros((dim, dim))
F = np.zeros(dim)

for j in range(n):
    for i in range(m):
        k = i + j * m
        A[k, k] = -2/dx**2 - 2/dy**2
        
        # Sjekk naboer og legg til randbidrag i F hvis vi er på kanten
        if i > 0: A[k, k-1] = 1/dx**2
        else: F[k] -= T_ovn/dx**2 # Venstre rand (200 grader)
        
        if i < m-1: A[k, k+1] = 1/dx**2
        else: F[k] -= T_ovn/dx**2 # Høyre rand (200 grader)
        
        if j > 0: A[k, k-m] = 1/dy**2
        else: F[k] -= T_ovn/dy**2 # Bunn-rand (200 grader)
        
        if j < n-1: A[k, k+m] = 1/dy**2
        else: F[k] -= T_ovn/dy**2 # Topp-rand (200 grader)

# Skaler med alfa
L = alpha * A
F_vec = alpha * F

# --- Tidsintegrasjon (Forlengs Euler) ---
dt = 0.5 
t_slutt = 3600 # 60 minutter
N_steps = int(t_slutt / dt)

u = np.full(dim, 15.0) # Initialbetingelse: 15 grader

# Lagring for plott (vi tar ut noen tidspunkter)
plots = {}
target_times = [300, 1200, 3600] # 5, 20 og 60 minutter

for s in range(N_steps + 1):
    t_curr = s * dt
    if int(t_curr) in target_times:
        plots[int(t_curr)] = u.copy()
    
    # u_{n+1} = u_n + dt * (L*u_n - F_vec)
    u = u + dt * (L @ u - F_vec)

# --- Visualisering ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (time, data) in enumerate(plots.items()):
    # Rekonstruer hele brødet (200 grader)
    full_grid = np.full((m + 2, n + 2), T_ovn)
    full_grid[1:-1, 1:-1] = data.reshape((m, n))
    
    im = axes[i].imshow(full_grid.T, origin="lower", extent=[0, Lx, 0, Ly], 
                        cmap="coolwarm", vmin=15, vmax=200)
    axes[i].set_title(f"Tid: {time//60} minutter")

plt.colorbar(im, ax=axes, label="Temperatur (°C)")
plt.show()




