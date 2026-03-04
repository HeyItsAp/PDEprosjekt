import numpy as np
import matplotlib.pyplot as plt

# --- Parametere fra 6a ---
Lx, Ly = 0.11, 0.10
m, n = 30, 30  # Antall indre gitterpunkter
dx, dy = Lx / (m + 1), Ly / (n + 1)
alpha = 4.1e-7
T_ovn = 200.0
T_start = 15.0
T_maal = 60.0

# --- Konstruksjon av matrisen A og vektoren F ---
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

# --- Simulering (6c) ---
dt = 0.5 
u = np.full(dim, T_start)
t = 0.0

# Finn indeksen til midtpunktet i vektoren u
mid_i, mid_j = m // 2, n // 2
mid_k = mid_i + mid_j * m

# Kjør simulering til midtpunktet når 60 grader
while u[mid_k] < T_maal:
    u = u + dt * (L @ u - F_vec)
    t += dt

print(f"Midten av brødet når {T_maal} grader etter {t/60:.2f} minutter.")

# --- Visualisering ---
plt.figure(figsize=(7, 6))

# Rekonstruer hele brødet inkludert randa
full_grid = np.full((m + 2, n + 2), T_ovn)
full_grid[1:-1, 1:-1] = u.reshape((m, n))

im = plt.imshow(full_grid.T, origin="lower", extent=[0, Lx, 0, Ly], 
                cmap="coolwarm", vmin=15, vmax=200)

plt.colorbar(im, label="Temperatur (°C)")
plt.title(f"Temperaturfordeling når midten er {T_maal}°C\nTid: {t/60:.1f} minutter")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()