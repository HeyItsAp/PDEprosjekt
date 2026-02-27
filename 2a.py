import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------
# Parameters
# -----------------------
xmin, xmax = -1000, 1000
nx = 2000
dx = (xmax - xmin) / nx
x = np.linspace(xmin, xmax, nx)

umax = 1.0          # Maximum density
T = 800             # Simulation time
dt = 0.4 * dx       # CFL condition
nt = int(T / dt)

# -----------------------
# Initial condition
# u(x,0) = umax for x<0
# u(x,0) = 0 for x>0
# -----------------------
u = np.zeros(nx)
u[x < 0] = umax

# Flux function
def flux(u):
    return u * (1 - u)

# Godunov scheme for LWR
def godunov(u):
    u_new = u.copy()
    for i in range(1, nx - 1):
        # Compute numerical flux
        if u[i-1] <= u[i]:
            f_left = min(flux(u[i-1]), flux(u[i]))
        else:
            f_left = max(flux(u[i-1]), flux(u[i]))

        if u[i] <= u[i+1]:
            f_right = min(flux(u[i]), flux(u[i+1]))
        else:
            f_right = max(flux(u[i]), flux(u[i+1]))

        u_new[i] = u[i] - dt/dx * (f_right - f_left)

    return u_new

# -----------------------
# Plot setup
# -----------------------
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot(x, u, lw=2)
ax.set_xlim(xmin, xmax)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Position x")
ax.set_ylabel("Tetthet (x,t)")
ax.set_title("Skisse til Trafikken ved start")
ax.axvline(0, linestyle="--")  # Traffic light position

plt.show()