import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

plt.close("all")

# ============================================================
# Oppgave 6: Varmeligning 2D – steking av dansk rugbrød
# Bygget på kodeeksemplene 1.py – 4.py
# ============================================================

# ------------------------------------------------------------
# a) Materialparametere og geometri
#    (tilsvarer oppsett i kodeeksempel 1.py)
# ------------------------------------------------------------
alpha = 0.5 / (450 * 2700)   # k/(rho*cp)  ≈ 4.12e-7 m²/s

Lx = 0.11   # m  (bredde)
Ly = 0.11   # m  (høyde)

T_rand  = 200.0   # randbetingelse [°C]
T_start =  15.0   # initialbetingelse [°C]

# Antall INDRE punkter (som i kodeeksempel 1.py: m, n)
m = 48   # indre punkter i x-retning
n = 48   # indre punkter i y-retning

# Gitter inkl. randpunkter (m+2 og n+2 punkter totalt)
x = np.linspace(0, Lx, m + 2)
y = np.linspace(0, Ly, n + 2)
h = x[1] - x[0]   # dx
k = y[1] - y[0]   # dy

x_in = x[1:-1]    # kun indre x-punkter
y_in = y[1:-1]    # kun indre y-punkter

X, Y = np.meshgrid(x_in, y_in, indexing='ij')   # form: (m, n)

print(f"α   = {alpha:.2e} m²/s")
print(f"dx  = dy = {h*100:.3f} cm")
print(f"Gitter (indre): {m} × {n} = {m*n} punkter")

# ------------------------------------------------------------
# 2D Laplace-operator L  (direkte fra kodeeksempel 1.py)
#    L = kron(Lx_mat, I_n) + kron(I_m, Ly_mat)
# ------------------------------------------------------------
Lx_mat = (1 / h**2) * (
    np.diag((m - 1) * [1], -1) +
    np.diag(m       * [-2], 0) +
    np.diag((m - 1) * [1],  1)
)

Ly_mat = (1 / k**2) * (
    np.diag((n - 1) * [1], -1) +
    np.diag(n       * [-2], 0) +
    np.diag((n - 1) * [1],  1)
)

L_mat = np.kron(Lx_mat, np.eye(n)) + np.kron(np.eye(m), Ly_mat)

print("Størrelse på L:", L_mat.shape)

# ------------------------------------------------------------
# Randvektor F  (direkte fra kodeeksempel 1.py)
#   Alle fire kanter holdes på T_rand = 200 °C
# ------------------------------------------------------------
Zm_l = np.zeros(m);  Zm_l[0]  = -1 / h**2   # x = 0   (venstre)
Zm_r = np.zeros(m);  Zm_r[-1] = -1 / h**2   # x = Lx  (høyre)
Zn_l = np.zeros(n);  Zn_l[0]  = -1 / k**2   # y = 0   (bunn)
Zn_r = np.zeros(n);  Zn_r[-1] = -1 / k**2   # y = Ly  (topp)

F_vec = (
    np.kron(T_rand * np.ones(m), Zn_l) +   # bunn
    np.kron(T_rand * np.ones(m), Zn_r) +   # topp
    np.kron(Zm_l, T_rand * np.ones(n)) +   # venstre
    np.kron(Zm_r, T_rand * np.ones(n))     # høyre
)

print("Størrelse på F:", F_vec.shape)

# ------------------------------------------------------------
# Initialtilstand  (som i kodeeksempel 2.py)
#   u(x,y,0) = T_start = 15 °C overalt i legemet
# ------------------------------------------------------------
U0    = np.full_like(X, T_start)   # 2D-felt (m × n)
u0    = U0.flatten()               # flat vektor (m*n)

# ------------------------------------------------------------
# Tidssteg – CFL-betingelse
# ------------------------------------------------------------
dt_max = h**2 * k**2 / (2 * alpha * (h**2 + k**2))
dt     = 0.9 * dt_max
T_total = 7200.0
Nt      = int(T_total / dt) + 1

print(f"dt  = {dt:.2f} s,   Nt = {Nt}")

# ------------------------------------------------------------
# Euler-funksjon  (direkte fra kodeeksempel 2.py)
#   Løser ODE-systemet  u'(t) = g(u, t)
# ------------------------------------------------------------
def euler(g, u0, t0, t1, N):
    """
    Forlengs Euler-metode for ODE-systemet u' = g(u,t).
    Returnerer: u (N × dim), t (N)
    """
    t  = np.linspace(t0, t1, N)
    dt_e = t[1] - t[0]

    u = np.zeros((N, u0.size))
    u[0, :] = u0

    for i in range(N - 1):
        u[i + 1, :] = u[i, :] + dt_e * g(u[i, :], t[i])

    return u, t


# Høyresiden i varmelikninga: u_t = alpha * (L u - F)
def g(u_vec, t_val):
    return alpha * (L_mat @ u_vec - F_vec)


# Midtpunkt-indeks i flat vektor
mx      = m // 2
my      = n // 2
mid_idx = mx * n + my

# ============================================================
# Kjør simulering
# ============================================================
print("Starter simulering …")
u_all, t_grid = euler(g, u0, 0.0, T_total, Nt)
print("Simulering ferdig.")

# ============================================================
# Hjelpefunksjon: lim inn resultatet på et fullstendig gitter
#   (legger til randpunkter med T_rand = 200 °C)
# ============================================================
def full_field(u_vec):
    """Gjenskaper (m+2) × (n+2)-feltet med randpunkter."""
    U = np.full((m + 2, n + 2), T_rand)
    U[1:-1, 1:-1] = u_vec.reshape(m, n)
    return U

# ============================================================
# b) Varmeplot for utvalgte tidspunkter
#    (bruker imshow-stilen fra kodeeksempel 4.py)
# ============================================================
plot_times = [300, 900, 1800, 3600]   # sekunder

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for idx, t_val in enumerate(plot_times):
    step  = int(t_val / dt)
    step  = min(step, Nt - 1)
    Z_full = full_field(u_all[step, :])

    ax = axes[idx]
    im = ax.imshow(
        Z_full.T, origin='lower', cmap='RdYlBu_r',
        extent=[0, Lx * 100, 0, Ly * 100],
        vmin=T_start, vmax=T_rand, aspect='equal'
    )
    ax.set_title(f"t = {t_val:.0f} s  ({t_val/60:.0f} min)")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    plt.colorbar(im, ax=ax, label="Temperatur [°C]")

fig.suptitle("Varmeplot – dansk rugbrød i ovn (200 °C)", fontsize=13)
plt.tight_layout()
plt.show()

# ============================================================
# c) Når når midten 60 °C?
#    (plukker ut fra u_all, som i kodeeksempel 3.py)
# ============================================================
temp_center = u_all[:, mid_idx]   # hele tidsserien for midtpunktet
idx_60 = np.argmax(temp_center >= 60.0)

if temp_center[idx_60] >= 60.0:
    t60_time  = t_grid[idx_60]
    t60_field = full_field(u_all[idx_60, :])
    print(f"Midten nådde 60 °C etter {t60_time:.0f} s  ({t60_time/60:.1f} min)")

    # Varmeplot for dette tidspunktet
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    im2 = ax2.imshow(
        t60_field.T, origin='lower', cmap='RdYlBu_r',
        extent=[0, Lx * 100, 0, Ly * 100],
        vmin=T_start, vmax=T_rand, aspect='equal'
    )
    ax2.set_title(
        f"Midten nådde 60 °C\n"
        f"t = {t60_time:.0f} s  ({t60_time/60:.1f} min)"
    )
    ax2.set_xlabel("x [cm]")
    ax2.set_ylabel("y [cm]")
    plt.colorbar(im2, ax=ax2, label="Temperatur [°C]")
    plt.tight_layout()
    plt.show()

    # Temperaturkurve for midtpunktet over tid
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(t_grid / 60, temp_center, 'b-', linewidth=1.5)
    ax3.axhline(60, color='r', linestyle='--', label='60 °C')
    ax3.axvline(t60_time / 60, color='g', linestyle='--',
                label=f't = {t60_time/60:.1f} min')
    ax3.set_xlabel("Tid [min]")
    ax3.set_ylabel("Temperatur i midten [°C]")
    ax3.set_title("Temperaturutvikling i midtpunktet")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("Midten nådde ikke 60 °C i simuleringen.")

# ============================================================
# d) Animasjon  (direkte fra kodeeksempel 4.py)
# ============================================================
stride    = max(1, Nt // 200)
frame_idx = [i * stride for i in range(200) if i * stride < Nt]

# Felles fargeomfang (som i 4.py)
U_frames = np.array([u_all[j, :] for j in frame_idx])
vmin_anim = T_start
vmax_anim = T_rand

fig4, ax4 = plt.subplots(figsize=(5, 5))

Z0 = full_field(u_all[frame_idx[0], :])
im_anim = ax4.imshow(
    Z0.T, origin='lower', cmap='RdYlBu_r',
    extent=[0, Lx * 100, 0, Ly * 100],
    vmin=vmin_anim, vmax=vmax_anim, aspect='equal'
)
plt.colorbar(im_anim, ax=ax4, label="Temperatur [°C]")
ax4.set_xlabel("x [cm]")
ax4.set_ylabel("y [cm]")
title_anim = ax4.set_title("")

# Oppdateringsfunksjon (som i kodeeksempel 4.py)
def animate(frame_number):
    j = frame_idx[frame_number]
    Z = full_field(u_all[j, :])
    im_anim.set_data(Z.T)
    title_anim.set_text(
        f"t ≈ {t_grid[j]:.0f} s  ({t_grid[j]/60:.1f} min)"
    )
    return im_anim, title_anim

ani = animation.FuncAnimation(
    fig4, animate,
    frames=len(frame_idx),
    interval=50,
    blit=True
)

HTML(ani.to_jshtml())
plt.show()
