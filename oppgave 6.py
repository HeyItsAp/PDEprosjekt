import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# ============================================================
# Oppgave 6: Varmeligning 2D – steking av dansk rugbrød
# ============================================================

# ------------------------------------------------------------
# a) Materialparametere og geometri
# ------------------------------------------------------------
# Materiale: dansk rugbrød
#   k   = 0.5  W/(m·K)   – varmeledningsevne
#   rho = 450  kg/m³      – tetthet
#   cp  = 2700 J/(kg·K)   – spesifikk varmekapasitet
#
# Termisk diffusivitet:
#   α = k / (rho * cp) = 0.5 / (450 * 2700) ≈ 4.12e-7 m²/s
alpha = 0.5 / (450 * 2700)   # m²/s  ≈ 4.12e-7

# Geometri: kvadratisk tverrsnitt 0.11 m × 0.11 m
Lx = 0.11   # m
Ly = 0.11   # m

# ------------------------------------------------------------
# b) Rand- og initialbetingelser
# ------------------------------------------------------------
# u(x,y,t) = 200 °C  på alle kanter  (Dirichlet-randbetingelse)
# u(x,y,0) = 15  °C  i hele legemet  (initialbetingelse)
T_rand  = 200.0   # °C
T_start =  15.0   # °C

# ------------------------------------------------------------
# Diskretisering
# ------------------------------------------------------------
Nx = 50   # antall gitterpunkter i x-retning
Ny = 50   # antall gitterpunkter i y-retning

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Stabilitetskrav for eksplisitt metode (CFL):  dt ≤ dx²dy² / (2α(dx²+dy²))
dt_max = dx**2 * dy**2 / (2 * alpha * (dx**2 + dy**2))
dt = 0.9 * dt_max   # ta litt sikkerhetsmargin

T_total = 7200.0   # simuler 2 timer (7200 s)
Nt = int(T_total / dt) + 1

print(f"α = {alpha:.2e} m²/s")
print(f"dx = dy = {dx*100:.3f} cm,   dt = {dt:.2f} s")
print(f"Antall tidssteg: {Nt}")

# ------------------------------------------------------------
# Sett opp gitter og initialbetingelse
# ------------------------------------------------------------
# u[i, j] = temperatur i punkt (x_i, y_j)
u = np.full((Nx, Ny), T_start)

# Påfør randbetingelsene (kantene holdes på 200 °C hele tiden)
u[0,  :] = T_rand   # venstre kant  (x = 0)
u[-1, :] = T_rand   # høyre kant    (x = Lx)
u[:,  0] = T_rand   # bunn          (y = 0)
u[:, -1] = T_rand   # topp          (y = Ly)

# Koeffisienter i differanseskjemaet
rx = alpha * dt / dx**2
ry = alpha * dt / dy**2

# Indekser for indre punkter
ix = slice(1, Nx - 1)
iy = slice(1, Ny - 1)

# Midtpunkt-indeks
mx = Nx // 2
my = Ny // 2

# Tidspunkter for varmeplot (b)
plot_times = [300, 900, 1800, 3600]   # sekunder
plot_nsteps = {int(t / dt): t for t in plot_times}

# Lagring
saved_fields = {}   # tid → temperaturfeltet (kopi)
time_center   = []  # (t, T_midt) for hvert tidssteg
t60_field     = None
t60_time      = None

# For animasjon: lagre hvert N-te steg (for å holde minne nede)
anim_skip = max(1, Nt // 200)
anim_frames = []
anim_times  = []

# ------------------------------------------------------------
# Tidsløkke – eksplisitt forward Euler
# u^{n+1}_{i,j} = u^n_{i,j}
#   + α·dt/dx² · (u_{i-1,j} - 2u_{i,j} + u_{i+1,j})
#   + α·dt/dy² · (u_{i,j-1} - 2u_{i,j} + u_{i,j+1})
# ------------------------------------------------------------
for n in range(Nt):
    T_mid = u[mx, my]
    time_center.append((n * dt, T_mid))

    # Lagre felt for varmeplot
    if n in plot_nsteps:
        saved_fields[plot_nsteps[n]] = u.copy()

    # Lagre tidspunkt for 60 °C i midten (første gang)
    if t60_time is None and T_mid >= 60.0:
        t60_time  = n * dt
        t60_field = u.copy()
        print(f"Midten nådde 60 °C etter {t60_time:.0f} s  ({t60_time/60:.1f} min)")

    # Lagre animasjonsbilder
    if n % anim_skip == 0:
        anim_frames.append(u.copy())
        anim_times.append(n * dt)

    # Tidssteg (vektorisert for hastighet)
    u_new = u.copy()
    u_new[ix, iy] = (
        u[ix, iy]
        + rx * (u[0:Nx-2, iy] - 2*u[ix, iy] + u[2:Nx, iy])
        + ry * (u[ix, 0:Ny-2] - 2*u[ix, iy] + u[ix, 2:Ny])
    )
    # Hold randbetingelsene
    u_new[0,  :] = T_rand
    u_new[-1, :] = T_rand
    u_new[:,  0] = T_rand
    u_new[:, -1] = T_rand
    u = u_new

# ------------------------------------------------------------
# b) Varmeplot for ulike tidspunkter
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for idx, t_val in enumerate(plot_times):
    ax = axes[idx]
    field = saved_fields.get(t_val, None)
    if field is None:
        continue
    im = ax.imshow(
        field.T, origin='lower', cmap='RdYlBu_r',
        extent=[0, Lx*100, 0, Ly*100],
        vmin=T_start, vmax=T_rand,
        aspect='equal'
    )
    ax.set_title(f"t = {t_val:.0f} s  ({t_val/60:.0f} min)")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    plt.colorbar(im, ax=ax, label="Temperatur [°C]")

fig.suptitle("Varmeplot – dansk rugbrød i ovn (200 °C)", fontsize=13)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# c) Varmeplot når midten når 60 °C
# ------------------------------------------------------------
if t60_field is not None:
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    im2 = ax2.imshow(
        t60_field.T, origin='lower', cmap='RdYlBu_r',
        extent=[0, Lx*100, 0, Ly*100],
        vmin=T_start, vmax=T_rand,
        aspect='equal'
    )
    ax2.set_title(
        f"Midten nådde 60 °C\nt = {t60_time:.0f} s  ({t60_time/60:.1f} min)"
    )
    ax2.set_xlabel("x [cm]")
    ax2.set_ylabel("y [cm]")
    plt.colorbar(im2, ax=ax2, label="Temperatur [°C]")
    plt.tight_layout()
    plt.show()

# Plott temperaturutvikling i midten over tid
tc_arr = np.array(time_center)
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.plot(tc_arr[:, 0] / 60, tc_arr[:, 1], 'b-', linewidth=1.5)
if t60_time is not None:
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

# ------------------------------------------------------------
# d) Animasjon – viser temperaturfeltet over tid
# ------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(5, 5))

im_anim = ax4.imshow(
    anim_frames[0].T, origin='lower', cmap='RdYlBu_r',
    extent=[0, Lx*100, 0, Ly*100],
    vmin=T_start, vmax=T_rand,
    aspect='equal'
)
plt.colorbar(im_anim, ax=ax4, label="Temperatur [°C]")
ax4.set_xlabel("x [cm]")
ax4.set_ylabel("y [cm]")
title_anim = ax4.set_title("")


def update(frame_idx):
    im_anim.set_data(anim_frames[frame_idx].T)
    title_anim.set_text(
        f"t = {anim_times[frame_idx]:.0f} s  ({anim_times[frame_idx]/60:.1f} min)"
    )
    return im_anim, title_anim


ani = animation.FuncAnimation(
    fig4, update,
    frames=len(anim_frames),
    interval=50,
    blit=True
)

plt.tight_layout()
plt.show()

# I Jupyter: bruk linjen under for å vise animasjonen inne i notebooken:
# HTML(ani.to_jshtml())