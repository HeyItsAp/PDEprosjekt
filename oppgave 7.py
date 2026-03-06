import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from IPython.display import HTML

# ============================================================
# Oppgave 7: Varmeligning 2D med luftlag rundt brødet
# ============================================================

# --- Materialparametere ---
alpha_bread = 4.1e-7    # termisk diffusivitet brød [m²/s]
alpha_air   = 2.0e-5    # termisk diffusivitet luft [m²/s]

# --- Domenegrenser (luftlag rundt brødet) ---
# Brød: 0 ≤ x ≤ 0.11, 0 ≤ y ≤ 0.11
# Luftlag: ~halvparten av brødets bredde på hver side → 0.055 m
x_min, x_max = -0.055, 0.165   # total bredde 0.22 m
y_min, y_max = -0.055, 0.165   # total høyde  0.22 m

x_brod_min, x_brod_max = 0.0, 0.11
y_brod_min, y_brod_max = 0.0, 0.11

# --- Temperaturer ---
T_rand  = 200.0   # randbetingelse [°C]
T_brod  =  15.0   # starttemperatur brød [°C]
T_luft  = 200.0   # starttemperatur luft [°C]

# --- Diskretisering: 111×111-gitter, dx = dy ≈ 2 mm ---
Nx = 111
Ny = 111

x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

print(f"dx = dy = {dx*1000:.2f} mm")
print(f"Gitter: {Nx} × {Ny} = {Nx*Ny} punkter")

# --- α-felt ---
X, Y = np.meshgrid(x, y, indexing='ij')
bread_mask = (X >= x_brod_min) & (X <= x_brod_max) & \
             (Y >= y_brod_min) & (Y <= y_brod_max)

alpha_field = np.where(bread_mask, alpha_bread, alpha_air)

# --- Stabilitetskrav (bruker α_max = α_luft) ---
alpha_max = alpha_air
dt_max = dx**2 * dy**2 / (2 * alpha_max * (dx**2 + dy**2))
dt = 0.9 * dt_max

T_total = 7200.0           # simuler 2 timer [s]
Nt = int(T_total / dt) + 1

print(f"dt = {dt:.4f} s,  Antall tidssteg: {Nt}")

# --- Initialfelt med rand- og initialbetingelser ---
u = np.where(bread_mask, T_brod, T_luft).astype(float)
u[0,  :] = T_rand
u[-1, :] = T_rand
u[:,  0] = T_rand
u[:, -1] = T_rand

ix = slice(1, Nx - 1)
iy = slice(1, Ny - 1)

# Midtpunkt av brødet
mx = np.argmin(np.abs(x - 0.055))
my = np.argmin(np.abs(y - 0.055))
print(f"Midtpunkt brød: x = {x[mx]*100:.1f} cm,  y = {y[my]*100:.1f} cm")

# --- Lagring ---
plot_times  = [300, 900, 1800, 3600]
plot_nsteps = {int(t / dt): t for t in plot_times}
saved_fields = {}
time_center  = []
t60_field    = None
t60_time     = None

anim_skip   = max(1, Nt // 200)
anim_frames = []
anim_times  = []

# Precompute alpha*dt/dx² og alpha*dt/dy²
rX = alpha_field[ix, iy] * dt / dx**2
rY = alpha_field[ix, iy] * dt / dy**2

# --- Tidsløkke: forward Euler med stedsavhengig α ---
for n in range(Nt):
    T_mid = u[mx, my]
    time_center.append((n * dt, T_mid))

    if n in plot_nsteps:
        saved_fields[plot_nsteps[n]] = u.copy()

    if t60_time is None and T_mid >= 60.0:
        t60_time  = n * dt
        t60_field = u.copy()
        print(f"Midten av brødet nådde 60 °C etter {t60_time:.0f} s  ({t60_time/60:.1f} min)")

    if n % anim_skip == 0:
        anim_frames.append(u.copy())
        anim_times.append(n * dt)

    u_new = u.copy()
    u_new[ix, iy] = (
        u[ix, iy]
        + rX * (u[0:Nx-2, iy] - 2*u[ix, iy] + u[2:Nx, iy])
        + rY * (u[ix, 0:Ny-2] - 2*u[ix, iy] + u[ix, 2:Ny])
    )
    u_new[0,  :] = T_rand
    u_new[-1, :] = T_rand
    u_new[:,  0] = T_rand
    u_new[:, -1] = T_rand
    u = u_new

print("Simulering ferdig.")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# --- a) Varmeplot for ulike tidspunkter ---

def add_bread_box(ax):
    """Tegner stiplede grenser for brødet."""
    rect = patches.Rectangle(
        (x_brod_min * 100, y_brod_min * 100),
        (x_brod_max - x_brod_min) * 100,
        (y_brod_max - y_brod_min) * 100,
        linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for idx, t_val in enumerate(plot_times):
    ax = axes.ravel()[idx]
    field = saved_fields.get(t_val)
    if field is None:
        ax.set_title(f"t = {t_val} s — ikke lagret")
        continue
    im = ax.imshow(
        field.T, origin='lower', cmap='RdYlBu_r',
        extent=[x_min * 100, x_max * 100, y_min * 100, y_max * 100],
        vmin=T_brod, vmax=T_rand, aspect='equal'
    )
    add_bread_box(ax)
    ax.set_title(f't = {t_val:.0f} s  ({t_val/60:.0f} min)')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    plt.colorbar(im, ax=ax, label='Temperatur [°C]')

fig.suptitle('Oppgave 7a — Varmeplot: brød med luftlag (200 °C ved rand)', fontsize=13)
plt.tight_layout()
plt.show()


# Oppgave b: plotting

# --- b) Animasjon: temperaturutvikling over tid ---
# Kildekoden er inkludert her; fjern kommentaren på HTML-linjen for å vise animasjonen.

fig5, ax5 = plt.subplots(figsize=(6, 6))

im_anim = ax5.imshow(
    anim_frames[0].T, origin='lower', cmap='RdYlBu_r',
    extent=[x_min * 100, x_max * 100, y_min * 100, y_max * 100],
    vmin=T_brod, vmax=T_rand, aspect='equal'
)
add_bread_box(ax5)
plt.colorbar(im_anim, ax=ax5, label='Temperatur [°C]')
ax5.set_xlabel('x [cm]')
ax5.set_ylabel('y [cm]')
title_anim = ax5.set_title('')


def update7(frame_idx):
    im_anim.set_data(anim_frames[frame_idx].T)
    title_anim.set_text(
        f't = {anim_times[frame_idx]:.0f} s  ({anim_times[frame_idx]/60:.1f} min)'
    )
    return im_anim, title_anim


ani7 = animation.FuncAnimation(
    fig5, update7,
    frames=len(anim_frames),
    interval=50,
    blit=True
)
plt.tight_layout()
plt.show()

# Fjern kommentaren under for å vise animasjonen som HTML i Jupyter:
# HTML(ani7.to_jshtml())


# oppgave c: 

# --- c) Tidspunkt for 60 °C i midten av brødet ---

if t60_field is not None:
    print(f"Midten av brødet nådde 60 °C etter {t60_time:.0f} s  ({t60_time/60:.1f} min)")

    # Varmeplot for dette tidspunktet
    fig6, ax6 = plt.subplots(figsize=(6, 6))
    im6 = ax6.imshow(
        t60_field.T, origin='lower', cmap='RdYlBu_r',
        extent=[x_min * 100, x_max * 100, y_min * 100, y_max * 100],
        vmin=T_brod, vmax=T_rand, aspect='equal'
    )
    add_bread_box(ax6)
    ax6.set_title(
        f'Oppgave 7c — Midten nådde 60 °C\n'
        f't = {t60_time:.0f} s  ({t60_time/60:.1f} min)'
    )
    ax6.set_xlabel('x [cm]')
    ax6.set_ylabel('y [cm]')
    plt.colorbar(im6, ax=ax6, label='Temperatur [°C]')
    plt.tight_layout()
    plt.show()
else:
    print("Midten av brødet nådde ikke 60 °C i løpet av simuleringen.")

# Temperaturutvikling i midtpunktet over tid
tc7 = np.array(time_center)
fig7, ax7 = plt.subplots(figsize=(7, 4))
ax7.plot(tc7[:, 0] / 60, tc7[:, 1], 'b-', linewidth=1.5, label='Midtpunkt brød')
ax7.axhline(60, color='r', linestyle='--', label='60 °C')
if t60_time is not None:
    ax7.axvline(t60_time / 60, color='g', linestyle='--',
                label=f't = {t60_time/60:.1f} min')
ax7.set_xlabel('Tid [min]')
ax7.set_ylabel('Temperatur i midten [°C]')
ax7.set_title('Oppgave 7c — Temperaturutvikling i brødets midtpunkt (med luftlag)')
ax7.legend()
ax7.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
