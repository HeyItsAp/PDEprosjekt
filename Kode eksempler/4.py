import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

plt.close("all")

# ------------------------------------------------------------
# Animasjon av 2D-løsningen u(x,y,t) som fargekart
#
# Forutsetter at du allerede har:
#   - u : løsning fra Euler (form: (t.size, m*n))
#   - x, y : gitter (inkl. randpunkter)
#   - m, n : antall indre punkter i hver retning
#
# Vi bruker samme animasjons-stil som tidligere:
#   HTML(ani.to_jshtml()); plt.show()
# ------------------------------------------------------------

# Hvor mange frames vil vi vise?
# Vi tar et hopp på 'stride' tidssteg mellom hver frame for å få en raskere animasjon.
stride = 100
n_frames = 90  # som i originalkoden

# Vi lager en liste med faktiske tidindekser vi vil bruke (sikrer at vi ikke går utenfor)
frame_idx = [i * stride for i in range(n_frames) if i * stride < u.shape[0]]

# ------------------------------------------------------------
# Forbered "fargeskala" (samme skala i alle frames gir roligere animasjon)
# ------------------------------------------------------------
# Vi bruker de frame-verdiene vi faktisk skal vise når vi bestemmer min/max.
U_frames = np.array([u[j, :] for j in frame_idx])
vmin = np.min(U_frames)
vmax = np.max(U_frames)

# ------------------------------------------------------------
# Sett opp figur
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Første frame som startbilde
Z0 = np.reshape(u[frame_idx[0], :], (m, n))

im = ax.imshow(
    -Z0.T,                 # .T for å få "riktig" orientering i imshow
    origin="lower",
    cmap="RdBu",
    vmin=-vmax, vmax=-vmin  # symmetrisk skala rundt 0 (nyttig for RdBu)
)

# Akse-etiketter (indre gitterpunkter)
ax.set_title("Tidsutvikling av u(x,y,t) (fargekart)")
ax.set_xlabel("x-indeks (indre punkter)")
ax.set_ylabel("y-indeks (indre punkter)")

# Fargebar gjør det lettere å tolke verdier
plt.colorbar(im, ax=ax, label="u-verdi")

# ------------------------------------------------------------
# Oppdateringsfunksjon (mer minnevennlig enn å lagre 90 bilder i en liste)
# ------------------------------------------------------------
def animate(frame_number):
    j = frame_idx[frame_number]
    Z = np.reshape(u[j, :], (m, n))
    im.set_data((-Z).T)
    ax.set_title(f"Tidsutvikling av u(x,y,t)  (t ≈ {j} tidssteg)")
    return (im,)

# ------------------------------------------------------------
# Lag animasjonen
# ------------------------------------------------------------
ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(frame_idx),
    interval=50,
    blit=True
)

HTML(ani.to_jshtml())
plt.show()