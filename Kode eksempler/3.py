import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

# ------------------------------------------------------------
# 3D-visualisering av løsningen ved et gitt tidspunkt
#
# Vi har løst ODE-systemet u'(t) = L u(t) - F med forlengs Euler.
# Her ser vi på løsningen etter et bestemt antall tidssteg.
# ------------------------------------------------------------

# Velg hvilket tidssteg vi vil visualisere
step = 500

print(f"Visualiserer løsning ved tidssteg {step}, t = {t[step]:.4f}")

# Reshape fra vektor til 2D-rutenett (indre punkter)
Z = np.reshape(u[step, :], (m, n))

# ------------------------------------------------------------
# 3D-plott av løsningen
# ------------------------------------------------------------
fig, ax = plt.subplots(
    subplot_kw={"projection": "3d"},
    figsize=(10, 8)
)

ax.plot_surface(X, Y, Z, cmap="viridis")

# Samme romlige skala som tidligere figurer
ax.set_xlim(x[1], x[-2])
ax.set_ylim(y[1], y[-2])
ax.set_zlim(np.min(Z), np.max(Z))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x,y,t)")
ax.set_title(f"Løsning etter {step} tidssteg")

plt.show()