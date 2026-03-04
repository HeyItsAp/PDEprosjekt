import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

# ------------------------------------------------------------
# Forlengs Euler for ODE-systemet u'(t) = g(u,t)
#
# Vi bruker denne etter romdiskretisering av et 2D-problem.
# Da blir u(t) en lang vektor med lengde m*n (indre punkter).
# ------------------------------------------------------------
def euler(g, u0, t0, t1, N):
    """
    Løser ODE-systemet u' = g(u,t) med forlengs Euler.

    Parametre:
      g  : høyreside g(u,t)
      u0 : initialtilstand (vektor)
      t0 : starttid
      t1 : sluttid
      N  : antall tidspunkter (inkl. start)

    Returnerer:
      u : løsning (N x dim)
      t : tidsgitter (N)
    """
    t = np.linspace(t0, t1, N)
    dt = t[1] - t[0]

    u = np.zeros((N, u0.size))
    u[0, :] = u0

    for i in range(N - 1):
        u[i + 1, :] = u[i, :] + dt * g(u[i, :], t[i])

    return u, t


# ------------------------------------------------------------
# Høyresiden i ODE-systemet
# Etter romdiskretisering får vi (typisk):
#   u'(t) = L u(t) - F
# ------------------------------------------------------------
def g(u, t):
    return L @ u - F


# ------------------------------------------------------------
# Rutenett for indre punkter (samme som tidligere)
# indexing='ij' gir:
#   X[i,j] = x_in[i],  Y[i,j] = y_in[j]
# ------------------------------------------------------------
x_in = x[1:-1]
y_in = y[1:-1]
X, Y = np.meshgrid(x_in, y_in, indexing="ij")

# ------------------------------------------------------------
# Initialtilstand u(x,y,0)
# Her velger vi u(x,y,0) = y (altså en "plan" som øker oppover).
# ------------------------------------------------------------
U0 = Y

# Vektorisering: viktig at rekkefølgen stemmer med L og F
u0 = np.reshape(U0, m * n)

print("Dimensjon på u0:", u0.shape)

# ------------------------------------------------------------
# Løs ODE-systemet i tid
# Vi trenger mange tidssteg fordi forlengs Euler har streng stabilitet.
# ------------------------------------------------------------
u, t = euler(g, u0, 0.0, 0.5, 10000)

print("dt =", t[1] - t[0])
print("Størrelse på u-matrisen:", u.shape)

# ------------------------------------------------------------
# Sjekk-plott: initialtilstanden (2D) med imshow
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.imshow(U0.T, origin="lower", aspect="auto",
           extent=[x_in[0], x_in[-1], y_in[0], y_in[-1]])
plt.colorbar(label="u(x,y,0)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Initialtilstand u(x,y,0) = y (på indre punkter)")
plt.show()