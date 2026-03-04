## Oppgave 3a — Analytisk løsning av Poissonlikningen

Vi skal løse:

$$u_{xx}(x) = f(x), \quad -1 \leq x \leq 1$$

med $f(x) = \cos(\pi x)$, $u(-1) = 0$ og $u(1) = 2$.

---

**Første integrasjon** gir $u_x$:

$$u_x = \int \cos(\pi x)\, dx = \frac{1}{\pi} \sin(\pi x) + C_1$$

**Andre integrasjon** gir $u$:

$$u(x) = \int \frac{1}{\pi} \sin(\pi x)\, dx + C_1 x = -\frac{1}{\pi^2} \cos(\pi x) + C_1 x + C_2$$

---

**Randbetingelse** $u(-1) = 0$:

$$-\frac{1}{\pi^2} \cos(-\pi) + C_1(-1) + C_2 = 0$$

$$\frac{1}{\pi^2} - C_1 + C_2 = 0 \tag{i}$$

**Randbetingelse** $u(1) = 2$:

$$-\frac{1}{\pi^2} \cos(\pi) + C_1 + C_2 = 2$$

$$\frac{1}{\pi^2} + C_1 + C_2 = 2 \tag{ii}$$

---

**Subtraher (i) fra (ii)**:

$$2C_1 = 2 \implies C_1 = 1$$

**Sett inn $C_1 = 1$ i (i)**:

$$C_2 = C_1 - \frac{1}{\pi^2} = 1 - \frac{1}{\pi^2}$$

---

**Analytisk løsning:**

$$u(x) = -\frac{1}{\pi^2}\cos(\pi x) + x + 1 - \frac{1}{\pi^2} = x + 1 - \frac{1 + \cos(\pi x)}{\pi^2}$$
