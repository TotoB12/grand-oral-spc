#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================
 Simulation du chaos dans le Système solaire (Terminale)
========================================================
Plan pédagogique
----------------
I.  Horlogerie céleste : lois de Newton/Kepler → stabilité à court terme.
II. Émergence du chaos : problème à N corps, sensibilité aux CI,
    temps de Lyapunov ~ 5–10 Ma pour planètes internes.
III.Exemple visuel : divergence entre deux systèmes ne différant
    que de 1 km sur la position initiale d’une planète.

Fonctionnement
--------------
* Choisir « internes » (Mercure → Mars) ou « externes » (Jupiter → Neptune).
* Le programme lance **deux** simulations parallèles :
  – la référence,                                              (ensemble A)
  – une copie légèrement perturbée sur une planète témoin,     (ensemble B)
  Les interactions gravitationnelles sont calculées
  avec la lois universelle de Newton (F = G m M / r²).
* La figure affiche :
  1. Les trajectoires XY vues du pôle nord de l’écliptique.
  2. Le log10 de la distance Δ(t) entre les positions de la planète témoin
     dans A et B → droite ↗️ si divergence exponentielle (signature du chaos).

Hypothèses simplificatrices
---------------------------
* Orbites initiales supposées circulaires, coplanaires (e=0, i=0),
  rayon égal au demi-grand axe.
* Le Soleil est fixé à l’origine (erreur << objectifs pédagogiques).
* Constantes : unités astronomiques, années, masses solaires.
  ⇒  G = 4π² AU³ · M☉⁻¹ · an⁻²  (donc Kepler 3 automatiquement satisfaite).

Auteurs : ChatGPT + <votre nom> – avril 2025
Licence : MIT – libre de modification pour votre Grand Oral.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import combinations

# ---------------------------------------------------------------------------
# 1. DONNÉES PLANÉTAIRES (a en UA, masse en M☉, couleur matplotlib)
#    Source : éléments orbitaux moyens (≈ J2000) – précision non critique ici
# ---------------------------------------------------------------------------
SOLAR_MASS = 1.0
G = 4 * np.pi ** 2                           # AU^3 / (M☉ · an^2)

PLANETS = {
    "Mercury":  (0.387, 1.660e-7, "darkgrey"),
    "Venus":    (0.723, 2.447e-6, "orange"),
    "Earth":    (1.000, 3.003e-6, "royalblue"),
    "Mars":     (1.524, 3.213e-7, "red"),
    "Jupiter":  (5.203, 9.545e-4, "peru"),
    "Saturn":   (9.537, 2.858e-4, "gold"),
    "Uranus":   (19.20, 4.366e-5, "turquoise"),
    "Neptune":  (30.05, 5.151e-5, "slateblue")
}


class Body:
    """Petit corps : position, vitesse, masse, couleur."""
    def __init__(self, name, a_AU, mass_Msun, color, phase=0.0):
        self.name = name
        self.m = mass_Msun
        # Position initiale sur l'axe Ox, vitesse sur l'axe Oy (orbite circulaire)
        self.r = np.array([a_AU * np.cos(phase), a_AU * np.sin(phase)], dtype=float)
        v = np.sqrt(G * SOLAR_MASS / a_AU)     # vitesse circulaire
        self.v = np.array([-v * np.sin(phase), v * np.cos(phase)], dtype=float)
        self.color = color

    def copy(self):
        clone = Body(self.name, 1, self.m, self.color)
        clone.r = self.r.copy()
        clone.v = self.v.copy()
        return clone


def acceleration(bodies, idx):
    """Accélération résultante sur le corps idx (sun fixed)."""
    ai = np.zeros(2)
    ri = bodies[idx].r
    for j, bj in enumerate(bodies):
        if j == idx:
            continue
        diff = bj.r - ri
        ai += G * bj.m * diff / np.linalg.norm(diff) ** 3
    return ai


def velocity_verlet(bodies, dt):
    """Un pas d’intégration symplectique."""
    # Demi-pas vitesse
    a0 = [acceleration(bodies, i) for i in range(len(bodies))]
    for i, b in enumerate(bodies):
        b.v += 0.5 * dt * a0[i]
    # Pas entier position
    for b in bodies:
        b.r += dt * b.v
    # Demi-pas vitesse
    a1 = [acceleration(bodies, i) for i in range(len(bodies))]
    for i, b in enumerate(bodies):
        b.v += 0.5 * dt * a1[i]


def build_system(selection="internes", twin=False):
    """Crée la liste de Body à simuler. twin=True → fait une copie (pour perturbation)."""
    if selection == "internes":
        names = ["Mercury", "Venus", "Earth", "Mars"]
    else:
        names = ["Jupiter", "Saturn", "Uranus", "Neptune"]
    system = [Body(n, *PLANETS[n]) for n in names]
    if twin:
        # Perturbe la planète la plus intérieure/extérieur de 1 km (≈6.7e-9 UA)
        delta = 6.68458712e-9
        system[0].r[0] += delta
    return system


def total_energy(bodies):
    """Énergie mécanique (vérification de la conservation)."""
    kinetic = sum(0.5 * b.m * np.dot(b.v, b.v) for b in bodies)
    potential = 0.0
    for i, j in combinations(range(len(bodies)), 2):
        rij = np.linalg.norm(bodies[i].r - bodies[j].r)
        potential -= G * bodies[i].m * bodies[j].m / rij
    return kinetic + potential


# ---------------------------------------------------------------------------
# 2. PARAMÈTRES GÉNÉRAUX
# ---------------------------------------------------------------------------
DT = 0.001          # pas de temps en années (~0.365 j)
STEPS = 50000       # nombre de pas (→ 50 ans simulés)
REFRESH = 10        # affichage chaque n pas

SELECTION = input("Choisissez le groupe (internes / externes) : ").strip().lower()
if SELECTION not in ("internes", "externes"):
    SELECTION = "internes"

sys_A = build_system(SELECTION)          # système de référence
sys_B = build_system(SELECTION, twin=True)  # système perturbé

target_name = sys_A[0].name  # planète perturbée

# Stockage pour la courbe Δ(t)
divergence = []

# ---------------------------------------------------------------------------
# 3. FIGURE & ANIMATION
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 5))
ax_orb = fig.add_subplot(1, 2, 1)
ax_div = fig.add_subplot(1, 2, 2)

ax_orb.set_title("Orbits – group: {}".format(SELECTION.capitalize()))
ax_orb.set_xlabel("x [AU]")
ax_orb.set_ylabel("y [AU]")
ax_orb.set_aspect("equal")

ax_div.set_title(f"Divergence for {target_name}")
ax_div.set_xlabel("Time [years]")
ax_div.set_ylabel(r"log$_{10}\,\Delta r$ [AU]")

# lignes de trajectoires et points
lines_A, lines_B, dots_A, dots_B = [], [], [], []
for b in sys_A:
    (lA,) = ax_orb.plot([], [], color=b.color, lw=1)
    (lB,) = ax_orb.plot([], [], color=b.color, lw=1, ls="--")
    (dA,) = ax_orb.plot([], [], color=b.color, marker="o", ls="")
    (dB,) = ax_orb.plot([], [], color=b.color, marker="x", ls="")
    lines_A.append(lA)
    lines_B.append(lB)
    dots_A.append(dA)
    dots_B.append(dB)

time_text = ax_orb.text(0.02, 0.95, "", transform=ax_orb.transAxes)


# ---------------------------------------------------------------------------
def init():
    """Initialisation des artistes."""
    for l in lines_A + lines_B:
        l.set_data([], [])
    for d in dots_A + dots_B:
        d.set_data([], [])
    ax_div.set_xlim(0, STEPS * DT)
    ax_div.set_ylim(-12, -3)
    (line_div,) = ax_div.plot([], [])
    return lines_A + lines_B + dots_A + dots_B + [line_div, time_text]


coords_A = [[] for _ in sys_A]
coords_B = [[] for _ in sys_B]
div_line, = ax_div.plot([], [], color="black")


def animate(frame):
    """Fonction appelée par FuncAnimation."""
    global sys_A, sys_B

    # Avance REFRESH pas
    for _ in range(REFRESH):
        velocity_verlet(sys_A, DT)
        velocity_verlet(sys_B, DT)

    t = frame * REFRESH * DT

    # mises à jour traces
    for i, b in enumerate(sys_A):
        coords_A[i].append(b.r.copy())
        ra = np.array(coords_A[i])
        lines_A[i].set_data(ra[:, 0], ra[:, 1])
        dots_A[i].set_data([b.r[0]], [b.r[1]])

    for i, b in enumerate(sys_B):
        coords_B[i].append(b.r.copy())
        rb = np.array(coords_B[i])
        lines_B[i].set_data(rb[:, 0], rb[:, 1])
        dots_B[i].set_data([b.r[0]], [b.r[1]])

    # divergence planète cible
    d_vec = sys_A[0].r - sys_B[0].r
    divergence.append(np.log10(np.linalg.norm(d_vec)))
    tt = np.arange(len(divergence)) * REFRESH * DT
    div_line.set_data(tt, divergence)

    time_text.set_text(f"t = {t:5.1f} a")

    return lines_A + lines_B + dots_A + dots_B + [div_line, time_text]


ani = FuncAnimation(fig, animate, frames=STEPS // REFRESH, init_func=init,
                    interval=30, blit=True, repeat=False)

plt.tight_layout()
plt.show()
