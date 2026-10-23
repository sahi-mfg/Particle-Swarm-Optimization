"""Tests de l'algorithme sur des fonctions de référence plus difficiles que la sphère.

Chaque fonction est optimisée sur 20 graines fixes. On mesure le taux de réussite
(écart à l'optimum connu < 1e-3) plutôt qu'un seul run : un PSO est stochastique.

Paramètres : coefficients de constriction de Clerc & Kennedy (2002),
w = 0.7298, phi1 = phi2 = 1.49618, le réglage standard qui garantit la convergence.
"""

import math

import numpy as np
import pytest

from pso import PSO

GRAINES = range(20)
TOLERANCE = 1e-3
CLERC = dict(w=0.7298, phi1=1.49618, phi2=1.49618)


def rastrigin(x, y):
    # très multimodale : une grille régulière de minima locaux, minimum global 0 en (0, 0)
    return 20 + x**2 - 10 * math.cos(2 * math.pi * x) + y**2 - 10 * math.cos(2 * math.pi * y)


def ackley(x, y):
    # plateau presque plat avec un trou étroit au centre, minimum global 0 en (0, 0)
    return (
        -20 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2)))
        - math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
        + math.e
        + 20
    )


def rosenbrock(x, y):
    # vallée étroite et courbe : facile à trouver, difficile à descendre ; minimum 0 en (1, 1)
    return 100 * (y - x**2) ** 2 + (1 - x) ** 2


def himmelblau(x, y):
    # quatre minima globaux de valeur 0
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def schwefel(x, y):
    # trompeuse : le meilleur minimum local est loin du global, minimum 0 en (420.97, 420.97)
    return 418.9829 * 2 - x * math.sin(math.sqrt(abs(x))) - y * math.sin(math.sqrt(abs(y)))


def eggholder(x, y):
    # très irrégulière, minimum global -959.6407 en (512, 404.2319), au bord du domaine
    return -(y + 47) * math.sin(math.sqrt(abs(x / 2 + y + 47))) - x * math.sin(math.sqrt(abs(x - (y + 47))))


def taux_de_reussite(fonction, borne, optimum):
    """Lance le PSO sur chaque graine ; renvoie le nombre de réussites et les résultats."""
    reussites, resultats = 0, []
    for graine in GRAINES:
        np.random.seed(graine)
        pso = PSO(fonction=fonction, dim=2, particle=40, iteration=150, min=-borne, max=borne, **CLERC)
        position, fitness = pso.run()
        # la fitness renvoyée doit correspondre à la position renvoyée
        assert fitness == pytest.approx(fonction(*position))
        reussites += fitness - optimum < TOLERANCE
        resultats.append((position, fitness))
    return reussites, resultats


@pytest.mark.parametrize(
    "fonction, borne, optimum",
    [
        (rastrigin, 5.12, 0.0),
        (ackley, 32.768, 0.0),
        (rosenbrock, 5.0, 0.0),
        (himmelblau, 5.0, 0.0),
    ],
    ids=["rastrigin", "ackley", "rosenbrock", "himmelblau"],
)
def test_converge_a_chaque_graine(fonction, borne, optimum):
    reussites, _ = taux_de_reussite(fonction, borne, optimum)
    assert reussites == len(GRAINES)


def test_himmelblau_trouve_un_des_quatre_minima():
    minima = np.array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])
    _, resultats = taux_de_reussite(himmelblau, 5.0, 0.0)
    for position, _ in resultats:
        assert np.min(np.linalg.norm(minima - position, axis=1)) < 1e-2


@pytest.mark.parametrize(
    "fonction, borne, optimum",
    [(schwefel, 500.0, 0.0), (eggholder, 512.0, -959.6407)],
    ids=["schwefel", "eggholder"],
)
def test_fonctions_trompeuses(fonction, borne, optimum):
    # Ici, garder la mémoire individuelle de chaque particule compte : l'essaim
    # explore plusieurs bassins au lieu de se rabattre sur la dernière position vue.
    # Mesuré : 18/20 avec le correctif, 11/20 avant.
    reussites, _ = taux_de_reussite(fonction, borne, optimum)
    assert reussites >= 16
