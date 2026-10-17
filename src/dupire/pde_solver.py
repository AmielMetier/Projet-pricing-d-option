"""Résolution de l'EDP forward de Dupire par θ-schéma.

Pour une volatilité locale σ_loc(K, T) donnée, le prix du call C(K, T)
vérifie, à S₀ fixé :

    ∂C/∂T = ½ σ_loc²(K, T) K² ∂²C/∂K² − r K ∂C/∂K,

avec la condition initiale C(K, 0) = (S₀ − K)⁺ et les conditions au bord
C(0, T) = S₀, C(K_max, T) = 0.

Discrétisation en T par un θ-schéma : θ = 0 explicite, θ = ½ Crank-Nicolson,
θ = 1 implicite, avec un lissage de Rannacher optionnel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Volatilité locale σ_loc(K, T) : prend un vecteur de strikes et une maturité,
#: renvoie un vecteur de volatilités de même taille.
LocalVolFunc = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class PDEGrid:
    """Grille de prix renvoyée par le solveur.

    Attributs
    ---------
    K : strikes, taille n_K + 1 (de 0 à K_max).
    T : maturités, taille n_T + 1 (de 0 à T_max).
    C : prix des calls, C[i, j] = C(K[j], T[i]), taille (n_T + 1, n_K + 1).
    """

    K: np.ndarray
    T: np.ndarray
    C: np.ndarray


def dupire_operator(
    K: np.ndarray, t: float, sigma_loc: LocalVolFunc, r: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrétise l'opérateur L C = ½ σ² K² ∂²C/∂K² − r K ∂C/∂K.

    Paramètres
    ----------
    K : grille de strikes (uniforme), bords compris.
    t : maturité à laquelle on évalue σ_loc.
    sigma_loc : volatilité locale σ_loc(K, t).
    r : taux sans risque.

    Retour
    ------
    (sous-diagonale, diagonale, sur-diagonale) de la matrice tridiagonale
    de L restreinte aux nœuds intérieurs.
    """
    raise NotImplementedError("Issue #3")


def solve_dupire_pde(
    S0: float,
    r: float,
    sigma_loc: LocalVolFunc,
    K_max: float,
    T_max: float,
    n_K: int,
    n_T: int,
    theta: float = 0.5,
    n_rannacher: int = 0,
) -> PDEGrid:
    """Résout l'EDP de Dupire sur [0, K_max] × [0, T_max].

    Paramètres
    ----------
    S0 : prix initial du sous-jacent.
    r : taux sans risque.
    sigma_loc : volatilité locale σ_loc(K, T).
    K_max : borne supérieure en strike (quelques S₀).
    T_max : maturité maximale.
    n_K : nombre de pas en strike.
    n_T : nombre de pas en maturité.
    theta : paramètre du θ-schéma (½ pour Crank-Nicolson).
    n_rannacher : nombre de pas de lissage de Rannacher au démarrage (0 : aucun lissage).

    Retour
    ------
    La grille (K, T, C), voir :class:`PDEGrid`.
    """
    raise NotImplementedError("Issue #3")
