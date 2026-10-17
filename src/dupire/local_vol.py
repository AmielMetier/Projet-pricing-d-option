"""Extraction de la volatilité locale par la formule de Dupire.

À partir d'une grille de prix de calls C(K, T) (sans dividende) :

    σ_loc²(K, T) = (∂C/∂T + r K ∂C/∂K) / (½ K² ∂²C/∂K²).

Les dérivées sont approchées par différences finies. Le dénominateur
(densité risque-neutre, à un facteur près) est petit loin de la monnaie,
ce qui rend la formule sensible au bruit sur les prix.
"""

from __future__ import annotations

import numpy as np


def price_derivatives(
    K: np.ndarray, T: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dérivées de C par différences finies centrées.

    Paramètres
    ----------
    K : strikes, taille n_K.
    T : maturités, taille n_T.
    C : prix, C[i, j] = C(K[j], T[i]), taille (n_T, n_K).

    Retour
    ------
    (∂C/∂T, ∂C/∂K, ∂²C/∂K²), chacun de taille (n_T, n_K) ; NaN sur les bords
    où le schéma centré n'est pas défini.
    """
    raise NotImplementedError("Issue #4")


def local_vol_from_prices(
    K: np.ndarray,
    T: np.ndarray,
    C: np.ndarray,
    r: float,
    min_gamma: float = 1e-10,
) -> np.ndarray:
    """Volatilité locale σ_loc(K, T) par la formule de Dupire.

    Paramètres
    ----------
    K, T, C : grille de prix, voir :func:`price_derivatives`.
    r : taux sans risque.
    min_gamma : seuil sous lequel ∂²C/∂K² est jugé trop petit ; la
        volatilité y vaut NaN plutôt qu'une valeur aberrante.

    Retour
    ------
    σ_loc, de taille (n_T, n_K).
    """
    raise NotImplementedError("Issue #4")
