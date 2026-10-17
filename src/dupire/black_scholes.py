"""Formules fermées de Black-Scholes et volatilité implicite.

Sert de référence pour valider le solveur EDP et le Monte Carlo : avec une
volatilité locale constante σ, le modèle de Dupire redonne Black-Scholes.

Toutes les fonctions acceptent des scalaires ou des ``np.ndarray`` (diffusion
numpy) et renvoient un ``np.ndarray``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


def d1_d2(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les quantités d1 et d2 de Black-Scholes.

    d1 = [ln(S/K) + (r + ½σ²) T] / (σ √T),   d2 = d1 − σ √T.

    Paramètres
    ----------
    S : prix du sous-jacent.
    K : strike.
    T : maturité (en années), T > 0.
    r : taux sans risque (continu).
    sigma : volatilité, σ > 0.

    Retour
    ------
    (d1, d2)
    """
    raise NotImplementedError("Issue #2")


def bs_call(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike
) -> np.ndarray:
    """Prix d'un call européen : C = S N(d1) − K e^{−rT} N(d2).

    Paramètres : voir :func:`d1_d2`.
    """
    raise NotImplementedError("Issue #2")


def bs_put(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike
) -> np.ndarray:
    """Prix d'un put européen : P = K e^{−rT} N(−d2) − S N(−d1).

    Paramètres : voir :func:`d1_d2`.
    """
    raise NotImplementedError("Issue #2")


def bs_vega(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike
) -> np.ndarray:
    """Vega (dérivée du prix par rapport à σ) : ∂C/∂σ = S φ(d1) √T.

    Identique pour le call et le put.
    """
    raise NotImplementedError("Issue #2")


def implied_vol(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    """Volatilité implicite : σ telle que le prix Black-Scholes égale ``price``.

    Paramètres
    ----------
    price : prix observé de l'option.
    S, K, T, r : voir :func:`d1_d2`.
    option_type : ``"call"`` ou ``"put"``.
    tol : tolérance sur l'écart de prix.
    max_iter : nombre maximal d'itérations.

    Retour
    ------
    Volatilité implicite (NaN si le prix sort des bornes de non-arbitrage).
    """
    raise NotImplementedError("Issue #2")
