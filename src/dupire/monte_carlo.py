"""Monte Carlo sous volatilité locale.

On simule dS_t = r S_t dt + σ_loc(S_t, t) S_t dW_t par un schéma d'Euler sur
X = log S :

    X_{n+1} = X_n + (r − ½ σ_loc²(S_n, t_n)) Δt + σ_loc(S_n, t_n) √Δt Z_n,

avec Z_n ~ N(0, 1). Le schéma en log garde S > 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dupire.pde_solver import LocalVolFunc


@dataclass
class MCResult:
    """Résultat d'un pricing Monte Carlo.

    Attributs
    ---------
    price : estimation du prix actualisé.
    std_error : erreur standard de l'estimateur.
    ci_low, ci_high : bornes de l'intervalle de confiance.
    """

    price: float
    std_error: float
    ci_low: float
    ci_high: float


def simulate_paths(
    S0: float,
    r: float,
    sigma_loc: LocalVolFunc,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simule des trajectoires de S sous volatilité locale (Euler sur log S).

    Paramètres
    ----------
    S0 : prix initial.
    r : taux sans risque.
    sigma_loc : volatilité locale σ_loc(S, t), évaluée en (S_t, t).
    T : horizon de simulation.
    n_steps : nombre de pas de temps.
    n_paths : nombre de trajectoires.
    rng : générateur aléatoire (``np.random.default_rng()`` si None).

    Retour
    ------
    Trajectoires S, taille (n_paths, n_steps + 1), la colonne 0 valant S0.
    """
    raise NotImplementedError("Issue #5")


def mc_price(
    payoffs: np.ndarray, r: float, T: float, confidence: float = 0.95
) -> MCResult:
    """Prix Monte Carlo et intervalle de confiance gaussien.

    prix = e^{−rT} moyenne(payoffs), IC = prix ± q · e^{−rT} écart-type / √N.

    Paramètres
    ----------
    payoffs : payoff de chaque trajectoire (call, barrière, digital…).
    r : taux sans risque.
    T : maturité.
    confidence : niveau de l'intervalle de confiance.

    Retour
    ------
    Voir :class:`MCResult`.
    """
    raise NotImplementedError("Issue #5")
