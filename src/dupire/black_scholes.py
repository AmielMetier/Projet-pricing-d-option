"""Pricer Black-Scholes analytique et volatilité implicite (issue #2).

Contenu
-------
- d1_d2, bs_call_price, bs_put_price, bs_vega : formules fermées (modèle avec dividende continu q)
- bs_implied_vol                              : vol implicite (Newton + fallback Brent)
- implied_vol_surface                         : surface sigma_imp(T, K) à partir d'une grille de prix

Convention : toutes les fonctions travaillent sur des scalaires.
Pour une grille, on appelle simplement ces fonctions dans une boucle
(voir `implied_vol_surface`).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Formules fermées
# ---------------------------------------------------------------------------
def d1_d2(S, K, T, r, q, sigma):
    """Calcule d1 et d2 de Black-Scholes (T > 0 et sigma > 0 requis)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S, K, T, r, q, sigma):
    """Prix analytique d'un Call européen.

    Cas limites : T <= 0 -> payoff ; sigma <= 0 -> valeur intrinsèque forward.
    """
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, q, sigma):
    """Prix analytique d'un Put européen.

    Cas limites : T <= 0 -> payoff ; sigma <= 0 -> valeur intrinsèque forward.
    """
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma):
    """Dérivée du prix par rapport à sigma (Vega), identique pour call et put."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


# ---------------------------------------------------------------------------
# Volatilité implicite
# ---------------------------------------------------------------------------
def bs_implied_vol(market_price, S, K, T, r, q, option_type="call", tol=1e-6, max_iter=50):
    """Calcule la volatilité implicite par Newton-Raphson avec fallback Brent.

    Paramètres
    ----------
    market_price : float
        Prix observé de l'option sur le marché.
    S, K, T, r, q : float
        Paramètres du modèle Black-Scholes.
    option_type : str, default 'call'
        Type d'option ('call' ou 'put').
    tol : float, default 1e-6
        Précision souhaitée sur le prix et sur la volatilité.
    max_iter : int, default 50
        Nombre maximal d'itérations pour Newton-Raphson.

    Retour
    ------
    float
        Volatilité implicite, ou np.nan si impossible (T <= 0, prix hors bornes
        d'arbitrage, vol hors de [1e-4, 5.0], non convergence).
    """
    # 1. Vérification des paramètres fondamentaux
    if T <= 0 or S <= 0 or K <= 0 or market_price < 0:
        return np.nan

    # 2. Si c'est un Put, conversion en prix de Call équivalent via la parité Call-Put
    if option_type.lower() == "put":
        market_price = market_price + S * np.exp(-q * T) - K * np.exp(-r * T)
    elif option_type.lower() != "call":
        raise ValueError(f"option_type doit être 'call' ou 'put', reçu : {option_type}")

    # 3. Vérification des bornes d'arbitrage
    c_min = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    c_max = S * np.exp(-q * T)
    if market_price <= c_min or market_price >= c_max:
        return np.nan

    # 4. Méthode de Newton-Raphson
    sigma = 0.20
    converged = False
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, q, sigma)
        diff = price - market_price
        vega = bs_vega(S, K, T, r, q, sigma)

        if vega < 1e-8:
            break

        step = diff / vega
        sigma -= step

        # Double critère d'arrêt : précision sur le prix ET sur la volatilité
        if abs(diff) < tol and abs(step) < tol:
            converged = True
            break

        # Sécurité pour ne pas diverger vers des valeurs négatives ou géantes
        if sigma <= 0 or sigma > 5.0:
            break

    if converged and sigma > 0:
        return float(sigma)

    # 5. Fallback : solveur robuste par encadrement (Brent) si Newton a échoué
    try:
        def obj(s):
            return bs_call_price(S, K, T, r, q, s) - market_price

        return float(brentq(obj, 1e-4, 5.0, xtol=tol))
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_surface(prices, S, strikes, maturities, r, q, option_type="call"):
    """Inverse une grille de prix en surface de vol implicite.

    Paramètres
    ----------
    prices : array (len(maturities), len(strikes))
        Prix d'options, une ligne par maturité, une colonne par strike.
    strikes, maturities : array 1D

    Retour
    ------
    array de même forme que `prices` (NaN là où l'inversion est impossible).
    """
    prices = np.asarray(prices, dtype=float)
    surface = np.full(prices.shape, np.nan)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            surface[i, j] = bs_implied_vol(prices[i, j], S, K, T, r, q, option_type)
    return surface


if __name__ == "__main__":
    S0, K0, T0, r0, q0, vol0 = 100.0, 100.0, 1.0, 0.03, 0.0, 0.20

    prix_call = bs_call_price(S0, K0, T0, r0, q0, vol0)
    vol_retrouvee = bs_implied_vol(prix_call, S0, K0, T0, r0, q0)

    print(f"Prix Call BS     : {prix_call:.4f}")
    print(f"Volatilité cible : {vol0:.2%}")
    print(f"Vol retrouvée    : {vol_retrouvee:.2%}")

