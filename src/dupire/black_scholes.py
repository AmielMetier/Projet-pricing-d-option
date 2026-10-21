"""Pricer Black-Scholes analytique et volatilité implicite (issue #2)."""
import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, q, sigma):
    """Calcule d1 et d2 de Black-Scholes avec NumPy."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S, K, T, r, q, sigma):
    """Prix analytique C d'un Call européen."""
    if np.all(T <= 0):
        return np.maximum(S - K, 0.0)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, q, sigma):
    """Prix analytique d'un Put européen."""
    if np.all(T <= 0):
        return np.maximum(K - S, 0.0)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma):
    """Dérivée du prix par rapport à sigma (Vega)."""
    if np.all(T <= 0):
        return 0.0
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def bs_implied_vol(market_price, S, K, T, r, q, tol=1e-6, max_iter=50):
    """Calcule la volatilité implicite par Newton-Raphson."""
    sigma = 0.20  # Initialisation standard à 20 %

    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, q, sigma)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = bs_vega(S, K, T, r, q, sigma)
        if abs(vega) < 1e-8:
            break

        sigma = sigma - diff / vega

    return sigma


if __name__ == "__main__":
    S0, K0, T0, r0, q0, vol0 = 100.0, 100.0, 1.0, 0.03, 0.0, 0.20

    prix_call = bs_call_price(S0, K0, T0, r0, q0, vol0)
    vol_retrouvee = bs_implied_vol(prix_call, S0, K0, T0, r0, q0)

    print(f"Prix Call BS     : {prix_call:.4f}")
    print(f"Volatilité cible : {vol0:.2%}")
    print(f"Vol retrouvée    : {vol_retrouvee:.2%}")
