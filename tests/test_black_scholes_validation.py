"""Tests de validation théorique et numérique complémentaire pour Black-Scholes.

Ces tests complètent test_black_scholes.py et test_black_scholes_robustness.py en validant :
1. Les valeurs de référence théoriques exactes (benchmarks analytiques avec dividende q).
2. La cohérence du Grec Vega avec son approximation par différences finies centrées.
3. L'homogénéité de degré 1 (scale invariance en spot et strike).
4. Les limites asymptotiques (sigma -> 0, sigma -> infini).
5. La reconstruction de surface de volatilité implicite sur des Puts.
"""

import numpy as np
import pytest
from dupire.black_scholes import (
    bs_call_price,
    bs_implied_vol,
    bs_put_price,
    bs_vega,
    d1_d2,
    implied_vol_surface,
)


def test_black_scholes_textbook_benchmark():
    """Valide les prix analytiques contre des valeurs de référence théoriques exactes.

    Paramètres avec dividende q :
    S = 100, K = 100, T = 1, r = 0.05, q = 0.02, sigma = 0.20
    d1 = (ln(1) + (0.05 - 0.02 + 0.5 * 0.04) * 1) / (0.20 * 1) = 0.05 / 0.20 = 0.25
    d2 = 0.25 - 0.20 = 0.05
    """
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20
    d1, d2 = d1_d2(S, K, T, r, q, sigma)

    assert np.isclose(d1, 0.25, atol=1e-12)
    assert np.isclose(d2, 0.05, atol=1e-12)

    call_price = bs_call_price(S, K, T, r, q, sigma)
    put_price = bs_put_price(S, K, T, r, q, sigma)
    vega = bs_vega(S, K, T, r, q, sigma)

    # N(0.25) ~= 0.5987063256829237, N(0.05) ~= 0.5199388058383725
    # Call = 100 * exp(-0.02) * N(0.25) - 100 * exp(-0.05) * N(0.05)
    expected_call = (
        100.0 * np.exp(-0.02) * 0.5987063256829237
        - 100.0 * np.exp(-0.05) * 0.5199388058383725
    )
    assert np.isclose(call_price, expected_call, atol=1e-6)

    # Parité Call-Put
    parity = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert np.isclose(call_price - put_price, parity, atol=1e-12)

    # Vega analytique : S * exp(-q*T) * sqrt(T) * norm.pdf(d1)
    # pdf(0.25) = exp(-0.25^2 / 2) / sqrt(2*pi) ~= 0.38666811685
    expected_vega = 100.0 * np.exp(-0.02) * 1.0 * 0.3866681168516086
    assert np.isclose(vega, expected_vega, atol=1e-6)

    # Volatilité implicite retrouvée avec tolérance stricte
    recovered_call_vol = bs_implied_vol(call_price, S, K, T, r, q, option_type="call", tol=1e-10)
    recovered_put_vol = bs_implied_vol(put_price, S, K, T, r, q, option_type="put", tol=1e-10)
    assert np.isclose(recovered_call_vol, sigma, atol=1e-8)
    assert np.isclose(recovered_put_vol, sigma, atol=1e-8)


def test_vega_finite_difference_approximation():
    """Vérifie que la formule analytique de Vega coïncide avec le quotient de différences finies centrées :

    Vega = dC/dsigma ~= (C(sigma + eps) - C(sigma - eps)) / (2 * eps)
    """
    S = 100.0
    K_values = [80.0, 100.0, 120.0]
    T_values = [0.25, 1.0, 2.0]
    r = 0.04
    q = 0.01
    sigma = 0.25
    eps = 1e-6

    for K in K_values:
        for T in T_values:
            analytical_vega = bs_vega(S, K, T, r, q, sigma)
            c_plus = bs_call_price(S, K, T, r, q, sigma + eps)
            c_minus = bs_call_price(S, K, T, r, q, sigma - eps)
            numerical_vega = (c_plus - c_minus) / (2.0 * eps)

            assert np.isclose(analytical_vega, numerical_vega, rtol=1e-5, atol=1e-7)


def test_scale_invariance_degree_one():
    """Vérifie l'homogénéité de degré 1 de Black-Scholes :

    C(lambda * S, lambda * K) = lambda * C(S, K)
    et invariance de la volatilité implicite.
    """
    S, K, T, r, q, sigma = 100.0, 105.0, 0.75, 0.03, 0.015, 0.22
    lambdas = [0.5, 2.0, 10.0]

    base_call = bs_call_price(S, K, T, r, q, sigma)
    base_put = bs_put_price(S, K, T, r, q, sigma)

    for lam in lambdas:
        scaled_call = bs_call_price(lam * S, lam * K, T, r, q, sigma)
        scaled_put = bs_put_price(lam * S, lam * K, T, r, q, sigma)

        assert np.isclose(scaled_call, lam * base_call, rtol=1e-10)
        assert np.isclose(scaled_put, lam * base_put, rtol=1e-10)

        # La vol implicite doit rester invariante sous changement d'échelle
        vol_scaled = bs_implied_vol(scaled_call, lam * S, lam * K, T, r, q)
        assert np.isclose(vol_scaled, sigma, atol=1e-5)


def test_asymptotic_volatility_limits():
    """Vérifie les limites asymptotiques quand sigma -> 0 et sigma -> grand."""
    S = 100.0
    K = 95.0
    T = 1.0
    r = 0.03
    q = 0.01

    # Limite sigma -> 0 : prix converge vers la valeur intrinsèque forward
    intrinsic_forward_call = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    low_vol_call = bs_call_price(S, K, T, r, q, sigma=1e-6)
    assert np.isclose(low_vol_call, intrinsic_forward_call, atol=1e-5)

    # Limite sigma -> infini : Call converge vers S * exp(-q*T)
    large_vol_call = bs_call_price(S, K, T, r, q, sigma=10.0)
    assert np.isclose(large_vol_call, S * np.exp(-q * T), atol=1e-3)


def test_implied_vol_surface_puts():
    """Vérifie la reconstruction d'une surface de volatilité implicite à partir de prix de Puts."""
    S = 100.0
    strikes = np.array([85.0, 100.0, 115.0])
    maturities = np.array([0.5, 1.0, 2.0])
    r = 0.02
    q = 0.01
    vol_target = 0.28

    # Grille de prix de Puts
    put_prices = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            put_prices[i, j] = bs_put_price(S, K, T, r, q, vol_target)

    # Inversion de la surface avec option_type='put'
    surface_put = implied_vol_surface(put_prices, S, strikes, maturities, r, q, option_type="put")

    assert surface_put.shape == (len(maturities), len(strikes))
    assert np.allclose(surface_put, vol_target, atol=1e-4)
