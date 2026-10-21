import numpy as np
import pytest
from dupire.black_scholes import (
    bs_call_price,
    bs_implied_vol,
    bs_put_price,
    bs_vega,
    d1_d2,
)


def test_d1_d2_relation():
    """Vérifie la relation d2 = d1 - sigma * sqrt(T)."""
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    assert np.isclose(d2, d1 - sigma * np.sqrt(T))


def test_put_call_parity():
    """Vérifie la relation de parité Call-Put sans arbitrage : C - P = S e^(-qT) - K e^(-rT)."""
    S = 100.0
    r = 0.04
    q = 0.01
    sigma = 0.25
    T = 1.5

    for K in [80.0, 100.0, 120.0]:
        call = bs_call_price(S, K, T, r, q, sigma)
        put = bs_put_price(S, K, T, r, q, sigma)
        theoretical_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert np.isclose(call - put, theoretical_diff, atol=1e-7)


def test_call_put_expiry_payoff():
    """À maturité (T = 0), les prix doivent converger vers le payoff intrinsèque."""
    S = 105.0
    K = 100.0
    r = 0.03
    q = 0.0
    sigma = 0.20

    # Call dans la monnaie
    assert np.isclose(bs_call_price(S, K, 0.0, r, q, sigma), 5.0)
    assert np.isclose(bs_put_price(S, K, 0.0, r, q, sigma), 0.0)

    # Put dans la monnaie
    S_low = 90.0
    assert np.isclose(bs_call_price(S_low, K, 0.0, r, q, sigma), 0.0)
    assert np.isclose(bs_put_price(S_low, K, 0.0, r, q, sigma), 10.0)


def test_vega_properties():
    """Le Vega doit être strictement positif pour T > 0 et sigma > 0."""
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.02, 0.0, 0.20
    vega = bs_vega(S, K, T, r, q, sigma)
    assert vega > 0.0

    # À T = 0, Vega = 0
    assert bs_vega(S, K, 0.0, r, q, sigma) == 0.0


@pytest.mark.parametrize("true_sigma", [0.10, 0.20, 0.35, 0.50])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_implied_vol_inversion(true_sigma, K):
    """Vérifie le cycle aller-retour : Prix(sigma) -> ImpliedVol(Prix) == sigma."""
    S = 100.0
    T = 0.75
    r = 0.03
    q = 0.01

    price = bs_call_price(S, K, T, r, q, true_sigma)
    recovered_vol = bs_implied_vol(price, S, K, T, r, q, tol=1e-7)

    assert recovered_vol is not None
    assert np.isclose(recovered_vol, true_sigma, atol=1e-5)


def test_vectorization_strikes():
    """Vérifie que les fonctions de pricing supportent des tableaux NumPy de strikes."""
    S = 100.0
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    T = 1.0
    r = 0.03
    q = 0.0
    sigma = 0.20

    calls = bs_call_price(S, strikes, T, r, q, sigma)
    puts = bs_put_price(S, strikes, T, r, q, sigma)

    # Doit renvoyer un tableau de même taille
    assert isinstance(calls, np.ndarray)
    assert calls.shape == strikes.shape
    # Les prix de calls doivent être strictement décroissants en fonction du strike
    assert np.all(np.diff(calls) < 0)
    # Les prix de puts doivent être strictement croissants en fonction du strike
    assert np.all(np.diff(puts) > 0)
