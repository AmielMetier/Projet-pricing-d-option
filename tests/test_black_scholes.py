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


def test_implied_vol_arbitrage_bounds():
    """Vérifie que les prix hors bornes d'arbitrage renvoient NaN et non une vol erronée."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0

    lower_bound = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper_bound = S * np.exp(-q * T)

    # Prix inférieur à la borne inférieure d'arbitrage
    assert np.isnan(bs_implied_vol(lower_bound - 1.0, S, K, T, r, q))
    assert np.isnan(bs_implied_vol(-5.0, S, K, T, r, q))

    # Prix supérieur à la borne supérieure d'arbitrage (S * e^(-qT))
    assert np.isnan(bs_implied_vol(upper_bound + 5.0, S, K, T, r, q))

    # Maturité nulle ou négative
    assert np.isnan(bs_implied_vol(5.0, S, K, 0.0, r, q))
    assert np.isnan(bs_implied_vol(5.0, S, K, -0.5, r, q))


def test_implied_vol_put():
    """Vérifie l'inversion de la volatilité implicite pour les Puts."""
    S = 100.0
    K = 95.0
    T = 0.5
    r = 0.02
    q = 0.01
    true_vol = 0.22

    put_price = bs_put_price(S, K, T, r, q, true_vol)
    recovered_vol = bs_implied_vol(put_price, S, K, T, r, q, option_type="put", tol=1e-7)

    assert not np.isnan(recovered_vol)
    assert np.isclose(recovered_vol, true_vol, atol=1e-5)


def test_implied_vol_list_of_strikes():
    """Vérifie le calcul de vol implicite sur une série de strikes via une compréhension de liste."""
    S = 100.0
    strikes = [90.0, 100.0, 110.0]
    T = 1.0
    r = 0.03
    q = 0.01
    vol = 0.25

    prices = [bs_call_price(S, k, T, r, q, vol) for k in strikes]
    recovered_vols = [bs_implied_vol(p, S, k, T, r, q, tol=1e-7) for p, k in zip(prices, strikes)]

    assert np.allclose(recovered_vols, vol, atol=1e-4)


def test_implied_vol_surface():
    """Vérifie l'inversion d'une grille 2D de prix via implied_vol_surface."""
    S = 100.0
    strikes = np.array([90.0, 100.0, 110.0])
    maturities = np.array([0.5, 1.0, 1.5])
    r = 0.03
    q = 0.01
    vol_target = 0.20

    # Grille de prix théoriques
    prices = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            prices[i, j] = bs_call_price(S, K, T, r, q, vol_target)

    # Inversion de la surface
    surface = implied_vol_surface(prices, S, strikes, maturities, r, q)

    assert surface.shape == (len(maturities), len(strikes))
    assert np.allclose(surface, vol_target, atol=1e-4)



def test_default_dividend_q_zero():
    """Vérifie que le dividende continu q vaut 0.0 par défaut dans toutes les fonctions."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.03
    sigma = 0.20

    # 1. d1_d2
    assert d1_d2(S, K, T, r, sigma) == d1_d2(S, K, T, r, 0.0, sigma)

    # 2. bs_call_price
    call_default = bs_call_price(S, K, T, r, sigma)
    call_explicit = bs_call_price(S, K, T, r, 0.0, sigma)
    assert np.isclose(call_default, call_explicit)

    # Appel avec sigma en mot-clé
    assert np.isclose(bs_call_price(S, K, T, r, sigma=sigma), call_explicit)

    # 3. bs_put_price
    put_default = bs_put_price(S, K, T, r, sigma)
    put_explicit = bs_put_price(S, K, T, r, 0.0, sigma)
    assert np.isclose(put_default, put_explicit)

    # 4. bs_vega
    vega_default = bs_vega(S, K, T, r, sigma)
    vega_explicit = bs_vega(S, K, T, r, 0.0, sigma)
    assert np.isclose(vega_default, vega_explicit)

    # 5. bs_implied_vol
    vol_call_default = bs_implied_vol(call_default, S, K, T, r)
    vol_call_explicit = bs_implied_vol(call_default, S, K, T, r, 0.0)
    assert np.isclose(vol_call_default, vol_call_explicit)

    vol_put_default = bs_implied_vol(put_default, S, K, T, r, option_type="put")
    vol_put_explicit = bs_implied_vol(put_default, S, K, T, r, 0.0, option_type="put")
    assert np.isclose(vol_put_default, vol_put_explicit)

    # 6. implied_vol_surface
    prices = np.array([[call_default]])
    surf_default = implied_vol_surface(prices, S, [K], [T], r)
    surf_explicit = implied_vol_surface(prices, S, [K], [T], r, 0.0)
    assert np.allclose(surf_default, surf_explicit)

    # 7. Absence de sigma doit lever un TypeError explicite
    with pytest.raises(TypeError, match="sigma"):
        bs_call_price(S, K, T, r)
