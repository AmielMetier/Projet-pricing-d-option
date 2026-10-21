"""Tests de robustesse approfondis et cas limites pour le pricer Black-Scholes."""
import numpy as np
import pytest
from dupire.black_scholes import (
    bs_call_price,
    bs_implied_vol,
    bs_put_price,
    bs_vega,
    d1_d2,
)


# ============================================================================
# 1. CAS LIMITES ET ENTRÉES ANORMALES
# ============================================================================


@pytest.mark.parametrize("T", [0.0, -0.01, -1.0])
def test_implied_vol_invalid_maturities(T):
    """Vérifie que toute maturité T <= 0 renvoie NaN pour la vol implicite."""
    assert np.isnan(bs_implied_vol(5.0, S=100.0, K=100.0, T=T, r=0.03, q=0.0))


@pytest.mark.parametrize("S", [0.0, -50.0])
def test_implied_vol_invalid_spot(S):
    """Vérifie qu'un spot non strictement positif renvoie NaN."""
    assert np.isnan(bs_implied_vol(5.0, S=S, K=100.0, T=1.0, r=0.03, q=0.0))


@pytest.mark.parametrize("K", [0.0, -100.0])
def test_implied_vol_invalid_strike(K):
    """Vérifie qu'un strike non strictement positif renvoie NaN."""
    assert np.isnan(bs_implied_vol(5.0, S=100.0, K=K, T=1.0, r=0.03, q=0.0))


def test_implied_vol_negative_price():
    """Un prix de marché strictement négatif doit renvoyer NaN."""
    assert np.isnan(bs_implied_vol(-0.5, S=100.0, K=100.0, T=1.0, r=0.03, q=0.0))


def test_implied_vol_invalid_option_type():
    """Un type d'option inconnu doit lever une ValueError explicite."""
    with pytest.raises(ValueError, match="option_type doit être 'call' ou 'put'"):
        bs_implied_vol(10.0, S=100.0, K=100.0, T=1.0, r=0.03, q=0.0, option_type="binary")


# ============================================================================
# 2. TESTS DE VIOLATION DES BORNES SANS ARBITRAGE
# ============================================================================


def test_arbitrage_bounds_call():
    """Vérifie que les prix de Call hors bornes sans arbitrage renvoient NaN."""
    S = 100.0
    K = 80.0
    T = 1.0
    r = 0.05
    q = 0.02

    lower_bound = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper_bound = S * np.exp(-q * T)

    # Sous la borne inférieure (valeur intrinsèque actualisée)
    assert np.isnan(bs_implied_vol(lower_bound - 0.01, S, K, T, r, q, option_type="call"))
    # Égal à la borne inférieure (valeur temps nulle, pas de vol implicite stricte)
    assert np.isnan(bs_implied_vol(lower_bound, S, K, T, r, q, option_type="call"))
    # Au-dessus de la borne supérieure (S * exp(-qT))
    assert np.isnan(bs_implied_vol(upper_bound + 0.01, S, K, T, r, q, option_type="call"))


def test_arbitrage_bounds_put():
    """Vérifie que les prix de Put hors bornes sans arbitrage renvoient NaN."""
    S = 80.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02

    lower_bound = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    upper_bound = K * np.exp(-r * T)

    # Sous la borne inférieure
    assert np.isnan(bs_implied_vol(lower_bound - 0.01, S, K, T, r, q, option_type="put"))
    # Au-dessus de la borne supérieure (K * exp(-rT))
    assert np.isnan(bs_implied_vol(upper_bound + 0.01, S, K, T, r, q, option_type="put"))


# ============================================================================
# 3. ROBUSTESSE DU ROUND-TRIP (PRIX -> VOL IMPLICITE)
# ============================================================================


@pytest.mark.parametrize("moneyness", [0.70, 0.85, 1.0, 1.15, 1.30])
@pytest.mark.parametrize("T", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("sigma", [0.08, 0.20, 0.50, 1.0])
def test_round_trip_grid_calls(moneyness, T, sigma):
    """Cycle aller-retour sur une large grille de moneyness, maturités et volatilités."""
    S = 100.0
    K = S * moneyness
    r = 0.03
    q = 0.01

    price = bs_call_price(S, K, T, r, q, sigma)
    vega = bs_vega(S, K, T, r, q, sigma)
    recovered = bs_implied_vol(price, S, K, T, r, q, option_type="call")

    if vega < 1e-7:
        # Option plate / valeur temps inférieure à la précision numérique
        assert np.isnan(recovered) or np.isclose(recovered, sigma, atol=0.05)
    else:
        assert not np.isnan(recovered)
        assert np.isclose(recovered, sigma, atol=1e-3)


@pytest.mark.parametrize("moneyness", [0.75, 1.0, 1.25])
@pytest.mark.parametrize("sigma", [0.12, 0.30, 0.70])
def test_round_trip_grid_puts(moneyness, sigma):
    """Cycle aller-retour pour les Puts européens."""
    S = 100.0
    K = S * moneyness
    T = 0.75
    r = 0.02
    q = 0.0

    price = bs_put_price(S, K, T, r, q, sigma)
    recovered = bs_implied_vol(price, S, K, T, r, q, option_type="put")

    assert not np.isnan(recovered)
    assert np.isclose(recovered, sigma, atol=1e-3)


# ============================================================================
# 4. PROPRIÉTÉS DU GREC VEGA
# ============================================================================


@pytest.mark.parametrize("S", [50.0, 100.0, 150.0])
@pytest.mark.parametrize("T", [0.0, 0.01, 1.0])
@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.30])
def test_vega_positivity_and_limits(S, T, sigma):
    """Le Vega doit toujours être >= 0, et exactement 0 si T <= 0 ou sigma <= 0."""
    vega = bs_vega(S, K=100.0, T=T, r=0.03, q=0.01, sigma=sigma)
    assert vega >= 0.0
    if T <= 0.0 or sigma <= 0.0:
        assert vega == 0.0


# ============================================================================
# 5. MONOTONIE ET CONVEXITÉ EN STRIKE
# ============================================================================


def test_strike_monotonicity_and_convexity():
    """Vérifie la décroissance en K du Call et sa convexité discrète."""
    S = 100.0
    T = 1.0
    r = 0.04
    q = 0.01
    sigma = 0.25

    strikes = np.linspace(50.0, 150.0, 51)
    calls = bs_call_price(S, strikes, T, r, q, sigma)
    puts = bs_put_price(S, strikes, T, r, q, sigma)

    # 1. Monotonie : Call décroissant, Put croissant
    assert np.all(np.diff(calls) < 0.0)
    assert np.all(np.diff(puts) > 0.0)

    # 2. Convexité du Call : dérivée seconde discrète >= 0 (absence d'arbitrage papillon)
    second_diff = np.diff(calls, n=2)
    assert np.all(second_diff >= -1e-12)
